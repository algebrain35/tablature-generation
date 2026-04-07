"""
generate.py — Model-driven guitar tablature generation

Uses the trained model's pitch_head to generate the next pitch autoregressively,
then uses A* (TrellisDecoder) to assign optimal fretboard positions.

The model predicts both what to play next (pitch) and where to play it (position)
from the same fused representation — so pitch choices are conditioned on past
trajectory, pitch context, and learned musical structure from training data.

Usage:
  python generate.py --length 32
  python generate.py --start 47 --length 64 --temp 1.2
  python generate.py --key A --scale minor_pentatonic --length 48
  python generate.py --length 32 --seed 42
"""

import os, sys, argparse, random, pickle
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fretboard_transformer import (
    FretboardTransformer, TrellisDecoder,
    POSITIONS, POS_TO_IDX, pos_to_midi, midi_to_positions,
    STANDARD_TUNING, NUM_POSITIONS, NUM_MIDI,
)

TUNINGS = {
    "standard": {1:64, 2:59, 3:55, 4:50, 5:45, 6:40},
    "eb":       {1:63, 2:58, 3:54, 4:49, 5:44, 6:39},
    "dropd":    {1:64, 2:59, 3:55, 4:50, 5:45, 6:38},
    "dropc":    {1:62, 2:57, 3:53, 4:48, 5:43, 6:36},
}

NOTE_NAMES = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
NOTE_TO_PC = {n: i for i, n in enumerate(NOTE_NAMES)}
NOTE_TO_PC.update({"Db":1,"D#":3,"Gb":6,"G#":8,"A#":10,"B#":0})

SCALES = {
    "major":             [0,2,4,5,7,9,11],
    "natural_minor":     [0,2,3,5,7,8,10],
    "harmonic_minor":    [0,2,3,5,7,8,11],
    "minor_pentatonic":  [0,3,5,7,10],
    "major_pentatonic":  [0,2,4,7,9],
    "blues":             [0,3,5,6,7,10],
    "dorian":            [0,2,3,5,7,9,10],
    "phrygian":          [0,1,3,5,7,8,10],
    "mixolydian":        [0,2,4,5,7,9,10],
    "chromatic":         list(range(12)),
}

REGISTERS = {
    "low":  (40, 60),
    "mid":  (52, 72),
    "high": (64, 84),
    "full": (40, 84),
}


def parse_args():
    p = argparse.ArgumentParser(description="Model-driven guitar tablature generation")
    p.add_argument("--model",     default="fretboard_transformer.pt")
    p.add_argument("--length",    type=int,   default=32)
    p.add_argument("--start",     type=int,   default=None,
                   help="Starting MIDI pitch (default: random from register)")
    p.add_argument("--temp",      type=float, default=1.2,
                   help="Sampling temperature for pitch generation")
    p.add_argument("--key",       default=None,
                   help="Constrain pitches to scale (e.g. A). Default: unconstrained.")
    p.add_argument("--scale",     default=None,
                   choices=list(SCALES.keys()) + [None],
                   help="Scale (default: chromatic when --key given, unconstrained otherwise)")
    p.add_argument("--register",  default="mid",
                   choices=sorted(REGISTERS.keys()))
    p.add_argument("--fret_bias", type=float, default=0.05)
    p.add_argument("--seed",      type=int,   default=None)
    p.add_argument("--tuning",    default="standard",
                   choices=sorted(TUNINGS.keys()))
    p.add_argument("--notes_per_line", type=int, default=16)
    p.add_argument("--step_size", type=int, default=2,
                   help="Max scale steps per note (1=stepwise, 3=leaps)")
    p.add_argument("--markov",   default=None,
                   help="Path to Markov model (.pkl from build_markov.py)")
    p.add_argument("--random_walk", action="store_true",
                   help="Use phrase random walk instead of model pitch_head")
    p.add_argument("--top_k",      type=int, default=0,
                   help="Top-K sampling (legacy, unused in phrase mode)")
    p.add_argument("--phrase_min", type=int, default=4,
                   help="Minimum notes per phrase (default 4)")
    p.add_argument("--phrase_max", type=int, default=8,
                   help="Maximum notes per phrase (default 8)")
    p.add_argument("--wav",      default=None,
                   help="Output WAV file path (e.g. output.wav)")
    p.add_argument("--bpm",      type=float, default=120.0,
                   help="Tempo for WAV synthesis (default 120)")
    p.add_argument("--no_postprocess", action="store_true")
    return p.parse_args()


def build_scale_mask(root_pc, intervals, lo, hi):
    """Boolean mask over NUM_MIDI — True where pitch is in scale and register."""
    mask = torch.zeros(NUM_MIDI, dtype=torch.bool)
    for p in range(lo, hi + 1):
        if (p - root_pc) % 12 in set(intervals):
            mask[p] = True
    return mask


# ── Model-driven pitch generation ────────────────────────────────────────────

@torch.no_grad()
def generate_pitches_model(model, scale_mask, length, start_pitch,
                            temperature=1.0, tonic_pc=None, tonic_weight=2.0,
                            device='cpu', repeat_penalty=0.1, repeat_window=4):
    """
    Generate pitches autoregressively using the model's pitch_head.

    repeat_penalty : weight multiplier for pitches seen in last repeat_window
                     steps (default 0.1 = 10× less likely to repeat)
    repeat_window  : how many recent pitches to penalise
    """
    model.eval()

    stable_pcs = set()
    if tonic_pc is not None:
        stable_pcs = {tonic_pc % 12,
                      (tonic_pc + 3) % 12, (tonic_pc + 4) % 12,
                      (tonic_pc + 7) % 12}

    def stability(p):
        return tonic_weight if (p % 12 in stable_pcs) else 1.0

    pitches   = [start_pitch]
    positions = []

    for t in range(1, length):
        pit_t   = torch.tensor(pitches, dtype=torch.long, device=device)
        pit_ctx = model.encode_pitches(pit_t)

        log_probs = model.decode_pitch_step(pit_ctx[-1], positions, t - 1)

        # Temperature
        lp = log_probs / max(temperature, 1e-6)

        # Scale mask
        if scale_mask is not None:
            lp = lp.masked_fill(~scale_mask.to(device), float('-inf'))

        # Tonal stability bonus
        if stable_pcs:
            bonus = torch.zeros_like(lp)
            for pc in stable_pcs:
                for octave in range(11):
                    p = pc + 12 * octave
                    if p < len(bonus):
                        bonus[p] += torch.log(torch.tensor(tonic_weight))
            lp = lp + bonus

        # Hard blacklist: zero out recently seen pitches entirely
        # This forces movement regardless of how confident the model is
        if repeat_window > 0:
            recent = set(pitches[-repeat_window:])
            for p in recent:
                if 0 <= p < len(lp):
                    lp[p] = float('-inf')

        # Safety: if everything is -inf (e.g. scale has only 1 pitch),
        # release the blacklist and just use temperature
        if torch.all(lp == float('-inf')):
            lp = log_probs / max(temperature, 1e-6)
            if scale_mask is not None:
                lp = lp.masked_fill(~scale_mask.to(device), float('-inf'))

        probs = torch.softmax(lp, dim=-1)

        if probs.isnan().any() or probs.sum() < 1e-9:
            valid = [p for p in range(128) if scale_mask is None or scale_mask[p]]
            next_pitch = random.choice(valid) if valid else pitches[-1]
        else:
            next_pitch = torch.multinomial(probs, 1).item()

        pitches.append(next_pitch)

    return pitches




def load_markov(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def markov_backoff(counts, context, order):
    """Katz backoff: try full context, fall back to shorter prefixes."""
    for length in range(order, 0, -1):
        ctx = tuple(context[-length:])
        if ctx in counts:
            return counts[ctx]
    return None


def generate_pitches_markov(model_data, scale_mask, length, start_pitch,
                             temperature=1.0, tonic_pc=None, tonic_weight=2.0,
                             repeat_penalty=0.1, repeat_window=4):
    """
    Generate pitches using the Markov chain, constrained to scale_mask.

    repeat_penalty : multiply weight of any pitch that appeared in the last
                     repeat_window notes (default 0.1 = strong suppression)
    repeat_window  : how many recent notes to penalise repetitions over
    """
    counts   = model_data["counts"]
    unigrams = model_data["unigrams"]
    order    = model_data["order"]

    stable_pcs = set()
    if tonic_pc is not None:
        stable_pcs = {tonic_pc % 12,
                      (tonic_pc + 3) % 12, (tonic_pc + 4) % 12,
                      (tonic_pc + 7) % 12}

    def stability(p):
        return tonic_weight if (p % 12 in stable_pcs) else 1.0

    context = [start_pitch] * order
    pitches = [start_pitch]

    for _ in range(length - 1):
        dist = markov_backoff(counts, context, order)
        if dist is None:
            dist = unigrams

        recent = set(pitches[-repeat_window:]) if repeat_window > 0 else set()

        candidates = []
        weights    = []
        for p, cnt in dist.items():
            if scale_mask is not None and not scale_mask[p]:
                continue
            w = (cnt ** (1.0 / max(temperature, 0.1))) * stability(p)
            if p in recent:
                w *= repeat_penalty
            candidates.append(p)
            weights.append(w)

        if not candidates:
            for p, cnt in dist.items():
                w = (cnt ** (1.0 / max(temperature, 0.1))) * stability(p)
                if p in recent:
                    w *= repeat_penalty
                candidates.append(p)
                weights.append(w)

        if not candidates:
            next_p = context[-1]
        else:
            total  = sum(weights)
            next_p = random.choices(candidates, weights=[w/total for w in weights])[0]

        pitches.append(next_p)
        context.append(next_p)

    return pitches





def pick_target(valid, stability, current_idx, contour, phrase_len):
    """
    Choose a melodic target for the phrase end based on contour.
    Returns an index into `valid`.
    """
    n = len(valid)
    # How far to travel — roughly proportional to phrase length
    reach = max(1, phrase_len // 2)

    if contour in ("ascending", "run_up"):
        lo = current_idx + 1
        hi = min(n - 1, current_idx + reach + 2)
    elif contour in ("descending", "run_down"):
        lo = max(0, current_idx - reach - 2)
        hi = current_idx - 1
    elif contour == "arch":
        # Peak somewhere above, resolve back near start
        lo = current_idx
        hi = min(n - 1, current_idx + reach)
    elif contour == "valley":
        lo = max(0, current_idx - reach)
        hi = current_idx
    else:  # flat
        lo = max(0, current_idx - 1)
        hi = min(n - 1, current_idx + 1)

    if lo > hi:
        lo, hi = hi, lo
    lo = max(0, lo)
    hi = min(n - 1, hi)

    # Among candidates, prefer stable tones
    candidates = list(range(lo, hi + 1))
    if not candidates:
        return current_idx
    weights = [stability[i] for i in candidates]
    total = sum(weights)
    return random.choices(candidates, weights=[w/total for w in weights])[0]


def generate_phrase(valid, stability, start_idx, length, contour, step_size, temperature):
    """
    Generate a single phrase that moves from start_idx toward a chosen target.

    The phrase is built in two stages for arch/valley contours:
      - Stage 1: move toward the peak/trough
      - Stage 2: move toward the resolution

    For linear contours (ascending, descending, run_up, run_down):
      - Move steadily toward the target, one step at a time

    For flat:
      - Ornament around the current pitch

    Every step is a weighted sample that favours moving toward the target
    while allowing occasional decorative steps in the opposite direction.
    """
    n       = len(valid)
    phrase  = [valid[start_idx]]
    idx     = start_idx

    if length <= 1:
        return phrase, idx

    if contour == "arch":
        # Split: first half ascends to peak, second half descends to resolution
        mid       = length // 2
        peak_idx  = pick_target(valid, stability, idx, "ascending", mid)
        end_idx   = pick_target(valid, stability, peak_idx, "descending", length - mid)
        waypoints = [(peak_idx, mid), (end_idx, length - mid)]
    elif contour == "valley":
        mid       = length // 2
        trough    = pick_target(valid, stability, idx, "descending", mid)
        end_idx   = pick_target(valid, stability, trough, "ascending", length - mid)
        waypoints = [(trough, mid), (end_idx, length - mid)]
    else:
        target = pick_target(valid, stability, idx, contour, length)
        waypoints = [(target, length)]

    pos = 1
    for (target_idx, segment_len) in waypoints:
        remaining_in_seg = segment_len - (1 if waypoints[0][1] == segment_len else 0)
        for step in range(remaining_in_seg):
            if pos >= length:
                break
            remaining = length - pos
            dist_to_target = target_idx - idx   # positive = need to go up

            # Build candidates with strong directional pull toward target
            candidates = []
            weights_c  = []
            for delta in range(-step_size, step_size + 1):
                if delta == 0:
                    continue
                ni = idx + delta
                if ni < 0 or ni >= n:
                    continue
                # Base weight by proximity
                w = max(0.1, 1.0 / (abs(delta) ** (1.0 / max(temperature, 0.1))))
                # Strong pull toward target direction
                if dist_to_target != 0:
                    aligned = (delta > 0) == (dist_to_target > 0)
                    w *= 4.0 if aligned else 0.5
                # Stability bonus
                w *= stability[ni]
                candidates.append(ni)
                weights_c.append(w)

            # On last note of phrase, snap to target
            if pos == length - 1:
                idx = target_idx
            elif candidates:
                total = sum(weights_c)
                idx = random.choices(candidates, weights=[w/total for w in weights_c])[0]

            phrase.append(valid[idx])
            pos += 1

    return phrase[:length], idx


def generate_pitches(scale_mask, length, start_pitch, step_size=2, top_k=0,
                     temperature=1.8, tonic_pc=None, tonic_weight=2.0,
                     phrase_len_range=(4, 8)):
    """
    Phrase-level pitch generation with target-directed melodic movement.

    Each phrase:
      1. Picks a contour (ascending, descending, arch, valley, flat, run)
      2. Picks a target pitch consistent with that contour
      3. Walks toward the target, preferring steps that close the gap
      4. Snaps to the target on the final note of the phrase

    Phrases chain together — the next phrase starts from the last note
    of the previous one, giving continuity across the full sequence.
    """
    valid = [p for p in range(128) if scale_mask[p]]
    if not valid:
        raise ValueError("No valid pitches in scale+register combination.")

    # Stability weights
    stability = [1.0] * len(valid)
    if tonic_pc is not None:
        stable_pcs = {tonic_pc % 12,
                      (tonic_pc + 3) % 12, (tonic_pc + 4) % 12,
                      (tonic_pc + 7) % 12}
        for i, p in enumerate(valid):
            if p % 12 in stable_pcs:
                stability[i] = tonic_weight

    idx       = min(range(len(valid)), key=lambda i: abs(valid[i] - start_pitch))
    pitches   = []
    remaining = length

    while remaining > 0:
        plen    = min(remaining, random.randint(*phrase_len_range))
        contour = random.choice(CONTOURS)
        phrase, idx = generate_phrase(valid, stability, idx, plen,
                                      contour, step_size, temperature)
        pitches.extend(phrase)
        remaining -= len(phrase)

    return pitches[:length]



def render_ascii(decoded, notes_per_line=16):
    STRING_LABELS = {1:"e", 2:"B", 3:"G", 4:"D", 5:"A", 6:"E"}
    lines = []
    for start in range(0, len(decoded), notes_per_line):
        chunk = decoded[start : start + notes_per_line]
        rows  = {s: STRING_LABELS[s] + "|" for s in range(1, 7)}
        for note in chunk:
            s, fret = note["string"], note["fret"]
            col = str(fret).ljust(3)
            for string in range(1, 7):
                rows[string] += col if string == s else "-" * len(col)
        for s in rows:
            rows[s] += "|"
        lines.append("")
        for s in range(1, 7):
            lines.append(rows[s])
    return "\n".join(lines)


def karplus_strong(freq, duration, sr=44100, decay=0.996, brightness=0.5):
    """
    Karplus-Strong plucked string synthesis.
    decay      : feedback coefficient (0.99=long sustain, 0.97=short)
    brightness : initial noise blend (0=dark, 1=bright)
    Returns (sr * duration,) float32 array normalised to [-1, 1].
    """
    import numpy as np
    n_samples   = int(sr * duration)
    buf_len     = max(2, int(sr / freq))
    # Excitation: short burst of noise
    buf         = np.random.uniform(-1, 1, buf_len).astype(np.float64)
    out         = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        out[i]      = buf[i % buf_len]
        # Low-pass filter + decay in feedback loop
        next_sample = decay * ((1 - brightness) * buf[i % buf_len]
                               + brightness    * buf[(i + 1) % buf_len])
        buf[i % buf_len] = next_sample
    # Normalise
    peak = np.max(np.abs(out))
    return (out / peak).astype(np.float32) if peak > 1e-9 else out.astype(np.float32)


def synthesize_wav(decoded, tuning, bpm=120.0, sr=44100,
                   note_dur_beats=0.5, gap_beats=0.02):
    """
    Synthesize decoded tablature to a numpy audio array using Karplus-Strong.
    Each note is note_dur_beats long at the given BPM, with a brief gap between.
    Returns (N,) float32 array.
    """
    import numpy as np
    beat_sec     = 60.0 / bpm
    note_sec     = note_dur_beats * beat_sec
    gap_sec      = gap_beats * beat_sec
    step_samples = int((note_sec + gap_sec) * sr)
    tail_samples = int(note_sec * sr * 3)          # generous tail for sustain
    total        = step_samples * len(decoded) + tail_samples
    audio        = np.zeros(total, dtype=np.float32)

    for i, note in enumerate(decoded):
        midi  = tuning[note["string"]] + note["fret"]
        freq  = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        grain = karplus_strong(freq, note_sec * 2.0, sr=sr)  # longer grain for sustain
        start = i * step_samples
        end   = start + len(grain)
        if end > len(audio):
            end   = len(audio)
            grain = grain[:end - start]
        audio[start:end] += grain

    # Final normalise
    peak = np.max(np.abs(audio))
    if peak > 1e-9:
        audio = audio / peak * 0.9
    return audio


def save_wav(audio, path, sr=44100):
    """Write float32 audio array to a 16-bit PCM WAV file."""
    import numpy as np
    try:
        from scipy.io import wavfile
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        wavfile.write(path, sr, pcm)
    except ImportError:
        # Fallback: manual WAV writer
        import struct, wave
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
    print(f"WAV saved : {path}  ({len(audio)/sr:.1f}s  {sr}Hz mono 16-bit PCM)")
    print(f"Play with : aplay -r {sr} -f S16_LE -c 1 {path}")





def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tuning = TUNINGS[args.tuning]

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if not os.path.exists(args.model):
        print(f"[ERROR] Checkpoint not found: {args.model}")
        sys.exit(1)

    model = FretboardTransformer(dropout=0.0)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Model : {args.model}  ({sum(p.numel() for p in model.parameters()):,} params)")

    # ── Scale mask ────────────────────────────────────────────────────────────
    scale_mask = None
    lo, hi = REGISTERS[args.register]
    if args.key:
        root_pc = NOTE_TO_PC.get(args.key)
        if root_pc is None:
            print(f"[ERROR] Unknown key '{args.key}'")
            sys.exit(1)
        scale_name = args.scale if args.scale else "chromatic"
        intervals  = SCALES[scale_name]
        scale_mask = build_scale_mask(root_pc, intervals, lo, hi)
        print(f"Key   : {args.key} {scale_name}  register={args.register}")
    else:
        # No key — all pitches in register
        scale_mask = torch.zeros(NUM_MIDI, dtype=torch.bool)
        scale_mask[lo:hi+1] = True
        print(f"Key   : unconstrained  register={args.register}")

    # ── Start pitch ───────────────────────────────────────────────────────────
    if args.start is not None:
        start_pitch = args.start
    else:
        valid = [p for p in range(lo, hi + 1)
                 if scale_mask is None or scale_mask[p]]
        start_pitch = random.choice(valid) if valid else (lo + hi) // 2

    name = NOTE_NAMES[start_pitch % 12] + str(start_pitch // 12 - 1)
    print(f"Start : MIDI {start_pitch}  ({name})")
    print(f"Length: {args.length}  temp={args.temp}")

    # ── Generate pitches ──────────────────────────────────────────────────────
    step_size = getattr(args, 'step_size', 2)
    tonic_pc  = root_pc if args.key else None

    if args.markov:
        if not os.path.exists(args.markov):
            print(f"[ERROR] Markov model not found: {args.markov}")
            print("  Run: python build_markov.py")
            sys.exit(1)
        markov_data = load_markov(args.markov)
        print(f"Pitch : Markov order={markov_data['order']}  "
              f"{len(markov_data['counts']):,} contexts")
        pitches = generate_pitches_markov(
            markov_data, scale_mask, args.length, start_pitch,
            temperature=args.temp, tonic_pc=tonic_pc, tonic_weight=2.0)
    elif args.random_walk:
        print(f"Pitch : random walk (phrase mode)")
        pitches = generate_pitches(scale_mask, args.length, start_pitch,
                                   step_size=step_size,
                                   top_k=args.top_k,
                                   temperature=args.temp,
                                   tonic_pc=tonic_pc,
                                   phrase_len_range=(args.phrase_min,
                                                     args.phrase_max))
    else:
        print(f"Pitch : model pitch_head (temp={args.temp})")
        pitches = generate_pitches_model(
            model, scale_mask, args.length, start_pitch,
            temperature=args.temp, tonic_pc=tonic_pc,
            tonic_weight=2.0, device=device)

    # ── Assign fretboard positions via A* ─────────────────────────────────────
    decoder = TrellisDecoder(model, tuning=tuning, fret_bias=args.fret_bias)
    decoded = decoder.decode(pitches)

    if not args.no_postprocess:
        decoded, n_fixed = decoder.postprocess(decoded)
        if n_fixed:
            print(f"Post-processing: {n_fixed} note(s) relocated")

    # ── Display ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 56)
    print(render_ascii(decoded, args.notes_per_line))
    print("─" * 56)

    print(f"\n{'#':>3}  {'Str':>3}  {'Fret':>4}  {'MIDI':>4}  Note")
    print("─" * 28)
    for i, note in enumerate(decoded):
        pitch = tuning[note["string"]] + note["fret"]
        nname = NOTE_NAMES[pitch % 12] + str(pitch // 12 - 1)
        print(f"{i+1:>3}  str {note['string']}  fret {note['fret']:>2}  "
              f"MIDI {pitch:>3}  {nname}{'  open' if note['is_open'] else ''}")

    print(f"\nPitch sequence (MIDI):")
    print(",".join(str(p) for p in pitches))

    # ── WAV synthesis ─────────────────────────────────────────────────────────
    if args.wav:
        audio = synthesize_wav(decoded, tuning, bpm=args.bpm)
        save_wav(audio, args.wav)


if __name__ == "__main__":
    main()

