"""
generate_diffusion.py — Masked-diffusion guitar tablature generation

Uses iterative unmasking (discrete masked diffusion) to assign fretboard
positions to a pitch sequence. The model sees all pitches bidirectionally
at every step, revealing the most confident position assignments first and
refining the rest in subsequent steps.

If no masked-LM checkpoint is found, the script can train one automatically
from the same data as the AR model, saving to fretboard_transformer_mlm.pt.

Usage:
    # Generate with existing MLM checkpoint
    python generate_diffusion.py --key E --scale minor_pentatonic --length 64

    # Train MLM model first (if fretboard_transformer_mlm.pt missing)
    python generate_diffusion.py --train --length 64

    # Same flags as generate.py
    python generate_diffusion.py --key A --scale blues --length 48 --temp 1.2 --seed 42
    python generate_diffusion.py --random_walk --key G --scale natural_minor --length 32
    python generate_diffusion.py --markov markov.pkl --key D --scale dorian --length 64

Diffusion parameters:
    --steps     Number of unmasking steps (default 10; more = slower but smoother)
    --anneal    Temperature annealing schedule: flat | cosine | linear (default cosine)
    --order     Token reveal order: confidence | random | left_right (default confidence)

Patches applied (vs original):
    1. CachedMLMModel     — pos_enc output cached per diffusion step; ~250x fewer
                            transformer calls for T=64, steps=10.
    2. Velocity dynamics  — vel_head scores velocity tokens during diffusion;
                            Karplus-Strong excitation scaled by amplitude proportional
                            to vel^0.6; string-dependent decay (low-E sustains longer).
    3. Pitch temperature  — separate softer annealing schedule for pitch tokens
                            (floor 0.5, not 0.1) preserves late melodic variety.
"""

import os, sys, argparse, random, pickle, math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fretboard_transformer import (
    FretboardTransformerMLM, FretboardTrainer,
    POSITIONS, POS_TO_IDX, pos_to_midi, midi_to_positions,
    STANDARD_TUNING, NUM_POSITIONS, NUM_MIDI,
)

# ── Constants shared with generate.py ────────────────────────────────────────

TUNINGS = {
    "standard": {1:64, 2:59, 3:55, 4:50, 5:45, 6:40},
    "eb":       {1:63, 2:58, 3:54, 4:49, 5:44, 6:39},
    "dropd":    {1:64, 2:59, 3:55, 4:50, 5:45, 6:38},
    "dropc":    {1:62, 2:57, 3:53, 4:48, 5:43, 6:36},
}

NOTE_NAMES  = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
NOTE_TO_PC  = {n: i for i, n in enumerate(NOTE_NAMES)}
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

CONTOURS = ["ascending","descending","arch","valley","flat","run_up","run_down"]

MLM_CHECKPOINT = "fretboard_transformer_mlm.pt"

# Velocity bin -> amplitude scale.  Bin 0 = unknown -> treated as mf (0.6).
# Perceptual curve: amplitude proportional to vel^0.6 (log loudness perception).
_VEL_BIN_AMP = [0.6] + [(b / 8) ** 0.6 for b in range(1, 9)]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Masked-diffusion guitar tablature generation")
    # Model / training
    p.add_argument("--model",      default=MLM_CHECKPOINT)
    p.add_argument("--train",      action="store_true",
                   help="Train a masked-LM model if checkpoint is missing or forced")
    p.add_argument("--force_train",action="store_true",
                   help="Force retrain even if checkpoint exists")
    p.add_argument("--train_epochs", type=int, default=20)
    p.add_argument("--scraped_tabs_dir", default=None,
                   help="Scraped GuitarPro tabs dir (e.g. ./assets/data/gprotab/)")
    p.add_argument("--cache_path", default="dataset_cache.pkl",
                   help="Dataset cache path -- pass full .v8 path to skip rebuild")
    # Generation
    p.add_argument("--length",     type=int,   default=32)
    p.add_argument("--start",      type=int,   default=None)
    p.add_argument("--temp",       type=float, default=1.0)
    p.add_argument("--key",        default=None)
    p.add_argument("--scale",      default=None, choices=list(SCALES.keys()))
    p.add_argument("--register",   default="mid", choices=sorted(REGISTERS.keys()))
    p.add_argument("--tuning",     default="standard", choices=sorted(TUNINGS.keys()))
    p.add_argument("--seed",       type=int,   default=None)
    p.add_argument("--fret_bias",  type=float, default=0.05)
    p.add_argument("--notes_per_line", type=int, default=16)
    # Pitch generation modes (mirrors generate.py)
    p.add_argument("--markov",     default=None,
                   help="Markov model .pkl from build_markov.py")
    p.add_argument("--random_walk",action="store_true",
                   help="Phrase random walk instead of model pitch_head")
    p.add_argument("--repeat_penalty", type=float, default=0.1)
    p.add_argument("--repeat_window",  type=int,   default=4)
    p.add_argument("--phrase_min", type=int, default=4)
    p.add_argument("--phrase_max", type=int, default=8)
    p.add_argument("--step_size",  type=int, default=2)
    # Diffusion-specific
    p.add_argument("--steps",      type=int,   default=10,
                   help="Unmasking steps (default 10)")
    p.add_argument("--anneal",     default="cosine",
                   choices=["flat","cosine","linear"],
                   help="Temperature annealing over steps (default cosine)")
    p.add_argument("--order",      default="confidence",
                   choices=["confidence","random","left_right"],
                   help="Token reveal order (default confidence)")
    # Audio
    p.add_argument("--wav",        default=None)
    p.add_argument("--bpm",        type=float, default=120.0)
    p.add_argument("--note_dur",   type=float, default=0.5,
                   help="Default note duration in beats when no duration bins available (default 0.5)")
    p.add_argument("--no_postprocess", action="store_true")
    return p.parse_args()


# =============================================================================
# PATCH 1 -- CachedMLMModel
# Wraps FretboardTransformerMLM and caches pos_enc output keyed on the
# current (positions, durations) state tuple.  The pos_enc transformer is
# the expensive redundant computation during masked diffusion: with T=64,
# 10 steps, and ~40 masked tokens per step the naive code runs pos_enc
# ~2,500 times.  After caching it runs ~10 times (once per step).
# =============================================================================

class CachedMLMModel:
    """
    Transparent wrapper around FretboardTransformerMLM that caches the
    position encoder (pos_enc) output keyed on the full position+duration
    state tuple.

    All existing method calls are proxied to the wrapped model unchanged,
    so this is a drop-in replacement anywhere the model is used.
    """

    def __init__(self, model):
        self._model = model
        self._cache = {}

    def __getattr__(self, name):
        return getattr(self._model, name)

    def clear_cache(self):
        """Call at the start of each diffusion step when tokens are revealed."""
        self._cache.clear()

    @torch.no_grad()
    def _get_pos_enc(self, cur_positions, cur_durations, device):
        """
        Return pos_enc output for the given state, computing and caching
        on first call.  Cache hit rate within a single diffusion step is
        100% -- all masked-token scoring calls share the same state.
        """
        key = (tuple(cur_positions), tuple(cur_durations))
        if key not in self._cache:
            pos_t = torch.tensor(
                cur_positions, dtype=torch.long, device=device).unsqueeze(0)
            traj  = self._model.pos_enc(pos_t)       # (1, T, D)
            self._cache[key] = traj.squeeze(0)        # (T, D)
        return self._cache[key]

    @torch.no_grad()
    def decode_joint_step(self, all_pitches, all_positions, target_t,
                          all_durations=None):
        """
        Cached decode_joint_step -- identical interface to the original.
        pos_enc is looked up from cache; pitch_enc is still computed per
        call since its inputs (cur_pitches) change as pitch tokens are revealed.
        """
        model    = self._model
        model.eval()
        dev      = next(model.parameters()).device
        T        = len(all_pitches)
        dur_list = all_durations if all_durations is not None else [0] * T

        pit_t     = torch.tensor(all_pitches, dtype=torch.long, device=dev).unsqueeze(0)
        dur_t     = torch.tensor(dur_list,    dtype=torch.long, device=dev).unsqueeze(0)
        pitch_ctx = model.pitch_enc(pit_t, durations=dur_t)          # (1, T, D)

        traj_ctx  = self._get_pos_enc(all_positions, dur_list, dev)  # (T, D) -- cached

        p_ctx  = pitch_ctx[0, target_t]    # (D,)
        t_ctx  = traj_ctx[target_t]        # (D,)
        fused  = model.fusion(torch.cat([p_ctx, t_ctx], dim=-1).unsqueeze(0))  # (1, D)

        pit_lp = F.log_softmax(model.pitch_head(fused), dim=-1).squeeze(0)
        pos_lp = F.log_softmax(model.head(fused),       dim=-1).squeeze(0)
        dur_lp = F.log_softmax(model.dur_head(fused),   dim=-1).squeeze(0)
        return pit_lp, pos_lp, dur_lp

    @torch.no_grad()
    def decode_step_bidirectional(self, pitch_ctx, all_positions, target_t):
        """Cached decode_step_bidirectional."""
        model = self._model
        model.eval()
        dev   = pitch_ctx.device
        traj  = self._get_pos_enc(all_positions, [0] * len(all_positions), dev)
        p_ctx = pitch_ctx[target_t] if pitch_ctx.dim() == 2 else pitch_ctx
        fused = model.fusion(
            torch.cat([p_ctx, traj[target_t]], dim=-1).unsqueeze(0))
        return F.log_softmax(model.head(fused), dim=-1).squeeze(0)


# ── Shared scale / pitch utilities ────────────────────────────────────────────

def build_scale_mask(root_pc, intervals, lo, hi):
    mask = torch.zeros(NUM_MIDI, dtype=torch.bool)
    for p in range(lo, hi + 1):
        if (p - root_pc) % 12 in set(intervals):
            mask[p] = True
    return mask


def build_stability_bonus(tonic_pc, tonic_weight=2.0):
    if tonic_pc is None:
        return None
    stable = {tonic_pc%12, (tonic_pc+3)%12, (tonic_pc+4)%12, (tonic_pc+7)%12}
    bonus  = torch.zeros(NUM_MIDI)
    logw   = math.log(tonic_weight)
    for p in range(NUM_MIDI):
        if p % 12 in stable:
            bonus[p] = logw
    return bonus


def sample_pitch(lp, temperature, scale_mask, stability_bonus, recent, repeat_window):
    """Hard-blacklist repeat penalty + temperature + scale + tonal stability."""
    lp = lp / max(temperature, 1e-6)
    if scale_mask is not None:
        lp = lp.masked_fill(~scale_mask.to(lp.device), float('-inf'))
    if stability_bonus is not None:
        lp = lp + stability_bonus.to(lp.device)
    if recent and repeat_window > 0:
        for p in list(recent)[-repeat_window:]:
            if 0 <= p < len(lp):
                lp[p] = float('-inf')
    if torch.all(lp == float('-inf')):
        lp = torch.zeros_like(lp)
        if scale_mask is not None:
            lp = lp.masked_fill(~scale_mask.to(lp.device), float('-inf'))
    probs = torch.softmax(lp, dim=-1)
    if probs.isnan().any() or probs.sum() < 1e-9:
        valid = [i for i in range(NUM_MIDI)
                 if scale_mask is None or scale_mask[i]]
        return random.choice(valid) if valid else 60
    return torch.multinomial(probs, 1).item()


# ── Pitch generation (mirrors generate.py) ───────────────────────────────────

@torch.no_grad()
def generate_pitches_model(model, scale_mask, length, start_pitch,
                           temperature, tonic_pc, device,
                           repeat_penalty, repeat_window, tuning,
                           ctx_window=64):
    """
    Autoregressive pitch_head generation with interleaved greedy positions.
    Mirrors generate.py -- defined here to avoid version mismatch on import.
    """
    model.eval()
    stability_bonus = build_stability_bonus(tonic_pc)

    pitches   = [start_pitch]
    positions = []

    pit_t     = torch.tensor(pitches, dtype=torch.long, device=device)
    pitch_ctx = model.encode_pitches(pit_t)
    cands0    = [POS_TO_IDX[p] for p in midi_to_positions(start_pitch, tuning)
                 if p in POS_TO_IDX]
    if cands0:
        all_pos = [model.MASK_IDX] * 1
        pos_lp  = model.decode_step_bidirectional(pitch_ctx[:1], all_pos, 0)
        positions.append(max(cands0, key=lambda c: pos_lp[c].item()))
    else:
        positions.append(FretboardTransformerMLM.MASK_IDX)

    for t in range(1, length):
        win_start = max(0, len(pitches) - ctx_window)
        pit_win   = torch.tensor(pitches[win_start:], dtype=torch.long, device=device)
        pitch_ctx = model.encode_pitches(pit_win)
        ctx_idx   = pitch_ctx.shape[0] - 1
        pos_win   = positions[win_start:]

        log_probs = model.decode_pitch_step(pitch_ctx[ctx_idx], pos_win, ctx_idx)
        next_pitch = sample_pitch(
            log_probs.clone(), temperature, scale_mask, stability_bonus,
            pitches, repeat_window)
        pitches.append(next_pitch)

        pit_new   = torch.tensor(pitches[win_start:], dtype=torch.long, device=device)
        pitch_ctx = model.encode_pitches(pit_new)
        cands     = [POS_TO_IDX[p] for p in midi_to_positions(next_pitch, tuning)
                     if p in POS_TO_IDX]
        if cands:
            all_pos = positions[win_start:] + [model.MASK_IDX]
            pos_lp  = model.decode_step_bidirectional(
                pitch_ctx, all_pos, pitch_ctx.shape[0] - 1)
            positions.append(max(cands, key=lambda c: pos_lp[c].item()))
        else:
            positions.append(model.MASK_IDX)

    return pitches


def load_markov(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def markov_backoff(counts, context, order):
    for length in range(order, 0, -1):
        ctx = tuple(context[-length:])
        if ctx in counts:
            return counts[ctx]
    return None


def generate_pitches_markov(model_data, scale_mask, length, start_pitch,
                             temperature, tonic_pc, tonic_weight,
                             repeat_penalty, repeat_window):
    counts   = model_data["counts"]
    unigrams = model_data["unigrams"]
    order    = model_data["order"]
    stable   = set()
    if tonic_pc is not None:
        stable = {tonic_pc%12,(tonic_pc+3)%12,(tonic_pc+4)%12,(tonic_pc+7)%12}
    stability = lambda p: tonic_weight if p%12 in stable else 1.0
    context  = [start_pitch] * order
    pitches  = [start_pitch]
    for _ in range(length - 1):
        dist = markov_backoff(counts, context, order) or unigrams
        recent = set(pitches[-repeat_window:]) if repeat_window > 0 else set()
        cands, weights = [], []
        for p, cnt in dist.items():
            if scale_mask is not None and not scale_mask[p]:
                continue
            w = cnt ** (1.0 / max(temperature, 0.1)) * stability(p)
            if p in recent:
                w *= repeat_penalty
            cands.append(p); weights.append(w)
        if not cands:
            cands = list(dist.keys()); weights = [1.0] * len(cands)
        total  = sum(weights) or 1.0
        next_p = random.choices(cands, weights=[w/total for w in weights])[0]
        pitches.append(next_p)
        context.append(next_p)
    return pitches


def pick_target(valid, stab, idx, contour, plen):
    n = len(valid); reach = max(1, plen // 2)
    if contour in ("ascending","run_up"):      lo,hi = idx+1, min(n-1,idx+reach+2)
    elif contour in ("descending","run_down"): lo,hi = max(0,idx-reach-2), idx-1
    elif contour == "arch":                    lo,hi = idx, min(n-1,idx+reach)
    elif contour == "valley":                  lo,hi = max(0,idx-reach), idx
    else:                                      lo,hi = max(0,idx-1), min(n-1,idx+1)
    if lo > hi: lo,hi = hi,lo
    lo,hi = max(0,lo), min(n-1,hi)
    cands = list(range(lo,hi+1)) or [idx]
    ws = [stab[i] for i in cands]; total = sum(ws) or 1.0
    return random.choices(cands, weights=[w/total for w in ws])[0]


def generate_phrase(valid, stab, idx, length, contour, step_size, temperature):
    n = len(valid); phrase = [valid[idx]]
    if length <= 1: return phrase, idx
    if contour == "arch":
        mid = length // 2
        peak = pick_target(valid,stab,idx,"ascending",mid)
        end  = pick_target(valid,stab,peak,"descending",length-mid)
        waypoints = [(peak,mid),(end,length-mid)]
    elif contour == "valley":
        mid    = length // 2
        trough = pick_target(valid,stab,idx,"descending",mid)
        end    = pick_target(valid,stab,trough,"ascending",length-mid)
        waypoints = [(trough,mid),(end,length-mid)]
    else:
        waypoints = [(pick_target(valid,stab,idx,contour,length),length)]
    pos = 1
    for (tgt,seg_len) in waypoints:
        steps_in_seg = seg_len - (1 if len(waypoints)>1 and waypoints[0][1]==seg_len else 0)
        for _ in range(steps_in_seg):
            if pos >= length: break
            dist = tgt - idx
            cands, ws = [], []
            for d in range(-step_size, step_size+1):
                if d == 0: continue
                ni = idx + d
                if ni < 0 or ni >= n: continue
                w = max(0.1, 1.0/(abs(d)**(1.0/max(temperature,0.1))))
                if dist != 0: w *= 4.0 if (d>0)==(dist>0) else 0.5
                w *= stab[ni]; cands.append(ni); ws.append(w)
            if pos == length-1: idx = tgt
            elif cands:
                total = sum(ws) or 1.0
                idx = random.choices(cands, weights=[w/total for w in ws])[0]
            phrase.append(valid[idx]); pos += 1
    return phrase[:length], idx


def generate_pitches_walk(scale_mask, length, start_pitch, step_size,
                           temperature, tonic_pc, tonic_weight, phrase_len_range):
    valid = [p for p in range(128) if scale_mask[p]]
    if not valid: raise ValueError("No valid pitches in scale+register.")
    stable = set()
    if tonic_pc is not None:
        stable = {tonic_pc%12,(tonic_pc+3)%12,(tonic_pc+4)%12,(tonic_pc+7)%12}
    stab = [tonic_weight if valid[i]%12 in stable else 1.0 for i in range(len(valid))]
    idx = min(range(len(valid)), key=lambda i: abs(valid[i]-start_pitch))
    pitches, remaining = [], length
    while remaining > 0:
        plen = min(remaining, random.randint(*phrase_len_range))
        phrase, idx = generate_phrase(valid,stab,idx,plen,
                                       random.choice(CONTOURS),step_size,temperature)
        pitches.extend(phrase); remaining -= len(phrase)
    return pitches[:length]


# ── Temperature schedules ─────────────────────────────────────────────────────

def annealed_temp(base_temp, step, total_steps, schedule):
    """
    Position/duration/velocity temperature -- anneals aggressively to near-
    deterministic selection (floor 0.1) so final assignments are the most
    playable/confident choices.
    """
    if total_steps <= 1 or schedule == "flat":
        return base_temp
    t = step / (total_steps - 1)
    if schedule == "linear":
        factor = 1.0 - 0.5 * t
    else:  # cosine
        factor = 0.5 * (1.0 + math.cos(math.pi * t))
    return max(0.1, base_temp * factor)


def annealed_temp_pitch(base_temp, step, total_steps, schedule):
    """
    PATCH 3 -- Pitch-specific temperature schedule.

    Softer anneal (floor 0.5, not 0.1) so late-revealed pitch tokens keep
    melodic variety.  The last few pitch tokens complete phrase endings where
    surprise is musically valuable -- forcing them to argmax produces stiff,
    repetitive cadences.
    """
    if total_steps <= 1 or schedule == "flat":
        return base_temp
    t = step / (total_steps - 1)
    if schedule == "linear":
        factor = 1.0 - 0.3 * t           # 1.0 -> 0.7
    else:
        factor = 0.5 * (1.0 + math.cos(math.pi * t * 0.7))  # slower cosine
    return max(0.5, base_temp * factor)   # floor at 0.5


# ── vel_head bootstrap ────────────────────────────────────────────────────────

def _ensure_vel_head(model, device):
    """
    Add vel_head to the inner model if absent.  Starts biased toward
    mezzo-forte (bin 4) so output is immediately usable.

    To train it properly, add a vel_loss term alongside dur_loss in the
    ft_data.py training loop:
        vel_loss = F.nll_loss(vel_logits[masked_v], velocities[masked_v])
        loss = pos_loss + 0.2*pit_loss + 0.3*dur_loss + 0.1*vel_loss
    """
    inner = model._model if isinstance(model, CachedMLMModel) else model
    if not hasattr(inner, 'vel_head'):
        embed_dim = inner.embed_dim
        inner.vel_head = nn.Linear(embed_dim, 9).to(device)  # bins 0-8
        nn.init.zeros_(inner.vel_head.weight)
        inner.vel_head.bias.data[4] = 2.0
        print("[INFO] vel_head not in checkpoint -- using mf-biased init. "
              "Add vel_loss to ft_data.py to train it properly.")


# =============================================================================
# PATCH 2 -- Velocity-aware Karplus-Strong synthesis
# =============================================================================

def karplus_strong(freq, duration, sr=44100, decay=0.996, brightness=0.5,
                   amplitude=1.0):
    """
    Karplus-Strong plucked-string synthesis with amplitude control.

    PATCH 2: amplitude parameter scales the initial noise excitation for
    velocity dynamics.  Louder notes have a stronger burst; the feedback
    loop keeps the decay envelope otherwise identical.
    """
    import numpy as np
    n_samples = int(sr * duration)
    buf_len   = max(2, int(sr / freq))
    buf       = np.random.uniform(-amplitude, amplitude, buf_len).astype(np.float64)
    out       = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        out[i]           = buf[i % buf_len]
        nxt              = decay * ((1 - brightness) * buf[i % buf_len]
                                    + brightness * buf[(i + 1) % buf_len])
        buf[i % buf_len] = nxt
    peak = np.max(np.abs(out))
    return (out / peak * amplitude).astype(np.float32) if peak > 1e-9 \
           else out.astype(np.float32)


def synthesize_wav(decoded, tuning, bpm=120.0, sr=44100,
                   note_dur_beats=0.5, gap_beats=0.02,
                   durations=None, velocities=None):
    """
    Duration-, velocity-, and string-aware WAV synthesis.

    PATCH 2 vs original:
      - velocities: each note's Karplus-Strong excitation is scaled by
        amplitude proportional to vel^0.6 (perceptual loudness curve).
        Bin 0 (unknown) defaults to mezzo-forte.
      - String-dependent decay: string 6 (low-E) sustains longer than
        string 1 (high-e) by +0.001 decay per string above 1.
      - Fret-dependent brightness and decay from original are preserved.
    """
    import numpy as np
    N_BINS = 16

    def dur_beats(bin_idx):
        if bin_idx is None or bin_idx <= 0:
            return note_dur_beats
        lo, hi = 0.125, 2.0
        t = (bin_idx - 1) / max(N_BINS - 1, 1)
        return lo * (hi / lo) ** t

    beat_sec  = 60.0 / bpm
    gap_sec   = gap_beats * beat_sec
    note_secs = [
        max(0.1, dur_beats(durations[i] if durations else None) * beat_sec)
        for i in range(len(decoded))
    ]
    steps_sam = [int((ns + gap_sec) * sr) for ns in note_secs]
    total     = sum(steps_sam) + int(max(note_secs) * sr * 3)
    audio     = np.zeros(total, dtype=np.float32)

    cursor = 0
    for i, note in enumerate(decoded):
        m    = tuning[note["string"]] + note["fret"]
        freq = 440.0 * (2.0 ** ((m - 69) / 12.0))
        f    = note["fret"]
        s    = note["string"]

        # Fret-dependent brightness (higher frets = duller)
        bright = max(0.2, 0.7 - f * 0.025)

        # Fret + string-dependent decay.
        # Low strings sustain longer: +0.001 per string above 1.
        string_sustain = 0.001 * (s - 1)
        dec = max(0.990, (0.998 - f * 0.0003) + string_sustain)

        # Velocity -> amplitude (perceptual curve vel^0.6)
        vel_bin = velocities[i] if velocities is not None else 0
        amp     = _VEL_BIN_AMP[vel_bin] if 0 <= vel_bin < len(_VEL_BIN_AMP) \
                  else _VEL_BIN_AMP[0]

        grain = karplus_strong(freq, note_secs[i] * 2.0, sr=sr,
                               decay=dec, brightness=bright, amplitude=amp)
        start = cursor
        end   = start + len(grain)
        if end > len(audio):
            grain = grain[:len(audio) - start]; end = len(audio)
        audio[start:end] += grain
        cursor += steps_sam[i]

    peak = np.max(np.abs(audio))
    if peak > 1e-9:
        audio = audio / peak * 0.9
    return audio


def save_wav(audio, path, sr=44100):
    import numpy as np
    try:
        from scipy.io import wavfile
        pcm = (np.clip(audio,-1.,1.)*32767).astype(np.int16)
        wavfile.write(path, sr, pcm)
    except ImportError:
        import wave
        pcm = (np.clip(audio,-1.,1.)*32767).astype(np.int16)
        with wave.open(path,'w') as wf:
            wf.setnchannels(1); wf.setsampwidth(2)
            wf.setframerate(sr); wf.writeframes(pcm.tobytes())
    print(f"WAV saved : {path}  ({len(audio)/sr:.1f}s  {sr}Hz mono 16-bit PCM)")


# ── Masked diffusion decoder ──────────────────────────────────────────────────

@torch.no_grad()
def masked_diffusion_decode(model, pitches, tuning,
                             steps=10, base_temp=1.0,
                             anneal="cosine", order="confidence",
                             device="cpu", scale_mask=None,
                             args_repeat_window=4):
    """
    Fully joint iterative unmasking over `steps` passes.

    Pitches, positions, durations, AND velocities start fully masked.
    At each step the model scores all masked tokens simultaneously using
    bidirectional context from all already-revealed tokens, then reveals
    the most confident ones.

    PATCHES applied vs original:
      1. Model is wrapped in CachedMLMModel -- pos_enc computed once per
         step instead of once per masked token (~250x fewer calls).
      2. Velocity tokens are scored and revealed alongside the other
         modalities; returned as the third value (before decoded).
      3. Pitch tokens use annealed_temp_pitch (floor 0.5) to preserve
         melodic variety in late-stage pitch reveals.

    Returns
    -------
    pitches    : list[int]   MIDI pitch per note
    durations  : list[int]   duration bin per note (1-16)
    velocities : list[int]   velocity bin per note (1-8)
    decoded    : list[dict]  {string, fret, midi, is_open}
    """
    # PATCH 1: wrap for pos_enc caching
    if not isinstance(model, CachedMLMModel):
        model = CachedMLMModel(model)

    inner = model._model
    inner.eval()

    # PATCH 2: ensure vel_head exists
    _ensure_vel_head(model, device)

    T = len(pitches) if pitches else 32

    cands_by_midi = {
        m: [POS_TO_IDX[p] for p in midi_to_positions(m, tuning) if p in POS_TO_IDX]
        for m in range(NUM_MIDI)
    }

    # Initial state -- all modalities masked
    cur_pitches    = [FretboardTransformerMLM.PITCH_MASK_IDX] * T
    cur_positions  = [FretboardTransformerMLM.MASK_IDX]       * T
    cur_durations  = [0] * T
    cur_velocities = [0] * T

    pit_masked = list(range(T))
    pos_masked = list(range(T))
    dur_masked = list(range(T))
    vel_masked = list(range(T))

    tokens_per_step = max(1, (4 * T) // steps)

    for step in range(steps):
        # PATCH 1: clear pos_enc cache -- new tokens revealed since last step
        model.clear_cache()

        # PATCH 3: separate temperatures for pitch vs everything else
        temp_pos = annealed_temp(base_temp, step, steps, anneal)
        temp_pit = annealed_temp_pitch(base_temp, step, steps, anneal)

        candidates = []

        def spatial_blacklist(t_idx):
            if args_repeat_window <= 0:
                return set()
            nearby = set()
            for dt in range(1, args_repeat_window + 1):
                for nb in (t_idx - dt, t_idx + dt):
                    if 0 <= nb < T:
                        p = cur_pitches[nb]
                        if p != FretboardTransformerMLM.PITCH_MASK_IDX:
                            nearby.add(p)
            return nearby

        # ── Score pitch tokens ─────────────────────────────────────────────
        for t_idx in pit_masked:
            pit_lp, _, _ = model.decode_joint_step(
                cur_pitches, cur_positions, t_idx, cur_durations)
            lp = pit_lp / max(temp_pit, 1e-6)   # PATCH 3
            if scale_mask is not None:
                lp = lp.masked_fill(~scale_mask.to(lp.device), float('-inf'))
            for p in spatial_blacklist(t_idx):
                if 0 <= p < len(lp):
                    lp[p] = float('-inf')
            if torch.all(lp == float('-inf')):
                lp = pit_lp / max(temp_pit, 1e-6)
                if scale_mask is not None:
                    lp = lp.masked_fill(~scale_mask.to(lp.device), float('-inf'))
            probs = torch.softmax(lp, dim=-1)
            if probs.isnan().any() or probs.sum() < 1e-9:
                continue
            conf, best_k = probs.max(0)
            candidates.append((conf.item(), 'pit', t_idx, best_k.item(), probs))

        # ── Score position tokens ──────────────────────────────────────────
        for t_idx in pos_masked:
            if cur_pitches[t_idx] == FretboardTransformerMLM.PITCH_MASK_IDX:
                continue
            midi_val = cur_pitches[t_idx]
            cands    = cands_by_midi.get(midi_val, [])
            if not cands:
                continue
            _, pos_lp, _ = model.decode_joint_step(
                cur_pitches, cur_positions, t_idx, cur_durations)
            lp_c  = pos_lp[cands] / max(temp_pos, 1e-6)
            probs = torch.softmax(lp_c, dim=-1)
            conf, best_k = probs.max(0)
            candidates.append((conf.item(), 'pos', t_idx, cands[best_k.item()], probs))

        # ── Score duration tokens ──────────────────────────────────────────
        for t_idx in dur_masked:
            _, _, dur_lp = model.decode_joint_step(
                cur_pitches, cur_positions, t_idx, cur_durations)
            lp    = dur_lp / max(temp_pos, 1e-6)
            probs = torch.softmax(lp, dim=-1)
            probs[0] = 0.0
            if probs.sum() > 1e-9:
                probs = probs / probs.sum()
            conf, best_k = probs.max(0)
            candidates.append((conf.item(), 'dur', t_idx, best_k.item(), probs))

        # ── Score velocity tokens (PATCH 2) ───────────────────────────────
        # vel_head shares the fused representation.  pos_enc comes from
        # cache (free); only pitch_enc needs a new forward pass.
        for t_idx in vel_masked:
            traj_ctx  = model._get_pos_enc(cur_positions, cur_durations, device)
            pit_t     = torch.tensor(
                cur_pitches, dtype=torch.long, device=device).unsqueeze(0)
            dur_t     = torch.tensor(
                cur_durations, dtype=torch.long, device=device).unsqueeze(0)
            pitch_ctx = inner.pitch_enc(pit_t, durations=dur_t)
            p_ctx     = pitch_ctx[0, t_idx]
            t_ctx     = traj_ctx[t_idx]
            fused     = inner.fusion(
                torch.cat([p_ctx, t_ctx], dim=-1).unsqueeze(0))
            vel_lp    = F.log_softmax(inner.vel_head(fused), dim=-1).squeeze(0)
            lp        = vel_lp / max(temp_pos, 1e-6)
            probs     = torch.softmax(lp, dim=-1)
            probs[0]  = 0.0
            if probs.sum() > 1e-9:
                probs = probs / probs.sum()
            conf, best_k = probs.max(0)
            candidates.append((conf.item(), 'vel', t_idx, best_k.item(), probs))

        if not candidates:
            break

        if order == "confidence":
            candidates.sort(key=lambda x: -x[0])
        elif order == "random":
            random.shuffle(candidates)
        else:
            candidates.sort(key=lambda x: x[2])

        n_reveal = len(candidates) if step == steps - 1 \
                   else min(tokens_per_step, len(candidates))

        revealed_pit = set(); revealed_pos = set()
        revealed_dur = set(); revealed_vel = set()

        for conf, kind, t_idx, best_val, probs in candidates[:n_reveal]:
            if kind == 'pit' and t_idx not in revealed_pit:
                cur_pitches[t_idx]    = best_val if temp_pit < 0.5 \
                    else torch.multinomial(probs, 1).item()   # PATCH 3
                revealed_pit.add(t_idx)
            elif kind == 'pos' and t_idx not in revealed_pos:
                midi_val = cur_pitches[t_idx]
                cands    = cands_by_midi.get(midi_val, [])
                if cands:
                    cur_positions[t_idx] = best_val if temp_pos < 0.2 \
                        else cands[torch.multinomial(probs, 1).item()]
                revealed_pos.add(t_idx)
            elif kind == 'dur' and t_idx not in revealed_dur:
                chosen = best_val if temp_pos < 0.2 \
                    else torch.multinomial(probs, 1).item()
                cur_durations[t_idx] = max(1, chosen)
                revealed_dur.add(t_idx)
            elif kind == 'vel' and t_idx not in revealed_vel:  # PATCH 2
                chosen = best_val if temp_pos < 0.2 \
                    else torch.multinomial(probs, 1).item()
                cur_velocities[t_idx] = max(1, chosen)
                revealed_vel.add(t_idx)

        pit_masked = [t for t in pit_masked if t not in revealed_pit]
        pos_masked = [t for t in pos_masked if t not in revealed_pos]
        dur_masked = [t for t in dur_masked if t not in revealed_dur]
        vel_masked = [t for t in vel_masked if t not in revealed_vel]

        if not any([pit_masked, pos_masked, dur_masked, vel_masked]):
            break

    # ── Final pass: fill any remaining masks ──────────────────────────────
    for t_idx in pit_masked:
        pit_lp, _, _ = model.decode_joint_step(
            cur_pitches, cur_positions, t_idx, cur_durations)
        cur_pitches[t_idx] = pit_lp.argmax().item()
    for t_idx in pos_masked:
        midi_val = cur_pitches[t_idx]
        if midi_val == FretboardTransformerMLM.PITCH_MASK_IDX:
            cur_pitches[t_idx] = 60; midi_val = 60
        cands = cands_by_midi.get(midi_val, [])
        if cands:
            _, pos_lp, _ = model.decode_joint_step(
                cur_pitches, cur_positions, t_idx, cur_durations)
            cur_positions[t_idx] = cands[pos_lp[cands].argmax().item()]
        else:
            cur_positions[t_idx] = 0
    for t_idx in dur_masked:
        _, _, dur_lp = model.decode_joint_step(
            cur_pitches, cur_positions, t_idx, cur_durations)
        dur_lp[0] = float('-inf')
        cur_durations[t_idx] = max(1, dur_lp.argmax().item())
    for t_idx in vel_masked:
        cur_velocities[t_idx] = 4   # fallback: mezzo-forte

    # ── Safety: fix out-of-range positions and still-masked pitches ───────
    for t_idx in range(T):
        p = cur_positions[t_idx]
        if p >= len(POSITIONS) or p < 0:
            midi_val = cur_pitches[t_idx] if cur_pitches[t_idx] < NUM_MIDI else 60
            cands    = cands_by_midi.get(midi_val, [0])
            cur_positions[t_idx] = cands[0] if cands else 0
        if cur_pitches[t_idx] >= NUM_MIDI:
            cur_pitches[t_idx] = 60

    decoded = [
        {
            "string":  POSITIONS[p][0],
            "fret":    POSITIONS[p][1],
            "midi":    pos_to_midi(*POSITIONS[p], tuning),
            "is_open": POSITIONS[p][1] == 0,
        }
        for p in cur_positions
    ]

    return cur_pitches, cur_durations, cur_velocities, decoded


# ── MLM training utility ─────────────────────────────────────────────────────

def train_mlm(save_path=MLM_CHECKPOINT, epochs=20,
              scraped_tabs_dir=None, cache_path="dataset_cache.pkl"):
    print(f"Training masked-LM model -> {save_path}")
    model = FretboardTransformerMLM(embed_dim=128, ffn_dim=512, dropout=0.3)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = FretboardTrainer(
        model,
        dadagp_dir        = "./assets/data/dadagp/",
        synthtab_dir      = "./assets/data/synthtab/outall/",
        scoreset_dir      = "./assets/data/scoreset/outall/",
        scraped_tabs_dir  = scraped_tabs_dir,
        proggp_dir        = None,
        midi_cache        = "./midi_sequences.pkl" if os.path.exists("./midi_sequences.pkl") else None,
        window            = 64,
        batch_size        = 32,
        lr                = 3e-4,
        genres            = ["metal", "rock", "hard_rock"],
        num_workers       = 16,
        queue_size        = 4096,
        steps_per_epoch   = 500,
        augment_semitones = (-3,-2,-1,1,2,3),
        val_split         = 0.1,
        mask_prob         = 0.40,
        dropout           = 0.3,
        training_mode     = 'masked_lm',
        max_source_fraction = 0.8,
        epochs_per_10k    = 20,
        max_epochs        = 30,
        cache_path        = cache_path,
    )

    n_train = len(trainer.ds)
    print(f"Running {epochs} epochs on {n_train} train sequences.")
    trainer.train(epochs=epochs, save_path=save_path)
    print(f"MLM checkpoint saved: {save_path}")
    return model


# ── Display ───────────────────────────────────────────────────────────────────

def render_ascii(decoded, notes_per_line=16):
    STRING_LABELS = {1:"e", 2:"B", 3:"G", 4:"D", 5:"A", 6:"E"}
    lines = []
    for start in range(0, len(decoded), notes_per_line):
        chunk = decoded[start:start+notes_per_line]
        rows  = {s: STRING_LABELS[s]+"|" for s in range(1,7)}
        for note in chunk:
            s,f = note["string"], note["fret"]
            col = str(f).ljust(3)
            for string in range(1,7):
                rows[string] += col if string==s else "-"*len(col)
        for s in rows: rows[s] += "|"
        lines.append("")
        for s in range(1,7): lines.append(rows[s])
    return "\n".join(lines)


def postprocess(decoded, tuning, max_deviation=5, max_fret=20):
    from fretboard_transformer import midi_to_positions
    result  = [dict(d) for d in decoded]
    n_fixed = 0
    for i in range(len(result)):
        fret = result[i]["fret"]
        if fret == 0:
            continue
        window  = result[max(0,i-5):i] + result[i+1:min(len(result),i+6)]
        frets_w = [n["fret"] for n in window if n["fret"] != 0]
        if not frets_w:
            continue
        mean_f = sum(frets_w) / len(frets_w)
        if abs(fret - mean_f) > max_deviation or fret > max_fret:
            midi  = result[i]["midi"]
            cands = midi_to_positions(midi, tuning)
            if not cands:
                continue
            best = min(cands, key=lambda pos: abs(pos[1] - mean_f))
            if abs(best[1] - mean_f) < abs(fret - mean_f):
                result[i] = {
                    "string":  best[0],
                    "fret":    best[1],
                    "midi":    midi,
                    "is_open": best[1] == 0,
                }
                n_fixed += 1
    return result, n_fixed


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    tuning = TUNINGS[args.tuning]

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    needs_train = (args.train or args.force_train or
                   not os.path.exists(args.model))
    if needs_train:
        if not args.force_train and not args.train and not os.path.exists(args.model):
            print(f"[INFO] Checkpoint {args.model} not found -- training now.")
            print(f"       Pass --train explicitly to suppress this message.")
        model = train_mlm(save_path=args.model, epochs=args.train_epochs,
                          scraped_tabs_dir=args.scraped_tabs_dir,
                          cache_path=args.cache_path)
        if args.train and args.length == 32 and not any([args.wav]):
            print("Training complete. Run without --train to generate.")
            return
    else:
        model = FretboardTransformerMLM(dropout=0.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not needs_train or os.path.exists(args.model):
        state = torch.load(args.model, map_location=device)
        model.load_state_dict(state)
    model.to(device).eval()

    # PATCH 1: wrap model for pos_enc caching
    model = CachedMLMModel(model)

    print(f"Model  : {args.model}  "
          f"({sum(p.numel() for p in model._model.parameters()):,} params)")
    print(f"Decoder: masked diffusion  steps={args.steps}  "
          f"anneal={args.anneal}  order={args.order}")

    lo, hi  = REGISTERS[args.register]
    root_pc = None
    if args.key:
        root_pc = NOTE_TO_PC.get(args.key)
        if root_pc is None:
            print(f"[ERROR] Unknown key '{args.key}'"); sys.exit(1)
        scale_name = args.scale if args.scale else "chromatic"
        intervals  = SCALES[scale_name]
        scale_mask = build_scale_mask(root_pc, intervals, lo, hi)
        print(f"Key    : {args.key} {scale_name}  register={args.register}")
    else:
        scale_mask = torch.zeros(NUM_MIDI, dtype=torch.bool)
        scale_mask[lo:hi+1] = True
        print(f"Key    : unconstrained  register={args.register}")

    if args.start is not None:
        start_pitch = args.start
    else:
        valid = [p for p in range(lo, hi+1) if scale_mask[p]]
        start_pitch = random.choice(valid) if valid else (lo+hi)//2
    name = NOTE_NAMES[start_pitch%12] + str(start_pitch//12-1)
    print(f"Start  : MIDI {start_pitch}  ({name})")
    print(f"Length : {args.length}  temp={args.temp}")

    tonic_pc   = root_pc if args.key else None
    joint_mode = not (args.markov or args.random_walk)

    if args.markov:
        if not os.path.exists(args.markov):
            print(f"[ERROR] Markov model not found: {args.markov}"); sys.exit(1)
        data = load_markov(args.markov)
        print(f"Pitch  : Markov order={data['order']}  {len(data['counts']):,} contexts")
        pitches = generate_pitches_markov(
            data, scale_mask, args.length, start_pitch,
            args.temp, tonic_pc, 2.0,
            args.repeat_penalty, args.repeat_window)
        joint_mode = False
    elif args.random_walk:
        print("Pitch  : phrase random walk")
        pitches = generate_pitches_walk(
            scale_mask, args.length, start_pitch, args.step_size,
            args.temp, tonic_pc, 2.0, (args.phrase_min, args.phrase_max))
        joint_mode = False
    else:
        pitches = None

    print(f"\nRunning {'joint' if joint_mode else 'position-only'} "
          f"masked diffusion  ({args.steps} steps)...")

    if joint_mode:
        print("Pitch  : joint masked diffusion (fully bidirectional)")
        pitches, durations, velocities, decoded = masked_diffusion_decode(
            model, [FretboardTransformerMLM.PITCH_MASK_IDX] * args.length,
            tuning, steps=args.steps, base_temp=args.temp,
            anneal=args.anneal, order=args.order, device=device,
            scale_mask=scale_mask, args_repeat_window=args.repeat_window)
    else:
        pitches, durations, velocities, decoded = masked_diffusion_decode(
            model, pitches, tuning,
            steps=args.steps, base_temp=args.temp,
            anneal=args.anneal, order=args.order, device=device,
            scale_mask=scale_mask, args_repeat_window=args.repeat_window)

    if not args.no_postprocess:
        decoded, n_fixed = postprocess(decoded, tuning)
        if n_fixed:
            print(f"Post-processing: {n_fixed} note(s) relocated")

    print("\n" + "-"*56)
    print(render_ascii(decoded, args.notes_per_line))
    print("-"*56)

    print(f"\n{'#':>3}  {'Str':>3}  {'Fret':>4}  {'MIDI':>4}  {'Vel':>3}  Note")
    print("-"*34)
    for i, note in enumerate(decoded):
        m     = tuning[note["string"]] + note["fret"]
        nname = NOTE_NAMES[m%12] + str(m//12-1)
        v     = velocities[i] if velocities else 0
        print(f"{i+1:>3}  str {note['string']}  fret {note['fret']:>2}  "
              f"MIDI {m:>3}  v{v}  {nname}{'  open' if note['is_open'] else ''}")

    print(f"\nPitch sequence (MIDI):")
    print(",".join(str(p) for p in pitches))

    if args.wav:
        audio = synthesize_wav(decoded, tuning, bpm=args.bpm,
                               note_dur_beats=args.note_dur,
                               durations=durations, velocities=velocities)
        save_wav(audio, args.wav)


if __name__ == "__main__":
    main()
