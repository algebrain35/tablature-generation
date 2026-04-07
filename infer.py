"""
infer.py — Guitar tablature inference for arbitrary-length pitch sequences

Accepts pitches from:
  1. A MIDI file  (--midi)
  2. A GP3/GP4/GP5 file  (--gp)       [uses GT pitches, ignores GT strings]
  3. A comma-separated list  (--pitches "47,54,55,54,47,54")
  4. Interactive prompt  (default)

Outputs:
  - ASCII tablature to stdout
  - Optional GP5 file  (--output out.gp5)   [requires guitarpro]
  - Optional MIDI file (--output out.mid)

Usage examples:
  python infer.py --model fretboard_transformer.pt --pitches "47,54,55,54,47,54,55,54,50,45"
  python infer.py --model fretboard_transformer.pt --midi my_song.mid
  python infer.py --model fretboard_transformer.pt --gp my_song.gp5
  python infer.py --model fretboard_transformer.pt   # interactive mode
"""

import os, sys, argparse, textwrap
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fretboard_transformer import (
    FretboardTransformer, TrellisDecoder,
    parse_scoreset_gp, pos_to_midi,
    STANDARD_TUNING, NUM_MIDI,
    quantize_ticks, quantize_velocity, N_TS_BINS, N_VEL_BINS, N_DUR_BINS,
)

# ── Constants ─────────────────────────────────────────────────────────────────

NOTE_NAMES = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]

WINDOW = 64   # max decode chunk — matches training window


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Guitar tablature inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
          Examples:
            python infer.py --pitches "47,54,55,54,47,54,55,54,50,45"
            python infer.py --midi my_song.mid
            python infer.py --gp  my_song.gp5 --output my_song_tab.gp5
        """)
    )
    p.add_argument("--model",    default="fretboard_transformer.pt")
    p.add_argument("--pitches",  default=None,
                   help="Comma-separated MIDI pitch numbers")
    p.add_argument("--midi",     default=None,
                   help="MIDI file to read pitches from")
    p.add_argument("--gp",       default=None,
                   help="GP3/GP4/GP5 file — uses its pitches (ignores string/fret)")
    p.add_argument("--output",   default=None,
                   help="Output file (.gp5 or .mid). Default: print ASCII only.")
    p.add_argument("--fret_bias", type=float, default=0.05)
    p.add_argument("--window",   type=int, default=WINDOW,
                   help="Decode chunk size (default 64, must match training window)")
    p.add_argument("--no_postprocess", action="store_true")
    p.add_argument("--tuning",   default="standard",
                   choices=["standard", "eb", "dropd", "dropc"],
                   help="Guitar tuning")
    return p.parse_args()


# ── Tunings ───────────────────────────────────────────────────────────────────

TUNINGS = {
    "standard": {1:64, 2:59, 3:55, 4:50, 5:45, 6:40},
    "eb":       {1:63, 2:58, 3:54, 4:49, 5:44, 6:39},
    "dropd":    {1:64, 2:59, 3:55, 4:50, 5:45, 6:38},
    "dropc":    {1:62, 2:57, 3:53, 4:48, 5:43, 6:36},
}


# ── Pitch input parsers ───────────────────────────────────────────────────────

def pitches_from_string(s):
    """Parse comma-separated MIDI pitch numbers."""
    try:
        pitches = [int(x.strip()) for x in s.split(",") if x.strip()]
    except ValueError:
        print("[ERROR] --pitches must be comma-separated integers, e.g. '47,54,55'")
        sys.exit(1)
    invalid = [p for p in pitches if not (0 <= p < NUM_MIDI)]
    if invalid:
        print(f"[ERROR] Out-of-range pitches: {invalid}  (valid: 0–127)")
        sys.exit(1)
    return pitches


def pitches_from_midi(path):
    """Extract pitch sequence from a single-track MIDI file using mido."""
    try:
        import mido
    except ImportError:
        print("[ERROR] mido required: pip install mido")
        sys.exit(1)
    mid    = mido.MidiFile(path)
    events = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                events.append((abs_tick, msg.note))
    events.sort()
    return [note for _, note in events]


def pitches_from_gp(path):
    """Extract pitch sequence from a GP file using the project's GP parser."""
    notes_raw, _ = parse_scoreset_gp(path)
    if not notes_raw:
        print(f"[ERROR] No notes parsed from {path}. Check the file and tuning.")
        sys.exit(1)
    return [pos_to_midi(s, f) for s, f, *_ in notes_raw]


def pitches_interactive():
    """Prompt user to enter MIDI pitches interactively."""
    print("\nEnter MIDI pitch numbers separated by spaces or commas.")
    print("Note reference: E2=40  A2=45  D3=50  G3=55  B3=59  E4=64")
    print("(Ctrl+C to exit)\n")
    while True:
        try:
            raw = input("Pitches > ").strip()
        except KeyboardInterrupt:
            print()
            sys.exit(0)
        if not raw:
            continue
        raw = raw.replace(",", " ")
        try:
            pitches = [int(x) for x in raw.split() if x]
            if pitches:
                return pitches
        except ValueError:
            print("  Invalid input — enter integers only.")


# ── Chunked decoding for long sequences ──────────────────────────────────────

def decode_long(decoder, pitches, window):
    """
    Decode a pitch sequence of arbitrary length by processing overlapping
    windows of size `window`. The full sequence is decoded in non-overlapping
    chunks — each chunk gets global bidirectional pitch context for its window.

    For sequences shorter than `window`, a single decode call is made.
    """
    T = len(pitches)
    if T <= window:
        return decoder.decode(pitches)

    result = []
    for start in range(0, T, window):
        chunk = pitches[start : start + window]
        decoded_chunk = decoder.decode(chunk)
        result.extend(decoded_chunk)
    return result


# ── ASCII tablature renderer ──────────────────────────────────────────────────

def render_ascii(decoded, tuning, notes_per_line=16):
    """
    Render decoded positions as ASCII guitar tab.
    String 1 = high-e (top row), string 6 = low-E (bottom row).
    """
    STRING_LABELS = {1:"e", 2:"B", 3:"G", 4:"D", 5:"A", 6:"E"}
    T = len(decoded)
    lines = []

    for start in range(0, T, notes_per_line):
        chunk = decoded[start : start + notes_per_line]
        rows = {s: STRING_LABELS[s] + "|" for s in range(1, 7)}
        for note in chunk:
            s    = note["string"]
            fret = note["fret"]
            label = str(fret) if fret > 0 else "0"
            for string in range(1, 7):
                if string == s:
                    rows[string] += label.ljust(3)
                else:
                    rows[string] += "-" * len(label.ljust(3))
        for string in rows:
            rows[string] += "|"
        lines.append("")
        for s in range(1, 7):
            lines.append(rows[s])
    return "\n".join(lines)


# ── MIDI output ───────────────────────────────────────────────────────────────

def save_midi(decoded, path, tuning, bpm=120):
    try:
        import mido
    except ImportError:
        print("[ERROR] mido required for MIDI output: pip install mido")
        return
    ticks_per_beat = 480
    note_ticks = ticks_per_beat // 2   # eighth notes
    mid   = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo",
                  tempo=mido.bpm2tempo(bpm), time=0))
    for note in decoded:
        pitch = tuning[note["string"]] + note["fret"]
        track.append(mido.Message("note_on",  note=pitch, velocity=80, time=0))
        track.append(mido.Message("note_off", note=pitch, velocity=0,
                                  time=note_ticks))
    mid.save(path)
    print(f"MIDI saved: {path}")


# ── GP5 output ────────────────────────────────────────────────────────────────

def save_gp5(decoded, path, bpm=120):
    try:
        import guitarpro
    except ImportError:
        print("[ERROR] guitarpro required for GP output: pip install pyguitarpro")
        return

    song  = guitarpro.models.Song()
    song.tempo = bpm
    track = song.tracks[0]
    track.strings = [
        guitarpro.models.GuitarString(i+1, STANDARD_TUNING[i+1])
        for i in range(6)
    ]
    measure = track.measures[0]
    voice   = measure.voices[0]

    for note_info in decoded:
        beat = guitarpro.models.Beat(voice)
        beat.duration = guitarpro.models.Duration()
        gp_note = guitarpro.models.Note(beat)
        gp_note.string = note_info["string"]
        gp_note.value  = note_info["fret"]
        beat.notes.append(gp_note)
        voice.beats.append(beat)

    guitarpro.write(song, path)
    print(f"GP5 saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tuning = TUNINGS[args.tuning]

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"[ERROR] Checkpoint not found: {args.model}")
        sys.exit(1)

    model = FretboardTransformer(dropout=0.0)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Model: {args.model}  ({sum(p.numel() for p in model.parameters()):,} params)  device={device}")

    decoder = TrellisDecoder(model, tuning=tuning, fret_bias=args.fret_bias)

    # ── Get pitches ───────────────────────────────────────────────────────────
    if args.pitches:
        pitches = pitches_from_string(args.pitches)
        source  = "command line"
    elif args.midi:
        pitches = pitches_from_midi(args.midi)
        source  = args.midi
    elif args.gp:
        pitches = pitches_from_gp(args.gp)
        source  = args.gp
    else:
        pitches = pitches_interactive()
        source  = "interactive"

    print(f"Source  : {source}")
    print(f"Notes   : {len(pitches)}")
    print(f"Pitches : {pitches[:16]}{'...' if len(pitches) > 16 else ''}")

    # ── Decode ────────────────────────────────────────────────────────────────
    decoded = decode_long(decoder, pitches, args.window)

    if not args.no_postprocess:
        decoded, n_fixed = decoder.postprocess(decoded)
        if n_fixed:
            print(f"Post-processing: {n_fixed} note(s) relocated")

    # ── Print ASCII tab ───────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(render_ascii(decoded, tuning))
    print("─" * 60)

    # Print note summary
    print(f"\n{'Pos':>4}  {'String':>6}  {'Fret':>4}  {'MIDI':>4}  {'Note'}")
    print("─" * 36)
    for i, note in enumerate(decoded):
        pitch     = tuning[note["string"]] + note["fret"]
        note_name = NOTE_NAMES[pitch % 12]
        octave    = pitch // 12 - 1
        print(f"{i+1:>4}  string {note['string']}  fret {note['fret']:>2}  "
              f"MIDI {pitch:>3}  {note_name}{octave}"
              f"{'  (open)' if note['is_open'] else ''}")

    # ── Optional output ───────────────────────────────────────────────────────
    if args.output:
        ext = os.path.splitext(args.output)[1].lower()
        if ext in (".gp3", ".gp4", ".gp5"):
            save_gp5(decoded, args.output)
        elif ext in (".mid", ".midi"):
            save_midi(decoded, args.output, tuning)
        else:
            print(f"[WARN] Unknown output extension '{ext}' — skipping file save.")

if __name__ == "__main__":
    main()
