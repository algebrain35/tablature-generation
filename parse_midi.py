"""
parse_midi.py — Extract pitch sequences from MIDI files for pitch_head training.

Reads .mid/.midi files, extracts monophonic pitch sequences from guitar tracks
(GM programs 24-31) or any melodic track, and adds them to the training cache
as source kind 'midi'. Position labels are set to 0 (dummy) and excluded from
position loss during training via source-conditional loss.

Usage:
    python parse_midi.py --midi_dir ./assets/data/lakh_midi/ --out midi_sequences.pkl
    python parse_midi.py --midi_dir ./assets/data/lakh_midi/ --guitar_only --workers 16
    python parse_midi.py --midi_dir ./assets/data/lakh_midi/ \
        --merge dataset_cache.pkl.*.v8 --out dataset_cache_midi_merged.pkl
"""

import os, sys, glob, argparse, pickle, random

try:
    import mido
except ImportError:
    print("[ERROR] mido required: pip install mido")
    sys.exit(1)

# GM guitar programs: Nylon, Steel, Jazz, Clean, Muted, Overdriven, Distortion, Harmonics
GUITAR_PROGRAMS  = set(range(24, 32))
MELODIC_PROGRAMS = set(range(0, 96)) - set(range(32, 40))


def parse_args():
    p = argparse.ArgumentParser(description="Extract pitch sequences from MIDI files")
    p.add_argument("--midi_dir",    required=True)
    p.add_argument("--out",         default="midi_sequences.pkl")
    p.add_argument("--merge",       default=None,
                   help="Existing cache .pkl to merge into")
    p.add_argument("--guitar_only", action="store_true",
                   help="Only extract guitar tracks (GM programs 24-31)")
    p.add_argument("--min_notes",   type=int, default=16)
    p.add_argument("--max_notes",   type=int, default=256)
    p.add_argument("--window",      type=int, default=64)
    p.add_argument("--stride",      type=int, default=32)
    p.add_argument("--max_files",   type=int, default=None)
    p.add_argument("--workers",     type=int, default=8,
                   help="Parallel worker processes (default 8)")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def extract_tracks(midi_path, guitar_only=False):
    try:
        mid = mido.MidiFile(midi_path)
    except Exception:
        return []

    sequences = []
    for track in mid.tracks:
        program  = 0
        channel  = None
        active   = {}
        notes    = []
        abs_tick = 0

        for msg in track:
            abs_tick += msg.time
            if msg.type == 'program_change':
                program = msg.program
                channel = msg.channel
            elif msg.type == 'note_on' and msg.velocity > 0:
                if channel is None:
                    channel = msg.channel
                active[msg.note] = abs_tick
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active:
                    notes.append((active.pop(msg.note), msg.note))

        if channel == 9:
            continue
        if guitar_only and program not in GUITAR_PROGRAMS:
            continue
        if not guitar_only and program not in MELODIC_PROGRAMS:
            continue
        if len(notes) < 4:
            continue

        notes.sort()
        pitches = [p for _, p in notes if 36 <= p <= 96]
        if len(pitches) >= 4:
            sequences.append(pitches)

    return sequences


def window_sequence(pitches, window, stride, min_notes):
    if len(pitches) < min_notes:
        return []
    if len(pitches) <= window:
        return [pitches]
    return [pitches[s:s+window] for s in range(0, len(pitches)-min_notes+1, stride)
            if len(pitches[s:s+window]) >= min_notes]


def make_cache_entry(pitches, source_name="midi"):
    n = len(pitches)
    return (source_name, 12, [0]*n, list(pitches), [0]*n, [0]*n, [0]*n)


def _process_file(args):
    path, guitar_only, window, stride, min_notes = args
    try:
        tracks = extract_tracks(path, guitar_only)
    except Exception:
        return []
    if not tracks:
        return []
    fname = os.path.splitext(os.path.basename(path))[0]
    results = []
    for pitches in tracks:
        for win in window_sequence(pitches, window, stride, min_notes):
            results.append(make_cache_entry(win, f"midi:{fname}"))
    return results


def main():
    args = parse_args()
    random.seed(args.seed)

    patterns  = ["**/*.mid", "**/*.midi", "**/*.MID", "**/*.MIDI"]
    all_files = list({f for pat in patterns
                      for f in glob.glob(os.path.join(args.midi_dir, pat), recursive=True)})
    random.shuffle(all_files)
    if args.max_files:
        all_files = all_files[:args.max_files]

    print(f"Found   : {len(all_files)} MIDI files")
    print(f"Mode    : {'guitar only (GM 24-31)' if args.guitar_only else 'all melodic'}")
    print(f"Workers : {args.workers}")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    worker_args = [(p, args.guitar_only, args.window, args.stride, args.min_notes)
                   for p in all_files]

    entries = []
    done = skipped = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_file, a): a for a in worker_args}
        for fut in as_completed(futures):
            done += 1
            if done % 5000 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}  ({len(entries)} sequences)...", flush=True)
            try:
                result = fut.result()
            except Exception:
                result = []
            if result:
                entries.extend(result)
            else:
                skipped += 1

    print(f"\nDone: {len(entries)} sequences  ({skipped} files skipped)")

    if args.merge:
        candidates = glob.glob(args.merge) if '*' in args.merge else [args.merge]
        candidates = [c for c in candidates if os.path.exists(c)]
        if candidates:
            merge_path = max(candidates, key=os.path.getmtime)
            print(f"Merging with: {merge_path}")
            with open(merge_path, 'rb') as f:
                existing = pickle.load(f)
            print(f"  Existing : {len(existing)} entries")
            entries = existing + entries
            print(f"  Combined : {len(entries)} entries")

    with open(args.out, 'wb') as f:
        pickle.dump(entries, f, protocol=pickle.HIGHEST_PROTOCOL)
    mb = os.path.getsize(args.out) / (1024*1024)
    print(f"\nSaved: {args.out}  ({len(entries)} entries, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
