"""
evaluate.py — Model evaluation on scraped GP tablatures

Samples GP files from the scraped tabs directory, filtering to artists in the
provided artist list. For each file, parses the ground truth (string, fret)
sequence and runs the TrellisDecoder. Reports per-file and aggregate accuracy.

Usage:
    python evaluate.py \
        --model  fretboard_transformer.pt \
        --tabs   ./assets/data/scraped_tabs/ \
        --artists artists_rock_metal.txt \
        --n      50 \
        --seed   42
"""

import os, sys, re, random, argparse, time
import torch

# ── Import model components from training script ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fretboard_transformer import (
    FretboardTransformer, TrellisDecoder,
    parse_scoreset_gp, POS_TO_IDX, POSITIONS, pos_to_midi,
    NUM_POSITIONS, NUM_STRINGS, midi_to_positions,
)


# ── Pure greedy decoder ───────────────────────────────────────────────────────

def greedy_decode(model, pitch_sequence, tuning=None, device=None):
    """
    True greedy argmax — no graph search, no path optimisation.
    For each note independently, encode all pitches bidirectionally
    then take the argmax position from the neural network output.
    Serves as a baseline to isolate the contribution of A* path search.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    T       = len(pitch_sequence)
    pitches = torch.tensor(pitch_sequence, dtype=torch.long, device=device)

    with torch.no_grad():
        pitch_ctx = model.encode_pitches(pitches, key=None)  # (T, D)

    results = []
    for t in range(T):
        # Get valid candidate positions for this pitch
        from ft_model import midi_to_positions as m2p, STANDARD_TUNING
        cands = m2p(pitch_sequence[t], STANDARD_TUNING if tuning is None else tuning)
        if not cands:
            # Fallback: open low E
            results.append({'string':6,'fret':0,'midi':pitch_sequence[t],'is_open':True})
            continue

        # Score each candidate independently — no conditioning on prior positions
        # Use a dummy empty past_path so decode_step sees no trajectory context
        with torch.no_grad():
            log_probs, _ = model.decode_step(pitch_ctx[t], [], t)

        # Argmax over valid positions only
        best_idx  = max(
            (POS_TO_IDX[p] for p in cands if p in POS_TO_IDX),
            key=lambda i: log_probs[i].item()
        )
        s, f = POSITIONS[best_idx]
        results.append({
            'string':  s,
            'fret':    f,
            'midi':    pos_to_midi(s, f),
            'is_open': f == 0,
        })
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate fretboard model on GP files")
    p.add_argument("--model",   default="fretboard_transformer.pt",
                   help="Path to saved model checkpoint")
    p.add_argument("--tabs",    default="./assets/data/scraped_tabs/",
                   help="Directory containing GP3/GP4/GP5 files")
    p.add_argument("--manifest", default=None,
                   help="Path to eval manifest file produced by build_eval_set.py "
                        "(overrides --tabs and --artists if provided)")
    p.add_argument("--artists", default="artists_rock_metal.txt",
                   help="Text file with one artist name per line (filter)")
    p.add_argument("--n",       type=int, default=50,
                   help="Number of GP files to evaluate")
    p.add_argument("--seed",    type=int, default=42,
                   help="Random seed for file sampling")
    p.add_argument("--min_notes", type=int, default=16,
                   help="Minimum note count to include a sequence")
    p.add_argument("--max_notes", type=int, default=128,
                   help="Maximum note count per sequence (truncates)")
    p.add_argument("--postprocess", action="store_true", default=True,
                   help="Apply Edwards et al. post-processing heuristic")
    p.add_argument("--no_postprocess", dest="postprocess", action="store_false")
    p.add_argument("--fret_bias", type=float, default=0.05,
                   help="A* fret bias weight")
    p.add_argument("--transition_bias", type=float, default=0.12,
                   help="A* horizontal transition bias weight")
    p.add_argument("--string_bias", type=float, default=0.0,
                   help="A* vertical string distance bias (default 0 — "
                        "explicit string penalty hurts accuracy; neural "
                        "network encodes string preference implicitly)")
    p.add_argument("--greedy", action="store_true",
                   help="Greedy argmax decoding — bypasses A* entirely. "
                        "Use to measure neural network contribution without "
                        "path optimisation.")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-file results")
    p.add_argument("--diffusion", action="store_true",
                   help="Use DiffusionDecoder (diffusion_tab.pt) instead of AR+A*")
    p.add_argument("--diffusion_model", default="diffusion_tab.pt",
                   help="Path to diffusion model checkpoint")
    p.add_argument("--T_diff", type=int, default=100,
                   help="Diffusion steps for DiffusionDecoder")
    p.add_argument("--temperature", type=float, default=0.5,
                   help="Sampling temperature for diffusion decoder")
    p.add_argument("--schedule", default="cosine",
                   choices=["cosine","linear"])
    return p.parse_args()


# ── Artist matching ───────────────────────────────────────────────────────────

def load_artists(path):
    """Load artist names, normalised to lowercase with no punctuation."""
    if not os.path.exists(path):
        return set()
    normalise = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    with open(path) as f:
        return {normalise(line.strip()) for line in f if line.strip()}


def artist_from_path(path, tabs_dir):
    """
    Infer artist name from the GP filename or parent directory.
    Scraped tabs are often named 'artist - song.gp5' or stored in artist/song.gp5.
    """
    rel  = os.path.relpath(path, tabs_dir)
    parts = re.split(r"[\\/]", rel)
    if len(parts) >= 2:
        return parts[0]  # directory name = artist
    # Flat layout: try 'artist - song.gp5'
    stem = os.path.splitext(parts[-1])[0]
    m = re.match(r"^(.*?)\s*[-–]\s*", stem)
    if m:
        return m.group(1).strip()
    return stem


def collect_gp_files(tabs_dir, artist_set):
    """Recursively collect GP files, filtered to artist list if provided."""
    GP_EXT = {".gp3", ".gp4", ".gp5"}
    normalise = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    files = []
    for root, _, fnames in os.walk(tabs_dir):
        for fname in fnames:
            if os.path.splitext(fname)[1].lower() not in GP_EXT:
                continue
            fpath = os.path.join(root, fname)
            if artist_set:
                artist = artist_from_path(fpath, tabs_dir)
                if normalise(artist) not in artist_set:
                    continue
            files.append(fpath)
    return files


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_file(path, decoder, min_notes, max_notes, postprocess,
                  args_verbose=False, use_diffusion=False, greedy_model=None):
    """
    Parse one GP file, run decoder, compute string agreement.
    Returns dict or None if file can't be parsed / too few notes.
    """
    notes_raw, _ = parse_scoreset_gp(path)
    if len(notes_raw) < min_notes:
        return None

    notes   = notes_raw[:max_notes]
    gt_pos  = [(s, f) for s, f, *_ in notes]
    pitches = [pos_to_midi(s, f) for s, f in gt_pos]

    try:
        if greedy_model is not None:
            decoded = greedy_decode(greedy_model, pitches)
        else:
            decoded = decoder.decode(pitches)
    except Exception as e:
        if args_verbose:
            print(f"  SKIP decode error: {type(e).__name__}: {e}  {os.path.basename(path)}")
        return None

    if postprocess and hasattr(decoder, 'postprocess'):
        decoded, n_fixed = decoder.postprocess(decoded)
    else:
        n_fixed = 0

    pred_pos = [(d["string"], d["fret"]) for d in decoded]
    n = len(gt_pos)
    exact = sum(g == p for g, p in zip(gt_pos, pred_pos))
    pitch_ok = sum(
        pos_to_midi(*g) == pos_to_midi(*p)
        for g, p in zip(gt_pos, pred_pos)
    )

    return {
        "file":         os.path.basename(path),
        "n_notes":      n,
        "exact":        exact,
        "pitch_ok":     pitch_ok,
        "n_fixed":      n_fixed,
        "exact_pct":    exact / n,
        "pitch_pct":    pitch_ok / n,
    }


def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.diffusion:
        # ── Diffusion decoder ─────────────────────────────────────────────────
        from diffusion_tab import DiffusionTabModel, DiffusionDecoder
        ckpt = args.diffusion_model
        if not os.path.exists(ckpt):
            print(f"[ERROR] Diffusion checkpoint not found: {ckpt}")
            sys.exit(1)
        diff_model = DiffusionTabModel(
            T_diff=args.T_diff, dropout=0.0,
            freeze_pitch_encoder=True, schedule=args.schedule)
        diff_model.load_state_dict(torch.load(ckpt, map_location=device))
        diff_model.to(device).eval()
        decoder = DiffusionDecoder(diff_model, device=device)
        # Wrap to match postprocess API expected by evaluate_file
        decoder.postprocess = lambda decoded: (decoded, 0)
        print(f"Loaded diffusion model: {ckpt}  "
              f"({sum(p.numel() for p in diff_model.parameters()):,} params)")
        print(f"Decoder: D3PM  T_diff={args.T_diff}  temp={args.temperature}  "
              f"schedule={args.schedule}")
        # Patch decode to pass temperature
        _orig_decode = decoder.decode
        decoder.decode = lambda pitches: _orig_decode(
            pitches, temperature=args.temperature)
        greedy_model = None
    else:
        # ── AR + A* decoder ───────────────────────────────────────────────────
        model = FretboardTransformer(dropout=0.0)
        if not os.path.exists(args.model):
            print(f"[ERROR] Model checkpoint not found: {args.model}")
            sys.exit(1)
        state = torch.load(args.model, map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()
        print(f"Loaded: {args.model}  "
              f"({sum(p.numel() for p in model.parameters()):,} params)")
        decoder = TrellisDecoder(model,
                                 fret_bias=0.0 if args.greedy else args.fret_bias,
                                 transition_bias=0.0 if args.greedy else args.transition_bias,
                                 string_bias=0.0)
        if args.greedy:
            print("Mode: greedy argmax (no graph search, no path optimisation)")
        greedy_model = model if args.greedy else None

    # ── Collect files ─────────────────────────────────────────────────────────
    if args.manifest:
        # Use pre-built clean manifest (produced by build_eval_set.py)
        if not os.path.exists(args.manifest):
            print(f"[ERROR] Manifest not found: {args.manifest}")
            sys.exit(1)
        with open(args.manifest) as f:
            all_files = [l.strip() for l in f if l.strip()]
        all_files = [f for f in all_files if os.path.exists(f)]
        print(f"Manifest: {args.manifest}  ({len(all_files)} files)")
    else:
        if not os.path.isdir(args.tabs):
            print(f"[ERROR] Tabs directory not found: {args.tabs}")
            sys.exit(1)
        artist_set = load_artists(args.artists)
        print(f"Artist filter: {len(artist_set)} artists loaded "
              f"({'no filter' if not artist_set else 'filtering enabled'})")
        all_files = collect_gp_files(args.tabs, artist_set)
        print(f"GP files found: {len(all_files)}")
        if not all_files:
            print("[ERROR] No GP files found — check --tabs and --artists paths.")
            sys.exit(1)

    seed = args.seed if args.seed != 0 else int(time.time())
    random.seed(seed)
    random.shuffle(all_files)
    sample = all_files

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = []
    skipped = 0
    t0 = time.time()

    for path in sample:
        if len(results) >= args.n:
            break
        r = evaluate_file(path, decoder, args.min_notes, args.max_notes,
                          args.postprocess, args.verbose,
                          use_diffusion=args.diffusion,
                          greedy_model=greedy_model)
        if r is None:
            skipped += 1
            continue
        results.append(r)
        if args.verbose:
            print(f"  {r['exact_pct']:5.1%} exact  {r['pitch_pct']:5.1%} pitch  "
                  f"n={r['n_notes']:3d}  fixed={r['n_fixed']}  {r['file']}")

    elapsed = time.time() - t0

    # ── Aggregate ─────────────────────────────────────────────────────────────
    if not results:
        print("[ERROR] No files evaluated successfully.")
        sys.exit(1)

    total_notes = sum(r["n_notes"]  for r in results)
    total_exact = sum(r["exact"]    for r in results)
    total_pitch = sum(r["pitch_ok"] for r in results)
    total_fixed = sum(r["n_fixed"]  for r in results)

    mean_exact = total_exact / total_notes
    mean_pitch = total_pitch / total_notes

    per_file_exact = [r["exact_pct"] for r in results]
    per_file_exact.sort()
    median_exact = per_file_exact[len(per_file_exact) // 2]

    print(f"\n{'═'*52}")
    print(f"  Files evaluated : {len(results)}  (skipped {skipped} — too short)")
    print(f"  Total notes     : {total_notes:,}")
    print(f"  Post-processing : {'ON' if args.postprocess else 'OFF'}  "
          f"({total_fixed} notes relocated, "
          f"{total_fixed/total_notes*100:.2f}%)")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  String agreement (exact match)")
    print(f"    Macro mean    : {mean_exact:6.1%}")
    print(f"    Micro median  : {median_exact:6.1%}")
    print(f"  Pitch accuracy  : {mean_pitch:6.1%}")
    print(f"  Time            : {elapsed:.1f}s  "
          f"({elapsed/len(results):.2f}s/file)")
    print(f"{'═'*52}")

    # Histogram of per-file exact accuracy
    bins = [(0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]
    labels = ["0–25%", "25–50%", "50–75%", "75–100%"]
    print("\n  Per-file distribution:")
    for (lo, hi), label in zip(bins, labels):
        count = sum(1 for x in per_file_exact if lo <= x < hi)
        bar = "█" * count
        print(f"    {label}  {bar} {count}")

    return results


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
