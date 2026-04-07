"""
build_eval_set.py — Build a clean, deduplicated Guitar Pro evaluation set.

Collects GP files from one or more evaluation directories, removes any files
that overlap with training directories (by filename slug, file hash, and
artist+song fingerprint), then samples a fixed-size held-out set and saves
the file list as a manifest.

Usage:
    # Build 50-file eval set, excluding anything in training dirs
    python build_eval_set.py \\
        --eval_dirs  ./assets/data/gprotab/ \\
        --train_dirs ./assets/data/synthtab/outall/ \\
                     ./assets/data/scoreset/outall/ \\
                     ./assets/data/scraped_tabs/ \\
        --n 50 --seed 42 \\
        --out eval_manifest.txt

    # Then evaluate using the clean manifest
    python evaluate.py \\
        --manifest eval_manifest.txt \\
        --model fretboard_transformer.pt

Output:
    eval_manifest.txt — one absolute filepath per line, deduplicated and
                        shuffled with the given seed. Safe to re-use across
                        evaluation runs.
"""

import os, re, sys, glob, hashlib, argparse, random, json
from collections import defaultdict


# ── Helpers ───────────────────────────────────────────────────────────────────

GP_EXT = {".gp3", ".gp4", ".gp5", ".gpx", ".gp"}


def slugify(s):
    """Lowercase alphanumeric only — used for fuzzy filename matching."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def file_hash(path, block=65536):
    """MD5 of file contents — catches exact duplicates regardless of filename."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(block):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def song_fingerprint(path):
    """
    (artist_slug, song_slug) tuple derived from filepath.
    Handles both:
      - Directory layout:  artist_slug/song_slug.gp5
      - Flat layout:       artist_slug - song_slug.gp5
    """
    stem   = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(path))

    # Directory layout — parent dir is the artist
    if slugify(parent) and slugify(parent) not in {"gprotab", "scraped", "tabs",
                                                    "outall", "data", "assets"}:
        return (slugify(parent), slugify(stem))

    # Flat layout — try "Artist - Song" naming
    m = re.match(r"^(.*?)\s*[-–]\s*(.+)$", stem)
    if m:
        return (slugify(m.group(1)), slugify(m.group(2)))

    return (None, slugify(stem))


def collect_gp_files(directories):
    """Recursively collect all GP files from a list of directories."""
    files = []
    for d in directories:
        if not os.path.isdir(d):
            print(f"  [WARN] Directory not found, skipping: {d}", flush=True)
            continue
        for ext in GP_EXT:
            files.extend(glob.glob(
                os.path.join(d, "**", f"*{ext}"), recursive=True))
    return [os.path.abspath(f) for f in files]


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Build a clean, deduplicated GP evaluation manifest")

    p.add_argument("--eval_dirs",  nargs="+", required=True,
                   help="Directories to source evaluation GP files from")
    p.add_argument("--train_dirs", nargs="+", default=[],
                   help="Training directories — any overlap will be excluded")
    p.add_argument("--n",          type=int,  default=50,
                   help="Number of files in the final eval set (default 50)")
    p.add_argument("--seed",       type=int,  default=42,
                   help="Random seed for sampling (default 42)")
    p.add_argument("--out",        default="eval_manifest.txt",
                   help="Output manifest filepath (default eval_manifest.txt)")
    p.add_argument("--min_size",   type=int,  default=1024,
                   help="Minimum file size in bytes — skip tiny/empty files")
    p.add_argument("--report",     default=None,
                   help="Optional JSON report of overlap statistics")
    p.add_argument("--verbose",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rng  = random.Random(args.seed)

    # ── Step 1: Collect training fingerprints ─────────────────────────────────
    print("Building training set fingerprints...")
    train_files = collect_gp_files(args.train_dirs)
    print(f"  {len(train_files)} training GP files found")

    train_hashes      = set()
    train_slugs       = set()      # filename slugs
    train_fingerprints = set()     # (artist_slug, song_slug) pairs

    for i, path in enumerate(train_files):
        if args.verbose and i % 500 == 0:
            print(f"  Hashing training files {i}/{len(train_files)}...",
                  flush=True)
        h = file_hash(path)
        if h:
            train_hashes.add(h)
        stem = os.path.splitext(os.path.basename(path))[0]
        train_slugs.add(slugify(stem))
        fp = song_fingerprint(path)
        train_fingerprints.add(fp)

    print(f"  {len(train_hashes)} unique file hashes")
    print(f"  {len(train_fingerprints)} unique (artist, song) fingerprints")

    # ── Step 2: Collect evaluation candidates ─────────────────────────────────
    print("\nCollecting evaluation candidates...")
    eval_candidates = collect_gp_files(args.eval_dirs)
    eval_candidates = [f for f in eval_candidates
                       if os.path.getsize(f) >= args.min_size]
    print(f"  {len(eval_candidates)} candidate files (≥{args.min_size}B)")

    # ── Step 3: Deduplicate within eval set itself ────────────────────────────
    print("\nDeduplicating evaluation candidates...")
    seen_hashes = set()
    seen_fps    = set()
    unique_eval = []
    n_dup_hash  = 0
    n_dup_fp    = 0

    for path in eval_candidates:
        h  = file_hash(path)
        fp = song_fingerprint(path)

        if h and h in seen_hashes:
            n_dup_hash += 1
            continue
        if fp in seen_fps and fp[0] is not None:
            n_dup_fp += 1
            continue

        if h:
            seen_hashes.add(h)
        seen_fps.add(fp)
        unique_eval.append(path)

    print(f"  Removed {n_dup_hash} exact duplicates (same file hash)")
    print(f"  Removed {n_dup_fp} near-duplicates (same artist+song slug)")
    print(f"  {len(unique_eval)} unique eval candidates remain")

    # ── Step 4: Remove training overlap ───────────────────────────────────────
    print("\nChecking for training/eval overlap...")
    clean_eval  = []
    n_overlap_hash = 0
    n_overlap_slug = 0
    n_overlap_fp   = 0
    overlap_files  = []

    for path in unique_eval:
        h    = file_hash(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        fp   = song_fingerprint(path)
        slug = slugify(stem)

        reason = None
        if h and h in train_hashes:
            reason = "exact hash match"
            n_overlap_hash += 1
        elif slug in train_slugs:
            reason = "filename slug match"
            n_overlap_slug += 1
        elif fp in train_fingerprints and fp[0] is not None:
            reason = "artist+song fingerprint match"
            n_overlap_fp += 1

        if reason:
            overlap_files.append((path, reason))
            if args.verbose:
                print(f"  OVERLAP ({reason}): {os.path.basename(path)}")
        else:
            clean_eval.append(path)

    total_overlap = n_overlap_hash + n_overlap_slug + n_overlap_fp
    print(f"  Removed {total_overlap} overlapping files:")
    print(f"    {n_overlap_hash} exact hash matches")
    print(f"    {n_overlap_slug} filename slug matches")
    print(f"    {n_overlap_fp} artist+song fingerprint matches")
    print(f"  {len(clean_eval)} clean candidates remain")

    # ── Step 5: Sample ────────────────────────────────────────────────────────
    if len(clean_eval) < args.n:
        print(f"\n[WARN] Only {len(clean_eval)} clean files available "
              f"(requested {args.n}). Using all of them.")
        args.n = len(clean_eval)

    rng.shuffle(clean_eval)
    final_set = clean_eval[:args.n]

    # ── Step 6: Write manifest ────────────────────────────────────────────────
    with open(args.out, "w") as f:
        for path in final_set:
            f.write(path + "\n")

    print(f"\n{'='*52}")
    print(f"  Eval candidates    : {len(eval_candidates)}")
    print(f"  After dedup        : {len(unique_eval)}")
    print(f"  After overlap check: {len(clean_eval)}")
    print(f"  Final eval set     : {len(final_set)}")
    print(f"  Overlap removed    : {total_overlap}")
    print(f"  Seed               : {args.seed}")
    print(f"  Manifest written   : {args.out}")
    print(f"{'='*52}")

    # ── Optional JSON report ──────────────────────────────────────────────────
    if args.report:
        report = {
            "seed":             args.seed,
            "n_requested":      args.n,
            "n_final":          len(final_set),
            "n_eval_candidates": len(eval_candidates),
            "n_after_dedup":    len(unique_eval),
            "n_clean":          len(clean_eval),
            "overlap": {
                "total":            total_overlap,
                "exact_hash":       n_overlap_hash,
                "filename_slug":    n_overlap_slug,
                "artist_song_fp":   n_overlap_fp,
            },
            "overlap_files": [
                {"path": p, "reason": r} for p, r in overlap_files
            ],
            "final_files": final_set,
        }
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  JSON report        : {args.report}")

    print(f"\nEvaluate with:")
    print(f"  python evaluate.py --manifest {args.out} "
          f"--model fretboard_transformer.pt")


if __name__ == "__main__":
    main()
