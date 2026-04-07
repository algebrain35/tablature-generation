"""
build_markov.py — Build a Markov chain pitch model from the training cache.

Reads the dataset cache (same format as fretboard_transformer.py), extracts
pitch sequences, and builds an n-gram transition model. Saves a compact
.pkl file that generate.py can load for model-driven pitch generation.

Usage:
    python build_markov.py
    python build_markov.py --cache dataset_cache.pkl --order 4 --out markov.pkl
"""

import os, sys, glob, argparse, pickle, random
from collections import defaultdict

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build Markov pitch model from cache")
    p.add_argument("--cache",   default="dataset_cache.pkl",
                   help="Base cache path (script finds the versioned file)")
    p.add_argument("--order",   type=int, default=3,
                   help="Markov order — context length (default 3)")
    p.add_argument("--out",     default="markov.pkl",
                   help="Output path for the Markov model")
    p.add_argument("--min_count", type=int, default=2,
                   help="Minimum transition count to keep (filters noise)")
    return p.parse_args()


# ── Cache loading ─────────────────────────────────────────────────────────────

def find_cache(base_path):
    """Find the versioned cache file matching the base path."""
    pattern = base_path + ".*.v*"
    matches = glob.glob(pattern)
    if not matches:
        # Try exact path too
        if os.path.exists(base_path):
            return base_path
        return None
    # Pick newest by mtime
    return max(matches, key=os.path.getmtime)


def load_pitch_sequences(cache_path):
    """Load pitch sequences from the training cache."""
    print(f"Loading cache: {cache_path}")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    print(f"  {len(data)} sequences loaded")

    sequences = []
    for entry in data:
        # Cache format: (source, key, pos, pit, ts, vel, dur)
        if len(entry) >= 4:
            pit = entry[3]   # pitch sequence (list of MIDI values)
            if len(pit) >= 4:
                sequences.append(list(pit))
    print(f"  {len(sequences)} valid pitch sequences")
    return sequences


# ── Markov model ──────────────────────────────────────────────────────────────

def build_markov(sequences, order=3, min_count=2):
    """
    Build n-gram transition counts from pitch sequences.

    counts[context_tuple][next_pitch] = count
    Also stores unigram distribution as fallback.
    """
    counts    = defaultdict(lambda: defaultdict(int))
    unigrams  = defaultdict(int)

    for seq in sequences:
        for p in seq:
            unigrams[p] += 1
        for i in range(len(seq) - order):
            context  = tuple(seq[i : i + order])
            next_p   = seq[i + order]
            counts[context][next_p] += 1

    # Prune low-count transitions
    pruned = 0
    for ctx in list(counts.keys()):
        for p in list(counts[ctx].keys()):
            if counts[ctx][p] < min_count:
                del counts[ctx][p]
                pruned += 1
        if not counts[ctx]:
            del counts[ctx]

    print(f"  {len(counts):,} contexts  ({pruned} transitions pruned)")
    return dict(counts), dict(unigrams)


def backoff_counts(counts, context, order):
    """
    Katz-style backoff: try full context, then progressively shorter.
    Returns the transition dict for the longest matching context found.
    """
    for length in range(order, 0, -1):
        ctx = tuple(context[-length:])
        if ctx in counts:
            return counts[ctx]
    return None   # no context found — caller uses unigrams


def save_markov(counts, unigrams, order, path):
    model = {"counts": counts, "unigrams": unigrams, "order": order}
    with open(path, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {path}  ({os.path.getsize(path) / 1024:.0f} KB)")


# ── Validation: quick sample ──────────────────────────────────────────────────

def sample_sequence(counts, unigrams, order, length=32, seed=42):
    """Generate a short sequence from the model to verify it works."""
    random.seed(seed)
    # Start from a random unigram
    pitches = random.choices(list(unigrams.keys()),
                             weights=list(unigrams.values()), k=order)
    for _ in range(length - order):
        ctx  = pitches[-order:]
        dist = backoff_counts(counts, ctx, order)
        if dist:
            nxt = random.choices(list(dist.keys()),
                                 weights=list(dist.values()))[0]
        else:
            nxt = random.choices(list(unigrams.keys()),
                                 weights=list(unigrams.values()))[0]
        pitches.append(nxt)
    return pitches


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    cache_path = find_cache(args.cache)
    if cache_path is None:
        print(f"[ERROR] Cache not found: {args.cache}")
        print("  Run fretboard_transformer.py once to build the cache first.")
        sys.exit(1)

    sequences          = load_pitch_sequences(cache_path)
    counts, unigrams   = build_markov(sequences, args.order, args.min_count)
    save_markov(counts, unigrams, args.order, args.out)

    # Quick sanity check
    sample = sample_sequence(counts, unigrams, args.order)
    print(f"\nSample sequence (MIDI): {sample}")
    print(f"  Range: {min(sample)}–{max(sample)}")
    print(f"\nDone. Use with generate.py --markov {args.out}")


if __name__ == "__main__":
    main()
