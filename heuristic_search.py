"""
heuristic_search.py — Grid or random search over A* heuristic parameters.

Searches over fret_bias and transition_bias to find the combination that
maximises string agreement on a held-out evaluation set.

Usage:
    # Grid search (exhaustive)
    python heuristic_search.py \
        --model fretboard_transformer.pt \
        --manifest eval_manifest.txt \
        --mode grid

    # Random search (faster, good for larger spaces)
    python heuristic_search.py \
        --model fretboard_transformer.pt \
        --manifest eval_manifest.txt \
        --mode random --n_trials 40

    # Fine-grained search around a known good point
    python heuristic_search.py \
        --model fretboard_transformer.pt \
        --manifest eval_manifest.txt \
        --mode grid \
        --fret_bias_range 0.02 0.08 \
        --transition_bias_range 0.08 0.16
"""

import os, sys, re, random, argparse, time, itertools
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fretboard_transformer import (
    FretboardTransformer, TrellisDecoder,
    parse_scoreset_gp, POS_TO_IDX, POSITIONS, pos_to_midi,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Grid/random search over A* heuristic params')
    p.add_argument('--model',    default='fretboard_transformer.pt')
    p.add_argument('--manifest', default=None,
                   help='Eval manifest from build_eval_set.py')
    p.add_argument('--tabs',     default='./assets/data/scraped_tabs/')
    p.add_argument('--n',        type=int, default=50,
                   help='Number of files to evaluate per trial')
    p.add_argument('--seed',     type=int, default=42)
    p.add_argument('--mode',     choices=['grid','random'], default='grid')
    p.add_argument('--n_trials', type=int, default=40,
                   help='Number of random trials (random mode only)')
    p.add_argument('--fret_bias_range', type=float, nargs=2, default=[0.0, 0.15],
                   metavar=('MIN','MAX'))
    p.add_argument('--transition_bias_range', type=float, nargs=2, default=[0.0, 0.25],
                   metavar=('MIN','MAX'))
    p.add_argument('--fret_steps',       type=int, default=6,
                   help='Grid steps for fret_bias')
    p.add_argument('--transition_steps', type=int, default=6,
                   help='Grid steps for transition_bias')
    p.add_argument('--min_notes', type=int, default=16)
    p.add_argument('--max_notes', type=int, default=128)
    return p.parse_args()


def linspace(lo, hi, n):
    if n == 1:
        return [lo]
    return [round(lo + (hi - lo) * i / (n - 1), 4) for i in range(n)]


def evaluate_params(model, files, fret_bias, transition_bias,
                    min_notes, max_notes, device):
    """Run evaluation for one (fret_bias, transition_bias) combination."""
    decoder = TrellisDecoder(model, fret_bias=fret_bias,
                              transition_bias=transition_bias, string_bias=0.0)
    total_exact = total_n = 0
    per_file = []

    for path in files:
        notes_raw, _ = parse_scoreset_gp(path)
        if len(notes_raw) < min_notes:
            continue
        notes   = notes_raw[:max_notes]
        gt_pos  = [(s, f) for s, f, *_ in notes]
        pitches = [pos_to_midi(s, f) for s, f in gt_pos]
        try:
            decoded = decoder.decode(pitches)
        except Exception:
            continue
        pred_pos = [(d['string'], d['fret']) for d in decoded]
        n     = len(gt_pos)
        exact = sum(g == p for g, p in zip(gt_pos, pred_pos))
        total_exact += exact
        total_n     += n
        per_file.append(exact / n)

    if total_n == 0:
        return 0.0, 0.0
    macro  = sum(per_file) / len(per_file)
    return macro, total_exact / total_n   # macro mean, micro mean


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model once
    model = FretboardTransformer(dropout=0.0)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f'Model loaded: {args.model}')

    # Load file list
    if args.manifest:
        with open(args.manifest) as f:
            all_files = [l.strip() for l in f if l.strip() and os.path.exists(l.strip())]
    else:
        GP_EXT = {'.gp3', '.gp4', '.gp5'}
        all_files = []
        for root, _, files in os.walk(args.tabs):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in GP_EXT:
                    all_files.append(os.path.join(root, fname))

    random.shuffle(all_files)
    files = all_files[:args.n]
    print(f'Evaluating on {len(files)} files per trial\n')

    # Build parameter list
    if args.mode == 'grid':
        fb_vals  = linspace(*args.fret_bias_range,       args.fret_steps)
        tb_vals  = linspace(*args.transition_bias_range, args.transition_steps)
        trials   = list(itertools.product(fb_vals, tb_vals))
        print(f'Grid search: {len(trials)} trials '
              f'({args.fret_steps} × {args.transition_steps})')
    else:
        trials = [
            (round(random.uniform(*args.fret_bias_range), 4),
             round(random.uniform(*args.transition_bias_range), 4))
            for _ in range(args.n_trials)
        ]
        print(f'Random search: {args.n_trials} trials')

    print(f'{"fret_bias":>10}  {"trans_bias":>10}  {"macro":>7}  {"micro":>7}')
    print('─' * 46)

    best_macro = -1
    best_params = (0.0, 0.0)
    results = []

    t0 = time.time()
    for i, (fb, tb) in enumerate(trials):
        macro, micro = evaluate_params(
            model, files, fb, tb, args.min_notes, args.max_notes, device
        )
        results.append((macro, micro, fb, tb))
        marker = ' ◀ best' if macro > best_macro else ''
        print(f'{fb:>10.4f}  {tb:>10.4f}  {macro:>7.2%}  {micro:>7.2%}{marker}')
        if macro > best_macro:
            best_macro  = macro
            best_params = (fb, tb)

    elapsed = time.time() - t0
    results.sort(reverse=True)

    print(f'\n{"═"*52}')
    print(f'  Search complete  ({elapsed:.1f}s  |  {elapsed/len(trials):.2f}s/trial)')
    print(f'  Best fret_bias       : {best_params[0]:.4f}')
    print(f'  Best transition_bias : {best_params[1]:.4f}')
    print(f'  Best macro mean      : {best_macro:.2%}')
    print(f'{"═"*52}')

    print('\nTop 5:')
    for macro, micro, fb, tb in results[:5]:
        print(f'  fret_bias={fb:.4f}  transition_bias={tb:.4f}  '
              f'macro={macro:.2%}  micro={micro:.2%}')

    print(f'\nTo reproduce best result:')
    print(f'  python evaluate.py --model {args.model} '
          f'--fret_bias {best_params[0]} --transition_bias {best_params[1]}')


if __name__ == '__main__':
    main()
