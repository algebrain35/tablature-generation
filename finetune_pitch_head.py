"""
finetune_pitch_head.py — Fine-tune pitch_head on raw MIDI pitch sequences.

The MLM model's pitch_head is an auxiliary output trained at 0.2× weight
alongside the primary position loss.  This script isolates pitch generation
quality by fine-tuning ONLY pitch_head and pitch_enc on Lakh MIDI sequences,
leaving the position decoder weights frozen.

Why this helps
--------------
During joint training, pos_loss dominates gradient flow through the shared
fusion layer.  pitch_head sees useful signal only on the ~50% of batches
that contain MIDI-only sequences, and even then at 0.2× weight.  This pass
gives pitch_head 100% of the gradient budget, letting it learn melodic
sequence priors (step motion, phrase shape, resolution) that the joint
training leaves partially learned.

Usage
-----
    # Fine-tune the MLM model (default)
    python finetune_pitch_head.py

    # Fine-tune the AR model's pitch_head instead
    python finetune_pitch_head.py --model fretboard_transformer.pt --ar

    # Point at your MIDI cache and run for 10 epochs
    python finetune_pitch_head.py --midi midi_sequences.pkl --epochs 10

    # Dry run — print dataset stats and exit without training
    python finetune_pitch_head.py --dry_run

Outputs
-------
    fretboard_transformer_mlm_ft.pt   (default output)

The original checkpoint is never overwritten unless you pass --in_place.
"""

import os, sys, argparse, random, pickle, math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ft_model import (
    FretboardTransformer, FretboardTransformerMLM,
    NUM_MIDI, NUM_POSITIONS, N_DUR_BINS,
    PITCH_MASK_IDX,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune pitch_head on MIDI pitch sequences")
    p.add_argument("--model",    default="fretboard_transformer_mlm.pt",
                   help="Checkpoint to fine-tune (default: MLM checkpoint)")
    p.add_argument("--out",      default=None,
                   help="Output path (default: <model>_ft.pt)")
    p.add_argument("--ar",       action="store_true",
                   help="Load as AR FretboardTransformer instead of MLM")
    p.add_argument("--midi",     default="midi_sequences.pkl",
                   help="MIDI cache pkl from parse_midi.py (default: midi_sequences.pkl)")
    p.add_argument("--epochs",   type=int,   default=5,
                   help="Fine-tuning epochs (default 5 — short, targeted)")
    p.add_argument("--lr",       type=float, default=3e-5,
                   help="Learning rate (default 3e-5 — lower than joint training)")
    p.add_argument("--batch",    type=int,   default=64)
    p.add_argument("--window",   type=int,   default=64,
                   help="Sequence window length (must match training window)")
    p.add_argument("--mask",     type=float, default=0.15,
                   help="Pitch mask probability for MLM fine-tuning (default 0.15)")
    p.add_argument("--steps",    type=int,   default=500,
                   help="Steps per epoch (default 500)")
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--dry_run",  action="store_true",
                   help="Print dataset stats and exit without training")
    p.add_argument("--in_place", action="store_true",
                   help="Overwrite input checkpoint (default: save to _ft.pt)")
    p.add_argument("--unfreeze_fusion", action="store_true",
                   help="Also unfreeze the fusion layer (slightly higher risk of "
                        "degrading position accuracy — use with low lr)")
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_midi_sequences(path, window):
    """
    Load MIDI cache produced by parse_midi.py.
    Returns a list of pitch-sequence windows (each a list of int MIDI pitches).

    The cache entries are either:
      - (source, key, pos_list, pit_list, ts_list, vel_list, dur_list)  [v8 format]
      - raw pitch lists                                                   [older format]

    We only need the pitch list — everything else is ignored here.
    """
    print(f"Loading MIDI cache: {path}", flush=True)
    with open(path, 'rb') as f:
        data = pickle.load(f)

    sequences = []
    skipped   = 0
    for entry in data:
        try:
            if isinstance(entry, (list, tuple)) and len(entry) >= 4:
                # v8 format: (source, key, pos, pit, ts, vel, dur)
                pit = list(entry[3]) if len(entry) > 3 else list(entry)
            else:
                pit = list(entry)
            pit = [int(p) for p in pit if 0 <= int(p) < NUM_MIDI]
            if len(pit) < 4:
                skipped += 1
                continue
            # Slice into windows (with random start for variety)
            for start in range(0, len(pit), window // 2):
                chunk = pit[start : start + window]
                if len(chunk) < 4:
                    break
                # Pad to window length if needed
                if len(chunk) < window:
                    chunk = chunk + [chunk[-1]] * (window - len(chunk))
                sequences.append(chunk)
        except Exception:
            skipped += 1

    print(f"  {len(sequences):,} windows from {len(data):,} sequences "
          f"({skipped} skipped)", flush=True)
    return sequences


def get_batch(sequences, batch_size, window, device):
    """Sample a random batch of pitch windows."""
    idxs = random.choices(range(len(sequences)), k=batch_size)
    batch = []
    for i in idxs:
        seq = sequences[i]
        # Random crop to exactly `window` notes
        if len(seq) > window:
            start = random.randint(0, len(seq) - window)
            seq = seq[start : start + window]
        batch.append(seq)
    return torch.tensor(batch, dtype=torch.long, device=device)  # (B, T)


# ── Frozen parameter logic ────────────────────────────────────────────────────

def freeze_for_pitch_finetuning(model, unfreeze_fusion=False):
    """
    Freeze everything except pitch_enc and pitch_head.
    Optionally unfreeze the fusion layer.

    Frozen:  pos_enc, head (position output), dur_head, tonic_head
    Unfrozen: pitch_enc, pitch_head, [fusion if --unfreeze_fusion]
    """
    # Freeze all parameters first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze pitch encoder and pitch prediction head
    for p in model.pitch_enc.parameters():
        p.requires_grad = True
    for p in model.pitch_head.parameters():
        p.requires_grad = True

    if unfreeze_fusion:
        for p in model.fusion.parameters():
            p.requires_grad = True

    frozen   = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen params   : {frozen:,}")
    print(f"Trainable params: {unfrozen:,}  (pitch_enc + pitch_head"
          f"{' + fusion' if unfreeze_fusion else ''})")


# ── Training ──────────────────────────────────────────────────────────────────

def compute_ar_pitch_loss(model, pitches_batch):
    """
    Autoregressive next-pitch loss on a batch of pitch sequences.

    Uses the model's pitch encoder to produce context vectors, then
    predicts the next pitch at each timestep using pitch_head.
    Positions are all set to MASK so the position decoder contributes
    nothing — this loss is purely about pitch sequence modelling.

    pitches_batch : (B, T) long tensor
    Returns       : scalar loss
    """
    B, T   = pitches_batch.shape
    device = pitches_batch.device

    # All positions masked — position decoder contributes nothing meaningful
    # This isolates pitch_enc + pitch_head from the position pathway
    mask_pos = torch.full((B, T - 1), model.MASK_IDX,
                          dtype=torch.long, device=device)

    pos_logits, dur_logits, pit_logits, _ = model(
        pitches_batch,   # full pitch sequence as input
        mask_pos,        # all positions masked
    )
    # pit_logits: (B, T-1, NUM_MIDI) — predicting pitch t+1 from context at t
    # Target   : pitches_batch[:, 1:]  — shift by 1 for next-pitch prediction
    tgt   = pitches_batch[:, 1:]           # (B, T-1)
    B, Tm, V = pit_logits.shape
    return F.nll_loss(pit_logits.reshape(B * Tm, V), tgt.reshape(B * Tm))


def compute_mlm_pitch_loss(model, pitches_batch, mask_prob):
    """
    Masked-LM pitch loss: randomly mask some pitch tokens, predict them
    from bidirectional context. Positions are all masked so the position
    pathway contributes nothing.

    pitches_batch : (B, T) long tensor
    mask_prob     : fraction of pitch tokens to mask (e.g. 0.15)
    Returns       : scalar loss
    """
    B, T   = pitches_batch.shape
    device = pitches_batch.device

    # Randomly mask pitch tokens
    pit_mask  = torch.rand(B, T, device=device) < mask_prob
    if not pit_mask.any():
        # Guarantee at least one masked token per batch to avoid zero-loss step
        pit_mask[torch.randint(B, (1,)), torch.randint(T, (1,))] = True

    inp_pit = pitches_batch.clone()
    inp_pit[pit_mask] = PITCH_MASK_IDX

    # All positions masked
    mask_pos = torch.full((B, T), model.MASK_IDX, dtype=torch.long, device=device)

    pos_logits, dur_logits, pit_logits, _ = model(inp_pit, mask_pos)

    # Loss only on masked pitch positions
    B2, T2, V = pit_logits.shape
    return F.nll_loss(
        pit_logits.reshape(B2 * T2, V)[pit_mask.reshape(B2 * T2)],
        pitches_batch.reshape(B2 * T2)[pit_mask.reshape(B2 * T2)],
    )


def run_training(model, sequences, args, device, is_mlm):
    """Main fine-tuning loop."""
    # Only optimise the unfrozen parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim     = torch.optim.Adam(trainable, lr=args.lr, weight_decay=1e-4)

    # Cosine annealing over the full run
    total_steps = args.epochs * args.steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=total_steps, eta_min=args.lr * 0.05)

    best_loss = float('inf')
    out_path  = args.out

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0

        for step in range(args.steps):
            pitches = get_batch(sequences, args.batch, args.window, device)

            optim.zero_grad()
            if is_mlm:
                loss = compute_mlm_pitch_loss(model, pitches, args.mask)
            else:
                loss = compute_ar_pitch_loss(model, pitches)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()
            sched.step()
            tot_loss += loss.item()

        avg = tot_loss / args.steps
        improved = " ✓" if avg < best_loss else ""
        print(f"Epoch {epoch:>2}/{args.epochs}  "
              f"pitch_loss={avg:.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}{improved}", flush=True)

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), out_path)
            print(f"  Saved: {out_path}", flush=True)

    return best_loss


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Output path ──────────────────────────────────────────────────────────
    if args.out is None:
        if args.in_place:
            args.out = args.model
        else:
            base, ext = os.path.splitext(args.model)
            args.out  = base + "_ft" + (ext or ".pt")

    # ── Load MIDI data ────────────────────────────────────────────────────────
    if not os.path.exists(args.midi):
        print(f"[ERROR] MIDI cache not found: {args.midi}")
        print("  Generate it with: python parse_midi.py --midi_dir ./assets/data/lakh_midi/ "
              "--guitar_only --out midi_sequences.pkl")
        sys.exit(1)

    sequences = load_midi_sequences(args.midi, args.window)
    if len(sequences) < args.batch:
        print(f"[ERROR] Only {len(sequences)} windows — need at least {args.batch}")
        sys.exit(1)

    if args.dry_run:
        print(f"\nDry run — {len(sequences):,} pitch windows available.")
        print(f"Would train {args.epochs} epochs × {args.steps} steps "
              f"(batch={args.batch}, window={args.window})")
        print(f"Output: {args.out}")
        return

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"[ERROR] Checkpoint not found: {args.model}")
        sys.exit(1)

    is_mlm = not args.ar
    ModelClass = FretboardTransformerMLM if is_mlm else FretboardTransformer
    model = ModelClass(dropout=0.1)  # small dropout for fine-tuning regularisation
    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model   : {args.model}  ({total_params:,} params, "
          f"{'MLM' if is_mlm else 'AR'})")

    # ── Freeze everything except pitch_enc + pitch_head ───────────────────────
    freeze_for_pitch_finetuning(model, unfreeze_fusion=args.unfreeze_fusion)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    model.to(device)

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nFine-tuning pitch_head: {args.epochs} epochs × {args.steps} steps")
    print(f"  lr={args.lr}  batch={args.batch}  window={args.window}")
    print(f"  loss={'masked-LM' if is_mlm else 'autoregressive next-pitch'}")
    print(f"  Output → {args.out}\n")

    best = run_training(model, sequences, args, device, is_mlm)
    print(f"\nDone. Best pitch_loss: {best:.4f}")
    print(f"Fine-tuned checkpoint saved to: {args.out}")
    print()
    print("To use in generate_diffusion.py:")
    print(f"  python generate_diffusion.py --model {args.out} "
          "--key A --scale minor_pentatonic --length 64")


if __name__ == "__main__":
    main()
