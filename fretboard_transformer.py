"""
fretboard_transformer.py — Re-export shim + training entrypoint.

All implementation lives in:
    ft_model.py  — neural architecture (FretboardTransformer, FretboardTransformerMLM, ...)
    ft_data.py   — data parsing, StreamingDataset, FretboardTrainer
    ft_decode.py — TrellisDecoder, postprocess

This file re-exports everything for backward compatibility with generate.py,
evaluate.py, infer.py, and generate_diffusion.py.
"""

# ── Re-exports ────────────────────────────────────────────────────────────────

from ft_model import (
    # Constants
    NUM_STRINGS, NUM_FRETS, NUM_POSITIONS, NUM_MIDI, PAD_IDX,
    N_TS_BINS, N_VEL_BINS, N_DUR_BINS,
    STANDARD_TUNING, POSITIONS, POS_TO_IDX,
    DADAGP_GUITAR_PREFIXES, _NOTE_RE,
    GUITARSET_OPEN_MIDI, SYNTHTAB_TRACK_OPEN_MIDI,
    _GP_DUR_TICKS,
    # Helpers
    quantize_ticks, quantize_velocity,
    pos_to_midi, midi_to_positions, dadagp_str,
    precompute_rope_freqs, apply_rope,
    # Architecture
    RoPEAttention, RoPETransformerLayer,
    PitchEncoder,
    PositionEncoder, BidirectionalPositionEncoder,
    KeyEmbedding,
    FretboardTransformer, FretboardTransformerMLM,
)

from ft_data import (
    estimate_key,
    parse_dadagp_file, parse_guitarset_jams,
    parse_scoreset_gp, parse_synthtab_dir,
    transpose_window,
    StreamingDataset, FretboardTrainer,
)

from ft_decode import TrellisDecoder

__all__ = [
    # Constants
    "NUM_STRINGS", "NUM_FRETS", "NUM_POSITIONS", "NUM_MIDI", "PAD_IDX",
    "N_TS_BINS", "N_VEL_BINS", "N_DUR_BINS",
    "STANDARD_TUNING", "POSITIONS", "POS_TO_IDX",
    # Helpers
    "quantize_ticks", "quantize_velocity",
    "pos_to_midi", "midi_to_positions", "dadagp_str",
    # Architecture
    "FretboardTransformer", "FretboardTransformerMLM",
    "PitchEncoder", "PositionEncoder", "BidirectionalPositionEncoder",
    # Training / data
    "StreamingDataset", "FretboardTrainer",
    # Decoding
    "TrellisDecoder",
]


# ── Training entrypoint ───────────────────────────────────────────────────────
# This configuration produced 68.2% GP string agreement (50-file eval, seed 42).
# Key decisions:
#   - BART+RoPE: bidirectional PitchEncoder (4 layers) + causal PositionEncoder (2 layers)
#   - 4 heads: position (126 classes), duration (17 bins), pitch (128 MIDI), tonic (12)
#   - Source-conditional pit_loss: only backpropped when MIDI seqs present in batch
#   - MIDI 1:1 cap: MIDI sequences capped to same count as GP/SynthTab sequences
#   - Shared MASK/PAD token (PAD_IDX = NUM_POSITIONS = 126) — dedicated PAD caused regression
#   - Loss: pos_loss + 0.3*dur_loss + 0.2*pit_loss + 0.1*tonic_loss
#   - A*: fret_bias=0.05, transition_bias=0.12 (inadmissible weighted A*, empirically better)

if __name__ == "__main__":
    import argparse, torch

    p = argparse.ArgumentParser(description="Train best-performing AR fretboard model")
    p.add_argument("--scraped_tabs_dir", default=None,
                   help="Path to scraped GuitarPro tabs. DISABLED by default — "
                        "scraped tabs introduce high-fret bias and low-quality "
                        "transcriptions that reduce GP generalisation by ~6pp. "
                        "Use only for ablation studies.")
    p.add_argument("--epochs",  type=int, default=20,
                   help="Training epochs (default 20 — more data warrants more training)")
    p.add_argument("--no_midi", action="store_true",
                   help="Disable MIDI augmentation even if midi_sequences.pkl exists")
    p.add_argument("--cache_path", default="dataset_cache.pkl")
    args = p.parse_args()

    midi_cache = None
    if not args.no_midi:
        import os
        midi_cache = "./midi_sequences.pkl" if os.path.exists("./midi_sequences.pkl") else None
        if midi_cache:
            print(f"MIDI augmentation: enabled ({midi_cache})")
        else:
            print("MIDI augmentation: disabled (midi_sequences.pkl not found)")

    def _show(name, decoded, dec):
        print(f"\n--- {name} ---")
        for note in decoded:
            tag = "(open)" if note["fret"] == 0 else "(fretted)"
            print(f"  string {note['string']}, fret {note['fret']:>2}  {tag}"
                  f"  MIDI {note['midi']}")
        result, n_fixed = dec.postprocess(decoded)
        if n_fixed:
            print(f"  [{n_fixed} note(s) relocated by post-processing]")

    # ── Model (best config: ~1.23M params) ───────────────────────────────────
    model = FretboardTransformer(
        embed_dim    = 256,
        num_heads    = 8,
        pitch_layers = 4,   # bidirectional pitch encoder depth
        pos_layers   = 2,   # causal position decoder depth
        ffn_dim      = 512,
        dropout      = 0.1, # model dropout (trainer overrides with 0.3 for training)
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Trainer (best config) ─────────────────────────────────────────────────
    trainer = FretboardTrainer(
        model,
        dadagp_dir        = "./assets/data/dadagp/",
        synthtab_dir      = "./assets/data/synthtab/outall/",
        scoreset_dir      = None,
        scraped_tabs_dir  = args.scraped_tabs_dir,  # None by default — see note above
        proggp_dir        = None,
        midi_cache        = midi_cache,
        window            = 64,
        batch_size        = 32,
        lr                = 3e-4,
        genres            = ["metal", "rock", "hard_rock"],
        num_workers       = 16,
        queue_size        = 4096,
        steps_per_epoch   = 500,
        augment_semitones = (-3, -2, -1, 1, 2, 3),
        val_split         = 0.1,
        mask_prob         = 0.40,   # masked-LM only — unused in AR mode
        dropout           = 0.3,    # training dropout
        training_mode     = 'ar',
        max_source_fraction = 0.8,  # no single source > 80% of training data
        epochs_per_10k    = 20,
        max_epochs        = 60,
        cache_path        = args.cache_path,
    )

    n_train = len(trainer.ds)
    print(f"Running {args.epochs} epochs on {n_train} train sequences.", flush=True)
    trainer.train(epochs=args.epochs)

    # ── Post-training evaluation on benchmark cases ───────────────────────────
    model.load_state_dict(
        torch.load("fretboard_transformer.pt", map_location='cpu'), strict=False)
    dec = TrellisDecoder(model, fret_bias=0.05, transition_bias=0.12)

    print("\n" + "═"*56)
    print("  Post-training benchmark (A* decoder, no post-processing)")
    print("═"*56)
    for name, pitches in [
        ("E major scale",                 [64,66,68,69,71,73,75,76]),
        ("Open string ascent",            [40,45,50,55,59,64]),
        ("Fade to Black — rhythm riff",   [47,54,55,54,47,54,55,54,50,45,54,55,54,45,54,55,54,50]),
        ("Fade to Black — lead run 1",    [59,66,67,69,67,69,71,69,67,69]),
        ("Fade to Black — lead run 2",    [69,71,69,67,66,67,66,64,66,67,66,66,64,64,62]),
        ("Fade to Black — high register", [83,78,79,78,74,74,76,74,73,71,69,71,69]),
        ("Fade to Black — ascending run", [57,59,60,62,64,65,67,69,71,72,74,76,77,79,76,77,76,74,81]),
        ("Reverend — main riff",          [42,49,52,57,61,45,52,59,59,59,59,40,52,59,59,59,59,50,64,61,59,61,59,64]),
        ("Reverend — solo lick 1",        [64,64,63,69,69,61,68,68,64,71,71,63]),
    ]:
        _show(name, dec.decode(pitches), dec)
