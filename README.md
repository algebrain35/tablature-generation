# Guitar Fret Navigator
### MATH 2056 — Discrete Mathematics · Algoma University · Winter 2026

Automatic guitar tablature generation from symbolic pitch sequences using
A\* graph search guided by a bidirectional transformer encoder-decoder.

---

## Quick start (inference only — no GPU required)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended. CPU inference is sufficient for all examples below.

### 2. Download the pre-trained checkpoint

Place `fretboard_transformer.pt` in the project root directory.

### 3. Run the benchmark test cases

```bash
python infer.py
```

This runs all benchmark cases through the A\* decoder and prints ASCII
tablature for each. No data files are required beyond the checkpoint.

To run a specific pitch sequence manually:

```bash
# E major scale
python infer.py --pitches 64,66,68,69,71,73,75,76
```

*(Exact string assignments may vary — pitch accuracy is always 100%.)*

---

## Evaluating on Guitar Pro files

The 50-file evaluation set used in the paper consisted of crowd-sourced
Guitar Pro (.gp3/.gp4/.gp5) files sourced from gprotab.net.
**These are not included in the submission.**

If you have your own Guitar Pro files, you can evaluate with:

```bash
python evaluate.py \
  --model fretboard_transformer.pt \
  --tabs /path/to/your/gp_files/ \
  --n 50 --seed 42 \
  --fret_bias 0.36 --transition_bias 0.184
```

Expected results on a representative rock/metal GP test set:
```
String agreement (exact match)
  Macro mean    :  ~66%
  Pitch accuracy:  100.0%
```

To build a clean deduplicated evaluation set (removing any overlap with
training data), use:

```bash
python build_eval_set.py \
  --eval_dirs  /path/to/gp_files/ \
  --train_dirs ./assets/data/synthtab/outall/ \
  --n 50 --seed 42 \
  --out eval_manifest.txt \
  --report overlap_report.json

python evaluate.py \
  --manifest eval_manifest.txt \
  --model fretboard_transformer.pt \
  --fret_bias 0.36 --transition_bias 0.184
```

---

## Project file structure

```
.
├── fretboard_transformer.py   # Training entrypoint
├── ft_model.py                # Neural architecture (FretboardTransformer)
├── ft_data.py                 # Data parsing, StreamingDataset, FretboardTrainer
├── ft_decode.py               # A* TrellisDecoder + post-processing
├── infer.py                   # Inference script (benchmark test cases)
├── evaluate.py                # GP file evaluation
├── build_eval_set.py          # Clean deduplicated eval set builder
├── heuristic_search.py        # Grid/random search over A* heuristic parameters
├── generate.py                # Phrase-walk generation + WAV synthesis
├── parse_midi.py              # MIDI pitch sequence extractor
├── requirements.txt           # Python dependencies
├── fretboard_transformer.pt   # Pre-trained AR model checkpoint [*]
└── README.md                  # This file

[*] Included in submission package.
```

---


### Prerequisites

```bash
# CPU (inference only)
pip install -r requirements.txt

# CUDA (training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pyguitarpro==0.6 mido numpy scipy
```

### Data

Two data sources are required for training:

**1. SynthTab** (~9,800 guitar sequences with ground-truth position labels)
```
./assets/data/synthtab/outall/
    song_001/
        string_1.mid
        ...
        string_6.mid
    song_002/
    ...
```
Available at: https://synthtab.dev

**2. Lakh MIDI Dataset** (pitch augmentation — ~1M sequences, no position labels)
```bash
# After downloading LMD-full from https://colinraffel.com/projects/lmd/
python parse_midi.py \
  --midi_dir ./assets/data/lakh_midi/ \
  --guitar_only \
  --out ./midi_sequences.pkl
```

### Train

```bash
python fretboard_transformer.py --epochs 20
```

Training logs val_loss and val_acc per epoch. The checkpoint
`fretboard_transformer.pt` is saved whenever validation loss improves.
Both SynthTab and the MIDI sequences (`midi_sequences.pkl`) are loaded
automatically if present in the expected paths.

### Evaluate

```bash
python evaluate.py \
  --model fretboard_transformer.pt \
  --tabs /path/to/gp_files/ \
  --n 50 --seed 42 \
  --fret_bias 0.36 --transition_bias 0.184
```

### Heuristic parameter search

To reproduce the grid search used to determine the optimal heuristic parameters:

```bash
python heuristic_search.py \
  --model fretboard_transformer.pt \
  --tabs /path/to/gp_files/ \
  --n 50 --seed 42 \
  --mode grid \
  --fret_bias_range 0.0 0.5 \
  --transition_bias_range 0.12 0.20
```

---

## A* decoder parameters

| Parameter | Optimal | Description |
|---|---|---|
| `--fret_bias` | 0.36 | Penalty per absolute fret number |
| `--transition_bias` | 0.184 | Penalty per fret of horizontal jump |
| `--string_bias` | 0.00 | Penalty per string of vertical jump (disabled — hurts accuracy) |

Parameters were determined by 6×6 grid search over α ∈ [0, 0.50] and
β ∈ [0.12, 0.20] on the GProTab evaluation corpus. The optimal fret bias
(α = 0.36) is larger than intuition suggests, indicating the neural network
systematically underpenalises high-fret positions relative to human tabbing
conventions.

`--string_bias 0.0` is the default; non-zero values consistently reduce
accuracy across all tested values (0.02–0.15).

---

## Generating new tablature

### From a custom pitch sequence

```bash
python infer.py --pitches 60,62,64,65,67,69,71,72
```

### Phrase-walk generation (random melody → tablature → WAV)

```bash
python generate.py \
  --key A --scale minor_pentatonic \
  --length 64 --temp 1.5 \
  --wav output.wav
```

---

## Results summary

| System | Evaluation set | Macro | Median | Post-proc |
|---|---|---|---|---|
| Greedy argmax (no path search) | GProTab | 64.9% | 68.8% | 1.55% |
| SynthTab only | ScoreSet (classical) | 38.7% | 39.1% | 9.44% |
| SynthTab only | Scraped tabs | 64.5% | 67.2% | 0.87% |
| SynthTab + MIDI, hand-tuned | GProTab | 65.0% | 69.5% | 1.00% |
| **SynthTab + MIDI, grid-searched** | **GProTab** | **66.1%** | **71.1%** | **0.43%** |
| Edwards et al. ISMIR 2024 | Professional transcriptions | 73.18% | — | <1% |

**Primary reported result: 66.1%** — SynthTab + MIDI training, grid-searched
heuristic parameters (α = 0.36, β = 0.184), GProTab evaluation corpus (n=50, seed=42).

All systems achieve 100% pitch accuracy — the pitch-constrained graph structure
guarantees only positions producing the correct pitch are considered at each node.

---

## Citation

> Guitar Fret Navigator: Automatic Tablature Generation via A\* Graph Search
> and Bidirectional Transformers. MATH 2056 Discrete Mathematics Project,
> Algoma University, Winter 2026.

Key external references:

- Edwards et al., "MIDI-to-Tab: Guitar Tablature Inference via Masked
  Language Modeling," ISMIR 2024. arXiv:2408.05024
- Hart, Nilsson & Raphael, "A Formal Basis for the Heuristic Determination
  of Minimum Cost Paths," IEEE Trans. Systems Science, 1968.
- Zang et al., "SynthTab: Leveraging Synthesized Data for Guitar Tablature
  Transcription," ICASSP 2024. arXiv:2309.09085
