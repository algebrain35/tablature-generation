"""
ft_data.py — Data parsing, caching, dataset, and training loop.

Classes:
    StreamingDataset   — In-memory dataset with disk cache (GP, SynthTab, MIDI)
    FretboardTrainer   — Training loop for FretboardTransformer (AR or masked-LM)
"""

import math, os, re, glob, random, hashlib, pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.nn.functional as F
from torch.optim import Adam

from ft_model import (
    FretboardTransformer, FretboardTransformerMLM,
    NUM_STRINGS, NUM_FRETS, NUM_POSITIONS, NUM_MIDI, N_DUR_BINS, N_TS_BINS, N_VEL_BINS,
    PAD_IDX, PITCH_MASK_IDX, STANDARD_TUNING, POSITIONS, POS_TO_IDX,
    quantize_ticks, quantize_velocity, pos_to_midi, midi_to_positions,
    dadagp_str, _GP_DUR_TICKS,
    DADAGP_GUITAR_PREFIXES, _NOTE_RE, GUITARSET_OPEN_MIDI, SYNTHTAB_TRACK_OPEN_MIDI,
)

try:
    import jams;      HAS_JAMS = True
except ImportError:   HAS_JAMS = False
try:
    import mido;      HAS_MIDO = True
except ImportError:   HAS_MIDO = False
try:
    import guitarpro; HAS_GP = True
except ImportError:   HAS_GP = False

_KS_MAJOR = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
_KS_MINOR = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]

def estimate_key(pitches):
    """
    Krumhansl-Schmuckler key-finding from a MIDI pitch list.
    Returns integer 0-11 (C=0, C#=1 ... B=11) for the most likely tonic,
    or 12 if the pitch list is empty (unknown key token).
    Ignores major vs minor — we care about tonic, not mode.
    """
    if not pitches:
        return 12  # unknown
    counts = [0] * 12
    for p in pitches:
        counts[int(p) % 12] += 1
    total = sum(counts)
    if total == 0:
        return 12
    dist = [c / total for c in counts]
    best_key, best_cor = 0, -999.0
    for root in range(12):
        for profile in (_KS_MAJOR, _KS_MINOR):
            rotated = profile[root:] + profile[:root]
            mean_d  = sum(dist) / 12
            mean_p  = sum(rotated) / 12
            num     = sum((d - mean_d) * (p - mean_p)
                          for d, p in zip(dist, rotated))
            den     = (sum((d - mean_d)**2 for d in dist) *
                       sum((p - mean_p)**2 for p in rotated)) ** 0.5
            cor     = num / den if den > 1e-9 else 0.0
            if cor > best_cor:
                best_cor = cor
                best_key = root
    return best_key


NUM_KEYS = 13   # 0-11 chromatic keys + 12 = unknown


def parse_dadagp_file(path):
    notes = []
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                m = _NOTE_RE.match(line.strip())
                if not m:
                    continue
                fret    = int(m.group(3))
                our_str = dadagp_str(int(m.group(2)))
                if fret < 0 or fret > NUM_FRETS or not (1 <= our_str <= NUM_STRINGS):
                    continue
                notes.append((our_str, fret))
    except Exception:
        pass
    return notes


def parse_guitarset_jams(path):
    """
    Parse a GuitarSet .jams file into a time-sorted list of (string, fret) tuples.

    GuitarSet stores 6 separate note_midi annotations — one per string.
    String numbering: 0 = low E (our string 6), 5 = high e (our string 1).
    Each annotation has onset/offset intervals and MIDI pitch values.

    We merge all 6 strings by onset time to get a monophonic-approximation
    sequence. Simultaneous notes (chords) are ordered by string number.

    Requires: pip install jams
    """
    if not HAS_JAMS:
        raise ImportError("jams library required for GuitarSet parsing. "
                          "Install with: pip install jams")
    try:
        jam = jams.load(path)
    except Exception as e:
        return []

    events = []  # list of (onset, guitarset_str, fret)

    for gs_str in range(6):
        our_str = 6 - gs_str   # remap: gs 0→our 6, gs 5→our 1
        open_midi = GUITARSET_OPEN_MIDI[gs_str]

        try:
            annos = jam.search(namespace='note_midi')
            anno  = annos.search(data_source=str(gs_str))
            if not anno:
                continue
            anno = anno[0]
        except Exception:
            continue

        try:
            intervals, values = anno.to_interval_values()
        except Exception:
            continue

        for (onset, _), midi in zip(intervals, values):
            midi  = int(round(float(midi)))
            fret  = midi - open_midi
            if fret < 0 or fret > NUM_FRETS:
                continue
            if not (1 <= our_str <= NUM_STRINGS):
                continue
            events.append((float(onset), our_str, fret))

    # Sort by onset time, then by string number for simultaneous notes
    events.sort(key=lambda e: (e[0], e[1]))
    return [(s, f) for _, s, f in events]


# GP tuning string name → MIDI mapping
_GP_STEP_TO_SEMITONE = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}

def _gp_string_midi(gs):
    """Convert a guitarpro GuitarString object to open-string MIDI pitch."""
    return (gs.value + 1) * 12 + _GP_STEP_TO_SEMITONE.get(gs.name, 0)


# GP key signature: sharps/flats value → semitone root (C major = 0)
# GP stores key as number of sharps (positive) or flats (negative)
# Maps to chromatic root: 0=C, 1=G, 2=D ... (circle of fifths)
_GP_SHARPS_TO_ROOT = {
    0: 0,   # C
    1: 7,   # G
    2: 2,   # D
    3: 9,   # A
    4: 4,   # E
    5: 11,  # B
    6: 6,   # F#/Gb
    7: 1,   # C#/Db
   -1: 5,   # F
   -2: 10,  # Bb
   -3: 3,   # Eb
   -4: 8,   # Ab
   -5: 1,   # Db
   -6: 6,   # Gb
   -7: 11,  # Cb
}


def _gp_key(song):
    """Extract key root (0-11) from a parsed guitarpro Song object."""
    try:
        sharps = song.key
        # pyguitarpro 0.6 stores key as integer sharps/flats
        if isinstance(sharps, int):
            return _GP_SHARPS_TO_ROOT.get(sharps, 12)
        # Some versions store as a Key object
        return _GP_SHARPS_TO_ROOT.get(getattr(sharps, 'value', 0), 12)
    except Exception:
        return 12  # unknown


def parse_scoreset_gp(path):
    """
    Parse a raw GuitarPro (.gp3/.gp4/.gp5) file using pyguitarpro.
    Returns a time-sorted list of (string, fret) tuples.

    Extracts notes from guitar tracks only (excludes bass, drums).
    Uses the track's actual tuning so downtuned files parse correctly.
    String numbering in GP: 1=high-e, 6=low-E — matches our convention directly.

    Requires: pip install pyguitarpro==0.6
    """
    if not HAS_GP:
        raise ImportError(
            "pyguitarpro required for .gp5 parsing. "
            "Install with: pip install pyguitarpro==0.6"
        )
    try:
        import signal
        def _timeout_handler(signum, frame):
            raise TimeoutError("guitarpro.parse timed out")
        old = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(10)  # 10 second hard kill
        try:
            song = guitarpro.parse(path)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    except Exception:
        return [], 12

    key_idx = _gp_key(song)

    # Use the first valid guitar track (4–7 strings, not drums).
    # GProTab files frequently contain 7-string guitars or 4-string bass —
    # we accept any track with enough strings and take the first 6.
    for track in song.tracks:
        try:
            channel = track.channel.channel
        except AttributeError:
            channel = 0
        if channel == 9:          # drums — skip
            continue
        n_strings = len(track.strings)
        if n_strings < 4:         # fewer than 4 strings — skip
            continue

        notes = []
        beat_idx = 0
        for measure in track.measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    try:
                        dur_name = beat.duration.value
                        dur_ticks = _GP_DUR_TICKS.get(str(dur_name), 480)
                    except Exception:
                        dur_ticks = 480
                    dur_bin = quantize_ticks(dur_ticks)
                    ts_bin  = quantize_ticks(dur_ticks)
                    for note in beat.notes:
                        s = note.string
                        f = note.value
                        # Remap strings > 6 to the closest valid string
                        if s > NUM_STRINGS:
                            s = NUM_STRINGS
                        if (s, f) in POS_TO_IDX:
                            notes.append((s, f, ts_bin, 4, dur_bin))
                    beat_idx += 1

        if notes:
            return notes, key_idx

    return [], key_idx


def parse_synthtab_dir(song_dir):
    """
    Parse one SynthTab song directory into a time-sorted (string, fret) sequence.

    SynthTab structure (confirmed from dataset inspection):
        song_dir/
            string_1.mid  ← high-e, our string 1, open MIDI 64
            string_2.mid  ← B,      our string 2, open MIDI 59
            string_3.mid  ← G,      our string 3, open MIDI 55
            string_4.mid  ← D,      our string 4, open MIDI 50
            string_5.mid  ← A,      our string 5, open MIDI 45
            string_6.mid  ← low-E,  our string 6, open MIDI 40

    Each file: Type 1, single track, absolute MIDI pitches.
    Pitch 24 (C1) = silence sentinel — always first note, skip it.
    Fret = pitch - open_string_midi.
    Notes from all strings merged and sorted by onset tick.

    Requires: pip install mido
    """
    if not HAS_MIDO:
        raise ImportError("mido required for SynthTab. pip install mido")

    events = []   # (abs_tick, our_string, fret)

    for our_str in range(1, NUM_STRINGS + 1):
        path = os.path.join(song_dir, f"string_{our_str}.mid")
        if not os.path.exists(path):
            continue
        try:
            mid   = mido.MidiFile(path)
            track = mid.tracks[0]
        except Exception:
            continue

        open_midi = STANDARD_TUNING[our_str]
        abs_tick  = 0

        # Track note_on times to compute duration via matching note_off
        pending = {}   # note_midi → (abs_tick, velocity)
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                if msg.note == 24:
                    continue
                pending[msg.note] = (abs_tick, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                onset_info = pending.pop(msg.note, None)
                if onset_info is None:
                    continue
                onset_tick, vel = onset_info
                fret = msg.note - open_midi
                if fret < 0 or fret > NUM_FRETS:
                    continue
                if (our_str, fret) not in POS_TO_IDX:
                    continue
                dur_ticks = abs_tick - onset_tick
                events.append((onset_tick, our_str, fret, vel, dur_ticks))
        # Flush any still-open notes at end of track
        for note, (onset_tick, vel) in pending.items():
            fret = note - open_midi
            if 0 <= fret <= NUM_FRETS and (our_str, fret) in POS_TO_IDX:
                events.append((onset_tick, our_str, fret, vel, 0))

    events.sort(key=lambda e: (e[0], e[1]))
    # Compute time shifts (inter-onset intervals)
    result = []
    prev_tick = 0
    for onset, s, f, vel, dur in events:
        ts = onset - prev_tick
        prev_tick = onset
        result.append((s, f, quantize_ticks(ts), quantize_velocity(vel), quantize_ticks(dur)))
    return result


def _worker_init():
    """Initialiser for dataset worker processes.
    Hides CUDA so workers never initialise the GPU — they are CPU-only
    and touching CUDA in a spawned child causes deadlocks on Linux."""
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def _parse_one(entry, window):
    """
    Module-level parse function — must be top-level for ProcessPoolExecutor pickling.
    Returns a ready-to-queue dict or None if the entry is invalid/too short.
    """
    import random
    kind, path = entry
    try:
        if kind == 'dadagp':
            notes = parse_dadagp_file(path)
        elif kind == 'jams':
            notes = parse_guitarset_jams(path)
        elif kind == 'gp':
            notes = parse_scoreset_gp(path)
        else:
            notes = parse_synthtab_dir(path)

        notes = [(s, f) for s, f in notes if (s, f) in POS_TO_IDX]
        if len(notes) < window + 1:
            return None

        pos       = [POS_TO_IDX[(s, f)] for s, f in notes]
        pit       = [pos_to_midi(s, f)  for s, f in notes]
        n_windows = len(pos) - window
        if n_windows <= 0:
            return None

        start   = random.randint(0, n_windows - 1)
        pos_win = pos[start : start + window + 1]
        pit_win = pit[start : start + window + 1]

        return dict(
            strings     = [POSITIONS[p][0] for p in pos_win[:-1]],
            frets       = [POSITIONS[p][1] for p in pos_win[:-1]],
            pitches_pos = pit_win[:-1],
            pitches_ctx = pit_win,
            targets     = pos_win[1:],
        )
    except Exception:
        return None


def _parse_entry_full(tagged_entry, window=64, _verbose=False):
    """
    Parse one entry and return (source_name, pos_list, pit_list).
    source_name is stored in cache so balancing can be exact.
    """
    source_name, kind, path = tagged_entry
    try:
        gp_key = 12
        if kind == 'dadagp':
            raw = parse_dadagp_file(path)
        elif kind == 'jams':
            raw = parse_guitarset_jams(path)
        elif kind == 'gp':
            raw, gp_key = parse_scoreset_gp(path)
        else:
            raw = parse_synthtab_dir(path)

        # Normalise to 5-tuples: (string, fret, ts_bin, vel_bin, dur_bin)
        # Older parsers (dadagp, jams) return 2-tuples — pad with zeros
        notes_5 = []
        for item in raw:
            if len(item) == 5:
                s, f, ts, vel, dur = item
            else:
                s, f = item[0], item[1]
                ts, vel, dur = 0, 0, 0
            if (s, f) in POS_TO_IDX:
                notes_5.append((s, f, ts, vel, dur))

        MIN_NOTES = 8
        if len(notes_5) < MIN_NOTES:
            if _verbose:
                print(f"  [parse] {kind} {os.path.basename(path)}: "
                      f"only {len(notes_5)} valid notes (need {MIN_NOTES})", flush=True)
            return None

        pos = [POS_TO_IDX[(s, f)]   for s, f, *_ in notes_5]
        pit = [pos_to_midi(s, f)    for s, f, *_ in notes_5]
        ts  = [item[2]              for item in notes_5]
        vel = [item[3]              for item in notes_5]
        dur = [item[4]              for item in notes_5]
        key = gp_key if gp_key != 12 else estimate_key(pit)
        return (source_name, key, pos, pit, ts, vel, dur)
    except Exception as e:
        if _verbose:
            import traceback
            print(f"  [parse] EXCEPT {kind} {os.path.basename(path)}: "
                  f"{type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        return None



# ─── Transposition augmentation ──────────────────────────────────────────────

def transpose_window(pos_win, pit_win, semitones):
    """
    Transpose a position window by semitones, keeping string assignments fixed.
    Returns (new_pos_win, new_pit_win) or None if any note shifts off fretboard.
    """
    new_pos, new_pit = [], []
    for p, pitch in zip(pos_win, pit_win):
        s, f  = POSITIONS[p]
        new_f = f + semitones
        new_p = pitch + semitones
        if new_f < 0 or new_f > NUM_FRETS:
            return None
        if new_p < 0 or new_p >= NUM_MIDI:
            return None
        key = (s, new_f)
        if key not in POS_TO_IDX:
            return None
        new_pos.append(POS_TO_IDX[key])
        new_pit.append(new_p)
    return new_pos, new_pit


class StreamingDataset:
    """
    In-memory dataset with optional disk cache.

    First run: parses all files, saves cache to disk (~30s-5min depending on dataset size).
    Subsequent runs: loads cache instantly (<1s).

    Training draws random windows from cached sequences — no I/O during training.
    """

    CACHE_VERSION = 8   # v8: pitch_head added, 4-head model

    def __init__(self, dadagp_dir=None, guitarset_dir=None, scoreset_dir=None,
                 synthtab_dir=None, scraped_tabs_dir=None, proggp_dir=None,
                 window=64, genres=None, max_files=None,
                 queue_size=2048, num_workers=4,
                 cache_path="dataset_cache.pkl",
                 augment_semitones=(-3,-2,-1,1,2,3),
                 val_split=0.1, max_source_fraction=0.8):

        if all(d is None for d in [dadagp_dir, guitarset_dir, scoreset_dir,
                                      synthtab_dir, scraped_tabs_dir, proggp_dir]):
            raise ValueError("At least one data directory must be provided.")

        self.window            = window
        self.augment_semitones = (-3,-2,-1,1,2,3)

        # ── Register entries per source ───────────────────────────────────────
        import random as _rnd
        sources = {}   # source_name → list of (kind, path) entries

        if dadagp_dir is not None:
            files = sorted(glob.glob(
                os.path.join(dadagp_dir, "**", "*.txt"), recursive=True))
            if genres:
                files = [f for f in files
                         if any(g.lower() in f.lower() for g in genres)]
            sources['DadaGP']    = [('dadagp', f) for f in files]

        if guitarset_dir is not None:
            files = sorted(glob.glob(
                os.path.join(guitarset_dir, "**", "*.jams"), recursive=True))
            sources['GuitarSet'] = [('jams', f) for f in files]

        if scoreset_dir is not None:
            files = sorted(
                glob.glob(os.path.join(scoreset_dir, "**", "*.gp5"), recursive=True) +
                glob.glob(os.path.join(scoreset_dir, "**", "*.gp4"), recursive=True) +
                glob.glob(os.path.join(scoreset_dir, "**", "*.gp3"), recursive=True))
            sources['ScoreSet']  = [('gp', f) for f in files]

        if scraped_tabs_dir is not None:
            files = sorted(
                glob.glob(os.path.join(scraped_tabs_dir, "**", "*.gp5"), recursive=True) +
                glob.glob(os.path.join(scraped_tabs_dir, "**", "*.gp4"), recursive=True) +
                glob.glob(os.path.join(scraped_tabs_dir, "**", "*.gp3"), recursive=True))
            sources['ScrapedTabs'] = [('gp', f) for f in files]

        if proggp_dir is not None:
            # ProgGP: 173 prog metal GP files from github.com/otnemrasordep/ProgGP
            # Same GP3/4/5 format — reuses parse_scoreset_gp directly.
            files = sorted(
                glob.glob(os.path.join(proggp_dir, "**", "*.gp5"), recursive=True) +
                glob.glob(os.path.join(proggp_dir, "**", "*.gp4"), recursive=True) +
                glob.glob(os.path.join(proggp_dir, "**", "*.gp3"), recursive=True))
            sources['ProgGP'] = [('gp', f) for f in files]

        if synthtab_dir is not None:
            try:
                entries = os.listdir(synthtab_dir)
            except FileNotFoundError:
                raise FileNotFoundError(f"SynthTab dir not found: '{synthtab_dir}'")
            song_dirs = sorted([
                os.path.join(synthtab_dir, e)
                for e in entries
                if os.path.isfile(os.path.join(synthtab_dir, e, "string_1.mid"))
            ])
            sources['SynthTab']  = [('synthtab', d) for d in song_dirs]

        if not sources:
            raise ValueError("At least one data directory must be provided.")

        # ── Print raw counts ──────────────────────────────────────────────────
        total_raw = sum(len(v) for v in sources.values())
        print("Dataset registration (before balancing):")
        for name, entries in sources.items():
            pct = 100 * len(entries) / total_raw if total_raw else 0
            print(f"  {name:<14}: {len(entries):>6} files ({pct:.1f}%)")

        # ── Merge all entries for cache (pre-balance) ───────────────────────
        # Tag each entry with its source name so the cache can enforce
        # exact per-source limits rather than proportional approximations.
        all_entries = []
        for source_name, entries in sources.items():
            for kind, path in entries:
                all_entries.append((source_name, kind, path))
        if max_files and len(all_entries) > max_files:
            _rnd.shuffle(all_entries)
            all_entries = all_entries[:max_files]
        self._entries = all_entries

        # ── Load or build cache ───────────────────────────────────────────────
        import random
        all_data = self._load_or_build_cache(cache_path, num_workers)

        # ── Exact per-source balancing ────────────────────────────────────────
        # Cache stores (source_name, pos, pit) — group by source for exact caps.
        rng = random.Random(42)

        source_buckets = {}
        for item in all_data:
            src = item[0]   # (source, key, pos, pit, ts, vel, dur)
            source_buckets.setdefault(src, []).append(item)

        # Shuffle each bucket with fixed seed for reproducibility
        for src in source_buckets:
            rng.shuffle(source_buckets[src])

        # Iteratively cap any source exceeding max_source_fraction.
        # Skip balancing if only one source has meaningful data (>10 sequences)
        # — a tiny dummy source (e.g. 1 DadaGP placeholder) must not cap a large one.
        meaningful = {n: v for n, v in source_buckets.items() if len(v) > 10}
        if len(meaningful) >= 2:
            thresh = max_source_fraction
            for _ in range(20):
                total = sum(len(v) for v in source_buckets.values())
                if total == 0:
                    break
                dominated = {n: v for n, v in source_buckets.items()
                             if len(v) / total > thresh}
                if not dominated:
                    break
                for name, seqs in dominated.items():
                    others = total - len(seqs)
                    cap    = max(1, int(others * thresh / (1 - thresh)))
                    source_buckets[name] = seqs[:cap]
        else:
            print(f"  [balance] Only one meaningful source — skipping cap.", flush=True)

        # Print exact per-source counts
        total_seq = sum(len(v) for v in source_buckets.values())
        print("Dataset after exact balancing:")
        for name, seqs in sorted(source_buckets.items()):
            pct = 100 * len(seqs) / total_seq if total_seq else 0
            raw = sum(1 for item in all_data if item[0] == name)
            print(f"  {name:<14}: {raw:>6} parsed → {len(seqs):>6} used ({pct:.1f}%)")
        print(f"  {'TOTAL':<14}: {total_seq:>6} sequences", flush=True)

        # Merge — keep source tag for source-conditional loss, then strip
        balanced = []
        for seqs in source_buckets.values():
            balanced.extend(seqs)
        rng.shuffle(balanced)
        # is_midi flag: True for entries from MIDI-only source (no position labels)
        balanced = [(key, pos, pit, ts, vel, dur,
                     source.startswith('midi'))
                    for source, key, pos, pit, ts, vel, dur in balanced]

        if len(balanced) < 2:
            raise RuntimeError(
                f"Dataset too small after balancing: only {len(balanced)} sequences. "
                f"Add more data or increase max_source_fraction.")

        n_val      = max(1, int(len(balanced) * val_split))
        n_val      = min(n_val, len(balanced) - 1)
        self._val  = balanced[:n_val]
        self._data = balanced[n_val:]
        self._index = list(range(len(self._data)))
        random.shuffle(self._index)
        print(f"  Split: {len(self._data)} train / {len(self._val)} val", flush=True)

    def _load_or_build_cache(self, cache_path, num_workers):
        import pickle, hashlib
        # Allow passing a fully-qualified versioned path directly
        if cache_path.endswith(f'.v{self.CACHE_VERSION}'):
            if os.path.exists(cache_path):
                print(f"Loading cache (direct): {cache_path}...", flush=True)
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"  Loaded {len(data)} sequences.", flush=True)
                return data
            else:
                # List available caches to help the user pick the right one
                import glob as _glob
                base = cache_path.rsplit('.', 2)[0]  # strip .HASH.vN
                available = _glob.glob(f"{base}.*.v{self.CACHE_VERSION}")
                if available:
                    print(f"[ERROR] Cache not found: {cache_path}", flush=True)
                    print(f"  Available caches:", flush=True)
                    for p in sorted(available):
                        size_mb = os.path.getsize(p) / (1024*1024)
                        print(f"    {p}  ({size_mb:.1f} MB)", flush=True)
                else:
                    print(f"[ERROR] Cache not found: {cache_path} — "
                          f"no .v{self.CACHE_VERSION} files found near that path.",
                          flush=True)
                raise FileNotFoundError(
                    f"Cache file not found: {cache_path}\n"
                    f"Run without specifying a versioned path to rebuild, "
                    f"or check the filenames above.")

        # Cache key: hash of sorted entry paths + window size
        key = hashlib.md5(
            (str(sorted(str(e) for e in self._entries)) + str(self.window)
             ).encode()).hexdigest()[:12]
        versioned_path = f"{cache_path}.{key}.v{self.CACHE_VERSION}"

        if os.path.exists(versioned_path):
            print(f"Loading cache: {versioned_path}...", flush=True)
            with open(versioned_path, 'rb') as f:
                data = pickle.load(f)
            print(f"  Loaded {len(data)} sequences from cache.", flush=True)
            return data

        # Build cache using process pool — separate processes bypass the GIL
        # giving true parallelism for CPU-bound MIDI/text parsing.
        print(f"Building cache ({len(self._entries)} files, {num_workers} workers)...",
              flush=True)

        # ── Diagnostic: test 5 random entries in-process to surface errors ──
        import random as _rng, traceback as _tb
        _sample = _rng.sample(self._entries, min(5, len(self._entries)))
        _n_ok = 0
        for _e in _sample:
            _src, _kind, _path = _e
            try:
                _r = _parse_entry_full(_e, self.window, _verbose=True)
                if _r:
                    _n_ok += 1
                    print(f"  [diag] OK   {_kind} {os.path.basename(_path)}", flush=True)
                else:
                    # Run parser directly to get more detail
                    try:
                        if _kind == 'gp':
                            _raw, _key = parse_scoreset_gp(_path)
                            print(f"  [diag] EMPTY {_kind} {os.path.basename(_path)}: "
                                  f"{len(_raw)} notes from parser", flush=True)
                        elif _kind == 'synthtab':
                            _raw = parse_synthtab_dir(_path)
                            print(f"  [diag] EMPTY synthtab {os.path.basename(_path)}: "
                                  f"{len(_raw)} notes", flush=True)
                        else:
                            print(f"  [diag] EMPTY {_kind} {os.path.basename(_path)}",
                                  flush=True)
                    except Exception as _ex2:
                        print(f"  [diag] PARSE ERR {_kind} {os.path.basename(_path)}: "
                              f"{type(_ex2).__name__}: {_ex2}", flush=True)
            except Exception as _ex:
                print(f"  [diag] EXCEPT {_kind} {os.path.basename(_path)}: "
                      f"{type(_ex).__name__}: {_ex}", flush=True)
                _tb.print_exc()
        print(f"  [diag] sample parse: {_n_ok}/5 valid", flush=True)
        # ── End diagnostic ──────────────────────────────────────────────────

        from concurrent.futures import ProcessPoolExecutor, as_completed
        import functools, multiprocessing
        data    = []
        skipped = 0
        done    = 0
        fn      = functools.partial(_parse_entry_full, window=self.window)
        # Workers are CPU-only — hide GPU so they never touch CUDA,
        # preventing deadlocks when the parent process has CUDA initialised.
        mp_ctx  = multiprocessing.get_context('spawn')

        with ProcessPoolExecutor(max_workers=num_workers,
                                 mp_context=mp_ctx,
                                 initializer=_worker_init) as pool:
            futures = {pool.submit(fn, e): i for i, e in enumerate(self._entries)}
            for fut in as_completed(futures):
                done += 1
                if done % 500 == 0 or done == len(futures):
                    print(f"  {done}/{len(futures)} "
                          f"({len(data)} valid, {skipped} skipped)...", flush=True)
                try:
                    result = fut.result(timeout=15)  # 15s per file — hangs killed
                except TimeoutError:
                    result = None
                    skipped += 1
                    continue
                except Exception:
                    result = None
                if result is None:
                    skipped += 1
                else:
                    data.append(result)

        print(f"Saving cache to {versioned_path}...", flush=True)
        with open(versioned_path, 'wb') as f:
            import pickle
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Cache saved. {len(data)} sequences, {skipped} skipped.", flush=True)
        return data

    def reshuffle(self):
        import random
        random.shuffle(self._index)

    def __len__(self):
        return len(self._data)

    def get_batch(self, batch_size):
        import random
        if len(self._data) == 0:
            raise RuntimeError("Training set is empty — cannot sample a batch.")
        batch    = []
        attempts = 0
        while len(batch) < batch_size and attempts < batch_size * 20:
            attempts += 1
            idx                              = random.randint(0, len(self._data) - 1)
            key, pos, pit, ts, vel, dur, is_midi = self._data[idx]
            # Use full window if long enough, otherwise use the whole sequence
            w     = min(self.window, len(pos) - 1)
            if w < 1:
                continue
            n     = len(pos) - w
            start = random.randint(0, max(0, n - 1))
            pos_win = pos[start : start + w + 1]
            pit_win = pit[start : start + w + 1]
            ts_win  = ts [start : start + w + 1]
            vel_win = vel[start : start + w + 1]
            dur_win = dur[start : start + w + 1]
            # Pad short windows
            if len(pos_win) < self.window + 1:
                pad_p = pos_win[-1]; pad_i = pit_win[-1]
                pad_t = ts_win[-1];  pad_v = vel_win[-1]; pad_d = dur_win[-1]
                pad_n = self.window + 1 - len(pos_win)
                pos_win = pos_win + [pad_p] * pad_n
                pit_win = pit_win + [pad_i] * pad_n
                ts_win  = ts_win  + [pad_t] * pad_n
                vel_win = vel_win + [pad_v] * pad_n
                dur_win = dur_win + [pad_d] * pad_n

            # Transposition augmentation
            if self.augment_semitones and random.random() < 0.7:
                st     = random.choice(self.augment_semitones)
                result = transpose_window(pos_win, pit_win, st)
                if result is not None:
                    pos_win, pit_win = result

            batch.append((key, pos_win[:-1], pit_win[:-1],
                          ts_win[:-1], vel_win[:-1], dur_win[:-1], is_midi))

        key_batch   = torch.tensor([b[0] for b in batch], dtype=torch.long)
        pos_batch   = torch.tensor([b[1] for b in batch], dtype=torch.long)
        pit_batch   = torch.tensor([b[2] for b in batch], dtype=torch.long)
        ts_batch    = torch.tensor([b[3] for b in batch], dtype=torch.long)
        vel_batch   = torch.tensor([b[4] for b in batch], dtype=torch.long)
        dur_batch   = torch.tensor([b[5] for b in batch], dtype=torch.long)
        midi_mask   = torch.tensor([b[6] for b in batch], dtype=torch.bool)  # (B,) True=midi-only
        return key_batch, pos_batch, pit_batch, ts_batch, vel_batch, dur_batch, midi_mask

    def get_val_batch(self, batch_size):
        """Sample a batch from held-out validation set (no augmentation)."""
        import random
        # Fall back to training data if val set is empty (tiny dataset edge case)
        pool = self._val if len(self._val) > 0 else self._data
        if len(pool) == 0:
            raise RuntimeError("Dataset is empty — cannot sample a validation batch.")
        batch = []
        attempts = 0
        while len(batch) < batch_size and attempts < batch_size * 20:
            attempts += 1
            idx      = random.randint(0, len(pool) - 1)
            key, pos, pit, ts, vel, dur, is_midi = pool[idx]
            w     = min(self.window, len(pos) - 1)
            if w < 1:
                continue
            n     = len(pos) - w
            start = random.randint(0, max(0, n - 1))
            pos_win = pos[start : start + w + 1]
            pit_win = pit[start : start + w + 1]
            ts_win  = ts [start : start + w + 1]
            vel_win = vel[start : start + w + 1]
            dur_win = dur[start : start + w + 1]
            if len(pos_win) < self.window + 1:
                pad_n = self.window + 1 - len(pos_win)
                pos_win = pos_win + [pos_win[-1]] * pad_n
                pit_win = pit_win + [pit_win[-1]] * pad_n
                ts_win  = ts_win  + [ts_win[-1]]  * pad_n
                vel_win = vel_win + [vel_win[-1]]  * pad_n
                dur_win = dur_win + [dur_win[-1]]  * pad_n
            batch.append((key, pos_win[:-1], pit_win[:-1],
                          ts_win[:-1], vel_win[:-1], dur_win[:-1], is_midi))
        if not batch:
            raise RuntimeError(
                f"Could not fill val batch — all {len(pool)} sequences shorter "
                f"than window size {self.window}.")
        while len(batch) < batch_size:
            batch.append(batch[-1])
        return (torch.tensor([b[0] for b in batch], dtype=torch.long),
                torch.tensor([b[1] for b in batch], dtype=torch.long),
                torch.tensor([b[2] for b in batch], dtype=torch.long),
                torch.tensor([b[3] for b in batch], dtype=torch.long),
                torch.tensor([b[4] for b in batch], dtype=torch.long),
                torch.tensor([b[5] for b in batch], dtype=torch.long),
                torch.tensor([b[6] for b in batch], dtype=torch.bool))

    def stop(self):
        pass  # no background threads to stop

    @property
    def queue_len(self):
        return len(self._data)  # always "full" — in-memory



class FretboardTrainer:
    """
    Trainer using StreamingDataset — background thread parses files into a
    deque while the main thread trains. No DataLoader needed.

    Training loop draws batches directly from the queue via get_batch(),
    which blocks only if the queue is empty (shouldn't happen after warmup).
    """

    def __init__(self, model, dadagp_dir=None, guitarset_dir=None, scoreset_dir=None,
                 synthtab_dir=None, scraped_tabs_dir=None, proggp_dir=None,
                 midi_cache=None,
                 window=64, batch_size=32, lr=3e-4,
                 genres=None, max_files=None, steps_per_epoch=500, queue_size=2048,
                 num_workers=4, augment_semitones=(-3,-2,-1,1,2,3),
                 val_split=0.1, mask_prob=0.40, dropout=0.3,
                 max_source_fraction=0.8,
                 epochs_per_10k=20, min_epochs=5, max_epochs=60,
                 training_mode='ar',
                 cache_path="dataset_cache.pkl"):
        self.batch_size      = batch_size
        self.steps_per_epoch = steps_per_epoch
        use_cuda = torch.cuda.is_available()
        print(f"Device: {'cuda' if use_cuda else 'cpu'}")
        self.ds    = StreamingDataset(
            dadagp_dir=dadagp_dir, guitarset_dir=guitarset_dir,
            scoreset_dir=scoreset_dir, synthtab_dir=synthtab_dir,
            scraped_tabs_dir=scraped_tabs_dir, proggp_dir=proggp_dir,
            window=window, genres=genres, max_files=max_files,
            queue_size=queue_size, num_workers=num_workers,
            augment_semitones=augment_semitones,
            val_split=val_split,
            max_source_fraction=max_source_fraction,
            cache_path=cache_path,
        )

        # ── Inject MIDI pitch sequences ───────────────────────────────────────
        # midi_cache is a pkl produced by parse_midi.py — entries have dummy
        # position labels (all zeros) and is_midi=True so the trainer skips
        # pos_loss for these sequences, only training pitch_head.
        if midi_cache:
            import glob as _glob
            # Support glob patterns (e.g. "midi_*.pkl")
            paths = _glob.glob(midi_cache) if '*' in midi_cache else [midi_cache]
            paths = [p for p in paths if os.path.exists(p)]
            if paths:
                midi_path = max(paths, key=os.path.getmtime)
                print(f"Loading MIDI cache: {midi_path}", flush=True)
                import pickle
                with open(midi_path, 'rb') as f:
                    midi_data = pickle.load(f)
                # Convert to (key, pos, pit, ts, vel, dur, is_midi=True) format
                import random as _random
                midi_entries = []
                for entry in midi_data:
                    # entry format: (source, key, pos, pit, ts, vel, dur)
                    _, key, pos, pit, ts, vel, dur = entry
                    midi_entries.append((key, pos, pit, ts, vel, dur, True))
                # Cap MIDI to 2× the existing GP/SynthTab dataset size
                # to prevent MIDI from drowning out position-labelled sequences
                max_midi = len(self.ds._data)  # 1:1 ratio — best empirical result
                if len(midi_entries) > max_midi:
                    import random as _random2
                    _random2.shuffle(midi_entries)
                    midi_entries = midi_entries[:max_midi]
                    print(f"  Capped to {max_midi} MIDI sequences (1:1 with GP/SynthTab)",
                          flush=True)
                _random.shuffle(midi_entries)
                n_val   = max(1, int(len(midi_entries) * val_split))
                self.ds._val  = self.ds._val  + midi_entries[:n_val]
                self.ds._data = self.ds._data + midi_entries[n_val:]
                _random.shuffle(self.ds._data)
                print(f"  Added {len(midi_entries)} MIDI sequences "
                      f"({len(midi_entries)-n_val} train, {n_val} val)", flush=True)
            else:
                print(f"[WARN] midi_cache not found: {midi_cache}", flush=True)
        # Now safe to initialise CUDA — dataset workers have already been spawned
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.model = model.to(self.device)   # move to GPU after dataset/cache build
        self.optim         = Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        self.mask_idx      = FretboardTransformer.MASK_IDX
        self.mask_prob     = mask_prob
        self.training_mode = training_mode
        self.best          = float('inf')
        print(f"Training mode : {training_mode}", flush=True)

        # ── Scale epochs to dataset size ─────────────────────────────────────
        # Target: epochs_per_10k epochs per 10k training sequences.
        # More data → more epochs needed to converge.
        # Fewer data → fewer epochs to avoid overfitting.
        n_train = len(self.ds)   # val sequences already excluded
        raw_epochs = epochs_per_10k * (n_train / 10_000)
        self.scaled_epochs = int(max(min_epochs, min(max_epochs, round(raw_epochs))))
        print(f"Dataset ready: {n_train} train sequences in memory.", flush=True)
        print(f"Scaled epochs : {self.scaled_epochs} "
              f"({epochs_per_10k} per 10k × {n_train/1000:.1f}k sequences, "
              f"clipped to [{min_epochs}, {max_epochs}])", flush=True)

    def train(self, epochs=20, save_path="fretboard_transformer.pt",
              warmup_epochs=3):
        """
        Linear warmup for warmup_epochs then cosine anneal to eta_min.
        Per-step scheduling stabilises early training when feature embeddings
        (ts/vel/dur) start from random and need time to settle.
        """
        import math
        total_steps   = epochs * self.steps_per_epoch
        warmup_steps  = warmup_epochs * self.steps_per_epoch
        base_lr       = self.optim.param_groups[0]['lr']
        eta_min_ratio = 1e-5 / base_lr

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return eta_min_ratio + (1 - eta_min_ratio) * cosine

        sched       = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)
        global_step = 0
        for ep in range(1, epochs + 1):
            self.model.train()
            tot_loss = correct = total = 0

            for step in range(self.steps_per_epoch):
                if ep == 1 and step % 10 == 0:
                    print(f"  step {step}/{self.steps_per_epoch} "
                          f"queue={self.ds.queue_len}", flush=True)

                keys, positions, pitches, tshifts, vels, durs, midi_mask = self.ds.get_batch(self.batch_size)
                # Key embedding disabled — use unknown token
                keys      = torch.full((positions.shape[0],), 12,
                                       dtype=torch.long, device=self.device)
                positions  = positions.to(self.device)
                pitches    = pitches.to(self.device)
                tshifts    = tshifts.to(self.device)
                vels       = vels.to(self.device)
                durs       = durs.to(self.device)
                midi_mask  = midi_mask.to(self.device)  # (B,) True = MIDI-only (no pos labels)
                has_pos    = ~midi_mask                  # (B,) True = has valid position labels
                B, T       = positions.shape

                self.optim.zero_grad()

                if self.training_mode == 'masked_lm':
                    # ── Joint Masked LM ────────────────────────────────────────
                    # Independently mask positions (40%), pitches (10%),
                    # durations (20%), velocities (20%).
                    rand_pos  = torch.rand(B, T, device=self.device)
                    rand_pit  = torch.rand(B, T, device=self.device)
                    rand_dur  = torch.rand(B, T, device=self.device)
                    rand_vel  = torch.rand(B, T, device=self.device)
                    masked_p  = rand_pos < self.mask_prob           # position mask ~40%
                    masked_i  = rand_pit < (self.mask_prob * 0.25)  # pitch mask ~10%
                    masked_d  = rand_dur < (self.mask_prob * 0.5)   # duration mask ~20%
                    masked_v  = rand_vel < (self.mask_prob * 0.5)   # velocity mask ~20%

                    inp_pos   = positions.clone()
                    inp_pit   = pitches.clone()
                    inp_durs  = durs.clone()
                    inp_vels  = vels.clone()
                    inp_pos[masked_p] = self.mask_idx
                    inp_pit[masked_i] = PITCH_MASK_IDX
                    inp_durs[masked_d] = 0   # bin 0 = unknown/masked duration
                    inp_vels[masked_v] = 0   # bin 0 = unknown/masked velocity

                    pos_logits, dur_logits, pit_logits, _, vel_logits = self.model(
                        inp_pit, inp_pos,
                        time_shifts=tshifts, velocities=inp_vels, durations=inp_durs)

                    pos_loss = F.nll_loss(
                        pos_logits[masked_p], positions[masked_p]) \
                        if masked_p.any() else torch.tensor(0.0, device=self.device)
                    pit_loss = F.nll_loss(
                        pit_logits[masked_i], pitches[masked_i]) \
                        if masked_i.any() else torch.tensor(0.0, device=self.device)
                    dur_loss = F.nll_loss(
                        dur_logits[masked_d], durs[masked_d]) \
                        if masked_d.any() else torch.tensor(0.0, device=self.device)
                    # Velocity loss: only on tokens where the label is known (bin > 0).
                    # MIDI-only sequences have vel_bin=0 (unknown) — exclude them so
                    # the head learns from GP/SynthTab velocity labels only.
                    vel_targets_valid = masked_v & (vels > 0)
                    vel_loss = F.nll_loss(
                        vel_logits[vel_targets_valid], vels[vel_targets_valid]) \
                        if vel_targets_valid.any() else torch.tensor(0.0, device=self.device)

                    loss  = pos_loss + 0.2 * pit_loss + 0.3 * dur_loss + 0.1 * vel_loss
                    preds = pos_logits[masked_p].argmax(-1)
                    correct += (preds == positions[masked_p]).sum().item()
                    total   += masked_p.sum().item() if masked_p.any() else 1
                else:
                    # ── Autoregressive ────────────────────────────────────────
                    # Input: positions 0..T-2 (teacher forcing).
                    # Target: positions 1..T-1 (next position prediction).
                    # The causal mask in PositionEncoder prevents future leakage.
                    inp    = positions[:, :-1]   # (B, T-1) known past
                    tgt    = positions[:, 1:]    # (B, T-1) next position
                    pos_logits, dur_logits, pit_logits, tonic_logits, vel_logits = self.model(
                                        pitches, inp,
                                        time_shifts=tshifts,
                                        velocities=vels,
                                        durations=durs)
                    dur_tgt   = durs[:, 1:]
                    pit_tgt   = pitches[:, 1:]
                    key_tgt   = keys % 12
                    has_key   = (keys < 12)
                    Bt, Tt, _ = pos_logits.shape

                    # Pitch loss: only when MIDI sequences present in batch
                    # When midi_cache=None all sequences are GP/SynthTab and
                    # pit_loss would compete with pos_loss for the same repr.
                    if midi_mask.any():
                        pit_loss = F.nll_loss(
                            pit_logits.reshape(Bt * Tt, NUM_MIDI),
                            pit_tgt.reshape(Bt * Tt))
                    else:
                        pit_loss = torch.tensor(0.0, device=self.device)

                    # Position + duration loss: only sequences with valid position labels
                    if has_pos.any():
                        pl  = pos_logits[has_pos]
                        tl  = tgt[has_pos]
                        dl  = dur_logits[has_pos]
                        dtl = dur_tgt[has_pos]
                        np_, tp_, _ = pl.shape
                        pos_loss = F.nll_loss(
                            pl.reshape(np_ * tp_, NUM_POSITIONS),
                            tl.reshape(np_ * tp_))
                        dur_loss = F.nll_loss(
                            dl.reshape(np_ * tp_, N_DUR_BINS + 1),
                            dtl.reshape(np_ * tp_))
                    else:
                        pos_loss = torch.tensor(0.0, device=self.device)
                        dur_loss = torch.tensor(0.0, device=self.device)

                    if has_key.any():
                        tonic_loss = F.nll_loss(
                            tonic_logits[has_key], key_tgt[has_key])
                    else:
                        tonic_loss = torch.tensor(0.0, device=self.device)

                    # Velocity loss: only on sequences with position labels and
                    # where the velocity label is known (bin > 0).
                    vel_tgt = vels[:, 1:]   # align with AR shift (predict vel t+1)
                    if has_pos.any():
                        vl      = vel_logits[has_pos]     # (B_pos, T-1, N_VEL_BINS+1)
                        vtl     = vel_tgt[has_pos]        # (B_pos, T-1)
                        nv_, tv_, _ = vl.shape
                        vl_flat = vl.reshape(nv_ * tv_, N_VEL_BINS + 1)
                        vtl_flat = vtl.reshape(nv_ * tv_)
                        known_vel = vtl_flat > 0   # exclude unknown bin 0
                        vel_loss = F.nll_loss(vl_flat[known_vel], vtl_flat[known_vel]) \
                            if known_vel.any() else torch.tensor(0.0, device=self.device)
                    else:
                        vel_loss = torch.tensor(0.0, device=self.device)

                    loss = pos_loss + 0.3 * dur_loss + 0.2 * pit_loss + 0.1 * tonic_loss + 0.1 * vel_loss

                    # Accuracy: only over sequences with position labels
                    if has_pos.any():
                        preds = pos_logits[has_pos].argmax(-1)
                        correct += (preds == tgt[has_pos]).sum().item()
                        total   += tgt[has_pos].numel()
                    else:
                        total += 1  # prevent division by zero

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()
                sched.step()
                global_step += 1
                tot_loss += loss.item()

            avg_loss = tot_loss / self.steps_per_epoch
            acc      = correct / total if total else 0

            # ── Validation ────────────────────────────────────────────────────
            self.model.eval()
            val_loss = val_cor = val_tot = 0
            val_steps = max(1, self.steps_per_epoch // 5)
            with torch.no_grad():
                for _ in range(val_steps):
                    vkeys, vpos, vpit, vts, vvel, vdur, vmidi_mask = self.ds.get_val_batch(self.batch_size)
                    vkeys      = torch.full((vpos.shape[0],), 12,
                                           dtype=torch.long, device=self.device)
                    vpos       = vpos.to(self.device); vpit = vpit.to(self.device)
                    vts        = vts.to(self.device);  vvel = vvel.to(self.device)
                    vdur       = vdur.to(self.device)
                    vmidi_mask = vmidi_mask.to(self.device)
                    vhas_pos   = ~vmidi_mask
                    B2, T2 = vpos.shape
                    if self.training_mode == 'masked_lm':
                        vmask_p  = torch.rand(B2, T2, device=self.device) < self.mask_prob
                        vmask_i  = torch.rand(B2, T2, device=self.device) < (self.mask_prob * 0.25)
                        vmask_d  = torch.rand(B2, T2, device=self.device) < (self.mask_prob * 0.5)
                        vinp_pos = vpos.clone();  vinp_pos[vmask_p] = self.mask_idx
                        vinp_pit = vpit.clone();  vinp_pit[vmask_i] = PITCH_MASK_IDX
                        vinp_dur = vdur.clone();  vinp_dur[vmask_d] = 0
                        vpos_logits, _, _, _, _ = self.model(vinp_pit, vinp_pos,
                                                          time_shifts=vts,
                                                          velocities=vvel,
                                                          durations=vinp_dur)
                        pos_vmask = vmask_p & vhas_pos.unsqueeze(1)
                        if pos_vmask.any():
                            val_loss += F.nll_loss(vpos_logits[pos_vmask], vpos[pos_vmask]).item()
                            val_cor  += (vpos_logits[pos_vmask].argmax(-1) == vpos[pos_vmask]).sum().item()
                            val_tot  += pos_vmask.sum().item()
                    else:
                        vinp = vpos[:, :-1]
                        vtgt = vpos[:, 1:]
                        vpos_logits, _, _, _, _ = self.model(vpit, vinp,
                                                 time_shifts=vts,
                                                 velocities=vvel,
                                                 durations=vdur)
                        # Only evaluate position loss on non-MIDI sequences
                        if vhas_pos.any():
                            pl  = vpos_logits[vhas_pos]
                            tl  = vtgt[vhas_pos]
                            Bv, Tv, _ = pl.shape
                            val_loss += F.nll_loss(
                                pl.reshape(Bv*Tv, NUM_POSITIONS),
                                tl.reshape(Bv*Tv)).item()
                            val_cor  += (pl.argmax(-1) == tl).sum().item()
                            val_tot  += tl.numel()
                        else:
                            val_loss += 0.0
                            val_tot  += 1  # prevent div/0
            val_loss /= val_steps
            val_acc   = val_cor / val_tot if val_tot else 0

            print(f"Ep {ep:3d}/{epochs}  "
                  f"loss={avg_loss:.4f} acc={acc:.3f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")

            if val_loss < self.best:
                self.best = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"  ✓ Saved (val_loss={val_loss:.4f})")

        self.ds.stop()
        print(f"Training complete. Best loss: {self.best:.4f}")

    def resume_training(self, epochs=20, save_path="fretboard_transformer.pt",
                        lr=3e-5):
        """
        Continue training from saved checkpoint with a fresh cosine LR schedule.
        Useful after initial training converges — restarts with a lower LR.
        """
        if os.path.exists(save_path):
            self.model.load_state_dict(
                torch.load(save_path, map_location=self.device))
            print(f"Resumed from {save_path}", flush=True)
        else:
            print("No checkpoint found — training from current weights.", flush=True)

        # Fresh optimizer + scheduler at lower LR
        self.optim = Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.best  = float('inf')
        self.train(epochs=epochs, save_path=save_path)


# ─── A* trellis decoder ───────────────────────────────────────────────────────
