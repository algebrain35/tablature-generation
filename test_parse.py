"""
test_parse.py
-------------
Tests the exact _parse_entry_full logic on scraped GP files.
Run from your project directory: python3 test_parse.py
"""
import glob, os, traceback

NUM_STRINGS = 6
NUM_FRETS   = 20
POSITIONS   = [(s, f) for s in range(1, NUM_STRINGS+1) for f in range(0, NUM_FRETS+1)]
POS_TO_IDX  = {p: i for i, p in enumerate(POSITIONS)}
STANDARD_TUNING = {1:64, 2:59, 3:55, 4:50, 5:45, 6:40}

def pos_to_midi(s, f, tuning=STANDARD_TUNING):
    return tuning[s] + f

# ── Copy of parse_scoreset_gp ─────────────────────────────────────────────────
import guitarpro

_GP_STEP_TO_SEMITONE = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}

def _gp_string_midi(gs):
    return (gs.value + 1) * 12 + _GP_STEP_TO_SEMITONE.get(gs.name, 0)

def parse_scoreset_gp_verbose(path):
    """Same as parse_scoreset_gp but prints what happens."""
    try:
        song = guitarpro.parse(path)
    except Exception as e:
        print(f"    PARSE FAIL: {e}")
        return []

    notes = []
    for ti, track in enumerate(song.tracks):
        try:
            channel = track.channel.channel
        except AttributeError:
            channel = 0   # GP3/GP4 — assume non-drums

        n_strings = len(track.strings)

        if channel == 9:
            continue
        if n_strings != NUM_STRINGS:
            continue

        tuning = {}
        for gs in track.strings:
            tuning[gs.number] = _gp_string_midi(gs)

        track_notes = 0
        for measure in track.measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    for note in beat.notes:
                        s = note.string
                        f = note.value
                        if f < 0 or f > NUM_FRETS:
                            continue
                        if not (1 <= s <= NUM_STRINGS):
                            continue
                        if s not in tuning:
                            continue
                        if (s, f) not in POS_TO_IDX:
                            continue
                        notes.append((s, f))
                        track_notes += 1

    return notes


# ── Test on first 20 scraped files ────────────────────────────────────────────
files = sorted(glob.glob('./assets/data/scraped_tabs/**/*.gp*', recursive=True))
files = [f for f in files if os.path.getsize(f) >= 1000][:20]

print(f"Testing {len(files)} files...\n")

ok = fail = empty = 0
for f in files:
    fname = os.path.basename(f)
    # Simulate _parse_entry_full exactly
    try:
        notes = parse_scoreset_gp_verbose(f)
        notes = [(s, ff) for s, ff in notes if (s, ff) in POS_TO_IDX]
        if len(notes) < 8:
            print(f"  SKIP  {fname}  ({len(notes)} notes < 8)")
            empty += 1
        else:
            print(f"  OK    {fname}  ({len(notes)} notes)")
            ok += 1
    except Exception as e:
        print(f"  ERROR {fname}: {e}")
        traceback.print_exc()
        fail += 1

print(f"\nResults: {ok} ok, {empty} too short, {fail} errors")
print(f"\nNow testing _parse_entry_full with exception logging...")

# Test with exception catching removed
for f in files[:5]:
    fname = os.path.basename(f)
    source_name = 'ScrapedTabs'
    kind = 'gp'
    tagged_entry = (source_name, kind, f)
    try:
        notes = parse_scoreset_gp_verbose(f)
        notes = [(s, ff) for s, ff in notes if (s, ff) in POS_TO_IDX]
        if len(notes) < 8:
            result = None
        else:
            pos = [POS_TO_IDX[(s, ff)] for s, ff in notes]
            pit = [pos_to_midi(s, ff) for s, ff in notes]
            result = (source_name, pos, pit)
        print(f"  {fname}: {'OK len=' + str(len(pos)) if result else 'None'}")
    except Exception as e:
        print(f"  {fname}: EXCEPTION — {e}")
        traceback.print_exc()
