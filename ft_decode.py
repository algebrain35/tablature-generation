"""
ft_decode.py — A* trellis decoder and post-processing.

Classes:
    TrellisDecoder — A* decoder for AR model (FretboardTransformer)

Functions:
    postprocess    — Edwards et al. neighbourhood heuristic
"""

import heapq
import torch
import torch.nn.functional as F

from ft_model import (
    NUM_POSITIONS, NUM_FRETS, POSITIONS, POS_TO_IDX, STANDARD_TUNING,
    pos_to_midi, midi_to_positions,
)

class TrellisDecoder:
    """
    A* trellis decoder using the masked prediction model.

    At each step the model predicts P(pos_t | all pitches, pos_{0:t-1})
    using bidirectional pitch encoder (precomputed once) and causal
    position encoder. A* finds the globally optimal path.

    Heuristic (inadmissible weighted A*):
        h = fret_bias * fret
          + transition_bias * horizontal_jump
          + string_bias * vertical_jump

    where horizontal_jump is fretwise distance between consecutive positions
    and vertical_jump is the string distance (1–6 scale). Open strings
    receive zero vertical cost — the fretting hand is free to reposition
    during an open note's sustain.
    """
    def __init__(self, model, tuning=STANDARD_TUNING, fret_bias=0.35,
                 transition_bias=0.2714, string_bias=0.0, device=None):
        """
        fret_bias       : penalty per absolute fret (prefers lower positions)
        transition_bias : penalty per fret of horizontal distance between notes
        string_bias     : penalty per string of vertical distance between notes.
                          Open strings incur zero vertical penalty — the hand
                          is free to reposition during an open note's sustain.
        """
        self.model           = model
        self.tuning          = tuning
        self.fret_bias       = fret_bias
        self.transition_bias = transition_bias
        self.string_bias     = string_bias
        self.device    = device if device is not None else next(model.parameters()).device
        self.model.eval()

    def decode(self, pitch_sequence):
        T       = len(pitch_sequence)
        pitches = torch.tensor(pitch_sequence, dtype=torch.long, device=self.device)

        # Estimate key from pitch sequence and encode with key conditioning
        # Key embedding disabled for AR SynthTab baseline
        pitch_ctx = self.model.encode_pitches(pitches, key=None)   # (T, D)

        cands = [
            [POS_TO_IDX[p] for p in midi_to_positions(m, self.tuning)
             if p in POS_TO_IDX]
            for m in pitch_sequence
        ]
        if any(len(c) == 0 for c in cands):
            raise ValueError("A pitch has no valid position under this tuning.")

        # Clear trajectory cache — safe to reuse across paths with same past_positions
        self.model.clear_decode_cache()

        def lp(pos_idx, t, past_path):
            log_probs, _ = self.model.decode_step(pitch_ctx[t], list(past_path), t)
            return log_probs[pos_idx].item()

        heap    = [(self.fret_bias * POSITIONS[p][1], 0.0, 0, p, (p,))
                   for p in cands[0]]
        heapq.heapify(heap)
        visited = set()   # (t, pos) pairs already expanded — skip duplicates

        while heap:
            _, g, t, pos, path = heapq.heappop(heap)

            if (t, pos) in visited:
                continue
            visited.add((t, pos))

            if t == T - 1:
                return [{'string' : POSITIONS[i][0],
                         'fret'   : POSITIONS[i][1],
                         'midi'   : pos_to_midi(*POSITIONS[i], self.tuning),
                         'is_open': POSITIONS[i][1] == 0}
                        for i in path]

            for nxt in cands[t + 1]:
                if (t + 1, nxt) in visited:
                    continue
                nf      = POSITIONS[nxt][1]   # next fret
                ns      = POSITIONS[nxt][0]   # next string
                cur_f   = POSITIONS[pos][1]   # current fret
                cur_s   = POSITIONS[pos][0]   # current string

                # ── Horizontal cost: fretwise distance ────────────────────
                # When moving TO an open string: if currently at fret <= 4
                # the hand is already near open position (no jump cost);
                # if far up the neck, penalise the repositioning distance.
                # When moving FROM an open string: pay the full fret distance.
                if nf == 0:
                    h_jump = 0 if cur_f <= 4 else cur_f
                elif cur_f == 0:
                    h_jump = nf
                else:
                    h_jump = abs(nf - cur_f)

                # ── Vertical cost: string distance ────────────────────────
                # Open strings incur zero vertical penalty — hand is free
                # to move laterally during an open note's sustain.
                if nf == 0 or cur_f == 0:
                    v_jump = 0
                else:
                    v_jump = abs(ns - cur_s)

                w   = -lp(nxt, t + 1, path)
                ng  = g + w
                # Inadmissible heuristic — may overestimate future cost but
                # empirically guides the search toward ergonomic paths.
                # The inadmissibility is acceptable: edge weights from the
                # neural network are already approximations of biomechanical
                # difficulty, so strict A* optimality is not the criterion.
                h   = (self.fret_bias       * nf
                     + self.transition_bias * h_jump
                     + self.string_bias     * v_jump)
                heapq.heappush(heap,
                    (ng + h, ng, t + 1, nxt, path + (nxt,)))

        raise RuntimeError("A* failed — no valid path found.")

    def postprocess(self, decoded, max_deviation=5, max_fret=20):
        """
        Post-processing heuristic from Edwards et al. (MIDI-to-Tab, ISMIR 2024).
        For each note in a window of 11 (5 past, 1 middle, 5 future), if the
        middle note's fret deviates more than max_deviation from the local
        neighbourhood mean (excluding open strings), or exceeds max_fret,
        relocate it to a better string.

        decoded : list of dicts with 'string', 'fret', 'midi' keys (from decode())
        Returns : corrected list of dicts (same format, modified in place copy)
        """
        result = [dict(d) for d in decoded]  # shallow copy
        n_fixed = 0
        for i in range(len(result)):
            lo = max(0, i - 5)
            hi = min(len(result), i + 6)
            window = result[lo:hi]

            # Mean fret of neighbourhood, excluding open strings
            fretted = [w['fret'] for w in window if w['fret'] > 0]
            mean_fret = sum(fretted) / len(fretted) if fretted else 0

            mid = result[i]
            if mid['fret'] > max_fret or (mid['fret'] > 0 and abs(mid['fret'] - mean_fret) > max_deviation):
                # Find all valid positions for this pitch
                cands = midi_to_positions(mid['midi'], self.tuning)
                if not cands:
                    continue

                # Prefer open strings first; then minimise distance from mean
                open_cands = [(s, f) for s, f in cands if f == 0]
                if open_cands:
                    best_s, best_f = open_cands[0]
                else:
                    best_s, best_f = min(cands, key=lambda sf: abs(sf[1] - mean_fret))

                result[i] = {
                    'string':  best_s,
                    'fret':    best_f,
                    'midi':    mid['midi'],
                    'is_open': best_f == 0,
                }
                n_fixed += 1

        return result, n_fixed


# ─── Entrypoint ───────────────────────────────────────────────────────────────
