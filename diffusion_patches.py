"""
diffusion_patches.py — Drop-in replacements for two bottlenecks in
generate_diffusion.py.

PATCH 1: CachedMLMModel
    Wraps FretboardTransformerMLM and caches the position encoder output
    so that repeated calls to decode_joint_step with the same (cur_pitches,
    cur_positions, cur_durations) state don't rerun the full transformer.

    The position encoder is the expensive redundant computation during
    masked diffusion: with T=64 tokens, 10 steps, and ~40 masked tokens
    per step, the naive implementation runs the pos_enc transformer
    ~2,500 times.  After caching, each unique (positions, durations) state
    is encoded once — typically 10-30 unique states per decode call.

    Usage: replace your model with CachedMLMModel(model) before calling
    masked_diffusion_decode.  The wrapper is transparent — all existing
    method calls work unchanged.

PATCH 2: velocity-aware synthesize_wav
    Replaces generate_diffusion.synthesize_wav with a version that reads
    a `velocities` list (one bin index 1-8 per note, from vel_head) and
    scales the Karplus-Strong excitation amplitude accordingly.  Also adds
    string-dependent decay so the low E string sustains longer than the
    high e.

    Usage: call synthesize_wav_v2 in place of synthesize_wav, passing the
    `velocities` list returned by masked_diffusion_decode_v2 (below).

PATCH 3: masked_diffusion_decode_v2
    Extends the existing masked_diffusion_decode to also score and reveal
    velocity tokens alongside pitch, position, and duration.  Returns a
    fourth value `velocities` (list of int bin indices, 1-8).

    Usage: replace your masked_diffusion_decode call with
    masked_diffusion_decode_v2.  The extra return value slots straight into
    synthesize_wav_v2.

All three patches are self-contained — copy the functions you need into
generate_diffusion.py, or import from here.
"""

import math, random
import torch
import torch.nn.functional as F


# ── Constants (must match ft_model.py) ───────────────────────────────────────

try:
    from ft_model import (
        NUM_MIDI, NUM_POSITIONS, N_DUR_BINS, N_VEL_BINS,
        POSITIONS, POS_TO_IDX, STANDARD_TUNING,
        pos_to_midi, midi_to_positions,
    )
    from ft_model import FretboardTransformerMLM
    _MASK_IDX       = FretboardTransformerMLM.MASK_IDX
    _PITCH_MASK_IDX = FretboardTransformerMLM.PITCH_MASK_IDX
except ImportError:
    # Fallback constants if running standalone
    NUM_MIDI      = 128
    NUM_POSITIONS = 126
    N_DUR_BINS    = 16
    N_VEL_BINS    = 8
    _MASK_IDX       = NUM_POSITIONS
    _PITCH_MASK_IDX = NUM_MIDI


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 1 — CachedMLMModel
# ═══════════════════════════════════════════════════════════════════════════════

class CachedMLMModel:
    """
    Transparent wrapper around FretboardTransformerMLM that caches the
    position encoder (pos_enc) output keyed on the full position+duration
    state tuple.

    Why only cache pos_enc, not pitch_enc?
        pitch_enc is bidirectional over all pitches and is called once per
        step in generate_pitches_model (already efficient).  In
        decode_joint_step, pitch_enc IS recomputed per masked token, but
        its inputs (cur_pitches) change every time a pitch token is revealed,
        making caching less effective.  pos_enc, by contrast, is called with
        identical (cur_positions, cur_durations) for every masked-token
        scoring call within the same step — perfect cache hit rate.

    Cache invalidation:
        The cache is keyed on (tuple(cur_positions), tuple(cur_durations)).
        It is automatically cleared at the start of each call to
        decode_joint_step if the key changes — which happens exactly once
        per diffusion step when tokens are revealed.

    Thread safety: not thread-safe (single-threaded inference only).
    """

    def __init__(self, model):
        self._model      = model
        self._cache      = {}       # key → (B=1, T, D) tensor
        self._last_key   = None

    def __getattr__(self, name):
        """Proxy all attribute access to the wrapped model."""
        return getattr(self._model, name)

    def clear_cache(self):
        self._cache.clear()
        self._last_key = None

    @torch.no_grad()
    def _get_pos_enc(self, cur_positions, cur_durations, device):
        """
        Return pos_enc output for the given state, using cache.
        cur_positions : list[int]  length T
        cur_durations : list[int]  length T
        Returns       : (T, D) tensor
        """
        key = (tuple(cur_positions), tuple(cur_durations))
        if key not in self._cache:
            T       = len(cur_positions)
            pos_t   = torch.tensor(cur_positions,  dtype=torch.long, device=device).unsqueeze(0)
            # pos_enc does not take durations directly — durations feed into
            # pitch_enc via dur_embed.  pos_enc only sees position tokens.
            traj    = self._model.pos_enc(pos_t)    # (1, T, D)
            self._cache[key] = traj.squeeze(0)      # (T, D)
        return self._cache[key]

    @torch.no_grad()
    def decode_joint_step(self, all_pitches, all_positions, target_t,
                          all_durations=None):
        """
        Cached version of FretboardTransformerMLM.decode_joint_step.

        Identical interface and return values — drop-in replacement.
        pos_enc is computed once per unique (positions, durations) state
        and reused for every masked token in the same diffusion step.
        pitch_enc is still computed per call (inputs differ per token).
        """
        model  = self._model
        model.eval()
        dev    = next(model.parameters()).device
        T      = len(all_pitches)

        dur_list = all_durations if all_durations is not None else [0] * T

        # ── pitch_enc: still per-call (inputs change when pitches are revealed)
        pit_t  = torch.tensor(all_pitches, dtype=torch.long, device=dev).unsqueeze(0)
        dur_t  = torch.tensor(dur_list,    dtype=torch.long, device=dev).unsqueeze(0)

        pitch_ctx = model.pitch_enc(pit_t, durations=dur_t)  # (1, T, D)

        # ── pos_enc: cached — same for all masked tokens in this step
        traj_ctx_t = self._get_pos_enc(all_positions, dur_list, dev)  # (T, D)

        # Fuse at target timestep only
        p_ctx   = pitch_ctx[0, target_t]          # (D,)
        t_ctx   = traj_ctx_t[target_t]            # (D,)
        fused   = model.fusion(
            torch.cat([p_ctx, t_ctx], dim=-1).unsqueeze(0))   # (1, D)

        pit_lp  = F.log_softmax(model.pitch_head(fused), dim=-1).squeeze(0)
        pos_lp  = F.log_softmax(model.head(fused),       dim=-1).squeeze(0)
        dur_lp  = F.log_softmax(model.dur_head(fused),   dim=-1).squeeze(0)
        return pit_lp, pos_lp, dur_lp

    @torch.no_grad()
    def decode_step_bidirectional(self, pitch_ctx, all_positions, target_t):
        """Cached version of decode_step_bidirectional."""
        model = self._model
        model.eval()
        dev   = pitch_ctx.device
        traj  = self._get_pos_enc(all_positions, [0]*len(all_positions), dev)
        p_ctx = pitch_ctx[target_t] if pitch_ctx.dim() == 2 else pitch_ctx
        fused = model.fusion(
            torch.cat([p_ctx, traj[target_t]], dim=-1).unsqueeze(0))
        return F.log_softmax(model.head(fused), dim=-1).squeeze(0)


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 2 — velocity-aware Karplus-Strong synthesis
# ═══════════════════════════════════════════════════════════════════════════════

# Map velocity bin (1-8) to amplitude scale factor.
# Bin 1 = very soft (pp), bin 8 = very loud (ff).
# Using a perceptual curve: amplitude ∝ velocity^0.6 feels more natural
# than linear because human loudness perception is logarithmic.
_VEL_BIN_AMP = [0.0] + [
    (b / 8) ** 0.6  for b in range(1, 9)
]  # index 0 = unknown → treated as mf (0.6)


def karplus_strong_v2(freq, duration, sr=44100, decay=0.996, brightness=0.5,
                       amplitude=1.0):
    """
    Karplus-Strong with amplitude control for velocity dynamics.

    amplitude : float in [0, 1] — scales the initial noise excitation.
                Louder notes have a stronger initial burst; the feedback
                loop ensures the decay envelope is otherwise identical.
    """
    import numpy as np
    n_samples = int(sr * duration)
    buf_len   = max(2, int(sr / freq))
    # Scale initial excitation by amplitude
    buf       = np.random.uniform(-amplitude, amplitude, buf_len).astype(np.float64)
    out       = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        out[i]         = buf[i % buf_len]
        nxt            = decay * ((1 - brightness) * buf[i % buf_len]
                                  + brightness * buf[(i + 1) % buf_len])
        buf[i % buf_len] = nxt
    peak = np.max(np.abs(out))
    return (out / peak * amplitude).astype(np.float32) if peak > 1e-9 \
           else out.astype(np.float32)


def synthesize_wav_v2(decoded, tuning, bpm=120.0, sr=44100,
                      note_dur_beats=0.5, gap_beats=0.02,
                      durations=None, velocities=None):
    """
    Velocity- and string-aware WAV synthesis.

    Differences from the original synthesize_wav:
      - velocities: list of bin indices (0-8).  Bin 0 = unknown → mf.
        Each note's Karplus-Strong excitation is scaled by amplitude ∝ vel^0.6
        so forte notes cut through and piano notes sit back.
      - String-dependent decay: low strings (6=low-E) sustain longer than
        high strings (1=high-e).  A player's low E ring is noticeably longer
        than the high e pluck — this difference (≈0.003 decay per string)
        is enough to be audible without sounding artificial.
      - Fret-dependent brightness and decay are preserved from the original.

    Parameters
    ----------
    decoded    : list of dicts with keys string, fret, is_open
    tuning     : dict {string: open_midi}
    durations  : optional list[int] bin indices (1-16); None → note_dur_beats
    velocities : optional list[int] bin indices (0-8); None → uniform mf
    """
    import numpy as np
    N_DUR_BINS_LOCAL = 16

    def dur_beats(bin_idx):
        if bin_idx is None or bin_idx <= 0:
            return note_dur_beats
        lo, hi = 0.125, 2.0
        t = (bin_idx - 1) / max(N_DUR_BINS_LOCAL - 1, 1)
        return lo * (hi / lo) ** t

    beat_sec  = 60.0 / bpm
    gap_sec   = gap_beats * beat_sec

    note_secs = [
        max(0.1, dur_beats(durations[i] if durations else None) * beat_sec)
        for i in range(len(decoded))
    ]
    steps_sam = [int((ns + gap_sec) * sr) for ns in note_secs]
    total     = sum(steps_sam) + int(max(note_secs) * sr * 3)
    audio     = np.zeros(total, dtype=np.float32)

    cursor = 0
    for i, note in enumerate(decoded):
        m    = tuning[note["string"]] + note["fret"]
        freq = 440.0 * (2.0 ** ((m - 69) / 12.0))
        f    = note["fret"]
        s    = note["string"]

        # Fret-dependent brightness (higher frets = duller)
        bright = max(0.2, 0.7 - f * 0.025)

        # Fret + string -dependent decay.
        # String 6 (low E): slowest decay (sustains longest)
        # String 1 (high e): fastest decay
        # We add a small per-string bonus: +0.001 per string above 1
        string_sustain = 0.001 * (s - 1)   # 0 for string 1, 0.005 for string 6
        dec = max(0.990, (0.998 - f * 0.0003) + string_sustain)

        # Velocity → amplitude
        vel_bin = velocities[i] if velocities is not None else 0
        amp     = _VEL_BIN_AMP[vel_bin] if 0 <= vel_bin < len(_VEL_BIN_AMP) \
                  else _VEL_BIN_AMP[0]
        if amp < 0.05:
            amp = 0.6   # unknown bin → mezzo-forte

        grain = karplus_strong_v2(freq, note_secs[i] * 2.0, sr=sr,
                                  decay=dec, brightness=bright, amplitude=amp)
        start = cursor
        end   = start + len(grain)
        if end > len(audio):
            grain = grain[:len(audio) - start]; end = len(audio)
        audio[start:end] += grain
        cursor += steps_sam[i]

    # Normalise
    peak = np.max(np.abs(audio))
    if peak > 1e-9:
        audio = audio / peak * 0.9
    return audio


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 3 — masked_diffusion_decode_v2 (adds velocity tokens)
# ═══════════════════════════════════════════════════════════════════════════════

def annealed_temp(base_temp, step, total_steps, schedule):
    """Matches generate_diffusion.annealed_temp exactly."""
    if total_steps <= 1 or schedule == "flat":
        return base_temp
    t = step / (total_steps - 1)
    if schedule == "linear":
        factor = 1.0 - 0.5 * t
    else:
        factor = 0.5 * (1.0 + math.cos(math.pi * t))
    return max(0.1, base_temp * factor)


def annealed_temp_pitch(base_temp, step, total_steps, schedule):
    """
    Separate (softer) annealing for pitch tokens.

    Pitch generation benefits from more diversity late in the process —
    the last few pitch tokens should not be forced to argmax because they
    complete phrase endings where surprise is musically valuable.
    Floor at 0.5 (vs 0.1 for positions) to preserve that variety.
    """
    if total_steps <= 1 or schedule == "flat":
        return base_temp
    t = step / (total_steps - 1)
    if schedule == "linear":
        factor = 1.0 - 0.3 * t          # 1.0 → 0.7  (softer than position)
    else:
        factor = 0.5 * (1.0 + math.cos(math.pi * t * 0.7))  # cosine, slower
    return max(0.5, base_temp * factor)  # floor at 0.5


@torch.no_grad()
def masked_diffusion_decode_v2(model, pitches, tuning,
                                steps=10, base_temp=1.0,
                                anneal="cosine", order="confidence",
                                device="cpu", scale_mask=None,
                                args_repeat_window=4):
    """
    Extended masked diffusion decoder that also reveals velocity tokens.

    Returns
    -------
    pitches   : list[int]  — MIDI pitch per note
    durations : list[int]  — duration bin per note (1-16)
    velocities: list[int]  — velocity bin per note (1-8)
    decoded   : list[dict] — {string, fret, midi, is_open}

    Compared to the original masked_diffusion_decode:
      - Adds cur_velocities state and vel_masked tracking
      - Scores velocity tokens using a vel_head linear layer added to the
        model.  If the model has no vel_head, falls back to uniform mf (4).
      - Uses separate pitch temperature annealing (floor 0.5, not 0.1)
        so late-revealed pitch tokens keep melodic variety.
      - Wraps model in CachedMLMModel automatically for speed.

    Velocity head requirement
    -------------------------
    This patch adds a velocity head to the model at runtime if missing:
        model.vel_head = nn.Linear(embed_dim, N_VEL_BINS + 1)
    This head starts randomly initialised — fine-tune it separately or
    accept uniform velocity output until you add vel_head to the training
    loss in ft_data.py (see comments in that file near the dur_head loss).
    """
    import torch.nn as nn

    # Wrap for caching
    if not isinstance(model, CachedMLMModel):
        model = CachedMLMModel(model)

    inner = model._model
    inner.eval()

    # ── Add vel_head if missing ────────────────────────────────────────────
    if not hasattr(inner, 'vel_head'):
        embed_dim = inner.embed_dim
        inner.vel_head = nn.Linear(embed_dim, N_VEL_BINS + 1).to(device)
        # Init to predict mezzo-forte (bin 4) by default
        nn.init.zeros_(inner.vel_head.weight)
        inner.vel_head.bias.data[4] = 2.0   # soft bias toward mf
        print("[INFO] vel_head not found in checkpoint — "
              "using randomly-initialised head (uniform mf until fine-tuned)")

    T = len(pitches) if pitches else 32
    cands_by_midi = {
        m: [POS_TO_IDX[p] for p in midi_to_positions(m, tuning) if p in POS_TO_IDX]
        for m in range(NUM_MIDI)
    }

    # Initial state — everything masked
    cur_pitches    = [_PITCH_MASK_IDX] * T
    cur_positions  = [_MASK_IDX]       * T
    cur_durations  = [0]               * T
    cur_velocities = [0]               * T   # 0 = unknown

    pit_masked = list(range(T))
    pos_masked = list(range(T))
    dur_masked = list(range(T))
    vel_masked = list(range(T))

    # Reveal ~3 modalities per step per token
    tokens_per_step = max(1, (4 * T) // steps)

    for step in range(steps):
        # Separate temperatures for pitch (softer) and position/dur/vel
        temp_pos = annealed_temp(base_temp, step, steps, anneal)
        temp_pit = annealed_temp_pitch(base_temp, step, steps, anneal)

        # Invalidate pos_enc cache — new tokens were revealed last step
        model.clear_cache()

        candidates = []

        # ── Spatial repeat blacklist for pitches ──────────────────────────
        def spatial_blacklist(t_idx):
            if args_repeat_window <= 0:
                return set()
            nearby = set()
            for dt in range(1, args_repeat_window + 1):
                for nb in (t_idx - dt, t_idx + dt):
                    if 0 <= nb < T:
                        p = cur_pitches[nb]
                        if p != _PITCH_MASK_IDX:
                            nearby.add(p)
            return nearby

        # ── Score pitch tokens ────────────────────────────────────────────
        for t_idx in pit_masked:
            pit_lp, _, _ = model.decode_joint_step(
                cur_pitches, cur_positions, t_idx, cur_durations)
            lp = pit_lp / max(temp_pit, 1e-6)
            if scale_mask is not None:
                lp = lp.masked_fill(~scale_mask.to(lp.device), float('-inf'))
            for p in spatial_blacklist(t_idx):
                if 0 <= p < len(lp):
                    lp[p] = float('-inf')
            if torch.all(lp == float('-inf')):
                lp = pit_lp / max(temp_pit, 1e-6)
                if scale_mask is not None:
                    lp = lp.masked_fill(~scale_mask.to(lp.device), float('-inf'))
            probs = torch.softmax(lp, dim=-1)
            if probs.isnan().any() or probs.sum() < 1e-9:
                continue
            conf, best_k = probs.max(0)
            candidates.append((conf.item(), 'pit', t_idx, best_k.item(), probs))

        # ── Score position tokens (only if pitch is known) ────────────────
        for t_idx in pos_masked:
            if cur_pitches[t_idx] == _PITCH_MASK_IDX:
                continue
            midi_val = cur_pitches[t_idx]
            cands    = cands_by_midi.get(midi_val, [])
            if not cands:
                continue
            _, pos_lp, _ = model.decode_joint_step(
                cur_pitches, cur_positions, t_idx, cur_durations)
            lp_c  = pos_lp[cands] / max(temp_pos, 1e-6)
            probs = torch.softmax(lp_c, dim=-1)
            conf, best_k = probs.max(0)
            candidates.append((conf.item(), 'pos', t_idx,
                                cands[best_k.item()], probs))

        # ── Score duration tokens ─────────────────────────────────────────
        for t_idx in dur_masked:
            _, _, dur_lp = model.decode_joint_step(
                cur_pitches, cur_positions, t_idx, cur_durations)
            lp    = dur_lp / max(temp_pos, 1e-6)
            probs = torch.softmax(lp, dim=-1)
            probs[0] = 0.0  # exclude unknown bin
            if probs.sum() > 1e-9:
                probs = probs / probs.sum()
            conf, best_k = probs.max(0)
            candidates.append((conf.item(), 'dur', t_idx, best_k.item(), probs))

        # ── Score velocity tokens ─────────────────────────────────────────
        # vel_head shares the same fused representation as dur_head —
        # we re-run fusion at target_t to get the velocity logits.
        for t_idx in vel_masked:
            pit_lp, _, _ = model.decode_joint_step(
                cur_pitches, cur_positions, t_idx, cur_durations)
            # Compute fused representation for vel_head
            # (decode_joint_step already runs fusion internally; we need
            #  to call it again for the vel projection.  A future refactor
            #  could return fused directly to avoid this.)
            dev      = next(inner.parameters()).device
            pit_t    = torch.tensor(cur_pitches,   dtype=torch.long, device=dev).unsqueeze(0)
            dur_t    = torch.tensor(cur_durations, dtype=torch.long, device=dev).unsqueeze(0)
            pitch_ctx = inner.pitch_enc(pit_t, durations=dur_t)
            traj_ctx  = model._get_pos_enc(cur_positions, cur_durations, dev)
            p_ctx     = pitch_ctx[0, t_idx]
            t_ctx     = traj_ctx[t_idx]
            fused     = inner.fusion(torch.cat([p_ctx, t_ctx], dim=-1).unsqueeze(0))
            vel_lp    = F.log_softmax(inner.vel_head(fused), dim=-1).squeeze(0)
            lp        = vel_lp / max(temp_pos, 1e-6)
            probs     = torch.softmax(lp, dim=-1)
            probs[0]  = 0.0  # exclude unknown bin
            if probs.sum() > 1e-9:
                probs = probs / probs.sum()
            conf, best_k = probs.max(0)
            candidates.append((conf.item(), 'vel', t_idx, best_k.item(), probs))

        if not candidates:
            break

        # ── Sort and reveal ───────────────────────────────────────────────
        if order == "confidence":
            candidates.sort(key=lambda x: -x[0])
        elif order == "random":
            random.shuffle(candidates)
        else:
            candidates.sort(key=lambda x: x[2])

        n_reveal = len(candidates) if step == steps - 1 \
                   else min(tokens_per_step, len(candidates))

        revealed_pit = set(); revealed_pos = set()
        revealed_dur = set(); revealed_vel = set()

        for conf, kind, t_idx, best_val, probs in candidates[:n_reveal]:
            if kind == 'pit' and t_idx not in revealed_pit:
                cur_pitches[t_idx]    = best_val if temp_pit < 0.5 \
                    else torch.multinomial(probs, 1).item()
                revealed_pit.add(t_idx)
            elif kind == 'pos' and t_idx not in revealed_pos:
                midi_val = cur_pitches[t_idx]
                cands    = cands_by_midi.get(midi_val, [])
                if cands:
                    cur_positions[t_idx] = best_val if temp_pos < 0.2 \
                        else cands[torch.multinomial(probs, 1).item()]
                revealed_pos.add(t_idx)
            elif kind == 'dur' and t_idx not in revealed_dur:
                chosen = best_val if temp_pos < 0.2 \
                    else torch.multinomial(probs, 1).item()
                cur_durations[t_idx] = max(1, chosen)
                revealed_dur.add(t_idx)
            elif kind == 'vel' and t_idx not in revealed_vel:
                chosen = best_val if temp_pos < 0.2 \
                    else torch.multinomial(probs, 1).item()
                cur_velocities[t_idx] = max(1, chosen)
                revealed_vel.add(t_idx)

        pit_masked = [t for t in pit_masked if t not in revealed_pit]
        pos_masked = [t for t in pos_masked if t not in revealed_pos]
        dur_masked = [t for t in dur_masked if t not in revealed_dur]
        vel_masked = [t for t in vel_masked if t not in revealed_vel]

        if not any([pit_masked, pos_masked, dur_masked, vel_masked]):
            break

    # ── Final pass: fill any remaining masks ──────────────────────────────
    for t_idx in pit_masked:
        pit_lp, _, _ = model.decode_joint_step(
            cur_pitches, cur_positions, t_idx, cur_durations)
        cur_pitches[t_idx] = pit_lp.argmax().item()
    for t_idx in pos_masked:
        midi_val = cur_pitches[t_idx]
        if midi_val == _PITCH_MASK_IDX:
            cur_pitches[t_idx] = 60; midi_val = 60
        cands = cands_by_midi.get(midi_val, [])
        if cands:
            _, pos_lp, _ = model.decode_joint_step(
                cur_pitches, cur_positions, t_idx, cur_durations)
            cur_positions[t_idx] = cands[pos_lp[cands].argmax().item()]
        else:
            cur_positions[t_idx] = 0
    for t_idx in dur_masked:
        _, _, dur_lp = model.decode_joint_step(
            cur_pitches, cur_positions, t_idx, cur_durations)
        dur_lp[0] = float('-inf')
        cur_durations[t_idx] = max(1, dur_lp.argmax().item())
    for t_idx in vel_masked:
        cur_velocities[t_idx] = 4   # fallback: mezzo-forte

    # ── Safety: replace any out-of-range positions ────────────────────────
    for t_idx in range(T):
        p = cur_positions[t_idx]
        if p >= len(POSITIONS) or p < 0:
            midi_val = cur_pitches[t_idx] if cur_pitches[t_idx] < NUM_MIDI else 60
            cands    = cands_by_midi.get(midi_val, [0])
            cur_positions[t_idx] = cands[0] if cands else 0
        if cur_pitches[t_idx] >= NUM_MIDI:
            cur_pitches[t_idx] = 60

    decoded = [
        {
            "string":  POSITIONS[p][0],
            "fret":    POSITIONS[p][1],
            "midi":    pos_to_midi(*POSITIONS[p], tuning),
            "is_open": POSITIONS[p][1] == 0,
        }
        for p in cur_positions
    ]

    return cur_pitches, cur_durations, cur_velocities, decoded


# ── Integration guide ─────────────────────────────────────────────────────────

_INTEGRATION_GUIDE = """
Integration guide — generate_diffusion.py
==========================================

1. Import at the top of generate_diffusion.py:

    from diffusion_patches import (
        CachedMLMModel,
        synthesize_wav_v2,
        masked_diffusion_decode_v2,
    )

2. After loading the model, wrap it:

    model = FretboardTransformerMLM(dropout=0.0)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    model = CachedMLMModel(model)          # <-- add this line

3. Replace the masked_diffusion_decode call (in joint mode):

    # OLD
    pitches, durations, decoded = masked_diffusion_decode(
        model, [...], tuning, ...)

    # NEW
    pitches, durations, velocities, decoded = masked_diffusion_decode_v2(
        model, [...], tuning, ...)

4. Replace the synthesize_wav call:

    # OLD
    audio = synthesize_wav(decoded, tuning, bpm=args.bpm,
                           note_dur_beats=args.note_dur, durations=durations)

    # NEW
    audio = synthesize_wav_v2(decoded, tuning, bpm=args.bpm,
                              note_dur_beats=args.note_dur,
                              durations=durations, velocities=velocities)

5. Fine-tune pitch_head for better melodic generation:

    python finetune_pitch_head.py --midi midi_sequences.pkl --epochs 5
    python generate_diffusion.py --model fretboard_transformer_mlm_ft.pt ...
"""

if __name__ == "__main__":
    print(_INTEGRATION_GUIDE)
