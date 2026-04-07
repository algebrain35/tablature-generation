"""
ft_model.py — Neural architecture for fretboard transcription.

Classes:
    RoPEAttention            — Multi-head attention with rotary position embeddings
    RoPETransformerLayer     — Single transformer layer using RoPEAttention
    PitchEncoder             — Bidirectional RoPE encoder over pitch sequences
    PositionEncoder          — Causal transformer over past fretboard positions (AR model)
    BidirectionalPositionEncoder — Bidirectional transformer over positions (MLM model)
    KeyEmbedding             — Tonic/key conditioning embedding
    FretboardTransformer     — Full AR model (BART-style, causal pos decoder)
    FretboardTransformerMLM  — Full MLM model (bidirectional pos encoder, for masked diffusion)
"""
import math, os, re, glob, heapq
try:
    import jams
    HAS_JAMS = True
except ImportError:
    HAS_JAMS = False

try:
    import mido
    HAS_MIDO = True
except ImportError:
    HAS_MIDO = False

try:
    import guitarpro
    HAS_GP = True
except ImportError:
    HAS_GP = False
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# ─── Constants ────────────────────────────────────────────────────────────────

NUM_STRINGS   = 6
NUM_FRETS     = 20
NUM_POSITIONS = 126
NUM_MIDI      = 128
PAD_IDX       = NUM_POSITIONS        # reuse MASK as PAD — simpler, proven stable
PITCH_MASK_IDX = NUM_MIDI             # index 128 — masked pitch token for joint MLM

STANDARD_TUNING = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

DADAGP_GUITAR_PREFIXES = (
    "distorted", "overdrive", "clean", "acoustic", "guitar", "lead", "rhythm",
)
_NOTE_RE = re.compile(
    r'^(' + '|'.join(DADAGP_GUITAR_PREFIXES) + r')[^:]*:note:s?(\d+):f?(\d+)$'
)

# GuitarSet: string_num 0=low-E → 5=high-e  (opposite of our convention)
# Mapping: our_string = 6 - guitarset_string_num
GUITARSET_OPEN_MIDI = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}  # standard tuning

# SynthTab: Format-1 MIDI, 6 tracks, one per string.
# Track index follows GuitarSet convention: 0=low-E, 5=high-e.
# Mapping to our convention (1=high-e, 6=low-E): our_str = 6 - track_idx
SYNTHTAB_TRACK_OPEN_MIDI = {
    0: 40,   # low  E  (our string 6)
    1: 45,   # A        (our string 5)
    2: 50,   # D        (our string 4)
    3: 55,   # G        (our string 3)
    4: 59,   # B        (our string 2)
    5: 64,   # high e   (our string 1)
}

POSITIONS  = [(s, f) for s in range(1, NUM_STRINGS + 1)
                      for f in range(0, NUM_FRETS + 1)]
POS_TO_IDX = {p: i for i, p in enumerate(POSITIONS)}

# ─── Note feature quantization ──────────────────────────────────────────────
# Each note is encoded with 4 features: pitch, time_shift, velocity, duration
# Time shift and duration are quantized to log-scale bins based on tick counts.
# Velocity is divided into 8 uniform levels.  Bin 0 = unknown/unavailable.

N_TS_BINS  = 16   # time-shift bins (inter-onset interval)
N_VEL_BINS =  8   # velocity bins
N_DUR_BINS = 16   # duration bins

# Tick thresholds (assuming ~480 ticks-per-beat as MIDI standard).
# Bin 0 = 0 ticks (simultaneous / unknown), bins 1-15 = log-scale intervals.
_TICK_THRESHOLDS = [0, 15, 30, 60, 120, 180, 240, 360, 480, 600, 720, 960,
                    1200, 1440, 1920]  # 15 thresholds → 16 bins

def quantize_ticks(ticks):
    """Map a tick count to a bin index 0-15."""
    for i, thresh in enumerate(_TICK_THRESHOLDS):
        if ticks <= thresh:
            return i
    return N_TS_BINS - 1

def quantize_velocity(vel):
    """Map MIDI velocity 0-127 to bin 0-7 (0=unknown)."""
    if vel <= 0:
        return 0
    return min(7, 1 + (vel - 1) * 7 // 127)

# GP duration value (in ticks at 960 tpq) → duration bin
_GP_DUR_TICKS = {
    'whole': 1920, 'half': 960, 'quarter': 480, 'eighth': 240,
    'sixteenth': 120, 'thirty-second': 60, 'sixty-fourth': 30,
}


def pos_to_midi(string, fret, tuning=STANDARD_TUNING):
    return tuning[string] + fret

def midi_to_positions(midi, tuning=STANDARD_TUNING):
    return [(s, f) for s, f in POSITIONS if pos_to_midi(s, f, tuning) == midi]

def dadagp_str(dadagp_string):
    return 7 - dadagp_string


# ─── RoPE helpers ────────────────────────────────────────────────────────────

def precompute_rope_freqs(head_dim: int, max_len: int = 512,
                          base: float = 10000.0) -> torch.Tensor:
    """
    Precompute RoPE frequency tensor.
    Returns: (max_len, head_dim/2) complex tensor of e^(i * m * theta_k)
    """
    assert head_dim % 2 == 0
    # theta_k = 1 / base^(2k / head_dim)  for k in [0, head_dim/2)
    k       = torch.arange(0, head_dim, 2, dtype=torch.float)
    thetas  = 1.0 / (base ** (k / head_dim))              # (head_dim/2,)
    pos     = torch.arange(max_len, dtype=torch.float)    # (max_len,)
    freqs   = torch.outer(pos, thetas)                    # (max_len, head_dim/2)
    return torch.polar(torch.ones_like(freqs), freqs)     # complex (max_len, head_dim/2)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to query or key tensor.
    x     : (B, T, H, head_dim)  float
    freqs : (T, head_dim/2)      complex
    Returns (B, T, H, head_dim) float
    """
    B, T, H, D = x.shape
    # View as complex: pair up consecutive dims
    x_c = torch.view_as_complex(x.float().reshape(B, T, H, D // 2, 2))  # (B,T,H,D/2) complex
    # freqs: (T, D/2) → (1, T, 1, D/2)
    f   = freqs[:T].unsqueeze(0).unsqueeze(2)                            # (1,T,1,D/2)
    x_r = torch.view_as_real(x_c * f).reshape(B, T, H, D)               # (B,T,H,D)
    return x_r.to(x.dtype)


# ─── Custom RoPE attention layer ─────────────────────────────────────────────

class RoPEAttention(nn.Module):
    """
    Multi-head self-attention with Rotary Position Embeddings (Su et al. 2021).

    RoPE encodes position by rotating the Q and K vectors in each head using
    position-dependent rotation matrices. This gives:
      - Relative position awareness: attention(q_m, k_n) depends only on m-n
      - No learned position parameters — pure geometric encoding
      - Generalises to longer sequences than training length
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.0, max_len: int = 512):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.drop = nn.Dropout(dropout)

        freqs = precompute_rope_freqs(self.head_dim, max_len)
        self.register_buffer('rope_freqs', freqs)   # (max_len, head_dim/2) complex

    def forward(self, x: torch.Tensor,
                attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x         : (B, T, D)
        attn_mask : (T, T) bool or float, optional causal/padding mask
        """
        B, T, D = x.shape
        H, Dh   = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, H, Dh)    # (B,T,3,H,Dh)
        q, k, v = qkv.unbind(dim=2)                    # each (B,T,H,Dh)

        # Apply RoPE to Q and K
        q = apply_rope(q, self.rope_freqs)
        k = apply_rope(k, self.rope_freqs)

        # Scaled dot-product attention
        q = q.transpose(1, 2)   # (B,H,T,Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,T,T)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            else:
                scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)

        attn = self.drop(torch.softmax(scores, dim=-1))
        out  = torch.matmul(attn, v)                 # (B,H,T,Dh)
        out  = out.transpose(1, 2).reshape(B, T, D)  # (B,T,D)
        return self.proj(out)


class RoPETransformerLayer(nn.Module):
    """Pre-norm transformer layer with RoPE self-attention."""
    def __init__(self, embed_dim: int, num_heads: int,
                 ffn_dim: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn  = RoPEAttention(embed_dim, num_heads, dropout, max_len)
        self.ff    = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                attn_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.ff(self.norm2(x))
        return x


# ─── Encoders ────────────────────────────────────────────────────────────────

class PitchEncoder(nn.Module):
    """
    Encodes per-note musical features into per-timestep context vectors using
    a bidirectional transformer with Rotary Position Embeddings (RoPE).

    RoPE (Su et al. 2021) encodes position by rotating Q and K vectors in
    each attention head. The inner product of rotated Q_m and K_n depends
    only on their relative distance (m - n), giving:

      - True relative position awareness in every attention head
      - No learned position parameters
      - Extrapolates to sequences longer than training length
      - Compatible with bidirectional (encoder) attention

    The absolute positional embedding (pos_embed) is removed entirely.
    """
    def __init__(self, embed_dim=128, num_heads=8, num_layers=4,
                 ffn_dim=512, dropout=0.1, max_len=512):
        super().__init__()
        self.num_heads = num_heads

        # Feature embeddings — no pos_embed (RoPE handles position)
        self.pitch_embed = nn.Embedding(NUM_MIDI + 2,    embed_dim)
        self.ts_embed    = nn.Embedding(N_TS_BINS + 1,   embed_dim, padding_idx=0)
        self.vel_embed   = nn.Embedding(N_VEL_BINS + 1,  embed_dim, padding_idx=0)
        self.dur_embed   = nn.Embedding(N_DUR_BINS + 1,  embed_dim, padding_idx=0)

        self.layers = nn.ModuleList([
            RoPETransformerLayer(embed_dim, num_heads, ffn_dim, dropout, max_len)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, pitches, time_shifts=None, velocities=None, durations=None):
        """
        pitches     : (B, T) int  — MIDI pitch per note (required)
        time_shifts : (B, T) int  — quantized IOI bins  (optional, default 0)
        velocities  : (B, T) int  — quantized vel bins  (optional, default 0)
        durations   : (B, T) int  — quantized dur bins  (optional, default 0)
        Returns     : (B, T, embed_dim)
        """
        B, T = pitches.shape
        dev  = pitches.device

        zeros = torch.zeros(B, T, dtype=torch.long, device=dev)
        ts  = time_shifts if time_shifts is not None else zeros
        vel = velocities  if velocities  is not None else zeros
        dur = durations   if durations   is not None else zeros

        # No pos_embed — RoPE handles relative position in attention
        x = (self.pitch_embed(pitches)
             + self.ts_embed(ts)
             + self.vel_embed(vel)
             + self.dur_embed(dur))   # (B, T, D)

        # Bidirectional — no mask needed
        for layer in self.layers:
            x = layer(x)

        return self.norm(x)


class PositionEncoder(nn.Module):
    """
    Embeds known past fretboard positions into per-timestep vectors.
    Uses a causal (left-to-right) transformer so position t only sees
    assignments 0..t-1 — prevents leaking future string assignments.

    At position t:
        - input  : pos_{0:t-1}  (known past assignments)
        - output : context vector summarising the trajectory so far
    A special MASK token is used for positions not yet assigned.
    """
    def __init__(self, embed_dim=128, num_heads=8, num_layers=2,
                 ffn_dim=256, dropout=0.1, max_len=512):
        super().__init__()
        self.MASK_IDX    = NUM_POSITIONS          # token for unknown/masked position
        self.pos_embed_t = nn.Embedding(max_len, embed_dim)
        self.pos_embed_p = nn.Embedding(NUM_POSITIONS + 1, embed_dim)  # +1 for MASK
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(embed_dim)

    def _causal_mask(self, T, device):
        """Upper-triangular mask — position t cannot attend to t+1..T-1."""
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(self, positions):
        """
        positions: (B, T) int — past assignments, MASK_IDX where unknown.
        Returns  : (B, T, embed_dim)
        """
        B, T    = positions.shape
        pos_ids = torch.arange(T, device=positions.device).unsqueeze(0)
        x       = self.pos_embed_p(positions) + self.pos_embed_t(pos_ids)
        mask    = self._causal_mask(T, positions.device)
        return self.norm(self.transformer(x, mask=mask))


# ─── Key embedding ───────────────────────────────────────────────────────────

class KeyEmbedding(nn.Module):
    """
    Learnable embedding for musical key (0-11 = C..B, 12 = unknown).
    Projected to embed_dim and broadcast across the sequence as a bias
    added to every timestep of the pitch encoder output.
    This conditions position predictions on tonal context — a guitarist
    in E minor will prefer open-position E-shape voicings; in Bb they barre.
    """
    def __init__(self, embed_dim=64):
        super().__init__()
        self.emb  = nn.Embedding(NUM_KEYS, embed_dim)   # NUM_KEYS = 13
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, key):
        """
        key: (B,) int tensor — key index per sequence
        Returns: (B, 1, embed_dim) — broadcast across T when added to (B, T, D)
        """
        return self.proj(self.emb(key)).unsqueeze(1)   # (B, 1, D)


# ─── Full model ───────────────────────────────────────────────────────────────

class FretboardTransformer(nn.Module):
    """
    Masked-prediction fretboard model — architecture follows MIDI-to-Tab
    (Edwards et al., 2024) adapted for next-position inference + A* decoding.

    Training (masked LM):
        - Full pitch sequence encoded bidirectionally → pitch context per timestep
        - Past position assignments encoded causally → trajectory context
        - Random 20% of positions masked → model predicts masked positions
        - Loss computed only on masked positions (same as BERT MLM)

    Inference (left-to-right with A*):
        1. Encode full pitch sequence once  → pitch_ctx  (B, T, D)  [O(T²)]
        2. At each timestep t, feed known past positions → traj_ctx  (B, T, D)
        3. Combine pitch_ctx[t] + traj_ctx[t] → log P(pos_t)         [O(1)]
        4. A* uses -log P as edge weights on pitch-constrained trellis

    Parameters: ~600K (larger than previous ~250K due to two transformer stacks)
    """
    MASK_IDX = NUM_POSITIONS   # shared mask token index

    def __init__(self, embed_dim=128, num_heads=8,
                 pitch_layers=4, pos_layers=2,
                 ffn_dim=512, dropout=0.1):
        super().__init__()
        self.embed_dim    = embed_dim
        self.pitch_enc    = PitchEncoder(embed_dim, num_heads, pitch_layers,
                                         ffn_dim, dropout)
        self.pos_enc      = PositionEncoder(embed_dim, num_heads, pos_layers,
                                            ffn_dim // 2, dropout)
        # Fusion + output heads
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.head       = nn.Linear(embed_dim, NUM_POSITIONS)  # string/fret prediction
        self.dur_head   = nn.Linear(embed_dim, N_DUR_BINS + 1) # duration prediction (0=unknown)
        self.tonic_head = nn.Linear(embed_dim, 12)              # tonic prediction (C..B)
        self.pitch_head = nn.Linear(embed_dim, NUM_MIDI)        # next-pitch prediction

    def forward(self, pitches, positions, key=None,
                time_shifts=None, velocities=None, durations=None):
        """
        BART-style: bidirectional encoder over all pitches, causal decoder
        over past positions.

        pitches     : (B, T_p) — full pitch sequence (all notes)
        positions   : (B, T_q) — past position assignments (T_q = T_p - 1 for
                                  AR, T_q = T_p for masked LM)
        time_shifts : (B, T_p) — quantized inter-onset bins  (optional)
        velocities  : (B, T_p) — quantized velocity bins     (optional)
        durations   : (B, T_p) — quantized duration bins     (optional)
        Returns     : (B, T_q, NUM_POSITIONS) log-probabilities

        AR mode (T_q = T_p - 1):
            pitch_ctx[:, 1:] aligns with traj_ctx — predicting pos[t] uses
            bidirectional context of all pitches centered on pitch[t].
        Masked LM mode (T_q = T_p):
            pitch_ctx and traj_ctx are the same length, used directly.
        """
        pitch_ctx = self.pitch_enc(pitches,
                                   time_shifts=time_shifts,
                                   velocities=velocities,
                                   durations=durations)  # (B, T_p, D) bidirectional
        traj_ctx  = self.pos_enc(positions)              # (B, T_q, D) causal

        # Align: if encoder saw one extra pitch (AR mode), drop first timestep
        if pitch_ctx.shape[1] == traj_ctx.shape[1] + 1:
            pitch_ctx = pitch_ctx[:, 1:]                 # (B, T_q, D)

        fused = self.fusion(
            torch.cat([pitch_ctx, traj_ctx], dim=-1))    # (B, T_q, D)
        pos_logits   = F.log_softmax(self.head(fused),      dim=-1)  # (B, T_q, NUM_POSITIONS)
        dur_logits   = F.log_softmax(self.dur_head(fused),  dim=-1)  # (B, T_q, N_DUR_BINS+1)
        pit_logits   = F.log_softmax(self.pitch_head(fused),dim=-1)  # (B, T_q, NUM_MIDI)
        # Tonic: pool full pitch encoder output → single key prediction per sequence
        tonic_logits = F.log_softmax(
            self.tonic_head(pitch_ctx.mean(dim=1)), dim=-1)           # (B, 12)
        return pos_logits, dur_logits, pit_logits, tonic_logits

    @torch.no_grad()
    def encode_pitches(self, pitches, key=None,
                       time_shifts=None, velocities=None, durations=None):
        """
        Encode full pitch sequence once, with optional note features.
        pitches     : (T,) or (1, T) int tensor
        time_shifts : (T,) or (1, T) int tensor — quantized IOI bins
        velocities  : (T,) or (1, T) int tensor — quantized velocity bins
        durations   : (T,) or (1, T) int tensor — quantized duration bins
        Returns: (T, embed_dim) — reused for all A* edge lookups.
        """
        self.eval()
        def _expand(x):
            if x is None: return None
            return x.unsqueeze(0) if x.dim() == 1 else x
        if pitches.dim() == 1:
            pitches = pitches.unsqueeze(0)
        return self.pitch_enc(pitches,
                              time_shifts=_expand(time_shifts),
                              velocities =_expand(velocities),
                              durations  =_expand(durations)).squeeze(0)

    @torch.no_grad()
    def decode_step(self, pitch_ctx_t, past_positions, t):
        """
        Predict log P(pos_t | all pitches, pos_{0:t-1}).
        Returns: (pos_lp, dur_lp) tensors of shape (NUM_POSITIONS,) and (N_DUR_BINS+1,).
        """
        self.eval()
        dev = pitch_ctx_t.device
        pos_seq  = torch.tensor(
            past_positions + [self.MASK_IDX],
            dtype=torch.long, device=dev).unsqueeze(0)
        traj_ctx = self.pos_enc(pos_seq)
        traj_t   = traj_ctx[0, -1]
        fused    = self.fusion(
            torch.cat([pitch_ctx_t, traj_t], dim=-1).unsqueeze(0))
        pos_lp = F.log_softmax(self.head(fused),     dim=-1).squeeze(0)
        dur_lp = F.log_softmax(self.dur_head(fused), dim=-1).squeeze(0)
        return pos_lp, dur_lp

    @torch.no_grad()
    def decode_step_cached(self, pitch_ctx_t, past_positions):
        """
        Like decode_step but caches the trajectory encoding keyed on the
        past_positions tuple. Safe within a single decode() call since
        the same past_positions always maps to the same trajectory context.
        Returns: (NUM_POSITIONS,) pos log-probs.
        """
        self.eval()
        dev   = pitch_ctx_t.device
        key   = tuple(past_positions)
        if not hasattr(self, '_traj_cache') or self._traj_cache_dev != dev:
            self._traj_cache     = {}
            self._traj_cache_dev = dev

        if key not in self._traj_cache:
            pos_seq = torch.tensor(
                list(past_positions) + [self.MASK_IDX],
                dtype=torch.long, device=dev).unsqueeze(0)
            traj_ctx = self.pos_enc(pos_seq)
            self._traj_cache[key] = traj_ctx[0, -1]  # (D,)

        traj_t = self._traj_cache[key]
        fused  = self.fusion(
            torch.cat([pitch_ctx_t, traj_t], dim=-1).unsqueeze(0))
        return F.log_softmax(self.head(fused), dim=-1).squeeze(0)

    def clear_decode_cache(self):
        """Call before each new decode() to reset the trajectory cache."""
        self._traj_cache = {}
        self._traj_cache_dev = None
    
    @torch.no_grad()
    def decode_pitch_step(self, pitch_ctx_t, past_positions, t):
        """
        Predict log P(next pitch | all pitches 0..t, pos_{0:t}).
        Returns: (NUM_MIDI,) log-probabilities.
        """
        self.eval()
        dev = pitch_ctx_t.device
        pos_seq  = torch.tensor(
            past_positions + [self.MASK_IDX],
            dtype=torch.long, device=dev).unsqueeze(0)
        traj_ctx = self.pos_enc(pos_seq)
        traj_t   = traj_ctx[0, -1]  # MASK is always last
        fused = self.fusion(
            torch.cat([pitch_ctx_t, traj_t], dim=-1).unsqueeze(0))
        return F.log_softmax(self.pitch_head(fused), dim=-1).squeeze(0)

    @torch.no_grad()
    def encode_context(self, pitches):
        """Precompute global context vector. One call per sequence at inference."""
        if pitches.dim() == 1:
            pitches = pitches.unsqueeze(0)
        return self.seq_encoder(pitches).squeeze(0)

    @torch.no_grad()
    def edge_logprobs(self, pos_idx, pitch, ctx):
        """
        O(1) — uses precomputed context, no recomputation per A* path.
        Returns (126,) log-probabilities over next positions.
        """
        s, f    = POSITIONS[pos_idx]
        dev     = ctx.device
        pos_emb = self.pos_encoder(
            torch.tensor([s],     dtype=torch.long, device=dev),
            torch.tensor([f],     dtype=torch.long, device=dev),
            torch.tensor([pitch], dtype=torch.long, device=dev),
        )
        return self.decoder(pos_emb, ctx.unsqueeze(0)).squeeze(0)


# ─── Dataset ──────────────────────────────────────────────────────────────────

# ─── Key estimation ──────────────────────────────────────────────────────────

# Krumhansl-Schmuckler key profiles (major and minor)
_KS_MAJOR = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
_KS_MINOR = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]



# ─── Bidirectional Position Encoder (for MLM / masked diffusion) ─────────────

class BidirectionalPositionEncoder(nn.Module):
    """
    Like PositionEncoder but WITHOUT a causal mask — attends to all positions
    in both directions. Designed for masked-LM training where the model sees
    the full position sequence with some tokens masked, and must predict each
    masked token conditioned on ALL surrounding context (past and future).

    This is the correct encoder for iterative unmasking / masked diffusion:
    when token at position t is being predicted, it can attend to already-
    revealed tokens at positions t+1, t+2, ... which causal masking forbids.

    MASK token (NUM_POSITIONS) is used for positions not yet assigned.
    The model learns to predict MASK tokens from their fully bidirectional
    neighbourhood context.
    """
    def __init__(self, embed_dim=128, num_heads=8, num_layers=2,
                 ffn_dim=256, dropout=0.1, max_len=512):
        super().__init__()
        self.MASK_IDX    = NUM_POSITIONS
        self.pos_embed_t = nn.Embedding(max_len,          embed_dim)
        self.pos_embed_p = nn.Embedding(NUM_POSITIONS + 1, embed_dim)  # +1 for MASK
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(embed_dim)

    def forward(self, positions):
        """
        positions : (B, T) int  — full position sequence, MASK_IDX where unknown.
        Returns   : (B, T, embed_dim)  — bidirectional context for every token.
        """
        B, T    = positions.shape
        pos_ids = torch.arange(T, device=positions.device).unsqueeze(0)
        x       = self.pos_embed_p(positions) + self.pos_embed_t(pos_ids)
        # No mask → full bidirectional attention
        return self.norm(self.transformer(x))


# ─── MLM model ───────────────────────────────────────────────────────────────

class FretboardTransformerMLM(nn.Module):
    """
    Masked-LM fretboard model for iterative unmasking (masked diffusion).

    Identical to FretboardTransformer except the PositionEncoder is replaced
    with BidirectionalPositionEncoder — no causal mask, attends to all
    already-revealed positions in both directions.

    Training:
        Random ~40% of position tokens are replaced with MASK.
        Model predicts all masked positions simultaneously.
        Loss computed only on masked tokens (BERT-style MLM).

    Inference (masked diffusion):
        1. Start with all positions = MASK
        2. Encode all pitches bidirectionally (same as AR model)
        3. Score all MASK tokens simultaneously — model sees full pitch context
           AND all already-revealed position neighbours
        4. Reveal N most-confident tokens per step
        5. Repeat until fully decoded

    This gives true masked diffusion semantics: each step refines positions
    conditioned on globally-coherent bidirectional context.
    """
    MASK_IDX       = NUM_POSITIONS   # position mask token
    PITCH_MASK_IDX = NUM_MIDI        # pitch mask token (index 128)

    def __init__(self, embed_dim=128, num_heads=8,
                 pitch_layers=4, pos_layers=2,
                 ffn_dim=512, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.pitch_enc = PitchEncoder(embed_dim, num_heads, pitch_layers,
                                      ffn_dim, dropout)
        self.pos_enc   = BidirectionalPositionEncoder(
            embed_dim, num_heads, pos_layers, ffn_dim // 2, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.head       = nn.Linear(embed_dim, NUM_POSITIONS)
        self.dur_head   = nn.Linear(embed_dim, N_DUR_BINS + 1)
        self.vel_head   = nn.Linear(embed_dim, N_VEL_BINS + 1)  # velocity prediction (0=unknown)
        self.tonic_head = nn.Linear(embed_dim, 12)
        self.pitch_head = nn.Linear(embed_dim, NUM_MIDI)

    def forward(self, pitches, positions,
                time_shifts=None, velocities=None, durations=None):
        """
        pitches   : (B, T) — pitch sequence; PITCH_MASK_IDX where masked
        positions : (B, T) — positions; MASK_IDX where masked

        Both can contain mask tokens simultaneously for joint MLM training.
        Returns: (pos_logits, dur_logits, pit_logits, tonic_logits, vel_logits)
        """
        pitch_ctx = self.pitch_enc(pitches,
                                   time_shifts=time_shifts,
                                   velocities=velocities,
                                   durations=durations)   # (B, T, D)
        traj_ctx  = self.pos_enc(positions)               # (B, T, D) — bidirectional
        fused     = self.fusion(
            torch.cat([pitch_ctx, traj_ctx], dim=-1))     # (B, T, D)
        pos_logits   = F.log_softmax(self.head(fused),       dim=-1)
        dur_logits   = F.log_softmax(self.dur_head(fused),   dim=-1)
        vel_logits   = F.log_softmax(self.vel_head(fused),   dim=-1)
        pit_logits   = F.log_softmax(self.pitch_head(fused), dim=-1)
        tonic_logits = F.log_softmax(
            self.tonic_head(pitch_ctx.mean(dim=1)), dim=-1)
        return pos_logits, dur_logits, pit_logits, tonic_logits, vel_logits

    @torch.no_grad()
    def encode_pitches(self, pitches, key=None,
                       time_shifts=None, velocities=None, durations=None):
        """Encode pitch sequence (may contain PITCH_MASK_IDX)."""
        self.eval()
        if pitches.dim() == 1:
            pitches = pitches.unsqueeze(0)
        ctx = self.pitch_enc(pitches, time_shifts=time_shifts,
                             velocities=velocities, durations=durations)
        return ctx.squeeze(0)  # (T, D)

    @torch.no_grad()
    def decode_joint_step(self, all_pitches, all_positions, target_t,
                          all_durations=None):
        """
        Predict P(pitch_t), P(pos_t), and P(dur_t) for a jointly-masked token.

        all_pitches   : list[int] length T — PITCH_MASK_IDX at masked
        all_positions : list[int] length T — MASK_IDX at masked
        all_durations : list[int] length T — 0 at masked (optional)
        target_t      : int — token to predict

        Returns: (pit_lp, pos_lp, dur_lp)
        """
        self.eval()
        dev   = next(self.parameters()).device
        T     = len(all_pitches)
        pit_t = torch.tensor(all_pitches,   dtype=torch.long, device=dev).unsqueeze(0)
        pos_t = torch.tensor(all_positions, dtype=torch.long, device=dev).unsqueeze(0)
        dur_t = torch.tensor(
            all_durations if all_durations is not None else [0]*T,
            dtype=torch.long, device=dev).unsqueeze(0)

        pitch_ctx = self.pitch_enc(pit_t, durations=dur_t)  # (1, T, D)
        traj_ctx  = self.pos_enc(pos_t)                     # (1, T, D)
        fused     = self.fusion(
            torch.cat([pitch_ctx, traj_ctx], dim=-1))       # (1, T, D)

        t_fused = fused[0, target_t]
        pit_lp  = F.log_softmax(self.pitch_head(t_fused), dim=-1)
        pos_lp  = F.log_softmax(self.head(t_fused),       dim=-1)
        dur_lp  = F.log_softmax(self.dur_head(t_fused),   dim=-1)
        return pit_lp, pos_lp, dur_lp

    @torch.no_grad()
    def decode_step_bidirectional(self, pitch_ctx, all_positions, target_t):
        """Position-only prediction given precomputed pitch context."""
        self.eval()
        dev   = pitch_ctx.device
        pos_t = torch.tensor(all_positions, dtype=torch.long, device=dev).unsqueeze(0)
        traj  = self.pos_enc(pos_t)
        fused = self.fusion(
            torch.cat([pitch_ctx.unsqueeze(0), traj], dim=-1))
        return F.log_softmax(self.head(fused[0, target_t]), dim=-1)
        pass  # no cache needed — bidirectional encoder is stateless per call

    @torch.no_grad()
    def decode_pitch_step(self, pitch_ctx_t, past_positions, t):
        """
        Predict log P(next pitch | all pitches 0..t, revealed positions).
        Mirrors FretboardTransformer.decode_pitch_step API.
        Returns: (NUM_MIDI,) log-probabilities.
        """
        self.eval()
        dev     = pitch_ctx_t.device
        T       = t + 1
        pos_seq = torch.tensor(
            list(past_positions) + [self.MASK_IDX],
            dtype=torch.long, device=dev).unsqueeze(0)   # (1, T)
        traj    = self.pos_enc(pos_seq)                  # (1, T, D) bidirectional
        fused   = self.fusion(
            torch.cat([pitch_ctx_t.unsqueeze(0).unsqueeze(0),
                       traj[:, -1:, :]], dim=-1))        # (1, 1, D)
        return F.log_softmax(
            self.pitch_head(fused.squeeze(1)), dim=-1).squeeze(0)  # (NUM_MIDI,)
