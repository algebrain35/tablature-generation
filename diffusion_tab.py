"""
diffusion_tab.py — Discrete diffusion model for guitar tablature position sequences.

Architecture:
    D3PM (Austin et al., 2021) with absorbing-state forward process.
    The MASK token from the existing model is the natural absorbing state.

    Condition encoder : PitchEncoder (frozen, from fretboard_transformer.py)
    Denoiser          : DenoisingTransformer — bidirectional transformer with
                        cross-attention to pitch context + sinusoidal diffusion
                        timestep embedding.  Predicts clean position x_0 at
                        every token from noisy input x_t.

Forward process:
    At diffusion timestep t ∈ [0, T_diff]:
        Each position token independently transitions to MASK with probability
        β_t (absorbing).  The cumulative corruption schedule α̅_t gives the
        probability that a token is still clean at step t:
            α̅_t = ∏_{s=1}^{t} (1 - β_s)
    At t=0 the sequence is fully clean; at t=T_diff it is nearly all MASK.

Loss:
    Cross-entropy between predicted x_0 logits and true clean positions,
    computed only on masked (corrupted) tokens.  This is the variational
    lower bound for the absorbing-state D3PM.

Inference:
    Start from x_{T_diff} = all MASK.  At each reverse step t → t-1:
        1. Predict p(x_0 | x_t, pitches) via the denoiser
        2. For still-masked tokens, sample x_0 ~ p(x_0 | x_t) and unmask
           with probability (1 - α̅_{t-1}) / (1 - α̅_t)  [reverse rate]
        3. Optionally apply pitch constraints: zero out positions that don't
           match the target pitch at each timestep
    After step 0, all tokens are unmasked → final position sequence.

Usage:
    # Train
    python diffusion_tab.py --train --dadagp_dir ./data/dadagp/ --epochs 30

    # Generate (with pitch constraint)
    python diffusion_tab.py --generate --pitches 64,66,68,69,71,73,75,76

    # Generate diverse samples (multiple valid tablatures for same melody)
    python diffusion_tab.py --generate --pitches 64,66,68,69 --n_samples 5
"""

import math, os, sys, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fretboard_transformer import (
    FretboardTransformer, PitchEncoder, RoPETransformerLayer,
    StreamingDataset, transpose_window, estimate_key,
    POSITIONS, POS_TO_IDX, pos_to_midi, midi_to_positions,
    STANDARD_TUNING, NUM_POSITIONS, NUM_MIDI,
    N_TS_BINS, N_VEL_BINS, N_DUR_BINS,
)

# ─── Constants ────────────────────────────────────────────────────────────────

MASK_IDX   = NUM_POSITIONS     # 126 — absorbing state
NUM_TOKENS = NUM_POSITIONS + 1 # 127 — vocabulary includes MASK


# ─── Noise schedule ──────────────────────────────────────────────────────────

def linear_alpha_bar(t, T):
    """Linear schedule — gentler than cosine, less extreme masking at high t."""
    return torch.clamp(1.0 - t.float() / T, min=1e-5, max=1.0)


def cosine_alpha_bar(t, T, s=0.008):
    """
    Cosine noise schedule (Nichol & Dhariwal, 2021).
    Returns α̅_t = probability that a token is still clean at step t.
    """
    f_t = torch.cos(((t / T) + s) / (1 + s) * (math.pi / 2)) ** 2
    f_0 = math.cos((s / (1 + s)) * (math.pi / 2)) ** 2
    return torch.clamp(f_t / f_0, min=1e-5, max=1.0)


def build_schedule(T_diff, schedule='cosine'):
    """
    Precompute schedule tensors for T_diff diffusion steps.
    schedule: 'cosine' (default) | 'linear' (gentler, fewer near-total-mask steps)
    """
    ts = torch.arange(T_diff + 1, dtype=torch.float)
    if schedule == 'linear':
        ab = linear_alpha_bar(ts, T_diff)
    else:
        ab = cosine_alpha_bar(ts, T_diff)
    beta     = torch.zeros(T_diff + 1)
    beta[1:] = 1.0 - ab[1:] / ab[:-1]
    beta     = torch.clamp(beta, min=0.0, max=0.999)
    return {'alpha_bar': ab, 'beta': beta}


# ─── Sinusoidal timestep embedding ──────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    """
    Sinusoidal embedding for diffusion timestep, projected to embed_dim.
    Same formulation as in the original DDPM (Ho et al., 2020).
    """
    def __init__(self, embed_dim, max_period=10000):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.embed_dim  = embed_dim
        self.max_period = max_period

    def forward(self, t):
        """
        t : (B,) int or float — diffusion timestep
        Returns : (B, embed_dim)
        """
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) *
            torch.arange(half, dtype=torch.float, device=t.device) / half
        )
        args  = t.float().unsqueeze(-1) * freqs.unsqueeze(0)   # (B, half)
        emb   = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, D)
        if self.embed_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)


# ─── Cross-attention layer ───────────────────────────────────────────────────

class CrossAttention(nn.Module):
    """
    Multi-head cross-attention: query from denoiser, key/value from pitch context.
    No positional encoding needed — position is already baked into both streams
    via RoPE (pitch encoder) and the denoiser's own position embeddings.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out    = nn.Linear(embed_dim, embed_dim, bias=False)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x, context):
        """
        x       : (B, T, D)  — denoiser hidden states (query)
        context : (B, T, D)  — pitch encoder output   (key, value)
        """
        B, T, D = x.shape
        H, Dh   = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, Dh).transpose(1, 2)        # (B,H,T,Dh)
        k = self.k_proj(context).reshape(B, -1, H, Dh).transpose(1, 2)  # (B,H,Tc,Dh)
        v = self.v_proj(context).reshape(B, -1, H, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn   = self.drop(torch.softmax(scores, dim=-1))
        out    = torch.matmul(attn, v)                                   # (B,H,T,Dh)
        return self.out(out.transpose(1, 2).reshape(B, T, D))


# ─── Denoiser block ─────────────────────────────────────────────────────────

class DenoiserBlock(nn.Module):
    """
    Single denoiser transformer block:
        1. Self-attention (bidirectional, with RoPE) over noisy positions
        2. Cross-attention to pitch context
        3. Feed-forward
    All with pre-norm residual connections.
    """
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1, max_len=512):
        super().__init__()
        self.self_attn  = RoPETransformerLayer(embed_dim, num_heads, ffn_dim,
                                                dropout, max_len)
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = CrossAttention(embed_dim, num_heads, dropout)
        self.cross_ff   = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context):
        """
        x       : (B, T, D)  — noisy position embeddings
        context : (B, T, D)  — pitch encoder output
        """
        x = self.self_attn(x)
        x = x + self.cross_attn(self.cross_norm(x), context)
        x = x + self.cross_ff(x)
        return x


# ─── Full denoising transformer ─────────────────────────────────────────────

class DenoisingTransformer(nn.Module):
    """
    Predicts clean position tokens x_0 from noisy input x_t, conditioned on:
        - Pitch context (from frozen PitchEncoder)
        - Diffusion timestep t

    Input:  noisy position sequence (B, T) with values in [0, NUM_POSITIONS]
            where NUM_POSITIONS = MASK token
    Output: (B, T, NUM_POSITIONS) logits over clean position classes
    """
    def __init__(self, embed_dim=128, num_heads=8, num_layers=6,
                 ffn_dim=512, dropout=0.1, max_len=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.tok_embed  = nn.Embedding(NUM_TOKENS, embed_dim)
        self.time_embed = TimestepEmbedding(embed_dim)
        self.layers = nn.ModuleList([
            DenoiserBlock(embed_dim, num_heads, ffn_dim, dropout, max_len)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, NUM_POSITIONS)

    def forward(self, x_t, t, pitch_ctx):
        """
        x_t       : (B, T) int — noisy position tokens (may contain MASK_IDX)
        t         : (B,)   int — diffusion timestep per sequence
        pitch_ctx : (B, T, D)  — precomputed pitch encoder output
        Returns   : (B, T, NUM_POSITIONS) logits over clean position classes
        """
        h = self.tok_embed(x_t) + self.time_embed(t).unsqueeze(1)
        for layer in self.layers:
            h = layer(h, pitch_ctx)
        return self.head(self.norm(h))


# ─── Full diffusion model ───────────────────────────────────────────────────

class DiffusionTabModel(nn.Module):
    """
    Complete discrete diffusion model for guitar tablature.

    Components:
        pitch_encoder : PitchEncoder (frozen) — encodes the melody
        denoiser      : DenoisingTransformer  — predicts clean positions

    The pitch encoder weights are loaded from a pretrained FretboardTransformer
    checkpoint, giving the diffusion model a strong starting condition encoder.
    """
    def __init__(self, embed_dim=128, num_heads=8,
                 pitch_layers=4, denoiser_layers=6,
                 ffn_dim=512, dropout=0.1, T_diff=100,
                 freeze_pitch_encoder=True, schedule='cosine'):
        super().__init__()
        self.T_diff = T_diff
        self.pitch_encoder = PitchEncoder(
            embed_dim, num_heads, pitch_layers, ffn_dim, dropout)
        if freeze_pitch_encoder:
            for p in self.pitch_encoder.parameters():
                p.requires_grad = False
        self.denoiser = DenoisingTransformer(
            embed_dim, num_heads, denoiser_layers, ffn_dim, dropout)
        sched = build_schedule(T_diff, schedule)
        self.register_buffer('alpha_bar', sched['alpha_bar'])
        self.register_buffer('beta',      sched['beta'])

    def load_pitch_encoder(self, checkpoint_path, device='cpu'):
        """Load pitch encoder weights from a pretrained FretboardTransformer."""
        state  = torch.load(checkpoint_path, map_location=device)
        prefix = 'pitch_enc.'
        pe_state = {k[len(prefix):]: v for k, v in state.items()
                    if k.startswith(prefix)}
        if not pe_state:
            print("[WARN] No pitch_enc keys found — pitch encoder trains from scratch.")
            return
        self.pitch_encoder.load_state_dict(pe_state, strict=False)
        print(f"Loaded pitch encoder: {len(pe_state)} tensors from {checkpoint_path}")

    def corrupt(self, x_0, t):
        """
        Forward (corruption) process: absorbing-state diffusion.
        x_0 : (B, T) int — clean position tokens
        t   : (B,)   int — diffusion timestep per sample
        Returns x_t : (B, T) — noisy tokens (some replaced with MASK_IDX)
        """
        ab       = self.alpha_bar[t].unsqueeze(-1)            # (B, 1)
        keep     = torch.rand_like(x_0.float()) < ab          # (B, T)
        x_t      = x_0.clone()
        x_t[~keep] = MASK_IDX
        return x_t

    def forward(self, x_0, pitches, time_shifts=None,
                velocities=None, durations=None):
        """
        Training forward pass with importance-sampled timesteps.

        Importance sampling: sample t proportional to (1 - α̅_t) so the model
        sees more partially-masked examples and fewer near-total-mask examples.
        This prevents the denoiser from spending most of its capacity on
        hopeless (nearly all MASK) inputs and collapsing to a local minimum.

        Returns: loss (scalar), accuracy (float)
        """
        B, T = x_0.shape
        dev  = x_0.device

        # Importance-sample t ∝ (1 - α̅_t) — weight toward higher noise levels
        # but avoid the hardest (near-total-mask) steps dominating
        weights = 1.0 - self.alpha_bar[1:]          # (T_diff,) — higher = more masked
        weights = weights / weights.sum()
        t_idx   = torch.multinomial(weights.expand(B, -1), 1).squeeze(-1) + 1  # (B,) in [1, T_diff]

        x_t = self.corrupt(x_0, t_idx)

        ctx = torch.no_grad() if not any(
            p.requires_grad for p in self.pitch_encoder.parameters()
        ) else torch.enable_grad()
        with ctx:
            pitch_ctx = self.pitch_encoder(
                pitches, time_shifts=time_shifts,
                velocities=velocities, durations=durations)

        logits    = self.denoiser(x_t, t_idx, pitch_ctx)
        corrupted = (x_t == MASK_IDX)
        if not corrupted.any():
            return torch.tensor(0.0, device=dev, requires_grad=True), 0.0

        # Label smoothing (0.1) prevents overconfident predictions which
        # collapse to the most frequent position token
        loss = F.cross_entropy(
            logits[corrupted], x_0[corrupted], label_smoothing=0.1)
        preds = logits[corrupted].argmax(-1)
        acc   = (preds == x_0[corrupted]).float().mean().item()
        return loss, acc


# ─── Inference ───────────────────────────────────────────────────────────────

class DiffusionDecoder:
    """
    Reverse-process sampler for discrete absorbing-state diffusion.

    Supports:
        - Pitch-constrained sampling (only allow positions matching each pitch)
        - Temperature control
        - Multiple diverse samples for the same input melody
    """
    def __init__(self, model, tuning=STANDARD_TUNING, device=None):
        self.model  = model
        self.tuning = tuning
        self.device = device or next(model.parameters()).device
        self.model.eval()

    def _pitch_constraint_mask(self, pitch_sequence):
        """(T, NUM_POSITIONS) bool — True where position matches pitch."""
        T    = len(pitch_sequence)
        mask = torch.zeros(T, NUM_POSITIONS, dtype=torch.bool)
        for t, midi in enumerate(pitch_sequence):
            for s, f in midi_to_positions(midi, self.tuning):
                if (s, f) in POS_TO_IDX:
                    mask[t, POS_TO_IDX[(s, f)]] = True
        return mask

    @torch.no_grad()
    def decode(self, pitch_sequence, temperature=0.8, constrain_pitches=True,
               time_shifts=None, velocities=None, durations=None):
        """
        Generate one position sequence by reverse diffusion.
        Returns: list of dicts with 'string', 'fret', 'midi', 'is_open'
        """
        T      = len(pitch_sequence)
        T_diff = self.model.T_diff
        dev    = self.device

        pitches   = torch.tensor(pitch_sequence, dtype=torch.long, device=dev).unsqueeze(0)

        def _prep(x):
            return None if x is None else \
                torch.tensor(x, dtype=torch.long, device=dev).unsqueeze(0)

        pitch_ctx  = self.model.pitch_encoder(
            pitches, time_shifts=_prep(time_shifts),
            velocities=_prep(velocities), durations=_prep(durations))

        valid_mask = self._pitch_constraint_mask(pitch_sequence).to(dev) \
                     if constrain_pitches else None

        x_t       = torch.full((1, T), MASK_IDX, dtype=torch.long, device=dev)
        alpha_bar = self.model.alpha_bar

        for step in range(T_diff, 0, -1):
            t_batch = torch.tensor([step], dtype=torch.long, device=dev)
            logits  = self.model.denoiser(x_t, t_batch, pitch_ctx).squeeze(0)
            logits  = logits / max(temperature, 1e-6)
            if valid_mask is not None:
                logits = logits.masked_fill(~valid_mask, float('-inf'))

            probs   = torch.softmax(logits, dim=-1)
            x0_pred = torch.multinomial(probs, 1).squeeze(-1)

            # Reverse unmasking rate: (α̅_{t-1} - α̅_t) / (1 - α̅_t)
            ab_t         = alpha_bar[step]
            ab_tm1       = alpha_bar[step - 1]
            unmask_prob  = torch.clamp((ab_tm1 - ab_t) / (1.0 - ab_t + 1e-8), 0.0, 1.0)

            is_masked = (x_t.squeeze(0) == MASK_IDX)
            do_unmask = is_masked & (torch.rand(T, device=dev) < unmask_prob)
            x_new     = x_t.squeeze(0).clone()
            x_new[do_unmask] = x0_pred[do_unmask]
            x_t = x_new.unsqueeze(0)

        # Final cleanup: argmax any remaining MASK tokens
        still_masked = (x_t.squeeze(0) == MASK_IDX)
        if still_masked.any():
            logits = self.model.denoiser(
                x_t, torch.tensor([1], device=dev), pitch_ctx).squeeze(0)
            if valid_mask is not None:
                logits = logits.masked_fill(~valid_mask, float('-inf'))
            x_t[0, still_masked] = logits[still_masked].argmax(-1)

        positions = x_t.squeeze(0).cpu().tolist()
        result    = []
        for t, pos_idx in enumerate(positions):
            pos_idx = min(pos_idx, NUM_POSITIONS - 1)
            s, f    = POSITIONS[pos_idx]
            result.append({'string': s, 'fret': f,
                           'midi': pos_to_midi(s, f, self.tuning),
                           'is_open': f == 0})
        return result

    @torch.no_grad()
    def decode_n(self, pitch_sequence, n_samples=5, temperature=0.8,
                 constrain_pitches=True, **kwargs):
        """
        Generate n_samples diverse tablatures for the same pitch sequence.
        Core advantage over A*: multiple valid arrangements, not just one.
        """
        return [self.decode(pitch_sequence, temperature=temperature,
                            constrain_pitches=constrain_pitches, **kwargs)
                for _ in range(n_samples)]


# ─── Trainer ─────────────────────────────────────────────────────────────────

class DiffusionTrainer:
    """Training loop for DiffusionTabModel. Reuses StreamingDataset."""

    def __init__(self, model, dadagp_dir=None, guitarset_dir=None,
                 scoreset_dir=None, synthtab_dir=None,
                 scraped_tabs_dir=None, proggp_dir=None,
                 window=64, batch_size=32, lr=3e-4,
                 genres=None, max_files=None,
                 steps_per_epoch=500, num_workers=4,
                 augment_semitones=(-3,-2,-1,1,2,3),
                 val_split=0.1, max_source_fraction=1.0,
                 pretrained_ckpt=None,
                 cache_path="dataset_cache.pkl"):

        self.device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size      = batch_size
        self.steps_per_epoch = steps_per_epoch
        print(f"Device: {self.device}")

        if pretrained_ckpt and os.path.exists(pretrained_ckpt):
            model.load_pitch_encoder(pretrained_ckpt, device='cpu')

        self.ds = StreamingDataset(
            dadagp_dir=dadagp_dir, guitarset_dir=guitarset_dir,
            scoreset_dir=scoreset_dir, synthtab_dir=synthtab_dir,
            scraped_tabs_dir=scraped_tabs_dir, proggp_dir=proggp_dir,
            window=window, genres=genres, max_files=max_files,
            num_workers=num_workers,
            augment_semitones=augment_semitones,
            val_split=val_split,
            max_source_fraction=max_source_fraction,
            cache_path=cache_path,
        )

        self.model = model.to(self.device)
        trainable  = [p for p in model.parameters() if p.requires_grad]
        self.optim = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-3)
        self.best  = float('inf')
        print(f"Trainable params : {sum(p.numel() for p in trainable):,}")
        print(f"Dataset          : {len(self.ds)} train sequences")

    def train(self, epochs=30, save_path="diffusion_tab.pt", warmup_epochs=3):
        total_steps  = epochs * self.steps_per_epoch
        warmup_steps = warmup_epochs * self.steps_per_epoch
        base_lr      = self.optim.param_groups[0]['lr']
        eta_min      = 1e-6

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine   = 0.5 * (1 + math.cos(math.pi * progress))
            return (eta_min / base_lr) + (1 - eta_min / base_lr) * cosine

        sched       = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)
        global_step = 0

        for ep in range(1, epochs + 1):
            self.model.train()
            for p in self.model.pitch_encoder.parameters():
                p.requires_grad = False

            tot_loss = tot_acc = 0

            for step in range(self.steps_per_epoch):
                keys, positions, pitches, tshifts, vels, durs, midi_mask = \
                    self.ds.get_batch(self.batch_size)

                has_pos   = ~midi_mask
                if not has_pos.any():
                    continue

                positions = positions[has_pos].to(self.device)
                pitches   = pitches[has_pos].to(self.device)
                tshifts   = tshifts[has_pos].to(self.device)
                vels      = vels[has_pos].to(self.device)
                durs      = durs[has_pos].to(self.device)

                self.optim.zero_grad()
                loss, acc = self.model(
                    positions, pitches,
                    time_shifts=tshifts, velocities=vels, durations=durs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()
                sched.step()
                global_step += 1
                tot_loss += loss.item()
                tot_acc  += acc

            avg_loss = tot_loss / self.steps_per_epoch
            avg_acc  = tot_acc  / self.steps_per_epoch

            # Validation
            self.model.eval()
            val_loss = val_acc = 0
            val_steps = max(1, self.steps_per_epoch // 5)
            with torch.no_grad():
                for _ in range(val_steps):
                    vkeys, vpos, vpit, vts, vvel, vdur, vmidi = \
                        self.ds.get_val_batch(self.batch_size)
                    vhas = ~vmidi
                    if not vhas.any():
                        continue
                    vpos = vpos[vhas].to(self.device)
                    vpit = vpit[vhas].to(self.device)
                    vts  = vts[vhas].to(self.device)
                    vvel = vvel[vhas].to(self.device)
                    vdur = vdur[vhas].to(self.device)
                    vl, va = self.model(
                        vpos, vpit,
                        time_shifts=vts, velocities=vvel, durations=vdur)
                    val_loss += vl.item()
                    val_acc  += va
            val_loss /= val_steps
            val_acc  /= val_steps

            print(f"Ep {ep:3d}/{epochs}  "
                  f"loss={avg_loss:.4f} acc={avg_acc:.3f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")

            if val_loss < self.best:
                self.best = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"  ✓ Saved (val_loss={val_loss:.4f})")

        self.ds.stop()
        print(f"Training complete. Best val loss: {self.best:.4f}")

    def resume(self, epochs=20, save_path="diffusion_tab.pt", lr=3e-5):
        if os.path.exists(save_path):
            self.model.load_state_dict(
                torch.load(save_path, map_location=self.device))
            print(f"Resumed from {save_path}")
        trainable  = [p for p in self.model.parameters() if p.requires_grad]
        self.optim = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-5)
        self.best  = float('inf')
        self.train(epochs=epochs, save_path=save_path)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def render_ascii(decoded, notes_per_line=16):
    STRING_LABELS = {1:"e", 2:"B", 3:"G", 4:"D", 5:"A", 6:"E"}
    lines = []
    for start in range(0, len(decoded), notes_per_line):
        chunk = decoded[start:start+notes_per_line]
        rows  = {s: STRING_LABELS[s]+"|" for s in range(1,7)}
        for note in chunk:
            s, fret = note["string"], note["fret"]
            col = str(fret).ljust(3)
            for string in range(1,7):
                rows[string] += col if string==s else "-"*len(col)
        for s in rows: rows[s] += "|"
        lines.append("")
        for s in range(1,7): lines.append(rows[s])
    return "\n".join(lines)


NOTE_NAMES = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]

TUNINGS = {
    "standard": {1:64, 2:59, 3:55, 4:50, 5:45, 6:40},
    "eb":       {1:63, 2:58, 3:54, 4:49, 5:44, 6:39},
    "dropd":    {1:64, 2:59, 3:55, 4:50, 5:45, 6:38},
    "dropc":    {1:62, 2:57, 3:53, 4:48, 5:43, 6:36},
}


def parse_args():
    p = argparse.ArgumentParser(description="Discrete diffusion tablature model (D3PM)")
    p.add_argument("--train",    action="store_true")
    p.add_argument("--generate", action="store_true")
    # Data
    p.add_argument("--dadagp_dir",       default=None)
    p.add_argument("--guitarset_dir",    default=None)
    p.add_argument("--scoreset_dir",     default=None)
    p.add_argument("--synthtab_dir",     default=None)
    p.add_argument("--scraped_tabs_dir", default=None)
    p.add_argument("--proggp_dir",       default=None)
    p.add_argument("--genres",           nargs='+', default=None)
    # Architecture
    p.add_argument("--embed_dim",        type=int,   default=128)
    p.add_argument("--denoiser_layers",  type=int,   default=6)
    p.add_argument("--T_diff",           type=int,   default=100,
                   help="Diffusion steps (default 100; cosine schedule)")
    p.add_argument("--schedule",         default="cosine",
                   choices=["cosine","linear"],
                   help="Noise schedule (linear is gentler, fewer hopeless steps)")
    # Training
    p.add_argument("--epochs",           type=int,   default=30)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--window",           type=int,   default=64)
    p.add_argument("--steps_per_epoch",  type=int,   default=500)
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--save_path",        default="diffusion_tab.pt")
    p.add_argument("--cache_path",       default="dataset_cache.pkl",
                   help="Dataset cache path — reuse existing v8 cache to skip rebuild")
    p.add_argument("--pretrained",       default="fretboard_transformer.pt",
                   help="Pretrained AR checkpoint for pitch encoder init")
    # Generation
    p.add_argument("--pitches",          default=None,
                   help="Comma-separated MIDI pitches, e.g. 64,66,68,69")
    p.add_argument("--temperature",      type=float, default=0.8)
    p.add_argument("--n_samples",        type=int,   default=1)
    p.add_argument("--tuning",           default="standard",
                   choices=["standard","eb","dropd","dropc"])
    p.add_argument("--notes_per_line",   type=int,   default=16)
    p.add_argument("--seed",             type=int,   default=None)
    # Audio
    p.add_argument("--wav",              default=None,
                   help="Output WAV file (e.g. out.wav). With --n_samples > 1, "
                        "saves sample_1.wav, sample_2.wav etc.")
    p.add_argument("--bpm",              type=float, default=120.0)
    p.add_argument("--note_dur",         type=float, default=0.5,
                   help="Default note duration in beats (default 0.5)")
    return p.parse_args()


# ─── Audio synthesis ─────────────────────────────────────────────────────────

def karplus_strong(freq, duration, sr=44100, decay=0.996, brightness=0.5):
    import numpy as np
    n       = int(sr * duration)
    buf_len = max(2, int(sr / freq))
    buf     = np.random.uniform(-1, 1, buf_len).astype(np.float64)
    out     = np.zeros(n, dtype=np.float64)
    for i in range(n):
        out[i]         = buf[i % buf_len]
        nxt            = decay * ((1-brightness)*buf[i % buf_len]
                                  + brightness*buf[(i+1) % buf_len])
        buf[i % buf_len] = nxt
    peak = np.max(np.abs(out))
    return (out / peak).astype(np.float32) if peak > 1e-9 else out.astype(np.float32)


def synthesize_wav(decoded, tuning, bpm=120.0, sr=44100, note_dur_beats=0.5):
    import numpy as np
    beat_sec  = 60.0 / bpm
    note_sec  = note_dur_beats * beat_sec
    gap_sec   = 0.02 * beat_sec
    step_sam  = int((note_sec + gap_sec) * sr)
    total     = step_sam * len(decoded) + int(note_sec * sr * 3)
    audio     = np.zeros(total, dtype=np.float32)
    for i, note in enumerate(decoded):
        m      = tuning[note["string"]] + note["fret"]
        freq   = 440.0 * (2.0 ** ((m - 69) / 12.0))
        f      = note["fret"]
        bright = max(0.2, 0.7 - f * 0.025)
        dec    = max(0.990, 0.998 - f * 0.0003)
        grain  = karplus_strong(freq, note_sec * 2.0, sr=sr,
                                decay=dec, brightness=bright)
        s, e   = i * step_sam, i * step_sam + len(grain)
        if e > len(audio): grain = grain[:len(audio)-s]; e = len(audio)
        audio[s:e] += grain
    peak = np.max(np.abs(audio))
    if peak > 1e-9: audio = audio / peak * 0.9
    return audio


def save_wav(audio, path, sr=44100):
    import numpy as np
    try:
        from scipy.io import wavfile
        pcm = (np.clip(audio, -1., 1.) * 32767).astype(np.int16)
        wavfile.write(path, sr, pcm)
    except ImportError:
        import wave
        pcm = (np.clip(audio, -1., 1.) * 32767).astype(np.int16)
        with wave.open(path, 'w') as wf:
            wf.setnchannels(1); wf.setsampwidth(2)
            wf.setframerate(sr); wf.writeframes(pcm.tobytes())
    print(f"WAV saved : {path}  ({len(audio)/sr:.1f}s)")


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        random.seed(args.seed); torch.manual_seed(args.seed)

    if args.train:
        if all(d is None for d in [args.dadagp_dir, args.guitarset_dir,
                                   args.scoreset_dir, args.synthtab_dir,
                                   args.scraped_tabs_dir, args.proggp_dir]):
            print("[ERROR] At least one data directory required."); sys.exit(1)

        model = DiffusionTabModel(
            embed_dim=args.embed_dim, denoiser_layers=args.denoiser_layers,
            T_diff=args.T_diff, dropout=0.1, freeze_pitch_encoder=True,
            schedule=args.schedule)
        print(f"Total params    : {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable params: "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        trainer = DiffusionTrainer(
            model,
            dadagp_dir=args.dadagp_dir, guitarset_dir=args.guitarset_dir,
            scoreset_dir=args.scoreset_dir, synthtab_dir=args.synthtab_dir,
            scraped_tabs_dir=args.scraped_tabs_dir, proggp_dir=args.proggp_dir,
            window=args.window, batch_size=args.batch_size, lr=args.lr,
            genres=args.genres or ["metal","rock","hard_rock"],
            steps_per_epoch=args.steps_per_epoch,
            num_workers=args.num_workers, pretrained_ckpt=args.pretrained,
            cache_path=args.cache_path,
            max_source_fraction=1.0)
        trainer.train(epochs=args.epochs, save_path=args.save_path)

    elif args.generate:
        if not args.pitches:
            print("[ERROR] --pitches required.  e.g. --pitches 64,66,68,69")
            sys.exit(1)

        pitches = [int(p) for p in args.pitches.split(',')]
        tuning  = TUNINGS[args.tuning]

        for i, midi in enumerate(pitches):
            if not midi_to_positions(midi, tuning):
                print(f"[ERROR] MIDI {midi} not playable in {args.tuning} tuning.")
                sys.exit(1)

        if not os.path.exists(args.save_path):
            print(f"[ERROR] Checkpoint not found: {args.save_path}"); sys.exit(1)

        model = DiffusionTabModel(
            embed_dim=args.embed_dim, denoiser_layers=args.denoiser_layers,
            T_diff=args.T_diff, dropout=0.0, freeze_pitch_encoder=True,
            schedule=args.schedule)
        model.load_state_dict(torch.load(args.save_path, map_location=device))
        model.to(device).eval()

        decoder = DiffusionDecoder(model, tuning=tuning, device=device)
        print(f"Model    : {args.save_path}")
        print(f"Pitches  : {pitches}")
        print(f"T_diff   : {args.T_diff}  temp={args.temperature}  samples={args.n_samples}")

        results = decoder.decode_n(pitches, n_samples=args.n_samples,
                                   temperature=args.temperature)

        for i, decoded in enumerate(results):
            header = f"Sample {i+1}" if args.n_samples > 1 else "Result"
            print(f"\n{'─'*56}\n {header}\n{'─'*56}")
            print(render_ascii(decoded, args.notes_per_line))
            print("─"*56)
            for j, note in enumerate(decoded):
                pitch = tuning[note["string"]] + note["fret"]
                nname = NOTE_NAMES[pitch % 12] + str(pitch // 12 - 1)
                ok    = "✓" if pitch == pitches[j] else "✗"
                print(f"  {j+1:>3}  str {note['string']}  fret {note['fret']:>2}  "
                      f"MIDI {pitch:>3}  {nname}  {ok}")
            n_ok = sum(1 for j, note in enumerate(decoded)
                       if tuning[note["string"]] + note["fret"] == pitches[j])
            print(f"\n  Pitch accuracy: {n_ok}/{len(pitches)}")

            # WAV export
            if args.wav:
                if args.n_samples > 1:
                    base, ext = os.path.splitext(args.wav)
                    wav_path  = f"{base}_{i+1}{ext or '.wav'}"
                else:
                    wav_path  = args.wav
                audio = synthesize_wav(decoded, tuning,
                                       bpm=args.bpm,
                                       note_dur_beats=args.note_dur)
                save_wav(audio, wav_path)
    else:
        print("Specify --train or --generate.  Use --help for options.")
        sys.exit(1)


if __name__ == "__main__":
    main()
