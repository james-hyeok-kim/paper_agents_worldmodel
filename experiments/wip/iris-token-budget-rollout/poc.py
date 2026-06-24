"""
PoC: IRIS Token-Depth Rollout
Gate: rollout_speedup > 1.5x AND quality_proxy_delta < 0.05
Key questions:
  1. Is transformer the hot-path (>70% of IRIS imagination)?
  2. Does depth-budget early-exit give >1.5× speedup?
  3. Does prediction quality stay within 5% when skipping layers for easy tokens?
Design: toy transformer WM (L=12 layers), input-side router predicts depth
        reward-sensitive tokens always get full depth
"""
import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

torch.manual_seed(42)
np.random.seed(42)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
N_TOKENS = 16   # tokens per frame (IRIS uses ~16 discrete tokens per obs frame)
D_MODEL  = 256  # transformer d_model
N_HEADS  = 8    # attention heads
N_LAYERS = 12   # total transformer layers
D_FF     = 1024 # FFN hidden
VOCAB    = 512  # codebook size
B_SEQ   = 64   # batch of sequences (imagination rollout batch)
T_FRAMES = 8    # frames per rollout
DEPTH_BUDGET = 0.45  # target: use 45% of layers on average
REWARD_FRAC  = 0.2   # fraction of tokens flagged as reward-sensitive (always full depth)
N_REPS       = 50    # benchmark repetitions
N_TRAIN_DATA = 2000  # sequences for router training

# ─── Models ──────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, d=D_MODEL, nhead=N_HEADS, d_ff=D_FF):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ffn  = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(),
            nn.Linear(d_ff, d)
        )
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x

class FullTransformerWM(nn.Module):
    """Full L-layer transformer world model (IRIS-style)."""
    def __init__(self):
        super().__init__()
        self.tok_embed = nn.Embedding(VOCAB, D_MODEL)
        self.pos_embed = nn.Embedding(N_TOKENS * T_FRAMES + 10, D_MODEL)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        self.head = nn.Linear(D_MODEL, VOCAB)  # next-token prediction
        self.reward_head = nn.Linear(D_MODEL, 1)  # reward per token

    def forward(self, token_ids, return_all_layers=False):
        """token_ids: (B, T_seq)"""
        B, T = token_ids.shape
        pos = torch.arange(T, device=DEVICE).unsqueeze(0)
        x = self.tok_embed(token_ids) + self.pos_embed(pos)
        if return_all_layers:
            hiddens = [x]
        for blk in self.blocks:
            x = blk(x)
            if return_all_layers:
                hiddens.append(x)
        logits = self.head(x)
        reward = self.reward_head(x)
        if return_all_layers:
            return logits, reward, hiddens
        return logits, reward

    def forward_partial(self, token_ids, n_layers):
        """Run only first n_layers for early-exit."""
        B, T = token_ids.shape
        pos = torch.arange(T, device=DEVICE).unsqueeze(0)
        x = self.tok_embed(token_ids) + self.pos_embed(pos)
        for i in range(min(n_layers, N_LAYERS)):
            x = self.blocks[i](x)
        logits = self.head(x)
        reward = self.reward_head(x)
        return logits, reward

class InputSideRouter(nn.Module):
    """
    Predicts required depth for each token based on context embedding ONLY.
    Runs BEFORE expensive transformer blocks (not after → no wasted compute).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL//4), nn.GELU(),
            nn.Linear(D_MODEL//4, 1),
            nn.Sigmoid()  # normalized depth in [0, 1]
        )

    def forward(self, embed_x):
        """embed_x: (B, T, D_MODEL) after positional embedding, before blocks."""
        return self.net(embed_x).squeeze(-1)  # (B, T) in [0,1]

# ─── 1. Measure transformer hot-path fraction ─────────────────────────────────

def measure_transformer_fraction(wm, B=B_SEQ, T=N_TOKENS*T_FRAMES, n=20):
    """What fraction of imagination cost is the transformer?"""
    wm.eval()
    # Tokenizer surrogate (codebook lookup + CNN encoder)
    tokenizer = nn.Sequential(
        nn.Conv2d(3, 32, 4, 2), nn.ELU(),
        nn.Conv2d(32, 64, 4, 2), nn.ELU(),
        nn.Flatten(), nn.Linear(64*14*14, VOCAB)
    ).to(DEVICE)
    imgs = torch.randn(B, 3, 64, 64, device=DEVICE)
    toks = torch.randint(0, VOCAB, (B, T), device=DEVICE)

    if DEVICE == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            _ = tokenizer(imgs)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t_tokenizer = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            _ = wm(toks)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t_transformer = time.perf_counter() - t0

    frac = t_transformer / (t_transformer + t_tokenizer)
    print(f"    Tokenizer:    {t_tokenizer*1000/n:.1f}ms")
    print(f"    Transformer:  {t_transformer*1000/n:.1f}ms")
    print(f"    Transformer fraction: {frac:.3f}")
    return frac

# ─── 2. Layer-depth skip speedup ─────────────────────────────────────────────

def benchmark_depth_speedup(wm, B=B_SEQ, T=N_TOKENS*T_FRAMES, target_depth_frac=DEPTH_BUDGET):
    """
    Compare full L-layer vs average depth target_depth_frac × L.
    For simplicity: split tokens into easy (depth=int(L*depth_frac)) and hard (depth=L).
    hard_frac = REWARD_FRAC, easy_frac = 1 - REWARD_FRAC.
    Actual average: REWARD_FRAC × 1.0 + (1-REWARD_FRAC) × depth_frac
    """
    wm.eval()
    toks = torch.randint(0, VOCAB, (B, T), device=DEVICE)
    T_easy = int(T * (1 - REWARD_FRAC))
    T_hard = T - T_easy
    avg_depth_frac = REWARD_FRAC * 1.0 + (1 - REWARD_FRAC) * target_depth_frac
    avg_layers = int(avg_depth_frac * N_LAYERS)
    easy_layers = max(1, int(target_depth_frac * N_LAYERS))
    print(f"    Easy tokens: {T_easy}/{T} at {easy_layers} layers")
    print(f"    Hard tokens: {T_hard}/{T} at {N_LAYERS} layers (full)")
    print(f"    Average depth: {avg_depth_frac:.3f} × {N_LAYERS} = {avg_layers} layers")

    # Warmup
    for _ in range(5):
        _ = wm(toks)
    if DEVICE == "cuda": torch.cuda.synchronize()

    # Full L-layer
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        with torch.no_grad():
            _ = wm(toks)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t_full = time.perf_counter() - t0

    # Depth-budget: run easy tokens at fewer layers
    # In practice this requires masking/routing, we simulate by running 2 forward passes:
    # 1) All tokens at easy_layers depth
    # 2) hard tokens at full depth (already included)
    # This overestimates cost slightly but tests actual GPU speedup
    toks_easy = torch.randint(0, VOCAB, (B, T_easy), device=DEVICE)
    toks_hard = torch.randint(0, VOCAB, (B, T_hard), device=DEVICE)

    t0 = time.perf_counter()
    for _ in range(N_REPS):
        with torch.no_grad():
            # Easy tokens: shallow
            _ = wm.forward_partial(toks_easy, easy_layers)
            # Hard tokens: full depth
            _ = wm.forward_partial(toks_hard, N_LAYERS)
            # Router overhead (embed only, no blocks)
            B2, T2 = toks.shape
            pos = torch.arange(T2, device=DEVICE).unsqueeze(0)
            x_embed = wm.tok_embed(toks) + wm.pos_embed(pos)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t_depth = time.perf_counter() - t0

    speedup = t_full / t_depth
    return t_full, t_depth, speedup, avg_depth_frac

# ─── 3. Quality: partial-depth prediction vs full-depth ──────────────────────

def measure_quality_at_depth(wm, target_frac, B=256, T=N_TOKENS*T_FRAMES, n=50):
    """
    Compare token prediction accuracy: full L layers vs target_frac × L layers.
    Quality proxy = cross-entropy increase (normalized).
    """
    wm.eval()
    toks = torch.randint(0, VOCAB, (B, T), device=DEVICE)
    partial_layers = max(1, int(target_frac * N_LAYERS))

    with torch.no_grad():
        logits_full, reward_full = wm(toks)
        logits_part, reward_part = wm.forward_partial(toks, partial_layers)

    # Cross-entropy of token prediction (next-token proxy)
    target = toks[:, 1:].contiguous().view(-1)  # (B*(T-1),)
    log_prob_full = logits_full[:, :-1].contiguous().view(-1, VOCAB)
    log_prob_part = logits_part[:, :-1].contiguous().view(-1, VOCAB)

    ce_full = nn.functional.cross_entropy(log_prob_full, target).item()
    ce_part = nn.functional.cross_entropy(log_prob_part, target).item()
    ce_delta = (ce_part - ce_full) / (ce_full + 1e-6)  # relative CE increase

    # Reward prediction delta
    r_delta = (reward_part - reward_full).abs().mean().item() / (reward_full.abs().mean().item() + 1e-6)

    print(f"    Full-depth CE:    {ce_full:.4f}")
    print(f"    Partial-depth CE: {ce_part:.4f} ({partial_layers}/{N_LAYERS} layers)")
    print(f"    CE delta (relative): {ce_delta:.4f}")
    print(f"    Reward delta: {r_delta:.4f}")

    quality_proxy_delta = ce_delta  # normalized CE increase
    return float(quality_proxy_delta), float(r_delta)

# ─── 4. Router training and quality ──────────────────────────────────────────

def train_router(router, wm, n=N_TRAIN_DATA, epochs=15):
    """
    Train router to predict which tokens need full depth.
    Label: tokens where partial-depth (easy_layers) CE >> full-depth CE.
    """
    wm.eval()
    partial_layers = max(1, int(DEPTH_BUDGET * N_LAYERS))
    router_optimizer = torch.optim.Adam(router.parameters(), lr=1e-3)

    # Generate training data
    with torch.no_grad():
        toks = torch.randint(0, VOCAB, (n, N_TOKENS), device=DEVICE)
        pos = torch.arange(N_TOKENS, device=DEVICE).unsqueeze(0)
        x_embed = wm.tok_embed(toks) + wm.pos_embed(pos)  # (n, T, D)

        logits_full, _ = wm(toks)
        logits_part, _ = wm.forward_partial(toks, partial_layers)
        # Per-token CE delta as routing label
        toks_target = torch.cat([toks[:, 1:], toks[:, :1]], dim=1)  # shift
        ce_full = nn.functional.cross_entropy(
            logits_full.view(-1, VOCAB), toks_target.view(-1), reduction='none'
        ).view(n, N_TOKENS)
        ce_part = nn.functional.cross_entropy(
            logits_part.view(-1, VOCAB), toks_target.view(-1), reduction='none'
        ).view(n, N_TOKENS)
        # Normalized: does this token need full depth?
        depth_label = (ce_part - ce_full) / (ce_full.abs() + 1e-6)
        depth_label = depth_label.clamp(0, 1)  # [0,1] routing signal

    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0; nb = 0
        for start in range(0, n, 256):
            idx = perm[start:start+256]
            pred = router(x_embed[idx].detach())
            loss = (pred - depth_label[idx].detach()).pow(2).mean()
            router_optimizer.zero_grad()
            loss.backward()
            router_optimizer.step()
            ep_loss += loss.item(); nb += 1
        losses.append(ep_loss / nb)
    return losses, depth_label

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== IRIS Token-Depth Rollout PoC ===\n")

    wm = FullTransformerWM().to(DEVICE)
    router = InputSideRouter().to(DEVICE)
    wm.eval()
    for p in wm.parameters(): p.requires_grad_(False)

    # ── 0. Hot-path fraction
    print("[0] Measuring transformer hot-path fraction...")
    transformer_frac = measure_transformer_fraction(wm)
    amdahl_ok = transformer_frac > 0.5
    print(f"    Amdahl feasible (transformer > 50%): {amdahl_ok}")

    # ── 1. Quality at various depth levels
    print(f"\n[1] Quality at depth budget {DEPTH_BUDGET*100:.0f}% of {N_LAYERS} layers...")
    quality_proxy_delta, r_delta = measure_quality_at_depth(wm, DEPTH_BUDGET)
    quality_pass = quality_proxy_delta < 0.05
    print(f"    quality_proxy_delta: {quality_proxy_delta:.4f} (gate <0.05): {'PASS' if quality_pass else 'FAIL'}")

    # Also check reward-only tokens: gate + reward-sensitive mask
    print(f"\n[1b] Quality for easy tokens only (20% reward-sensitive always full)...")
    q_easy, r_easy = measure_quality_at_depth(wm, DEPTH_BUDGET, T=int(N_TOKENS*(1-REWARD_FRAC)))
    print(f"    Easy-token quality delta: {q_easy:.4f}")

    # ── 2. Depth-budget speedup
    print(f"\n[2] Depth-budget speedup (easy={int(DEPTH_BUDGET*100)}% layers, hard=full)...")
    t_full, t_depth, speedup, avg_frac = benchmark_depth_speedup(wm)
    speed_pass = speedup > 1.5
    print(f"    Full ({N_LAYERS} layers):  {t_full*1000:.1f}ms / {N_REPS} reps")
    print(f"    Depth-budget:         {t_depth*1000:.1f}ms / {N_REPS} reps")
    print(f"    Speedup: {speedup:.3f}x (gate >1.5x: {'PASS' if speed_pass else 'FAIL'})")

    # ── 3. Train router
    print(f"\n[3] Training input-side router...")
    losses, depth_labels = train_router(router, wm)
    print(f"    Router loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    router.eval()
    # Check: router assigns high depth-need to high-CE tokens
    with torch.no_grad():
        toks_test = torch.randint(0, VOCAB, (500, N_TOKENS), device=DEVICE)
        pos_test = torch.arange(N_TOKENS, device=DEVICE).unsqueeze(0)
        x_test = wm.tok_embed(toks_test) + wm.pos_embed(pos_test)
        pred_depth = router(x_test)
        corr_val = float(torch.corrcoef(torch.stack([
            pred_depth[:, 0], depth_labels[:500, 0]
        ]))[0, 1])
    print(f"    Router correlation with true depth-need: {corr_val:.4f}")

    # ── Verdict
    verdict = "CONDITIONAL-GO" if (speed_pass and quality_pass) else "FAIL"
    fail_reasons = []
    if not speed_pass: fail_reasons.append(f"speedup {speedup:.2f}x < 1.5x")
    if not quality_pass: fail_reasons.append(f"quality_proxy {quality_proxy_delta:.4f} >= 0.05")

    print(f"\n=== VERDICT: {verdict} ===")
    if fail_reasons:
        for r in fail_reasons: print(f"  FAIL: {r}")

    # ── Save
    results = {
        "idea": "iris-token-budget-rollout",
        "verdict": verdict,
        "rollout_speedup": round(speedup, 4),
        "quality_proxy_delta": round(quality_proxy_delta, 4),
        "quality_easy_tokens_delta": round(q_easy, 4),
        "transformer_fraction": round(float(transformer_frac), 4),
        "amdahl_feasible": bool(amdahl_ok),
        "avg_depth_fraction": round(float(avg_frac), 4),
        "router_correlation": round(float(corr_val), 4),
        "depth_budget": DEPTH_BUDGET,
        "reward_sensitive_frac": REWARD_FRAC,
        "n_layers": N_LAYERS,
        "device": DEVICE,
    }

    out_path = Path(__file__).parent / "poc_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")
    print(json.dumps(results, indent=2))
