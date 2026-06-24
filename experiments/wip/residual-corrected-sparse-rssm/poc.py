"""
PoC: Residual-Corrected Sparse RSSM
Gate: rollout_speedup > 1.5x AND quality_proxy_delta < 0.05
Key question: Can rank-r corrector estimate Δh without carry-forward drift?
- vs BL-06 (carry-forward Δ=0): corrector here uses Δ≠0
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
D = 512        # GRU hidden dim (DreamerV3-scale)
Z = 32         # stochastic z dim (discrete 32×32 → 32 categories, flatten to 32 for simplicity)
A = 6          # action dim
R = 64         # corrector rank (r=D/8)
K = 4          # anchor interval
N_SEQ = 2000   # number of rollout sequences
T = 32         # steps per sequence
BATCH = 256    # batch size for speed measurement
N_EPOCHS = 20  # corrector training epochs

# ─── Models ──────────────────────────────────────────────────────────────────

class FullGRU(nn.Module):
    """Full RSSM dynamics (anchor)"""
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(Z + A, D)
        # prior MLP (~30% of RSSM transition cost)
        self.prior = nn.Sequential(
            nn.Linear(D, D//2), nn.ELU(),
            nn.Linear(D//2, Z*2)  # mean/std for z
        )

    def forward(self, h, z, a):
        inp = torch.cat([z, a], dim=-1)
        h_new = self.gru(inp, h)
        _ = self.prior(h_new)   # prior cost (always runs)
        return h_new

class RankRCorrector(nn.Module):
    """Cheap rank-r residual corrector for Δh estimation"""
    def __init__(self):
        super().__init__()
        # P: D→R downproject, small MLP, U: R→D upproject
        self.P = nn.Linear(D, R, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(R + Z + A, R*2), nn.ELU(),
            nn.Linear(R*2, R)
        )
        self.U = nn.Linear(R, D, bias=False)

    def forward(self, h, z, a):
        h_low = self.P(h)
        r_in = torch.cat([h_low, z, a], dim=-1)
        delta_low = self.mlp(r_in)
        delta_h = self.U(delta_low)
        return delta_h  # Δh estimate (NOT carry-forward)

# ─── Data generation ──────────────────────────────────────────────────────────

def generate_rollouts(gru, n_seq=N_SEQ, t=T):
    """Generate ground-truth h trajectories with full GRU."""
    gru.eval()
    all_h = []  # shape: (n_seq, t+1, D)
    all_z = []
    all_a = []
    with torch.no_grad():
        h = torch.randn(n_seq, D, device=DEVICE) * 0.1
        for step in range(t):
            z = torch.randn(n_seq, Z, device=DEVICE)
            a = torch.randn(n_seq, A, device=DEVICE)
            if step == 0:
                all_h.append(h)
            h_new = gru(h, z, a)
            all_h.append(h_new)
            all_z.append(z)
            all_a.append(a)
            h = h_new
    # all_h: list of (n_seq, D) × (t+1)
    H = torch.stack(all_h, dim=1)   # (n_seq, t+1, D)
    Z_t = torch.stack(all_z, dim=1) # (n_seq, t, Z)
    A_t = torch.stack(all_a, dim=1) # (n_seq, t, A)
    return H, Z_t, A_t

# ─── 1. Measure Δh dimensionality (SVD energy) ───────────────────────────────

def measure_delta_h_rank(H):
    """What rank captures 95% of Δh energy?"""
    # H shape: (n_seq, t+1, D)
    delta_H = H[:, 1:, :] - H[:, :-1, :]  # (n_seq, t, D)
    delta_flat = delta_H.reshape(-1, D)     # (N, D)
    # subsample for SVD speed
    idx = torch.randperm(delta_flat.size(0))[:4096]
    delta_sub = delta_flat[idx].cpu().float()
    delta_sub -= delta_sub.mean(0)
    _, S, _ = torch.linalg.svd(delta_sub, full_matrices=False)
    S = S.numpy()
    energy = (S**2).cumsum() / (S**2).sum()
    rank_90 = int(np.searchsorted(energy, 0.90)) + 1
    rank_95 = int(np.searchsorted(energy, 0.95)) + 1
    rank_99 = int(np.searchsorted(energy, 0.99)) + 1
    top32_energy = float(energy[min(31, len(energy)-1)])
    return rank_90, rank_95, rank_99, top32_energy

# ─── 2. Train corrector ───────────────────────────────────────────────────────

def train_corrector(corrector, H, Z_t, A_t):
    """Train corrector to predict Δh = h_{t} - h_{t-1}."""
    optimizer = torch.optim.Adam(corrector.parameters(), lr=1e-3)
    n, t, _ = Z_t.shape
    corrector.train()
    losses = []
    for epoch in range(N_EPOCHS):
        # flatten sequences
        h_prev = H[:, :-1, :].reshape(-1, D)      # (N, D)
        h_curr = H[:, 1:, :].reshape(-1, D)
        z = Z_t.reshape(-1, Z)
        a = A_t.reshape(-1, A)
        delta_true = h_curr - h_prev               # target

        perm = torch.randperm(h_prev.size(0))
        epoch_loss = 0
        n_batches = 0
        for start in range(0, len(perm), 2048):
            idx = perm[start:start+2048]
            delta_pred = corrector(h_prev[idx], z[idx], a[idx])
            loss = (delta_pred - delta_true[idx]).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / n_batches)
    return losses

# ─── 3. Compare drift: carry-forward vs corrector ────────────────────────────

def compare_drift(gru, corrector, n_eval=512, t=T):
    """
    Run K-step rollout with:
    - full GRU every step (ground truth)
    - carry-forward every step (BL-06: Δ=0)
    - corrector (Δ≠0, re-anchor every K steps)
    Measure drift = ||h_approx - h_full||_2 / ||h_full||_2
    """
    gru.eval(); corrector.eval()
    with torch.no_grad():
        h_true = torch.randn(n_eval, D, device=DEVICE) * 0.1
        h_carry = h_true.clone()
        h_corrector = h_true.clone()

        drift_carry = []
        drift_corr = []

        for step in range(t):
            z = torch.randn(n_eval, Z, device=DEVICE)
            a = torch.randn(n_eval, A, device=DEVICE)
            # ground truth
            h_true_new = gru(h_true, z, a)

            # carry-forward (BL-06: skip GRU, Δ=0)
            h_carry_new = h_carry.clone()  # Δ=0

            # corrector (Δ≠0, K-step re-anchor)
            if step % K == 0:
                # full anchor
                h_corrector_new = gru(h_corrector, z, a)
            else:
                # cheap corrector
                delta = corrector(h_corrector, z, a)
                h_corrector_new = h_corrector + delta

            # compute drift
            ref_norm = h_true_new.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            d_carry = ((h_carry_new - h_true_new).norm(dim=-1) / ref_norm.squeeze()).mean().item()
            d_corr  = ((h_corrector_new - h_true_new).norm(dim=-1) / ref_norm.squeeze()).mean().item()
            drift_carry.append(d_carry)
            drift_corr.append(d_corr)

            h_true = h_true_new
            h_carry = h_carry_new
            h_corrector = h_corrector_new

    return drift_carry, drift_corr

# ─── 4. Speed benchmark ───────────────────────────────────────────────────────

def benchmark_speed(gru, corrector, batch=BATCH, t=32, n_reps=50):
    """Measure steps/sec: full GRU vs K-anchor + corrector."""
    gru.eval(); corrector.eval()

    # WARMUP
    h = torch.randn(batch, D, device=DEVICE)
    for _ in range(10):
        z = torch.randn(batch, Z, device=DEVICE)
        a = torch.randn(batch, A, device=DEVICE)
        h = gru(h, z, a)
    if DEVICE == "cuda": torch.cuda.synchronize()

    # Full GRU
    h = torch.randn(batch, D, device=DEVICE)
    t0 = time.perf_counter()
    for _ in range(n_reps):
        for step in range(t):
            z = torch.randn(batch, Z, device=DEVICE)
            a = torch.randn(batch, A, device=DEVICE)
            with torch.no_grad():
                h = gru(h, z, a)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t_full = time.perf_counter() - t0

    # K-anchor + corrector
    h = torch.randn(batch, D, device=DEVICE)
    t0 = time.perf_counter()
    for _ in range(n_reps):
        for step in range(t):
            z = torch.randn(batch, Z, device=DEVICE)
            a = torch.randn(batch, A, device=DEVICE)
            with torch.no_grad():
                if step % K == 0:
                    h = gru(h, z, a)
                else:
                    delta = corrector(h, z, a)
                    h = h + delta
    if DEVICE == "cuda": torch.cuda.synchronize()
    t_sparse = time.perf_counter() - t0

    speedup = t_full / t_sparse
    return t_full, t_sparse, speedup

# ─── 5. Measure GRU vs prior MLP FLOP ratio ──────────────────────────────────

def measure_flop_ratio(gru, batch=256, n=200):
    """Estimate GRU vs prior MLP time ratio."""
    gru.eval()
    h = torch.randn(batch, D, device=DEVICE)
    z = torch.randn(batch, Z, device=DEVICE)
    a = torch.randn(batch, A, device=DEVICE)
    inp = torch.cat([z, a], dim=-1)

    # warmup
    for _ in range(10):
        _ = gru.gru(inp, h)
        _ = gru.prior(h)
    if DEVICE == "cuda": torch.cuda.synchronize()

    # GRU only
    t0 = time.perf_counter()
    for _ in range(n):
        with torch.no_grad():
            _ = gru.gru(inp, h)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t_gru = time.perf_counter() - t0

    # prior MLP only
    t0 = time.perf_counter()
    for _ in range(n):
        with torch.no_grad():
            _ = gru.prior(h)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t_prior = time.perf_counter() - t0

    return t_gru, t_prior, t_gru / (t_gru + t_prior)

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Residual-Corrected Sparse RSSM PoC ===\n")

    # Initialize models
    gru = FullGRU().to(DEVICE)
    corrector = RankRCorrector().to(DEVICE)

    # Pretend GRU is pretrained (random but fixed)
    gru.eval()
    for p in gru.parameters():
        p.requires_grad_(False)
    # Re-enable for GRU rollout usage:
    for p in gru.parameters():
        p.requires_grad_(False)

    # ── 1. GRU vs prior FLOP ratio
    print("[1] Measuring GRU vs prior MLP time ratio...")
    t_gru, t_prior, gru_frac = measure_flop_ratio(gru)
    print(f"    GRU time:   {t_gru*1000:.2f}ms")
    print(f"    Prior time: {t_prior*1000:.2f}ms")
    print(f"    GRU fraction of transition: {gru_frac:.3f}")
    amdahl_feasible = gru_frac > 0.5
    print(f"    Amdahl feasible (GRU>50%): {amdahl_feasible}")

    # ── 2. Generate rollouts
    print("\n[2] Generating ground-truth rollouts...")
    H, Z_t, A_t = generate_rollouts(gru)
    print(f"    H shape: {H.shape}")

    # ── 3. SVD energy of Δh
    print("\n[3] Measuring Δh dimensionality (SVD)...")
    rank_90, rank_95, rank_99, top32_energy = measure_delta_h_rank(H)
    print(f"    Rank for 90% energy: {rank_90} / {D}")
    print(f"    Rank for 95% energy: {rank_95} / {D}")
    print(f"    Rank for 99% energy: {rank_99} / {D}")
    print(f"    Top-32 dims capture {top32_energy:.3f} of energy")
    delta_h_low_rank = top32_energy > 0.7
    print(f"    Δh is low-rank (top-32 > 70% energy): {delta_h_low_rank}")

    # ── 4. Train corrector
    print("\n[4] Training rank-r corrector (teacher residual)...")
    losses = train_corrector(corrector, H, Z_t, A_t)
    print(f"    Loss: {losses[0]:.4f} → {losses[-1]:.4f}")

    # ── 5. Compare drift
    print("\n[5] Comparing drift: carry-forward (BL-06) vs corrector...")
    drift_carry, drift_corr = compare_drift(gru, corrector)
    mean_carry = float(np.mean(drift_carry))
    mean_corr  = float(np.mean(drift_corr))
    max_carry  = float(np.max(drift_carry))
    max_corr   = float(np.max(drift_corr))
    print(f"    Carry-forward  — mean drift: {mean_carry:.4f}, max: {max_carry:.4f}")
    print(f"    Corrector      — mean drift: {mean_corr:.4f}, max: {max_corr:.4f}")
    print(f"    Drift reduction: {mean_carry/max(mean_corr,1e-8):.2f}x")
    quality_proxy_delta = mean_corr  # normalized drift as quality proxy
    quality_pass = quality_proxy_delta < 0.05
    print(f"    quality_proxy_delta: {quality_proxy_delta:.4f} (gate <0.05): {'PASS' if quality_pass else 'FAIL'}")

    # ── 6. Speed benchmark
    print("\n[6] Speed benchmark...")
    t_full, t_sparse, speedup = benchmark_speed(gru, corrector)
    print(f"    Full GRU:   {t_full*1000:.1f}ms / {50*32} steps = {50*32/(t_full):.0f} steps/s")
    print(f"    Sparse K={K}: {t_sparse*1000:.1f}ms / {50*32} steps = {50*32/(t_sparse):.0f} steps/s")
    print(f"    Rollout speedup: {speedup:.3f}x")
    speed_pass = speedup > 1.5
    print(f"    Gate >1.5x: {'PASS' if speed_pass else 'FAIL'}")

    # ── Verdict
    verdict = "CONDITIONAL-GO" if (speed_pass and quality_pass) else "FAIL"
    fail_reasons = []
    if not speed_pass: fail_reasons.append(f"speedup {speedup:.2f}x < 1.5x")
    if not quality_pass: fail_reasons.append(f"quality_proxy {quality_proxy_delta:.4f} >= 0.05")

    print(f"\n=== VERDICT: {verdict} ===")
    if fail_reasons:
        for r in fail_reasons: print(f"  FAIL: {r}")

    # ── Save results
    results = {
        "idea": "residual-corrected-sparse-rssm",
        "verdict": verdict,
        "rollout_speedup": round(speedup, 4),
        "quality_proxy_delta": round(quality_proxy_delta, 4),
        "gru_fraction_of_transition": round(float(gru_frac), 4),
        "amdahl_feasible": bool(amdahl_feasible),
        "delta_h_rank_90": int(rank_90),
        "delta_h_rank_95": int(rank_95),
        "delta_h_top32_energy": round(float(top32_energy), 4),
        "delta_h_is_low_rank": bool(delta_h_low_rank),
        "drift_carry_mean": round(mean_carry, 4),
        "drift_corrector_mean": round(mean_corr, 4),
        "drift_reduction_ratio": round(mean_carry / max(mean_corr, 1e-8), 2),
        "corrector_rank": R,
        "anchor_interval_K": K,
        "device": DEVICE,
    }

    out_path = Path(__file__).parent / "poc_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")
    print(json.dumps(results, indent=2))
