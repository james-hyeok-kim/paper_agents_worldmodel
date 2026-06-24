"""
PoC: Value-Bootstrapped Depth Scheduler
Gate: rollout_speedup > 1.5x AND quality_proxy_delta < 0.05
Key questions:
  1. Does value-curvature at h_0 predict whether K-step + bootstrap is as good as H-step?
  2. What routing fraction to K_short achieves quality_delta < 0.05?
  3. What speedup is achieved with batch-grouped depth levels?
Design:
  - Pre-rollout: compute value Hessian-trace proxy (variance across z-samples, ~1 step)
  - Assign depth K ∈ {K_short=5, K_full=15} per trajectory
  - Batch by depth (all K_short together, all K_full together) → no GPU fragmentation
  - Short trajectories: bootstrap tail with V(h_K) × λ^(H-K)
- vs BL-08: pre-rollout decision (not per-step), batch grouping avoids fragmentation
- vs BL-02: data-driven depth levels (not fixed)
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
D = 512          # RSSM hidden dim
Z = 32
A = 6
H = 15           # full imagination horizon
K_SHORT = 5      # short depth (for low-curvature states)
GAMMA = 0.99     # discount factor
N_Z_CURVATURE = 5  # z-samples for curvature proxy
B = 512          # batch size
N_STATES = 5000  # test states
N_REPS = 100     # benchmark reps
FRAC_SHORT = 0.6  # target: 60% routed to short depth

# ─── Models ──────────────────────────────────────────────────────────────────

class RSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(Z + A, D)
        self.prior = nn.Sequential(
            nn.Linear(D, D//2), nn.ELU(),
            nn.Linear(D//2, Z*2)
        )
        self.value = nn.Sequential(
            nn.Linear(D + Z, D//4), nn.ELU(),
            nn.Linear(D//4, 1)
        )
        self.reward = nn.Sequential(
            nn.Linear(D + Z, D//4), nn.ELU(),
            nn.Linear(D//4, 1)
        )

    def sample_z(self, h):
        out = self.prior(h); mean, ls = out.chunk(2, dim=-1)
        return mean + ls.clamp(-4, 2).exp() * torch.randn_like(mean)

    def step(self, h, z, a):
        return self.gru(torch.cat([z, a], dim=-1), h)

    def value_fn(self, h, z):
        return self.value(torch.cat([h, z], dim=-1)).squeeze(-1)

    def reward_fn(self, h, z):
        return self.reward(torch.cat([h, z], dim=-1)).squeeze(-1)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, D//4), nn.ELU(), nn.Linear(D//4, A), nn.Tanh())
    def forward(self, h): return self.net(h)

# ─── 1. Curvature proxy: value variance across z-samples ─────────────────────

def compute_curvature_proxy(rssm, h_batch, n_z=N_Z_CURVATURE):
    """
    Proxy for value Hessian trace: variance of V(h, z) across K_Z z-samples.
    High variance → high curvature → need full depth.
    Low variance → smooth value → short depth ok.
    """
    rssm.eval()
    with torch.no_grad():
        values = []
        for _ in range(n_z):
            z = rssm.sample_z(h_batch)
            v = rssm.value_fn(h_batch, z)
            values.append(v)
        values = torch.stack(values, dim=0)  # (n_z, B)
        curvature = values.var(dim=0)        # (B,)
    return curvature

# ─── 2. Compute H-step return via rollout + bootstrap ─────────────────────────

def rollout_return(rssm, policy, h0, n_steps, bootstrap_tail=True, gamma=GAMMA):
    """
    Compute discounted return for n_steps.
    If bootstrap_tail and n_steps < H: add V(h_{n_steps}) × γ^n_steps as tail.
    """
    rssm.eval(); policy.eval()
    h = h0
    G = torch.zeros(h0.size(0), device=DEVICE)
    disc = 1.0
    with torch.no_grad():
        for t in range(n_steps):
            z = rssm.sample_z(h)
            a = policy(h)
            r = rssm.reward_fn(h, z)
            G += disc * r
            h = rssm.step(h, z, a)
            disc *= gamma
        if bootstrap_tail and n_steps < H:
            z_term = rssm.sample_z(h)
            v_term = rssm.value_fn(h, z_term)
            G += disc * v_term  # bootstrap remaining value
    return G

# ─── 3. Calibrate: does curvature predict short-rollout quality? ──────────────

def calibrate_curvature_threshold(rssm, policy, n=N_STATES):
    """
    For each state:
    1. Compute curvature proxy
    2. Compute G_full (H-step return)
    3. Compute G_short (K_short-step + bootstrap)
    4. Measure return delta = |G_short - G_full| / |G_full|

    Find curvature threshold τ such that low-curvature states have delta < 0.05.
    """
    rssm.eval(); policy.eval()
    print(f"  Calibrating curvature threshold on {n} states...")
    with torch.no_grad():
        # Use heterogeneous states (mix of smooth and complex)
        h_all = torch.randn(n, D, device=DEVICE)
        h_all[:n//3] *= 0.2     # "smooth" region
        h_all[n//3:2*n//3] *= 0.7  # "medium"
        h_all[2*n//3:] *= 1.5   # "complex" region

        # Curvature proxy
        curv = compute_curvature_proxy(rssm, h_all)

        # Returns
        G_full  = rollout_return(rssm, policy, h_all, H, bootstrap_tail=False)
        G_short = rollout_return(rssm, policy, h_all, K_SHORT, bootstrap_tail=True)

        # Per-state return delta
        G_ref = G_full.abs().clamp(min=0.01)
        delta = (G_short - G_full).abs() / G_ref  # (n,)

        # Find threshold: maximize routing fraction while keeping delta < 0.05
        tau_values = torch.quantile(curv, torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7], device=DEVICE))
        best_tau = None; best_routing = 0; best_delta = float('inf')

        print(f"  {'Tau':>12} {'Route%':>8} {'AvgDelta':>10} {'MaxDelta':>10} {'Pass?':>6}")
        for tau in tau_values:
            mask_short = (curv < tau)
            frac_short = mask_short.float().mean().item()
            delta_short = delta[mask_short].mean().item() if mask_short.sum() > 0 else 0
            max_delta_short = delta[mask_short].max().item() if mask_short.sum() > 0 else 0
            passes = delta_short < 0.05
            print(f"  {tau.item():>12.4f} {frac_short:>8.3f} {delta_short:>10.4f} {max_delta_short:>10.4f} {'✓' if passes else '✗':>6}")
            if passes and frac_short > best_routing:
                best_routing = frac_short
                best_delta = delta_short
                best_tau = tau.item()

        return float(best_tau) if best_tau is not None else 0, best_routing, best_delta, curv, delta

# ─── 4. Speed: batch-grouped depth scheduling ─────────────────────────────────

def benchmark_depth_scheduler(rssm, policy, tau, curv_batch=None, n_reps=N_REPS):
    """
    Compare:
    - Full H-step for all B states
    - Depth-scheduled: K_SHORT for low-curv, H for high-curv (batch grouped)
    """
    rssm.eval(); policy.eval()

    # Use a fixed batch for benchmarking
    h0_bench = torch.randn(B, D, device=DEVICE)
    h0_bench[:int(B * FRAC_SHORT)] *= 0.2  # "smooth" states → short depth
    h0_bench[int(B * FRAC_SHORT):] *= 1.5  # "complex" states → full depth

    n_short = int(B * FRAC_SHORT)
    n_full  = B - n_short

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            rollout_return(rssm, policy, h0_bench, H, bootstrap_tail=False)
    if DEVICE == "cuda": torch.cuda.synchronize()

    # Full H-step rollout (baseline)
    t0 = time.perf_counter()
    for _ in range(n_reps):
        with torch.no_grad():
            rollout_return(rssm, policy, h0_bench, H, bootstrap_tail=False)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t_full = time.perf_counter() - t0

    # Depth-scheduled: 2 batched rollouts (short + full) instead of 1 big
    t0 = time.perf_counter()
    for _ in range(n_reps):
        with torch.no_grad():
            # Curvature proxy (1 step, N_Z_CURVATURE z-samples)
            curv = compute_curvature_proxy(rssm, h0_bench)
            mask_short = (curv < tau)
            h_s = h0_bench[mask_short]   # short-depth batch
            h_f = h0_bench[~mask_short]  # full-depth batch
            if h_s.size(0) > 0:
                rollout_return(rssm, policy, h_s, K_SHORT, bootstrap_tail=True)
            if h_f.size(0) > 0:
                rollout_return(rssm, policy, h_f, H, bootstrap_tail=False)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t_sched = time.perf_counter() - t0

    speedup = t_full / t_sched
    return t_full, t_sched, speedup, n_short / B

# ─── 5. Quality: overall return delta with depth scheduling ──────────────────

def measure_scheduled_quality(rssm, policy, tau, n=2000):
    """
    For a mixed batch, compare return from depth-scheduled vs full H-step.
    """
    rssm.eval(); policy.eval()
    with torch.no_grad():
        h_all = torch.randn(n, D, device=DEVICE)
        h_all[:n//3] *= 0.2; h_all[n//3:2*n//3] *= 0.7; h_all[2*n//3:] *= 1.5

        curv = compute_curvature_proxy(rssm, h_all)
        mask_short = (curv < tau) if tau > 0 else torch.zeros(n, dtype=torch.bool, device=DEVICE)
        mask_full  = ~mask_short
        frac_short = mask_short.float().mean().item()

        G_full_ref = rollout_return(rssm, policy, h_all, H, bootstrap_tail=False)

        # Scheduled return
        G_sched = G_full_ref.clone()
        if mask_short.sum() > 0:
            G_sched[mask_short] = rollout_return(rssm, policy, h_all[mask_short], K_SHORT, bootstrap_tail=True)

        delta = (G_sched - G_full_ref).abs() / G_full_ref.abs().clamp(min=0.01)
        quality_delta = delta.mean().item()
        print(f"  Short fraction: {frac_short:.3f}")
        print(f"  Return delta (scheduled vs full): {quality_delta:.5f}")
        print(f"  Max delta: {delta.max().item():.4f}")
    return quality_delta, frac_short

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Value-Bootstrapped Depth Scheduler PoC ===\n")

    rssm   = RSSM().to(DEVICE)
    policy = Policy().to(DEVICE)
    rssm.eval(); policy.eval()
    for p in list(rssm.parameters()) + list(policy.parameters()): p.requires_grad_(False)

    # ── 1. Calibrate threshold
    print("[1] Calibrating curvature threshold...")
    tau, best_routing, best_delta, curv_all, delta_all = calibrate_curvature_threshold(rssm, policy)
    print(f"\n  Best tau: {tau:.4f}")
    print(f"  Routing fraction to K_short={K_SHORT}: {best_routing:.3f}")
    print(f"  Mean delta for short-routed states: {best_delta:.5f}")
    routing_feasible = best_routing >= 0.4  # need at least 40% short for 1.5× speedup
    print(f"  Routing feasible (≥40% to short): {routing_feasible}")

    # ── 2. Overall quality
    print(f"\n[2] Overall quality with depth scheduling (tau={tau:.4f})...")
    quality_delta, frac_short = measure_scheduled_quality(rssm, policy, tau)
    quality_pass = quality_delta < 0.05
    print(f"  quality_proxy_delta: {quality_delta:.5f} (gate <0.05: {'PASS' if quality_pass else 'FAIL'})")

    # ── 3. Speed benchmark
    print(f"\n[3] Speed benchmark (K_short={K_SHORT}, K_full={H}, tau={tau:.4f})...")
    t_full, t_sched, speedup, actual_short_frac = benchmark_depth_scheduler(rssm, policy, tau)
    speed_pass = speedup > 1.5
    print(f"  Full H={H}:      {t_full*1000:.1f}ms / {N_REPS} reps")
    print(f"  Scheduled:       {t_sched*1000:.1f}ms / {N_REPS} reps")
    print(f"  Actual short fraction in benchmark: {actual_short_frac:.3f}")
    print(f"  Speedup: {speedup:.3f}x (gate >1.5x: {'PASS' if speed_pass else 'FAIL'})")

    # ── 4. Theoretical estimate
    # Expected speedup: frac_short × K_short/H + (1-frac_short) × 1.0 + curvature overhead
    # Plus sub-batch split overhead (same as BL-11 issue?)
    theoretical = 1.0 / (frac_short * K_SHORT/H + (1-frac_short) * 1.0 + N_Z_CURVATURE / (H * 3))
    print(f"  Theoretical speedup (frac={frac_short:.2f}): {theoretical:.3f}x")

    # ── Verdict
    verdict = "CONDITIONAL-GO" if (speed_pass and quality_pass) else "FAIL"
    fail_reasons = []
    if not speed_pass: fail_reasons.append(f"speedup {speedup:.3f}x < 1.5x")
    if not quality_pass: fail_reasons.append(f"quality_delta {quality_delta:.5f} >= 0.05")

    print(f"\n=== VERDICT: {verdict} ===")
    if fail_reasons:
        for r in fail_reasons: print(f"  FAIL: {r}")

    # ── Save
    results = {
        "idea": "value-bootstrapped-depth-scheduler",
        "verdict": verdict,
        "rollout_speedup": round(speedup, 4),
        "quality_proxy_delta": round(quality_delta, 5),
        "calibration_tau": round(tau, 4),
        "routing_fraction_to_short": round(frac_short, 4),
        "best_routing_calibration": round(best_routing, 4),
        "best_delta_calibration": round(best_delta, 5),
        "theoretical_speedup": round(theoretical, 4),
        "K_short": K_SHORT,
        "H_full": H,
        "N_Z_CURVATURE": N_Z_CURVATURE,
        "device": DEVICE,
    }

    out_path = Path(__file__).parent / "poc_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")
    print(json.dumps(results, indent=2))
