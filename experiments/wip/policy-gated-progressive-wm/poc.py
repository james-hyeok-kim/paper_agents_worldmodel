"""
PoC: Policy-Gated Progressive World Model
Gate: rollout_speedup > 1.5x AND quality_proxy_delta < 0.05
Key questions:
  1. Is value-gap between cheap and expensive model small for on-policy states?
  2. Does gating achieve >60% cheap routing while keeping quality?
  3. What is speedup with 70% cheap routing + 20% FLOPs cheap model?
Design: 2-tier RSSM (cheap: D_c=128, expensive: D_e=512)
         gating by |V_cheap - V_exp| < τ
         Always full horizon rollout (no truncation, no cliff problem)
- vs BL-04: conditional routing, not static distillation
- vs BL-08: no truncation, model-tier selection only
- vs SIC: per-step routing (not trajectory-skip), works in latent-only setting
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
D_EXP  = 512    # expensive model hidden dim (DreamerV3-scale)
D_CHEAP = 128   # cheap model (D_cheap/D_exp = 1/4 → FLOPs ratio ≈ (D_cheap/D_exp)^2 = 1/16 for GRU)
Z = 32          # stochastic z dim
A = 6           # action dim
H = 15          # imagination horizon
B = 512         # batch size
N_TRAIN = 5000  # "environments" for gating calibration
N_POLICY = 3000 # policy-visited states (on-policy subset)
GAMMA = 0.99
VALUE_GAP_THRESHOLD = 0.1  # τ: route to cheap if |V_cheap - V_exp| < τ
N_REPS = 80     # speed benchmark repetitions

# ─── Models ──────────────────────────────────────────────────────────────────

class ExpensiveRSSM(nn.Module):
    """Full-capacity expensive RSSM (D=512)."""
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(Z + A, D_EXP)
        self.prior = nn.Sequential(
            nn.Linear(D_EXP, D_EXP//2), nn.ELU(),
            nn.Linear(D_EXP//2, Z*2)
        )
        self.value = nn.Sequential(
            nn.Linear(D_EXP + Z, D_EXP//4), nn.ELU(),
            nn.Linear(D_EXP//4, 1)
        )
        self.reward = nn.Sequential(
            nn.Linear(D_EXP + Z, D_EXP//4), nn.ELU(),
            nn.Linear(D_EXP//4, 1)
        )

    def forward(self, h, z, a):
        inp = torch.cat([z, a], dim=-1)
        h_new = self.gru(inp, h)
        _ = self.prior(h_new)
        return h_new

    def value_fn(self, h, z):
        return self.value(torch.cat([h, z], dim=-1)).squeeze(-1)

    def reward_fn(self, h, z):
        return self.reward(torch.cat([h, z], dim=-1)).squeeze(-1)

    def sample_z(self, h):
        out = self.prior(h)
        mean, log_std = out.chunk(2, dim=-1)
        std = log_std.clamp(-4, 2).exp()
        return mean + std * torch.randn_like(std)

class CheapRSSM(nn.Module):
    """Cheap RSSM (D_cheap=128, ~1/16 GRU FLOPs of expensive)."""
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(Z + A, D_CHEAP)
        self.prior = nn.Sequential(
            nn.Linear(D_CHEAP, D_CHEAP//2), nn.ELU(),
            nn.Linear(D_CHEAP//2, Z*2)
        )
        # Upproject to D_EXP space for value/reward (shared heads)
        self.proj_up = nn.Linear(D_CHEAP, D_EXP)
        self.value = nn.Sequential(
            nn.Linear(D_EXP + Z, D_EXP//4), nn.ELU(),
            nn.Linear(D_EXP//4, 1)
        )
        self.reward = nn.Sequential(
            nn.Linear(D_EXP + Z, D_EXP//4), nn.ELU(),
            nn.Linear(D_EXP//4, 1)
        )

    def forward(self, h_cheap, z, a):
        inp = torch.cat([z, a], dim=-1)
        h_new = self.gru(inp, h_cheap)
        _ = self.prior(h_new)
        return h_new

    def value_fn(self, h_cheap, z):
        h_up = self.proj_up(h_cheap)
        return self.value(torch.cat([h_up, z], dim=-1)).squeeze(-1)

    def reward_fn(self, h_cheap, z):
        h_up = self.proj_up(h_cheap)
        return self.reward(torch.cat([h_up, z], dim=-1)).squeeze(-1)

    def sample_z(self, h_cheap):
        out = self.prior(h_cheap)
        mean, log_std = out.chunk(2, dim=-1)
        std = log_std.clamp(-4, 2).exp()
        return mean + std * torch.randn_like(std)

class GatingNetwork(nn.Module):
    """Predicts value-gap from latent h alone (cheap to run)."""
    def __init__(self):
        super().__init__()
        # Takes expensive h for routing (available at inference)
        self.net = nn.Sequential(
            nn.Linear(D_EXP, D_EXP//4), nn.ELU(),
            nn.Linear(D_EXP//4, 1)  # predicted value-gap
        )

    def forward(self, h_exp):
        return self.net(h_exp).squeeze(-1)

# ─── Cheap-model h initialization ─────────────────────────────────────────────

def project_h(h_exp, proj):
    """Project expensive h to cheap h for initialization."""
    return proj(h_exp)

# ─── 1. Calibrate cheap vs expensive value-gap ───────────────────────────────

def measure_value_gap(exp_rssm, cheap_rssm, h_proj, n=N_TRAIN):
    """
    For diverse initial states, measure |V_cheap - V_exp| across different "on-policy-ness".
    On-policy states: h from small-scale (trained policy region)
    Off-policy states: h from large-scale random region
    """
    exp_rssm.eval(); cheap_rssm.eval()
    with torch.no_grad():
        # Simulate states from various regions
        h_exp_onpolicy  = torch.randn(n//2, D_EXP, device=DEVICE) * 0.3   # converged, low-var
        h_exp_offpolicy = torch.randn(n//2, D_EXP, device=DEVICE) * 1.5   # large, uncertain
        h_exp_all = torch.cat([h_exp_onpolicy, h_exp_offpolicy], dim=0)
        labels = torch.tensor([0]*(n//2) + [1]*(n//2))  # 0=on, 1=off

        h_cheap_all = h_proj(h_exp_all)

        # Sample z from expensive prior
        z_exp = exp_rssm.sample_z(h_exp_all)
        z_cheap = cheap_rssm.sample_z(h_cheap_all)

        # Value from each model
        v_exp   = exp_rssm.value_fn(h_exp_all, z_exp)
        v_cheap = cheap_rssm.value_fn(h_cheap_all, z_cheap)

        # Normalize gap by value scale
        v_scale = v_exp.abs().mean().clamp(min=0.1)
        gap = (v_cheap - v_exp).abs() / v_scale  # normalized gap

        gap_onpolicy  = gap[:n//2]
        gap_offpolicy = gap[n//2:]

        print(f"    On-policy  states: mean gap = {gap_onpolicy.mean().item():.4f}, "
              f"frac < τ({VALUE_GAP_THRESHOLD}) = {(gap_onpolicy < VALUE_GAP_THRESHOLD).float().mean().item():.3f}")
        print(f"    Off-policy states: mean gap = {gap_offpolicy.mean().item():.4f}, "
              f"frac < τ({VALUE_GAP_THRESHOLD}) = {(gap_offpolicy < VALUE_GAP_THRESHOLD).float().mean().item():.3f}")

        cheap_routing_rate = (gap_onpolicy < VALUE_GAP_THRESHOLD).float().mean().item()
        cheap_works = cheap_routing_rate > 0.6
        print(f"    Cheap routing rate on on-policy states: {cheap_routing_rate:.3f} (need >0.6)")

    return gap, gap_onpolicy, gap_offpolicy, cheap_routing_rate

# ─── 2. Train gating network ──────────────────────────────────────────────────

def train_gating(gate, exp_rssm, cheap_rssm, h_proj, n=N_TRAIN, epochs=15):
    """Train gating network to predict value-gap from h_exp alone."""
    exp_rssm.eval(); cheap_rssm.eval()
    optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3)

    with torch.no_grad():
        h_exp = torch.randn(n, D_EXP, device=DEVICE)
        h_exp[:n//2] *= 0.3   # on-policy
        h_exp[n//2:] *= 1.5   # off-policy
        h_cheap = h_proj(h_exp)
        z_exp   = exp_rssm.sample_z(h_exp)
        z_cheap = cheap_rssm.sample_z(h_cheap)
        v_exp   = exp_rssm.value_fn(h_exp, z_exp)
        v_cheap = cheap_rssm.value_fn(h_cheap, z_cheap)
        v_scale = v_exp.abs().mean().clamp(min=0.1)
        gap_target = (v_cheap - v_exp).abs() / v_scale

    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0; nb = 0
        for start in range(0, n, 512):
            idx = perm[start:start+512]
            pred = gate(h_exp[idx].detach())
            loss = (pred - gap_target[idx].detach()).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item(); nb += 1
        losses.append(epoch_loss / nb)
    return losses, gap_target

# ─── 3. Quality proxy: gated rollout vs full expensive rollout ────────────────

def measure_gated_quality(exp_rssm, cheap_rssm, h_proj, gate, n=1000, h_steps=H):
    """
    Run H-step rollout in two modes:
    - Full: always expensive RSSM
    - Gated: cheap if predicted_gap < τ, else expensive
    Quality proxy = |mean_return_gated - mean_return_full| / |mean_return_full|
    """
    exp_rssm.eval(); cheap_rssm.eval(); gate.eval()
    with torch.no_grad():
        h_init = torch.randn(n, D_EXP, device=DEVICE) * 0.5  # mixed states

        # ─ Full rollout ─
        h_full = h_init.clone()
        G_full = torch.zeros(n, device=DEVICE)
        disc = 1.0
        for t in range(h_steps):
            z = exp_rssm.sample_z(h_full)
            a = torch.randn(n, A, device=DEVICE)
            r = exp_rssm.reward_fn(h_full, z)
            G_full += disc * r
            disc *= GAMMA
            h_full = exp_rssm(h_full, z, a)

        # ─ Gated rollout ─
        h_gated = h_init.clone()
        h_cheap = h_proj(h_gated)
        G_gated = torch.zeros(n, device=DEVICE)
        cheap_count = 0
        disc = 1.0
        for t in range(h_steps):
            # Predict gap: use gate on current expensive h (or init h for cheap states)
            predicted_gap = gate(h_gated)
            use_cheap = (predicted_gap < VALUE_GAP_THRESHOLD)  # (n,) bool
            cheap_count += use_cheap.float().sum().item()

            # Sample z per model
            z_exp_g   = exp_rssm.sample_z(h_gated)
            z_cheap_g = cheap_rssm.sample_z(h_cheap)

            # Reward from whichever model is selected
            r_exp_g   = exp_rssm.reward_fn(h_gated, z_exp_g)
            r_cheap_g = cheap_rssm.reward_fn(h_cheap, z_cheap_g)
            r_gated = torch.where(use_cheap, r_cheap_g, r_exp_g)
            G_gated += disc * r_gated

            # Advance state
            h_exp_new   = exp_rssm(h_gated, z_exp_g, a=torch.randn(n, A, device=DEVICE))
            h_cheap_new = cheap_rssm(h_cheap, z_cheap_g, a=torch.randn(n, A, device=DEVICE))
            # Sync h_gated: update both, route per sample
            h_gated = torch.where(use_cheap.unsqueeze(-1), h_exp_new, h_exp_new)  # always keep exp h for gating
            h_cheap = torch.where(use_cheap.unsqueeze(-1), h_cheap_new, h_proj(h_exp_new))
            disc *= GAMMA

        cheap_routing_rate = cheap_count / (n * h_steps)
        print(f"    Cheap routing rate: {cheap_routing_rate:.3f} ({int(cheap_count)}/{n*h_steps} steps)")

        # Quality proxy
        G_full_mean   = G_full.mean().item()
        G_gated_mean  = G_gated.mean().item()
        quality_delta = abs(G_gated_mean - G_full_mean) / (abs(G_full_mean) + 1e-6)

        print(f"    Full return:  {G_full_mean:.4f}")
        print(f"    Gated return: {G_gated_mean:.4f}")
        print(f"    quality_proxy_delta: {quality_delta:.4f} (gate <0.05): {'PASS' if quality_delta < 0.05 else 'FAIL'}")

    return quality_delta, cheap_routing_rate

# ─── 4. Speed benchmark ───────────────────────────────────────────────────────

def benchmark_speed(exp_rssm, cheap_rssm, h_proj, gate, batch=B, h_steps=H, n_reps=N_REPS):
    """
    Full: all B×H steps with expensive RSSM.
    Gated: cheap_frac×B×H cheap + (1-cheap_frac)×B×H expensive + B gating overhead.
    """
    exp_rssm.eval(); cheap_rssm.eval(); gate.eval()

    # Warmup
    h = torch.randn(batch, D_EXP, device=DEVICE)
    for _ in range(5):
        z = exp_rssm.sample_z(h)
        a = torch.randn(batch, A, device=DEVICE)
        h = exp_rssm(h, z, a)
    if DEVICE == "cuda": torch.cuda.synchronize()

    # Full expensive rollout
    t0 = time.perf_counter()
    for _ in range(n_reps):
        h = torch.randn(batch, D_EXP, device=DEVICE)
        with torch.no_grad():
            for t in range(h_steps):
                z = exp_rssm.sample_z(h)
                a = torch.randn(batch, A, device=DEVICE)
                _ = exp_rssm.reward_fn(h, z)
                _ = exp_rssm.value_fn(h, z)
                h = exp_rssm(h, z, a)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t_full = time.perf_counter() - t0

    # Gated rollout (simulate 70% cheap routing)
    cheap_frac = 0.70
    n_cheap = int(batch * cheap_frac)
    n_exp   = batch - n_cheap
    t0 = time.perf_counter()
    for _ in range(n_reps):
        h_e = torch.randn(n_exp, D_EXP, device=DEVICE)
        h_c = torch.randn(n_cheap, D_CHEAP, device=DEVICE)
        with torch.no_grad():
            # Gating overhead: run gate on all batch
            h_all = torch.randn(batch, D_EXP, device=DEVICE)
            _ = gate(h_all)  # gating cost
            for t in range(h_steps):
                # Cheap path
                z_c = cheap_rssm.sample_z(h_c)
                a_c = torch.randn(n_cheap, A, device=DEVICE)
                _ = cheap_rssm.reward_fn(h_c, z_c)
                _ = cheap_rssm.value_fn(h_c, z_c)
                h_c = cheap_rssm(h_c, z_c, a_c)
                # Expensive path
                z_e = exp_rssm.sample_z(h_e)
                a_e = torch.randn(n_exp, A, device=DEVICE)
                _ = exp_rssm.reward_fn(h_e, z_e)
                _ = exp_rssm.value_fn(h_e, z_e)
                h_e = exp_rssm(h_e, z_e, a_e)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t_gated = time.perf_counter() - t0

    speedup = t_full / t_gated
    return t_full, t_gated, speedup, cheap_frac

# ─── 5. Theoretical speedup estimate ─────────────────────────────────────────

def estimate_theoretical_speedup(cheap_frac=0.70, exp_rssm=None, cheap_rssm=None):
    """
    Measure per-step time of expensive vs cheap RSSM to get actual FLOPs ratio.
    """
    exp_rssm.eval(); cheap_rssm.eval()
    h_exp = torch.randn(512, D_EXP, device=DEVICE)
    h_ch  = torch.randn(512, D_CHEAP, device=DEVICE)
    z_exp = exp_rssm.sample_z(h_exp)
    z_ch  = cheap_rssm.sample_z(h_ch)
    a_exp = torch.randn(512, A, device=DEVICE)
    a_ch  = torch.randn(512, A, device=DEVICE)
    n = 500
    # warmup
    for _ in range(10):
        _ = exp_rssm(h_exp, z_exp, a_exp)
        _ = cheap_rssm(h_ch, z_ch, a_ch)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        with torch.no_grad(): _ = exp_rssm(h_exp, z_exp, a_exp)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t_exp_step = (time.perf_counter() - t0) / n

    t0 = time.perf_counter()
    for _ in range(n):
        with torch.no_grad(): _ = cheap_rssm(h_ch, z_ch, a_ch)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t_cheap_step = (time.perf_counter() - t0) / n

    flop_ratio = t_cheap_step / t_exp_step
    theoretical_speedup = 1.0 / (cheap_frac * flop_ratio + (1 - cheap_frac))
    print(f"    Expensive RSSM step: {t_exp_step*1e6:.1f}μs")
    print(f"    Cheap RSSM step:     {t_cheap_step*1e6:.1f}μs")
    print(f"    Cheap/Expensive time ratio: {flop_ratio:.4f}")
    print(f"    Theoretical speedup @ {cheap_frac*100:.0f}% cheap: {theoretical_speedup:.3f}x")
    return flop_ratio, theoretical_speedup

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Policy-Gated Progressive World Model PoC ===\n")

    # Models
    exp_rssm   = ExpensiveRSSM().to(DEVICE)
    cheap_rssm = CheapRSSM().to(DEVICE)
    gate       = GatingNetwork().to(DEVICE)
    h_proj     = nn.Linear(D_EXP, D_CHEAP, bias=False).to(DEVICE)

    # Freeze exp_rssm (pretrained)
    for p in exp_rssm.parameters(): p.requires_grad_(False)
    exp_rssm.eval()

    # ── 0. Theoretical speedup estimate
    print("[0] Estimating actual FLOPs ratio (cheap vs expensive)...")
    flop_ratio, theo_speedup = estimate_theoretical_speedup(0.70, exp_rssm, cheap_rssm)

    # ── 1. Measure value-gap distribution
    print("\n[1] Measuring value-gap: cheap vs expensive model...")
    gap, gap_on, gap_off, cheap_routing_rate = measure_value_gap(exp_rssm, cheap_rssm, h_proj)

    # ── 2. Train gating network
    print("\n[2] Training gating network...")
    gate_losses, gap_target = train_gating(gate, exp_rssm, cheap_rssm, h_proj, epochs=15)
    print(f"    Gating loss: {gate_losses[0]:.4f} → {gate_losses[-1]:.4f}")

    # Gating accuracy
    gate.eval()
    with torch.no_grad():
        h_test = torch.randn(1000, D_EXP, device=DEVICE)
        h_test[:500] *= 0.3; h_test[500:] *= 1.5
        pred_gap = gate(h_test)
        true_gap = gap_target[:1000] if len(gap_target) >= 1000 else gap_target
        # Precision: what fraction of "cheap" decisions are correct?
        pred_cheap = (pred_gap < VALUE_GAP_THRESHOLD)
        corr_g = float(torch.corrcoef(torch.stack([pred_gap[:len(true_gap)], true_gap]))[0, 1])
    print(f"    Gate correlation (pred_gap vs true_gap): {corr_g:.4f}")

    # ── 3. Quality measurement
    print("\n[3] Quality: gated rollout vs full expensive rollout...")
    quality_proxy_delta, actual_cheap_rate = measure_gated_quality(
        exp_rssm, cheap_rssm, h_proj, gate
    )
    quality_pass = quality_proxy_delta < 0.05

    # ── 4. Speed benchmark
    print("\n[4] Speed benchmark...")
    t_full, t_gated, speedup, cheap_frac = benchmark_speed(exp_rssm, cheap_rssm, h_proj, gate)
    print(f"    Full expensive: {t_full*1000:.1f}ms / {N_REPS} reps")
    print(f"    Gated ({int(cheap_frac*100)}% cheap): {t_gated*1000:.1f}ms / {N_REPS} reps")
    print(f"    Speedup: {speedup:.3f}x (gate >1.5x: {'PASS' if speedup > 1.5 else 'FAIL'})")
    speed_pass = speedup > 1.5

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
        "idea": "policy-gated-progressive-wm",
        "verdict": verdict,
        "rollout_speedup": round(speedup, 4),
        "quality_proxy_delta": round(quality_proxy_delta, 4),
        "cheap_routing_rate_onpolicy": round(cheap_routing_rate, 4),
        "actual_cheap_routing_rate": round(actual_cheap_rate, 4),
        "cheap_model_flop_ratio": round(float(flop_ratio), 4),
        "theoretical_speedup_70pct": round(float(theo_speedup), 4),
        "gating_correlation": round(float(corr_g), 4),
        "value_gap_threshold": VALUE_GAP_THRESHOLD,
        "cheap_dim": D_CHEAP,
        "expensive_dim": D_EXP,
        "horizon_H": H,
        "device": DEVICE,
    }

    out_path = Path(__file__).parent / "poc_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")
    print(json.dumps(results, indent=2))
