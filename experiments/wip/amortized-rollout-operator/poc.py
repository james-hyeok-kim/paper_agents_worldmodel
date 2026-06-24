"""
PoC: Amortized Rollout Operator
Gate: rollout_speedup > 1.5x AND quality_proxy_delta < 0.05
Key question: Can sequential H-step GRU calls be replaced by a single
              parallel transformer pass with comparable quality?
Key insight: sequential H calls on GPU = H small kernel launches (bad).
             1 transformer pass = 1 large matmul (GPU-friendly).
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
Z = 32           # stochastic z dim
A = 6            # action dim
H = 15           # imagination horizon
B = 512          # batch (replay batch in DreamerV3)
D_OP = 256       # amortized operator model dim (small transformer)
N_HEADS = 4      # transformer heads
N_LAYERS_OP = 3  # operator transformer layers (lightweight)
N_SEQ = 4000     # training sequences for distillation
N_EPOCHS = 25    # distillation epochs
N_REPS = 100     # benchmark repetitions

# ─── Models ──────────────────────────────────────────────────────────────────

class TeacherRSSM(nn.Module):
    """Full sequential RSSM (teacher = ground truth)."""
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(Z + A, D)
        self.prior = nn.Sequential(
            nn.Linear(D, D//2), nn.ELU(),
            nn.Linear(D//2, Z*2)
        )

    def sample_z(self, h):
        out = self.prior(h)
        mean, log_std = out.chunk(2, dim=-1)
        std = log_std.clamp(-4, 2).exp()
        return mean + std * torch.randn_like(std)

    def step(self, h, z, a):
        inp = torch.cat([z, a], dim=-1)
        return self.gru(inp, h)

    def rollout_sequential(self, h0, policy_fn):
        """H sequential GRU steps — the baseline we want to replace."""
        h = h0
        trajectory = []
        for t in range(H):
            a = policy_fn(h)
            z = self.sample_z(h)
            h = self.step(h, z, a)
            trajectory.append(h)
        return torch.stack(trajectory, dim=1)  # (B, H, D)


class RandomPolicy(nn.Module):
    """Random policy for PoC (in practice this is the learned actor)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D//4), nn.ELU(),
            nn.Linear(D//4, A), nn.Tanh()
        )

    def forward(self, h):
        return self.net(h)


class AmortizedOperator(nn.Module):
    """
    Single-pass operator: h0 → (h1,...,hH) parallel transformer.
    Uses causal self-attention so h_t can attend to h_{0..t-1} context.
    Policy is absorbed via the trajectory distillation.
    """
    def __init__(self):
        super().__init__()
        # Project h0 to operator dim
        self.proj_in  = nn.Linear(D, D_OP)
        self.proj_out = nn.Linear(D_OP, D)
        # Step embedding (time context)
        self.step_embed = nn.Embedding(H + 1, D_OP)
        # Causal transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_OP, nhead=N_HEADS, dim_feedforward=D_OP*4,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS_OP)
        # Causal mask (each position only attends to previous + self)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((H+1, H+1), float('-inf')), diagonal=1)
        )

    def forward(self, h0):
        """
        h0: (B, D) initial latent
        Returns: (B, H, D) predicted trajectory
        """
        B = h0.size(0)
        # Expand h0 as position-0 token
        h0_proj = self.proj_in(h0).unsqueeze(1)   # (B, 1, D_OP)
        # Create H query positions for future steps
        steps = torch.arange(H+1, device=h0.device)  # [0, 1, ..., H]
        step_emb = self.step_embed(steps).unsqueeze(0).expand(B, -1, -1)  # (B, H+1, D_OP)
        # Input: [h0_token, step_1_query, ..., step_H_query]
        x = step_emb.clone()
        x[:, 0, :] = x[:, 0, :] + h0_proj[:, 0, :]  # inject h0 at position 0
        # Causal transformer: each step predicts next latent
        x = self.transformer(x, mask=self.causal_mask)
        # Extract predictions for steps 1..H
        h_pred = self.proj_out(x[:, 1:, :])  # (B, H, D)
        return h_pred


# ─── 1. Generate teacher trajectories ─────────────────────────────────────────

def generate_trajectories(teacher, policy, n=N_SEQ):
    """Generate H-step closed-loop GRU trajectories for distillation."""
    teacher.eval(); policy.eval()
    all_h0 = []
    all_traj = []
    with torch.no_grad():
        for start in range(0, n, 512):
            batch_n = min(512, n - start)
            h0 = torch.randn(batch_n, D, device=DEVICE)
            traj = teacher.rollout_sequential(h0, policy)
            all_h0.append(h0)
            all_traj.append(traj)
    H0   = torch.cat(all_h0, dim=0)   # (N, D)
    Traj = torch.cat(all_traj, dim=0) # (N, H, D)
    return H0, Traj

# ─── 2. Train amortized operator ──────────────────────────────────────────────

def train_operator(operator, H0, Traj, epochs=N_EPOCHS):
    """Distill sequential rollout into single-pass operator."""
    optimizer = torch.optim.Adam(operator.parameters(), lr=1e-3)
    N = H0.size(0)
    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(N)
        ep_loss = 0; nb = 0
        for start in range(0, N, 256):
            idx = perm[start:start+256]
            h0_b = H0[idx]
            traj_b = Traj[idx]                    # (B, H, D) teacher
            pred_b = operator(h0_b)               # (B, H, D) student
            # MSE loss in normalized space
            loss = (pred_b - traj_b).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item(); nb += 1
        losses.append(ep_loss / nb)
    return losses

# ─── 3. Quality: trajectory MSE ───────────────────────────────────────────────

def measure_quality(teacher, operator, policy, n=1000):
    """Compare operator trajectory vs teacher sequential rollout."""
    teacher.eval(); operator.eval(); policy.eval()
    with torch.no_grad():
        h0 = torch.randn(n, D, device=DEVICE)
        traj_teacher = teacher.rollout_sequential(h0, policy)  # (n, H, D)
        traj_operator = operator(h0)                            # (n, H, D)
        # Normalized MSE
        ref_norm = traj_teacher.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        mse_per_step = ((traj_operator - traj_teacher) / ref_norm).pow(2).mean(dim=-1)  # (n, H)
        mse_mean = mse_per_step.mean().item()
        mse_final = mse_per_step[:, -1].mean().item()  # last-step error (accumulates)
    return mse_mean, mse_final, mse_per_step.mean(dim=0).cpu().numpy()

# ─── 4. GPU latency benchmark ─────────────────────────────────────────────────

def benchmark_latency(teacher, operator, policy, batch=B, n_reps=N_REPS):
    """
    Core question: H sequential GRU calls vs 1 transformer pass on GPU.
    On GPU: small matmuls with H kernel launches vs 1 large batched forward.
    """
    teacher.eval(); operator.eval(); policy.eval()

    # Warmup
    h = torch.randn(batch, D, device=DEVICE)
    for _ in range(10):
        with torch.no_grad():
            traj = teacher.rollout_sequential(h, policy)
            pred = operator(h)
    if DEVICE == "cuda": torch.cuda.synchronize()

    # Sequential H GRU calls (teacher)
    t0 = time.perf_counter()
    for _ in range(n_reps):
        h = torch.randn(batch, D, device=DEVICE)
        with torch.no_grad():
            _ = teacher.rollout_sequential(h, policy)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t_sequential = time.perf_counter() - t0

    # Single operator pass
    t0 = time.perf_counter()
    for _ in range(n_reps):
        h = torch.randn(batch, D, device=DEVICE)
        with torch.no_grad():
            _ = operator(h)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t_operator = time.perf_counter() - t0

    speedup = t_sequential / t_operator
    return t_sequential, t_operator, speedup

# ─── 5. Policy drift test ─────────────────────────────────────────────────────

def test_policy_drift(teacher, operator, n=500):
    """
    Simulate policy update: slightly perturb policy weights.
    Test how much operator quality degrades before re-distillation.
    """
    teacher.eval(); operator.eval()

    # Policy 1: original
    policy1 = RandomPolicy().to(DEVICE)
    policy1.eval()
    with torch.no_grad():
        h0 = torch.randn(n, D, device=DEVICE)
        traj1 = teacher.rollout_sequential(h0, policy1)
        pred1 = operator(h0)
        mse1 = ((pred1 - traj1) / traj1.norm(dim=-1, keepdim=True).clamp(min=1e-6)).pow(2).mean().item()

    # Policy 2: perturbed (simulate policy update)
    policy2 = RandomPolicy().to(DEVICE)
    policy2.eval()  # different random weights
    with torch.no_grad():
        traj2 = teacher.rollout_sequential(h0, policy2)
        pred_unchanged = operator(h0)  # operator not retrained
        mse_drift = ((pred_unchanged - traj2) / traj2.norm(dim=-1, keepdim=True).clamp(min=1e-6)).pow(2).mean().item()

    print(f"    MSE with original policy: {mse1:.4f}")
    print(f"    MSE with new policy (no retrain): {mse_drift:.4f}")
    print(f"    Policy drift degradation: {mse_drift/max(mse1,1e-8):.2f}x")
    return mse1, mse_drift

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Amortized Rollout Operator PoC ===\n")

    teacher  = TeacherRSSM().to(DEVICE)
    policy   = RandomPolicy().to(DEVICE)
    operator = AmortizedOperator().to(DEVICE)

    # Teacher is pretrained (random but fixed)
    for p in teacher.parameters(): p.requires_grad_(False)
    for p in policy.parameters():  p.requires_grad_(False)
    teacher.eval(); policy.eval()

    # ── 1. Latency benchmark (BEFORE training, just checking architecture)
    print("[1] GPU latency: sequential H GRU vs operator (untrained)...")
    t_seq, t_op, speedup_raw = benchmark_latency(teacher, operator, policy)
    print(f"    Sequential {H} GRU calls: {t_seq*1000:.2f}ms / {N_REPS} reps = {N_REPS*B/(t_seq):.0f} states/s")
    print(f"    Operator 1 pass:          {t_op*1000:.2f}ms / {N_REPS} reps = {N_REPS*B/(t_op):.0f} states/s")
    print(f"    Raw speedup (arch-only):  {speedup_raw:.3f}x")

    # ── 2. Generate teacher trajectories
    print(f"\n[2] Generating {N_SEQ} teacher trajectories...")
    H0, Traj = generate_trajectories(teacher, policy)
    print(f"    H0 shape: {H0.shape}, Traj shape: {Traj.shape}")

    # Normalize trajectories for training stability
    traj_std = Traj.std().clamp(min=1e-6)

    # ── 3. Train operator
    print(f"\n[3] Training amortized operator ({N_EPOCHS} epochs)...")
    losses = train_operator(operator, H0, Traj)
    print(f"    Loss: {losses[0]:.5f} → {losses[-1]:.5f}")

    # ── 4. Quality measurement
    print(f"\n[4] Quality: operator vs sequential teacher...")
    mse_mean, mse_final, mse_per_step = measure_quality(teacher, operator, policy)
    print(f"    Mean normalized MSE (all steps): {mse_mean:.5f}")
    print(f"    Final-step normalized MSE:        {mse_final:.5f}")
    print(f"    Per-step MSE: {[f'{x:.4f}' for x in mse_per_step[::3]]}")  # every 3rd step
    quality_proxy_delta = mse_mean
    quality_pass = quality_proxy_delta < 0.05
    print(f"    quality_proxy_delta: {quality_proxy_delta:.5f} (gate <0.05): {'PASS' if quality_pass else 'FAIL'}")

    # ── 5. Final latency benchmark (trained operator)
    print(f"\n[5] Latency benchmark (trained operator)...")
    t_seq2, t_op2, speedup_final = benchmark_latency(teacher, operator, policy)
    print(f"    Sequential {H} GRU calls: {t_seq2*1000:.2f}ms")
    print(f"    Trained operator 1 pass:  {t_op2*1000:.2f}ms")
    print(f"    Final speedup: {speedup_final:.3f}x")
    speed_pass = speedup_final > 1.5
    print(f"    Gate >1.5x: {'PASS' if speed_pass else 'FAIL'}")

    # ── 6. Policy drift test
    print(f"\n[6] Policy drift test...")
    mse_orig, mse_drift = test_policy_drift(teacher, operator)
    drift_factor = mse_drift / max(mse_orig, 1e-8)
    print(f"    Drift factor: {drift_factor:.2f}x")

    # ── Verdict
    verdict = "CONDITIONAL-GO" if (speed_pass and quality_pass) else "FAIL"
    fail_reasons = []
    if not speed_pass: fail_reasons.append(f"speedup {speedup_final:.2f}x < 1.5x")
    if not quality_pass: fail_reasons.append(f"quality_proxy {quality_proxy_delta:.5f} >= 0.05")

    print(f"\n=== VERDICT: {verdict} ===")
    if fail_reasons:
        for r in fail_reasons: print(f"  FAIL: {r}")

    # ── Save
    results = {
        "idea": "amortized-rollout-operator",
        "verdict": verdict,
        "rollout_speedup": round(speedup_final, 4),
        "quality_proxy_delta": round(quality_proxy_delta, 5),
        "raw_speedup_untrained": round(speedup_raw, 4),
        "mse_mean_normalized": round(mse_mean, 5),
        "mse_final_step": round(mse_final, 5),
        "policy_drift_factor": round(float(drift_factor), 3),
        "sequential_ms_per_batch": round(t_seq2*1000/N_REPS, 3),
        "operator_ms_per_batch": round(t_op2*1000/N_REPS, 3),
        "operator_layers": N_LAYERS_OP,
        "operator_dim": D_OP,
        "horizon_H": H,
        "batch_B": B,
        "device": DEVICE,
    }

    out_path = Path(__file__).parent / "poc_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")
    print(json.dumps(results, indent=2))
