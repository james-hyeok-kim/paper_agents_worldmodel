"""
Predictive Horizon Controller — Synthetic PoC
==============================================
아이디어: RSSM prior entropy 변화율(ΔH)과 reward-head 분산을 입력으로 받는
경량 horizon-controller가 imagination rollout을 조기 종료시켜
1.5×+ 가속과 <5% quality 하락을 동시에 달성할 수 있는가?

Gate 기준:
  - rollout_speedup > 1.5×   (step-count 기준, 같은 per-step cost 가정)
  - quality_proxy_delta < 0.05

advisor 지적 반영:
  1. post-plateau에서도 실제 return이 있는 에피소드 30~40% 존재
     → quality gate에 실제 패널티 부여 (over-truncation이 실제로 costly)
  2. fixed-horizon baseline 추가 (learned controller vs trivial early cut)
  3. speedup = baseline_steps / modified_steps (steps/sec 아님)
  4. quality_proxy_delta = |E[return_err]| / std(full_return)  (near-zero mean 회피)
"""

import json
import time
import random
import math
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ──────────────────────────────────────────────
# 0. 설정
# ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

STATE_DIM   = 16
ACTION_DIM  = 4
HORIZON     = 15
GAMMA       = 0.99

# 에피소드 유형 비율
# Type A (65%): plateau 후 reward ≈ 0  → 조기 종료 safe
# Type B (35%): plateau 후에도 significat return 존재 → 조기 종료 costly
FRAC_B = 0.35

N_TRAIN     = 2000   # controller 학습용 에피소드
N_EVAL      = 500    # 평가용 에피소드

# ──────────────────────────────────────────────
# 1. Toy RSSM 시뮬레이터
# ──────────────────────────────────────────────

def simulate_episode(episode_type: str, horizon: int = HORIZON):
    """
    Returns per-step signals (entropy, reward_var, reward) for one episode.

    episode_type:
      'A' — prior entropy plateaus at t=5~8, post-plateau reward ≈ N(0, 0.1)
      'B' — prior entropy plateaus at t=5~8, but post-plateau reward follows
             a nonzero drift (e.g. +0.5 per step) representing genuine future value
    """
    plateau_t = random.randint(5, 8)
    entropies = []
    reward_vars = []
    rewards = []

    # 초기 entropy (높음)
    H = 3.0 + np.random.randn() * 0.2

    for t in range(horizon):
        # Prior entropy 시뮬레이션
        if t < plateau_t:
            # 감소 단계: 일정 비율로 drop
            dH = -0.4 * H + np.random.randn() * 0.05
            H = max(H + dH, 0.3)
        else:
            # Plateau: 소량 노이즈만
            dH = np.random.randn() * 0.03
            H = max(H + dH, 0.2)
        entropies.append(float(H))

        # Reward-head 분산 (entropy와 유사하게 감소 후 plateau)
        rv = max(1.0 * math.exp(-0.3 * t) + np.random.randn() * 0.05, 0.02)
        reward_vars.append(float(rv))

        # 실제 reward
        if episode_type == 'A':
            r = np.random.randn() * 0.1  # post-plateau: near-zero
        else:  # type B
            if t < plateau_t:
                r = np.random.randn() * 0.2
            else:
                # post-plateau: 유의미한 양의 return (조기 종료 시 손해)
                r = 0.5 + np.random.randn() * 0.15
        rewards.append(float(r))

    return entropies, reward_vars, rewards, plateau_t


def compute_discounted_return(rewards, gamma=GAMMA):
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
    return G


# ──────────────────────────────────────────────
# 2. Controller: 2-layer MLP
#    Input:  [ΔH, reward_var, t/horizon]  (dim=3)
#    Output: stop_logit (scalar) — sigmoid → stop probability
# ──────────────────────────────────────────────

class HorizonController(nn.Module):
    def __init__(self, input_dim=3, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))  # stop probability


# ──────────────────────────────────────────────
# 3. Controller 학습
#    타깃: full-horizon return의 "안전 절사" 여부를 회귀
#    plateau 도달 여부 + episode type으로 ground-truth 생성
# ──────────────────────────────────────────────

def build_training_data(n=N_TRAIN):
    """
    각 (episode, step) 쌍에 대해 (feature, should_stop) 생성.
    should_stop = 1 if 남은 return이 full_return의 5% 미만 (= 조기 종료 safe)
    """
    X, y = [], []
    for _ in range(n):
        ep_type = 'B' if random.random() < FRAC_B else 'A'
        H_seq, rv_seq, r_seq, plateau_t = simulate_episode(ep_type)
        full_return = compute_discounted_return(r_seq)
        return_std_proxy = abs(full_return) + 1e-3  # 정규화용

        for t in range(1, HORIZON):
            dH = H_seq[t] - H_seq[t-1]
            rv = rv_seq[t]
            t_norm = t / HORIZON

            feat = [dH, rv, t_norm]

            # 나머지 return
            remaining_return = compute_discounted_return(r_seq[t:]) * (GAMMA ** t)
            # 조기 종료가 "safe"한지: 남은 return이 미미할 때
            safe_to_stop = 1.0 if abs(remaining_return) / return_std_proxy < 0.05 else 0.0

            X.append(feat)
            y.append(safe_to_stop)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return X, y


def train_controller(controller, X, y, epochs=30, lr=1e-3):
    optimizer = optim.Adam(controller.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    dataset = torch.utils.data.TensorDataset(X, y)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            pred = controller(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

    return controller


# ──────────────────────────────────────────────
# 4. 평가: 3가지 variant
#    (a) Baseline    — 항상 full horizon (t=15)
#    (b) Learned     — controller stop_prob > 0.5 시 조기 종료
#    (c) Fixed-cut   — 항상 t=floor(HORIZON/2)에서 종료 (trivial baseline)
# ──────────────────────────────────────────────

FIXED_CUT_STEP = HORIZON // 2  # = 7


def evaluate_variants(controller, n=N_EVAL):
    """
    Returns per-episode metrics for all three variants.
    quality_proxy = |bootstrap_return - full_return| / std(full_returns)
    """
    results_baseline = []
    results_learned  = []
    results_fixed    = []

    full_returns_all = []

    episodes = []
    for _ in range(n):
        ep_type = 'B' if random.random() < FRAC_B else 'A'
        H_seq, rv_seq, r_seq, plateau_t = simulate_episode(ep_type)
        full_return = compute_discounted_return(r_seq)
        full_returns_all.append(full_return)
        episodes.append((H_seq, rv_seq, r_seq, full_return))

    full_return_std = statistics.stdev(full_returns_all) + 1e-6

    controller.eval()
    with torch.no_grad():
        for H_seq, rv_seq, r_seq, full_return in episodes:
            # ── (a) Baseline ──
            results_baseline.append({
                "steps":       HORIZON,
                "full_return": full_return,
                "used_return": full_return,
                "return_err":  0.0,
            })

            # ── (b) Learned controller ──
            stopped_at = HORIZON
            for t in range(1, HORIZON):
                dH    = H_seq[t] - H_seq[t-1]
                rv    = rv_seq[t]
                t_norm = t / HORIZON
                feat  = torch.tensor([[dH, rv, t_norm]], dtype=torch.float32)
                p_stop = controller(feat).item()
                if p_stop > 0.5:
                    stopped_at = t
                    break
            partial_return = compute_discounted_return(r_seq[:stopped_at])
            results_learned.append({
                "steps":       stopped_at,
                "full_return": full_return,
                "used_return": partial_return,
                "return_err":  abs(partial_return - full_return) / full_return_std,
            })

            # ── (c) Fixed-cut ──
            partial_fixed = compute_discounted_return(r_seq[:FIXED_CUT_STEP])
            results_fixed.append({
                "steps":       FIXED_CUT_STEP,
                "full_return": full_return,
                "used_return": partial_fixed,
                "return_err":  abs(partial_fixed - full_return) / full_return_std,
            })

    return results_baseline, results_learned, results_fixed, full_return_std


def summarize(results, name):
    mean_steps  = statistics.mean(r["steps"] for r in results)
    mean_err    = statistics.mean(r["return_err"] for r in results)
    return {"name": name, "mean_steps": round(mean_steps, 3), "mean_return_err": round(mean_err, 4)}


# ──────────────────────────────────────────────
# 5. 메인
# ──────────────────────────────────────────────

def main():
    t_start = time.perf_counter()

    print("=== Predictive Horizon Controller PoC ===")
    print(f"env: horizon={HORIZON}, state_dim={STATE_DIM}, frac_B={FRAC_B}")
    print(f"     N_TRAIN={N_TRAIN}, N_EVAL={N_EVAL}")
    print()

    # 학습 데이터 생성
    print("[1/3] Generating training data...")
    X, y = build_training_data(N_TRAIN)
    pos_frac = y.mean().item()
    print(f"      samples={len(X)}, positive_frac(safe_to_stop)={pos_frac:.3f}")

    # Controller 학습
    print("[2/3] Training HorizonController...")
    controller = HorizonController(input_dim=3, hidden=32)
    n_params   = sum(p.numel() for p in controller.parameters())
    print(f"      params={n_params}")
    controller = train_controller(controller, X, y, epochs=30)

    # 평가
    print("[3/3] Evaluating variants...")
    res_base, res_learned, res_fixed, full_return_std = evaluate_variants(controller, N_EVAL)

    s_base    = summarize(res_base,    "baseline")
    s_learned = summarize(res_learned, "learned_controller")
    s_fixed   = summarize(res_fixed,   "fixed_cut")

    # ── Gate 지표 계산 ──
    speedup_learned = round(s_base["mean_steps"] / s_learned["mean_steps"], 3)
    speedup_fixed   = round(s_base["mean_steps"] / s_fixed["mean_steps"], 3)

    # quality_proxy_delta = normalized return error (mean over episodes)
    # 낮을수록 good
    quality_learned = s_learned["mean_return_err"]
    quality_fixed   = s_fixed["mean_return_err"]

    gate_speedup_pass = speedup_learned > 1.5
    gate_quality_pass = quality_learned < 0.05

    if gate_speedup_pass and gate_quality_pass:
        verdict = "CONDITIONAL-GO"
    else:
        verdict = "FAIL"

    elapsed = round(time.perf_counter() - t_start, 1)

    # ── 출력 ──
    print()
    print("─" * 55)
    print("Results")
    print("─" * 55)
    print(f"{'Variant':<22} {'Mean steps':>11} {'Return err':>11} {'Speedup':>8}")
    print(f"{'Baseline':<22} {s_base['mean_steps']:>11.2f} {s_base['mean_return_err']:>11.4f} {'1.000':>8}")
    print(f"{'Learned controller':<22} {s_learned['mean_steps']:>11.2f} {s_learned['mean_return_err']:>11.4f} {speedup_learned:>8.3f}x")
    print(f"{'Fixed cut (H/2)':<22} {s_fixed['mean_steps']:>11.2f} {s_fixed['mean_return_err']:>11.4f} {speedup_fixed:>8.3f}x")
    print()
    print(f"Gate 1 — rollout_speedup > 1.50x  : {speedup_learned:.3f}x  {'PASS' if gate_speedup_pass else 'FAIL'}")
    print(f"Gate 2 — quality_proxy_delta < 0.05: {quality_learned:.4f}   {'PASS' if gate_quality_pass else 'FAIL'}")
    print()
    print(f"Verdict: {verdict}")
    print(f"Elapsed: {elapsed}s")

    # ── JSON 저장 ──
    out = {
        "slug": "predictive-horizon-controller",
        "verdict": verdict,
        "elapsed_sec": elapsed,
        "config": {
            "horizon": HORIZON,
            "state_dim": STATE_DIM,
            "frac_B_episodes": FRAC_B,
            "n_train": N_TRAIN,
            "n_eval": N_EVAL,
            "fixed_cut_step": FIXED_CUT_STEP,
        },
        "variants": {
            "baseline":           s_base,
            "learned_controller": s_learned,
            "fixed_cut":          s_fixed,
        },
        "gate_results": {
            "rollout_speedup": {
                "learned": speedup_learned,
                "fixed":   speedup_fixed,
                "threshold": 1.50,
                "pass": gate_speedup_pass,
            },
            "quality_proxy_delta": {
                "learned": quality_learned,
                "fixed":   quality_fixed,
                "threshold": 0.05,
                "pass": gate_quality_pass,
            },
        },
        "full_return_std": round(full_return_std, 4),
    }

    out_path = Path(__file__).parent / "poc_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to: {out_path}")
    return out


if __name__ == "__main__":
    main()
