"""
branch-shared-imagination: Synthetic PoC for KL-based merge tree in MPPI planning.

핵심 질문: MPPI에서 같은 초기 state + 다른 action sequence를 가진 N개 분기가
초기 몇 step 동안 latent space에서 수렴(KL이 작은)하는가?
그리고 merge 적용 시 forward 절감이 return 추정 오차를 허용 범위 내에서 이끌어내는가?

설계 원칙 (advisor review 반영):
1. tau sweep → Pareto frontier에서 두 게이트 동시 만족 여부 판정
2. action_scale/noise_scale ratio 3개 regime sensitivity
3. speedup = forward_count 비율 (sleep 타이밍 아님)
4. quality_proxy = return estimation error (MPPI weighted return 기반)
5. non-degenerate planning 전제 검증 (return std across branches 확인)
"""

import json
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import itertools

# ─── 재현성 ──────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ─── 설정 ─────────────────────────────────────────────────────────────────────
N_BRANCHES = 100
HORIZON = 10
LATENT_DIM = 64
ACTION_DIM = 4
N_EPISODES = 50
TEMPERATURE = 1.0  # MPPI softmax temperature
TAU_SWEEP = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

# Regime: (action_scale, noise_scale) — advisor 지적 반영
REGIMES = {
    "low_action"   : (0.1, 1.0),   # action 영향 작음 — degenerate 가능성
    "balanced"     : (0.5, 0.5),   # 균형 잡힌 regime
    "high_action"  : (1.0, 0.1),   # action 영향 큼 — non-trivial planning
}

# ─── Toy Stochastic World Model (RSSM-like, without NN) ───────────────────────

@dataclass
class ToyRSSM:
    """
    z_{t+1} = A @ z_t + B @ a_t + eps,   eps ~ N(0, noise_scale^2 I)
    reward   = -||z_t||^2 * reward_scale + r_noise
    """
    latent_dim: int
    action_dim: int
    action_scale: float
    noise_scale: float
    reward_scale: float = 0.1
    forward_count: int = 0

    def __post_init__(self):
        rng = np.random.RandomState(SEED)
        # A: stable dynamics (eigenvalues < 1)
        A_raw = rng.randn(self.latent_dim, self.latent_dim) * 0.1
        # make spectral radius < 0.9
        eigvals = np.linalg.eigvals(A_raw)
        sr = np.max(np.abs(eigvals))
        self.A = A_raw / (sr / 0.9) if sr > 0 else A_raw
        # B: action projection
        self.B = rng.randn(self.latent_dim, self.action_dim) * self.action_scale

    def step(self, z: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Returns (z_next_mean, reward, z_next_sample).
        z_next_mean: deterministic part (used for KL estimation)
        z_next_sample: actual sample (adds stochasticity)
        """
        self.forward_count += 1
        mean = self.A @ z + self.B @ a
        eps = np.random.randn(self.latent_dim) * self.noise_scale
        z_next = mean + eps
        reward = float(-np.dot(z, z) * self.reward_scale + np.random.randn() * 0.01)
        return mean, reward, z_next

    def reset_counter(self):
        self.forward_count = 0


# ─── KL divergence between two Gaussians with same (diagonal) variance ────────

def kl_divergence_same_var(mu1: np.ndarray, mu2: np.ndarray, var: float) -> float:
    """KL(N(mu1, varI) || N(mu2, varI)) = ||mu1 - mu2||^2 / (2 * var)"""
    return float(np.sum((mu1 - mu2) ** 2) / (2.0 * var + 1e-8))


# ─── Baseline: 완전 독립 rollout ──────────────────────────────────────────────

def baseline_rollout(
    wm: ToyRSSM, z0: np.ndarray, actions_per_branch: np.ndarray
) -> Tuple[int, np.ndarray]:
    """
    N개 분기를 완전히 독립적으로 rollout.
    actions_per_branch: (N, horizon, action_dim)
    Returns: (forward_count, returns: shape (N,))
    """
    wm.reset_counter()
    N = actions_per_branch.shape[0]
    returns = np.zeros(N)
    for i in range(N):
        z = z0.copy()
        total_reward = 0.0
        for h in range(HORIZON):
            a = actions_per_branch[i, h]
            _, r, z = wm.step(z, a)
            total_reward += r
        returns[i] = total_reward
    return wm.forward_count, returns


# ─── Modified: KL-based merge tree rollout ────────────────────────────────────

def merged_rollout(
    wm: ToyRSSM,
    z0: np.ndarray,
    actions_per_branch: np.ndarray,
    tau: float,
) -> Tuple[int, np.ndarray, Dict]:
    """
    KL < tau인 분기를 같은 그룹으로 merge, 그룹의 대표 분기만 forward.
    나머지 분기는 대표 분기의 latent를 공유.

    Returns: (forward_count, returns_approx: shape (N,), diagnostics)
    """
    wm.reset_counter()
    N = actions_per_branch.shape[0]
    var = wm.noise_scale ** 2

    # 각 분기의 현재 mean latent (결정론적 부분)
    z_means = np.tile(z0, (N, 1))  # (N, latent_dim)
    z_samples = np.tile(z0, (N, 1))  # (N, latent_dim)

    returns = np.zeros(N)
    total_unique_nodes_per_step = []
    merge_rates_per_step = []

    for h in range(HORIZON):
        # ── step 1: merge 그룹 결정 (pairwise KL on means) ──
        # O(N^2) overhead를 PoC에서는 직접 계산 (real impl은 nearest-neighbor approx 사용)
        group_of = list(range(N))  # group_of[i] = representative index
        for i in range(N):
            for j in range(i):
                kl = kl_divergence_same_var(z_means[i], z_means[j], var)
                if kl < tau and group_of[j] == j:
                    group_of[i] = j
                    break  # 첫 번째 merge 가능한 대표에 연결

        # unique representatives
        reps = list(set(group_of))
        n_unique = len(reps)
        merge_rate = 1.0 - n_unique / N
        total_unique_nodes_per_step.append(n_unique)
        merge_rates_per_step.append(merge_rate)

        # ── step 2: 대표 분기만 forward ──
        new_z_means = np.zeros_like(z_means)
        new_z_samples = np.zeros_like(z_samples)
        rep_mean_map = {}
        rep_sample_map = {}

        for rep in reps:
            a = actions_per_branch[rep, h]
            mean_next, r, z_next = wm.step(z_samples[rep], a)
            rep_mean_map[rep] = mean_next
            rep_sample_map[rep] = z_next
            returns[rep] += r

        # ── step 3: non-rep 분기는 대표의 latent 복사 + 자신의 action offset ──
        for i in range(N):
            rep = group_of[i]
            if i == rep:
                new_z_means[i] = rep_mean_map[rep]
                new_z_samples[i] = rep_sample_map[rep]
            else:
                # representative latent에 action 차이로 인한 mean offset만 적용
                a_i = actions_per_branch[i, h]
                a_rep = actions_per_branch[rep, h]
                delta_a = a_i - a_rep
                delta_z = wm.B @ delta_a  # linear action contribution difference
                new_z_means[i] = rep_mean_map[rep] + delta_z
                new_z_samples[i] = rep_sample_map[rep] + delta_z
                # reward 근사 (representative reward 그대로)
                returns[i] += returns[rep] - sum(
                    returns[rep] for _ in range(1)  # placeholder — use rep reward
                )
                # 실제로는 representative의 마지막 reward step만 재활용
                # returns[i]에 이미 반영됨 (아래에서 수정)

        # reward correction: non-rep의 returns는 대표의 reward + delta_z 기반 reward 근사
        for i in range(N):
            rep = group_of[i]
            if i != rep:
                # z_approx에서 reward 재계산 (no forward count)
                z_approx = new_z_samples[i]
                r_approx = float(-np.dot(z_approx, z_approx) * wm.reward_scale
                                 + np.random.randn() * 0.01)
                returns[i] += r_approx

        z_means = new_z_means
        z_samples = new_z_samples

    diagnostics = {
        "unique_nodes_per_step": total_unique_nodes_per_step,
        "merge_rates_per_step": merge_rates_per_step,
        "mean_unique": float(np.mean(total_unique_nodes_per_step)),
        "mean_merge_rate": float(np.mean(merge_rates_per_step)),
    }
    return wm.forward_count, returns, diagnostics


# ─── MPPI return estimation quality ───────────────────────────────────────────

def mppi_weighted_return(returns: np.ndarray, temperature: float) -> float:
    """MPPI softmax weighted sum of returns."""
    weights = np.exp(returns / (temperature + 1e-8))
    weights /= weights.sum() + 1e-8
    return float(np.dot(weights, returns))


def top_k_action_overlap(
    returns_base: np.ndarray, returns_mod: np.ndarray, k: int = 10
) -> float:
    """Top-k 분기 선택 overlap (quality indicator)."""
    top_base = set(np.argsort(returns_base)[-k:])
    top_mod = set(np.argsort(returns_mod)[-k:])
    return len(top_base & top_mod) / k


# ─── 비선형성 검증: return spread ─────────────────────────────────────────────

def check_non_degenerate(returns: np.ndarray, threshold: float = 0.1) -> bool:
    """분기별 return std가 의미 있는지 확인 (degenerate setup 방지)."""
    normalized_std = np.std(returns) / (abs(np.mean(returns)) + 1e-6)
    return bool(normalized_std > threshold)


# ─── 메인 실험 루프 ───────────────────────────────────────────────────────────

def run_experiment(regime_name: str, action_scale: float, noise_scale: float) -> Dict:
    print(f"\n{'='*60}")
    print(f"Regime: {regime_name}  (action_scale={action_scale}, noise_scale={noise_scale})")
    print(f"{'='*60}")

    rng = np.random.RandomState(SEED)

    # ── 1. non-degenerate 검증 ──
    wm_check = ToyRSSM(LATENT_DIM, ACTION_DIM, action_scale, noise_scale)
    z0_check = rng.randn(LATENT_DIM)
    actions_check = rng.randn(N_BRANCHES, HORIZON, ACTION_DIM)
    _, returns_check = baseline_rollout(wm_check, z0_check, actions_check)
    is_non_degenerate = check_non_degenerate(returns_check)
    return_std = float(np.std(returns_check))
    normalized_std = float(np.std(returns_check) / (abs(np.mean(returns_check)) + 1e-6))
    print(f"  Non-degenerate check: {is_non_degenerate} "
          f"(return_std={return_std:.4f}, normalized_std={normalized_std:.4f})")
    if not is_non_degenerate:
        print(f"  WARNING: This regime may be degenerate — action has little effect")

    # ── 2. tau sweep 실험 ──
    tau_results = {}

    for tau in TAU_SWEEP:
        speedups = []
        quality_deltas = []
        top_k_overlaps = []
        diag_list = []

        for ep in range(N_EPISODES):
            rng_ep = np.random.RandomState(SEED + ep)
            z0 = rng_ep.randn(LATENT_DIM)
            actions = rng_ep.randn(N_BRANCHES, HORIZON, ACTION_DIM)

            wm_base = ToyRSSM(LATENT_DIM, ACTION_DIM, action_scale, noise_scale)
            wm_mod = ToyRSSM(LATENT_DIM, ACTION_DIM, action_scale, noise_scale)
            # 같은 A, B 행렬 공유
            wm_mod.A = wm_base.A.copy()
            wm_mod.B = wm_base.B.copy()

            fc_base, returns_base = baseline_rollout(wm_base, z0, actions)
            fc_mod, returns_mod, diag = merged_rollout(wm_mod, z0, actions, tau)

            speedup = fc_base / max(fc_mod, 1)
            speedups.append(speedup)

            # MPPI quality: weighted return error
            mppi_base = mppi_weighted_return(returns_base, TEMPERATURE)
            mppi_mod = mppi_weighted_return(returns_mod, TEMPERATURE)
            rel_err = abs(mppi_base - mppi_mod) / (abs(mppi_base) + 1e-6)
            quality_deltas.append(rel_err)

            overlap = top_k_action_overlap(returns_base, returns_mod, k=10)
            top_k_overlaps.append(overlap)
            diag_list.append(diag)

        mean_speedup = float(np.mean(speedups))
        mean_quality_delta = float(np.mean(quality_deltas))
        mean_overlap = float(np.mean(top_k_overlaps))
        mean_merge_rate = float(np.mean([d["mean_merge_rate"] for d in diag_list]))
        mean_unique = float(np.mean([d["mean_unique"] for d in diag_list]))

        gate_speedup = mean_speedup > 1.5
        gate_quality = mean_quality_delta < 0.05
        both_pass = gate_speedup and gate_quality

        tau_results[tau] = {
            "tau": tau,
            "mean_speedup": mean_speedup,
            "mean_quality_delta": mean_quality_delta,
            "mean_top_k_overlap": mean_overlap,
            "mean_merge_rate": mean_merge_rate,
            "mean_unique_nodes": mean_unique,
            "gate_speedup_pass": gate_speedup,
            "gate_quality_pass": gate_quality,
            "both_gates_pass": both_pass,
        }
        print(f"  tau={tau:.2f} | speedup={mean_speedup:.3f}x | "
              f"q_delta={mean_quality_delta:.4f} | "
              f"merge_rate={mean_merge_rate:.3f} | "
              f"speedup_gate={'PASS' if gate_speedup else 'FAIL'} | "
              f"quality_gate={'PASS' if gate_quality else 'FAIL'} | "
              f"{'** BOTH **' if both_pass else ''}")

    # Pareto: tau where both gates pass
    passing_taus = [v for v in tau_results.values() if v["both_gates_pass"]]
    has_valid_operating_point = len(passing_taus) > 0

    # best tau: highest speedup among quality-passing taus
    quality_passing = [v for v in tau_results.values() if v["gate_quality_pass"]]
    if quality_passing:
        best_by_speedup = max(quality_passing, key=lambda x: x["mean_speedup"])
    else:
        best_by_speedup = None

    regime_result = {
        "regime": regime_name,
        "action_scale": action_scale,
        "noise_scale": noise_scale,
        "is_non_degenerate": is_non_degenerate,
        "return_std": return_std,
        "normalized_return_std": normalized_std,
        "tau_sweep": tau_results,
        "has_valid_operating_point": has_valid_operating_point,
        "n_passing_taus": len(passing_taus),
        "best_operating_point": best_by_speedup,
        "passing_taus": [v["tau"] for v in passing_taus],
    }

    if has_valid_operating_point:
        print(f"\n  RESULT: Valid operating points exist at tau={[v['tau'] for v in passing_taus]}")
    else:
        print(f"\n  RESULT: No single tau satisfies both gates simultaneously")

    return regime_result


# ─── 전체 실험 실행 ───────────────────────────────────────────────────────────

def main():
    all_results = {}
    for regime_name, (action_scale, noise_scale) in REGIMES.items():
        result = run_experiment(regime_name, action_scale, noise_scale)
        all_results[regime_name] = result

    # ── 판정 로직 ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL VERDICT ANALYSIS")
    print("="*60)

    # 기준: balanced 이상 non-trivial regime에서 valid operating point 존재
    non_trivial_regimes = [
        all_results["balanced"],
        all_results["high_action"],
    ]
    trivial_regime = all_results["low_action"]

    # 조건 1: non-trivial regime 중 하나 이상에서 두 게이트 동시 통과
    non_trivial_pass = any(r["has_valid_operating_point"] for r in non_trivial_regimes)

    # 조건 2: 통과가 degenerate regime에서만 나오는지 확인
    only_trivial_pass = (
        trivial_regime["has_valid_operating_point"] and not non_trivial_pass
    )

    # 조건 3: non-degenerate 확인
    balanced_non_degen = all_results["balanced"]["is_non_degenerate"]
    high_action_non_degen = all_results["high_action"]["is_non_degenerate"]

    # 최종 판정
    if non_trivial_pass and (balanced_non_degen or high_action_non_degen):
        verdict = "CONDITIONAL-GO"
        verdict_reason = (
            "Non-trivial planning regime에서 두 게이트(rollout_speedup > 1.5×, "
            "quality_delta < 0.05)를 동시에 만족하는 tau가 존재함. "
            "단, speedup은 forward-count 상한선 기준이며 pairwise KL overhead 미포함."
        )
    elif only_trivial_pass:
        verdict = "FAIL"
        verdict_reason = (
            "두 게이트 동시 통과가 degenerate(low_action) regime에서만 발생. "
            "action이 latent에 실질적 영향을 주는 환경에서는 merge로 인한 오차가 gate를 초과."
        )
    else:
        verdict = "FAIL"
        verdict_reason = (
            "어떤 tau에서도 rollout_speedup > 1.5× 와 quality_delta < 0.05를 동시에 만족하지 못함. "
            "KL 기반 merge는 충분한 속도 향상을 만들기에 분기 수렴이 부족하거나, "
            "merge 시 return 추정 오차가 허용 범위를 초과함."
        )

    # 대표 수치 (balanced regime, best operating point 기준)
    balanced = all_results["balanced"]
    if balanced["best_operating_point"]:
        rep_speedup = balanced["best_operating_point"]["mean_speedup"]
        rep_quality = balanced["best_operating_point"]["mean_quality_delta"]
        rep_tau = balanced["best_operating_point"]["tau"]
    elif balanced["tau_sweep"]:
        # quality gate 통과 없으면 speedup 가장 높은 tau 사용
        best_tau_data = max(balanced["tau_sweep"].values(), key=lambda x: x["mean_speedup"])
        rep_speedup = best_tau_data["mean_speedup"]
        rep_quality = best_tau_data["mean_quality_delta"]
        rep_tau = best_tau_data["tau"]
    else:
        rep_speedup, rep_quality, rep_tau = 0, 0, 0

    summary = {
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "config": {
            "n_branches": N_BRANCHES,
            "horizon": HORIZON,
            "latent_dim": LATENT_DIM,
            "action_dim": ACTION_DIM,
            "n_episodes": N_EPISODES,
            "tau_sweep": TAU_SWEEP,
        },
        "gate_criteria": {
            "rollout_speedup_threshold": 1.5,
            "quality_proxy_delta_threshold": 0.05,
        },
        "representative_numbers": {
            "regime": "balanced",
            "tau": rep_tau,
            "rollout_speedup": round(rep_speedup, 3),
            "quality_proxy_delta": round(rep_quality, 4),
        },
        "regime_verdicts": {
            r: {
                "is_non_degenerate": all_results[r]["is_non_degenerate"],
                "has_valid_operating_point": all_results[r]["has_valid_operating_point"],
                "passing_taus": all_results[r]["passing_taus"],
                "n_passing_taus": all_results[r]["n_passing_taus"],
            }
            for r in REGIMES
        },
        "key_assumptions": [
            "toy RSSM uses linear dynamics (A, B matrices); real RSSM is nonlinear",
            "action-offset correction is linear (B @ delta_a); true correction is nonlinear",
            "pairwise KL overhead O(N^2 * latent_dim) is NOT counted in speedup",
            "noise is isotropic Gaussian; real RSSM has non-isotropic posterior",
            "KL convergence in early steps is ASSUMED for toy model — real RSSM requires empirical validation",
        ],
        "full_regime_results": all_results,
    }

    print(f"\nVERDICT: {verdict}")
    print(f"Reason: {verdict_reason}")
    print(f"\nRepresentative numbers (balanced regime, tau={rep_tau}):")
    print(f"  rollout_speedup   = {rep_speedup:.3f}x  (gate: >1.5x)")
    print(f"  quality_delta     = {rep_quality:.4f}  (gate: <0.05)")

    return summary


if __name__ == "__main__":
    result = main()

    out_path = "/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/branch-shared-imagination/poc_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
