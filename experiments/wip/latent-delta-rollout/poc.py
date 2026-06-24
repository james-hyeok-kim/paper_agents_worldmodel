"""
latent-delta-rollout PoC
========================
핵심 가설: RSSM deterministic state h의 step-to-step delta가 작을 때
GRU update를 skip하고 h carry-forward하면 rollout을 1.5x+ 가속,
quality_proxy_delta < 0.05 유지 가능한가?

설계 원칙 (advisor 검토 반영):
  - time.sleep 대신 실제 GRU matmul 연산으로 속도 측정
  - regime sweep (delta_scale): dense→sparse 범위를 외부 파라미터로 sweep
  - τ (skip threshold) sweep: 하나의 τ가 speedup과 quality를 동시에 결정
  - predictor 비용 (2-layer linear probe) speedup에 포함
  - quality proxy: h accumulation drift MSE → PSNR drop 환산
  - skip 오류 전파: wrong-skip 시 delta를 zero로 처리하여 누적 오차 반영
  - z-diversity: secondary diagnostic (entfopy under frozen vs live h)

게이트 기준:
  - rollout_speedup > 1.5x AND quality_proxy_delta < 0.05
  - regime sweep 중 2개 이상에서 어떤 τ가 동시에 통과 → CONDITIONAL-GO
  - 실패 시 어떤 가정이 깨졌는지 명시
"""

import time
import json
import statistics
import random
import numpy as np
import math
from pathlib import Path

np.random.seed(42)
random.seed(42)

# ─── 하이퍼파라미터 ───────────────────────────────────────────────
H_DIM = 256        # RSSM deterministic h 차원 (toy)
Z_DIM = 32         # stochastic z 차원
ACT_DIM = 4        # action 차원
N_EPISODES = 50
HORIZON = 15
PRED_HIDDEN = 64   # delta-predictor hidden dim

# regime sweep: step-to-step delta의 평균 크기 조절
DELTA_SCALES = [0.01, 0.05, 0.1, 0.2, 0.5]

# τ sweep: skip if predicted ||Δh|| < τ
TAU_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

# ─── 장난감 RSSM (실제 GRU matmul 사용) ─────────────────────────────

class ToyGRU:
    """GRU 핵심 연산을 numpy로 구현 (실제 matmul 비용 반영)."""
    def __init__(self, h_dim, input_dim):
        s = 1.0 / math.sqrt(h_dim)
        # Reset, update, new gate weights
        self.Wr = np.random.randn(h_dim, input_dim + h_dim).astype(np.float32) * s
        self.Wz = np.random.randn(h_dim, input_dim + h_dim).astype(np.float32) * s
        self.Wn = np.random.randn(h_dim, input_dim + h_dim).astype(np.float32) * s
        self.br = np.zeros(h_dim, dtype=np.float32)
        self.bz = np.zeros(h_dim, dtype=np.float32)
        self.bn = np.zeros(h_dim, dtype=np.float32)

    def step(self, h, x):
        """실제 GRU forward pass (numpy matmul)."""
        xh = np.concatenate([x, h])
        r = 1.0 / (1.0 + np.exp(-(self.Wr @ xh + self.br)))
        z = 1.0 / (1.0 + np.exp(-(self.Wz @ xh + self.bz)))
        xh_r = np.concatenate([x, r * h])
        n = np.tanh(self.Wn @ xh_r + self.bn)
        h_new = (1 - z) * n + z * h
        return h_new


class ToyRSSM:
    """Toy RSSM: deterministic h (GRU) + stochastic z (prior head)."""
    def __init__(self, h_dim=H_DIM, z_dim=Z_DIM, act_dim=ACT_DIM):
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.gru = ToyGRU(h_dim, z_dim + act_dim)
        # prior head: h -> z_mean, z_logvar
        s = 1.0 / math.sqrt(h_dim)
        self.W_prior_mean = np.random.randn(z_dim, h_dim).astype(np.float32) * s
        self.W_prior_logv = np.random.randn(z_dim, h_dim).astype(np.float32) * s
        # synthetic decoder proxy: h+z -> obs_recon
        self.W_dec = np.random.randn(h_dim, h_dim + z_dim).astype(np.float32) * s

    def prior_step(self, h, z, action):
        """GRU update: h' = GRU(h, [z, action])"""
        x = np.concatenate([z, action]).astype(np.float32)
        return self.gru.step(h, x)

    def sample_z(self, h):
        """z ~ N(mu(h), sigma(h)) — prior"""
        mu = self.W_prior_mean @ h
        logv = np.clip(self.W_prior_logv @ h, -4.0, 4.0)
        eps = np.random.randn(self.z_dim).astype(np.float32)
        z = mu + np.exp(0.5 * logv) * eps
        return z, mu, logv

    def decode(self, h, z):
        """Lightweight reconstruction proxy."""
        hz = np.concatenate([h, z]).astype(np.float32)
        return self.W_dec @ hz

    def z_entropy(self, logvar):
        """Gaussian entropy from logvar."""
        return 0.5 * np.sum(1 + logvar)


class DeltaPredictor:
    """
    2-layer linear probe: [h_prev, action] -> predicted ||Δh||
    비용이 speedup에 포함됨.
    """
    def __init__(self, h_dim=H_DIM, act_dim=ACT_DIM, hidden=PRED_HIDDEN):
        s = 1.0 / math.sqrt(h_dim + act_dim)
        self.W1 = np.random.randn(hidden, h_dim + act_dim).astype(np.float32) * s
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = np.random.randn(1, hidden).astype(np.float32) * s
        self.b2 = np.zeros(1, dtype=np.float32)

    def forward(self, h, action):
        x = np.concatenate([h, action]).astype(np.float32)
        hidden = np.maximum(0, self.W1 @ x + self.b1)   # ReLU
        out = float((self.W2 @ hidden + self.b2).item())
        return max(out, 0.0)

    def fit(self, X, y, lr=0.01, epochs=200):
        """SGD로 ||Δh|| 예측 학습."""
        n = len(y)
        for _ in range(epochs):
            idx = random.randint(0, n - 1)
            xi, yi = X[idx], y[idx]
            # forward
            h1 = np.maximum(0, self.W1 @ xi + self.b1)
            pred = float((self.W2 @ h1 + self.b2).item())
            err = pred - yi
            # backward
            dW2 = err * h1.reshape(1, -1)
            dh1 = err * self.W2.flatten() * (h1 > 0)
            dW1 = np.outer(dh1, xi)
            # update
            self.W1 -= lr * dW1
            self.b1 -= lr * dh1
            self.W2 -= lr * dW2
            self.b2 -= lr * np.array([err])


# ─── 데이터 생성: 한 번만 ─────────────────────────────────────────

def generate_trajectory(rssm, delta_scale, horizon=HORIZON):
    """
    Piecewise dynamics: delta_scale 이 sparse/dense 정도를 조절.
    크게 변하는 구간(transition)과 안정 구간(plateau)이 섞임.
    """
    h = np.random.randn(rssm.h_dim).astype(np.float32) * 0.1
    h = h / (np.linalg.norm(h) + 1e-8)

    traj = []  # (h_t, z_t, action_t, delta_norm_t, recon_t)
    for t in range(horizon):
        action = np.random.randn(ACT_DIM).astype(np.float32)
        z, mu, logv = rssm.sample_z(h)
        h_prev = h.copy()
        h_new = rssm.prior_step(h, z, action)

        # delta_scale로 h 변화량 조절: 실제 환경에서 settled state 시뮬레이션
        # plateau 확률 (1-delta_scale): 낮은 delta_scale → 많은 plateau
        if np.random.rand() < (1.0 - min(delta_scale * 2, 1.0)):
            # carry-forward (plateau): h_new ≈ h_prev
            h_new = h_prev + np.random.randn(rssm.h_dim).astype(np.float32) * delta_scale * 0.1
        # else: full transition, keep h_new as-is

        delta_norm = float(np.linalg.norm(h_new - h_prev))
        recon = rssm.decode(h_new, z)
        entropy = rssm.z_entropy(logv)

        traj.append({
            "h_prev": h_prev,
            "h_new": h_new,
            "z": z,
            "action": action,
            "delta_norm": delta_norm,
            "recon": recon,
            "entropy": entropy,
        })
        h = h_new

    return traj


# ─── predictor 학습 데이터 수집 ──────────────────────────────────

def collect_training_data(rssm, delta_scale, n=500, horizon=HORIZON):
    X, y = [], []
    for _ in range(n // horizon + 1):
        traj = generate_trajectory(rssm, delta_scale, horizon)
        for step in traj:
            x = np.concatenate([step["h_prev"], step["action"]])
            X.append(x.astype(np.float32))
            y.append(step["delta_norm"])
    return np.array(X[:n]), np.array(y[:n], dtype=np.float32)


# ─── 롤아웃 벤치마크 ─────────────────────────────────────────────

def run_baseline_rollout(rssm, trajectories):
    """
    Baseline: 모든 step에서 GRU를 실행 (skip 없음).
    반환: (latency_ms, gru_calls, recon_mse=0, z_entropies)
    """
    gru_calls = 0
    latency_ms = 0.0
    z_entropies = []
    recons = []

    for traj in trajectories:
        h = traj[0]["h_prev"].copy()
        for step in traj:
            action = step["action"]
            t0 = time.perf_counter()
            z, _, logv = rssm.sample_z(h)
            h = rssm.prior_step(h, z, action)
            elapsed = (time.perf_counter() - t0) * 1000
            latency_ms += elapsed
            gru_calls += 1
            recon = rssm.decode(h, z)
            recons.append(recon.copy())
            z_entropies.append(rssm.z_entropy(logv))

    return {
        "latency_ms": latency_ms,
        "gru_calls": gru_calls,
        "recons": np.array(recons),
        "z_entropies": z_entropies,
    }


def run_modified_rollout(rssm, predictor, trajectories, tau):
    """
    Modified: predictor로 ||Δh|| 예측 후 τ 미만이면 GRU skip.
    predictor 비용은 항상 포함 (skip 여부와 무관하게 predict 호출).
    skip 시 h carry-forward, z는 frozen h에서 재샘플.
    skip error 전파: 실제 delta를 0으로 처리하여 누적 drift 발생.
    """
    gru_calls = 0
    skip_count = 0
    pred_calls = 0
    latency_ms = 0.0
    z_entropies = []
    recons = []

    for traj in trajectories:
        h = traj[0]["h_prev"].copy()
        for step in traj:
            action = step["action"]

            # predictor 비용 항상 포함
            t0 = time.perf_counter()
            pred_delta = predictor.forward(h, action)
            pred_elapsed = (time.perf_counter() - t0) * 1000
            pred_calls += 1

            if pred_delta < tau:
                # SKIP: h carry-forward
                t1 = time.perf_counter()
                z, _, logv = rssm.sample_z(h)  # frozen h에서 z 샘플
                skip_elapsed = (time.perf_counter() - t1) * 1000
                skip_count += 1
                latency_ms += pred_elapsed + skip_elapsed
            else:
                # FULL GRU update
                t1 = time.perf_counter()
                z, _, logv = rssm.sample_z(h)
                h = rssm.prior_step(h, z, action)
                gru_elapsed = (time.perf_counter() - t1) * 1000
                gru_calls += 1
                latency_ms += pred_elapsed + gru_elapsed

            recon = rssm.decode(h, z)
            recons.append(recon.copy())
            z_entropies.append(rssm.z_entropy(logv))

    return {
        "latency_ms": latency_ms,
        "gru_calls": gru_calls,
        "skip_count": skip_count,
        "pred_calls": pred_calls,
        "recons": np.array(recons),
        "z_entropies": z_entropies,
    }


def mse_to_psnr_drop(mse_baseline, mse_modified, signal_range=2.0):
    """
    PSNR drop 계산 (proxy).
    PSNR = 10 * log10(signal_range^2 / MSE)
    """
    eps = 1e-10
    psnr_base = 10 * math.log10(signal_range**2 / (mse_baseline + eps))
    psnr_mod  = 10 * math.log10(signal_range**2 / (mse_modified + eps))
    return psnr_base - psnr_mod  # positive = drop


def compute_recon_mse(recons_base, recons_mod):
    """reconstruction 간 평균 MSE."""
    diff = recons_base - recons_mod
    return float(np.mean(diff**2))


def predictor_vs_trivial_baseline(predictor, X_test, y_test):
    """
    predictor가 trivial baseline (mean predictor) 보다 나은지 검사.
    Trivial: 항상 mean(y_train) 예측.
    """
    preds = np.array([predictor.forward(X_test[i, :H_DIM], X_test[i, H_DIM:]) for i in range(len(X_test))])
    pred_mse = float(np.mean((preds - y_test) ** 2))
    trivial_pred = np.full_like(y_test, np.mean(y_test))
    trivial_mse = float(np.mean((trivial_pred - y_test) ** 2))
    return pred_mse, trivial_mse, pred_mse < trivial_mse


# ─── 메인 실험 ────────────────────────────────────────────────────

def run_experiment():
    print("=== latent-delta-rollout PoC ===")
    print(f"h_dim={H_DIM}, z_dim={Z_DIM}, episodes={N_EPISODES}, horizon={HORIZON}")
    print(f"delta_scales={DELTA_SCALES}")
    print(f"tau_values={TAU_VALUES}")
    print()

    rssm = ToyRSSM()
    results = {}
    gate_pass_count = 0
    regime_results = []

    for delta_scale in DELTA_SCALES:
        print(f"--- Regime: delta_scale={delta_scale} ---")

        # 데이터 생성 (predictor 학습용 + 벤치마크용)
        print("  Generating training data for predictor...", flush=True)
        X_train, y_train = collect_training_data(rssm, delta_scale, n=1000)
        X_test,  y_test  = collect_training_data(rssm, delta_scale, n=200)

        # delta 분포 분석
        delta_mean = float(np.mean(y_train))
        delta_std  = float(np.std(y_train))
        delta_p50  = float(np.percentile(y_train, 50))
        delta_p90  = float(np.percentile(y_train, 90))
        print(f"  Delta distribution: mean={delta_mean:.4f}, std={delta_std:.4f}, "
              f"p50={delta_p50:.4f}, p90={delta_p90:.4f}")

        # predictor 학습
        print("  Training delta predictor...", flush=True)
        predictor = DeltaPredictor()
        predictor.fit(X_train, y_train, lr=0.005, epochs=2000)
        pred_mse, trivial_mse, pred_beats_trivial = predictor_vs_trivial_baseline(predictor, X_test, y_test)
        print(f"  Predictor: MSE={pred_mse:.6f}, Trivial MSE={trivial_mse:.6f}, "
              f"Beats trivial: {pred_beats_trivial}")

        # 벤치마크 trajectory 생성 (고정)
        print("  Generating benchmark trajectories...", flush=True)
        bench_trajectories = [generate_trajectory(rssm, delta_scale) for _ in range(N_EPISODES)]

        # Baseline 실행
        print("  Running baseline rollout...", flush=True)
        base_result = run_baseline_rollout(rssm, bench_trajectories)
        baseline_mse_ref = float(np.mean(base_result["recons"]**2))

        # τ sweep
        tau_sweep = []
        best_tau_result = None
        print(f"  Sweeping τ={TAU_VALUES}...")

        for tau in TAU_VALUES:
            mod_result = run_modified_rollout(rssm, predictor, bench_trajectories, tau)

            total_steps = N_EPISODES * HORIZON
            skip_rate = mod_result["skip_count"] / total_steps

            # speedup: baseline_latency / modified_latency
            speedup = base_result["latency_ms"] / (mod_result["latency_ms"] + 1e-9)

            # quality_proxy_delta: reconstruction MSE 증가율
            recon_mse_diff = compute_recon_mse(base_result["recons"], mod_result["recons"])
            quality_proxy_delta = recon_mse_diff / (baseline_mse_ref + 1e-8)

            # PSNR drop
            mod_mse = float(np.mean(mod_result["recons"]**2))
            psnr_drop = mse_to_psnr_drop(baseline_mse_ref, mod_mse)

            # z entropy ratio
            base_ent = statistics.mean(base_result["z_entropies"])
            mod_ent  = statistics.mean(mod_result["z_entropies"])
            z_entropy_ratio = mod_ent / (base_ent + 1e-8)

            gate_speed_pass = speedup > 1.5
            gate_quality_pass = quality_proxy_delta < 0.05
            gate_pass = gate_speed_pass and gate_quality_pass

            tau_entry = {
                "tau": tau,
                "skip_rate": round(skip_rate, 4),
                "gru_calls": mod_result["gru_calls"],
                "skip_count": mod_result["skip_count"],
                "speedup": round(speedup, 4),
                "quality_proxy_delta": round(quality_proxy_delta, 6),
                "psnr_drop_db": round(psnr_drop, 4),
                "z_entropy_ratio": round(z_entropy_ratio, 4),
                "gate_speed_pass": gate_speed_pass,
                "gate_quality_pass": gate_quality_pass,
                "gate_pass": gate_pass,
            }
            tau_sweep.append(tau_entry)
            print(f"    τ={tau:.3f}: skip={skip_rate:.2%}, speedup={speedup:.2f}x, "
                  f"q_delta={quality_proxy_delta:.4f}, psnr_drop={psnr_drop:.2f}dB, "
                  f"gate={'PASS' if gate_pass else 'FAIL'}")

        # 이 regime에서 통과 τ 존재하는가?
        passing_taus = [t for t in tau_sweep if t["gate_pass"]]
        regime_pass = len(passing_taus) > 0

        if regime_pass:
            gate_pass_count += 1
            best_tau_result = passing_taus[0]  # 가장 낮은 τ (가장 공격적인 skip)

        regime_entry = {
            "delta_scale": delta_scale,
            "delta_mean": round(delta_mean, 6),
            "delta_std": round(delta_std, 6),
            "delta_p50": round(delta_p50, 6),
            "delta_p90": round(delta_p90, 6),
            "baseline_latency_ms": round(base_result["latency_ms"], 2),
            "baseline_gru_calls": base_result["gru_calls"],
            "pred_mse": round(pred_mse, 8),
            "trivial_mse": round(trivial_mse, 8),
            "pred_beats_trivial": pred_beats_trivial,
            "tau_sweep": tau_sweep,
            "regime_pass": regime_pass,
            "best_tau": best_tau_result,
            "passing_tau_count": len(passing_taus),
        }
        regime_results.append(regime_entry)
        print(f"  Regime {'PASS' if regime_pass else 'FAIL'} "
              f"({'게이트 통과 τ 존재' if regime_pass else '모든 τ에서 실패'})\n")

    # ─── 최종 판정 ────────────────────────────────────────────────
    # 기준: regime sweep 중 2개 이상에서 통과 τ 존재 → CONDITIONAL-GO
    verdict = "CONDITIONAL-GO" if gate_pass_count >= 2 else "FAIL"

    # predictor 신뢰도 (대부분 regime에서 trivial 이겨야 함)
    pred_beats_count = sum(1 for r in regime_results if r["pred_beats_trivial"])
    predictor_reliable = pred_beats_count >= 3  # 5개 중 3개 이상

    # 가정 위반 분석
    broken_assumptions = []
    if gate_pass_count < 2:
        sparse_regimes = [r for r in regime_results if r["delta_mean"] < 0.1]
        if all(not r["regime_pass"] for r in sparse_regimes):
            broken_assumptions.append(
                "RSSM h delta가 sparse해도 skip으로 인한 누적 drift가 quality를 크게 훼손 (quality gate 실패)"
            )
        dense_regimes = [r for r in regime_results if r["delta_mean"] >= 0.1]
        if all(not r["regime_pass"] for r in dense_regimes):
            broken_assumptions.append(
                "Dense regime에서는 skip 비율이 너무 낮아 speedup 1.5x 달성 불가 (speedup gate 실패)"
            )
        if not predictor_reliable:
            broken_assumptions.append(
                "Delta predictor가 trivial baseline (mean predictor)보다 나쁨 — predictor 비용이 speedup을 잠식"
            )

    # 요약 통계
    best_overall = None
    for r in regime_results:
        for t in r["tau_sweep"]:
            if t["gate_pass"]:
                if best_overall is None or t["speedup"] > best_overall["speedup"]:
                    best_overall = {**t, "delta_scale": r["delta_scale"]}

    summary = {
        "verdict": verdict,
        "gate_pass_count_of_5": gate_pass_count,
        "gate_required": 2,
        "predictor_reliable": predictor_reliable,
        "pred_beats_trivial_count_of_5": pred_beats_count,
        "broken_assumptions": broken_assumptions,
        "best_overall_gate_pass": best_overall,
        "regimes": regime_results,
    }

    # 결과 출력
    print("\n" + "="*60)
    print(f"최종 판정: {verdict}")
    print(f"게이트 통과 regime 수: {gate_pass_count}/5 (기준: 2 이상)")
    print(f"Predictor reliable: {predictor_reliable} ({pred_beats_count}/5 regime에서 trivial 이김)")
    if best_overall:
        print(f"Best gate pass: delta_scale={best_overall['delta_scale']}, "
              f"tau={best_overall['tau']}, speedup={best_overall['speedup']}x, "
              f"q_delta={best_overall['quality_proxy_delta']}")
    if broken_assumptions:
        print("\n위반된 가정:")
        for ba in broken_assumptions:
            print(f"  - {ba}")
    print("="*60)

    return summary


# ─── 실행 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    t_start = time.perf_counter()
    summary = run_experiment()
    elapsed_total = (time.perf_counter() - t_start) / 60

    summary["total_runtime_min"] = round(elapsed_total, 2)
    print(f"\n총 실행 시간: {elapsed_total:.2f}분")

    out_path = Path(__file__).parent / "poc_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"결과 저장: {out_path}")
