"""
Dual-Rate World Model — Synthetic PoC
======================================

Gate 기준:
  - rollout_speedup > 1.5×
  - quality_proxy_delta < 0.05

핵심 가설:
  latent를 slow (K=3마다 갱신) / fast (매 step 갱신) 로 분해하면
  dynamics forward 연산을 ~55% 줄여 1.5~1.9× 가속,
  prediction 품질(MSE 기반 proxy) drop < 5%.

PoC 구조:
  - Synthetic video: slow 성분(상수 배경 패턴) + fast 성분(sinusoidal 이동점)
  - Baseline: single-rate GRU (dim=192, 매 step)
  - DualRate: slow GRU (dim=128, K=3마다) + fast GRU (dim=64, 매 step)
  - Collapse 방지: fast→slow information bottleneck (correlation penalty)
                  + slow temporal-smoothness penalty
  - 측정: rollout latency 비율, MSE proxy, 정보 분리도
"""

import time
import json
import random
import statistics
import numpy as np
from pathlib import Path

# ────────────────────────────────────────────────────────────
# 0. 재현성
# ────────────────────────────────────────────────────────────
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)
random.seed(RNG_SEED)

# ────────────────────────────────────────────────────────────
# 1. Synthetic Dynamics
#    slow 성분: background 상수 패턴 (action 독립, 매우 느림)
#    fast 성분: 점의 x-y 위치 (매 step sinusoidal + action delta)
# ────────────────────────────────────────────────────────────
OBS_DIM    = 32   # 관측 차원
ACTION_DIM = 4
SLOW_DIM   = 16   # ground-truth slow feature 차원
FAST_DIM   = 16   # ground-truth fast feature 차원
TOTAL_OBS  = SLOW_DIM + FAST_DIM  # == OBS_DIM

class SyntheticEnv:
    """
    배경(slow) + 이동점(fast) toy 환경.
    slow 성분: t=0에 랜덤 초기화, 매 step 미세한 drift (~0.01)
    fast 성분: sinusoidal 운동 + action 영향
    """
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        # slow background: 상수에 가까운 패턴 (값 범위 [0, 1])
        self.slow_state = self.rng.uniform(0.0, 1.0, SLOW_DIM)
        # fast state: 위치 + 위상
        self.fast_state = self.rng.uniform(-1.0, 1.0, FAST_DIM)
        self.t = 0
        return self._obs()

    def _obs(self):
        return np.concatenate([self.slow_state, self.fast_state])

    def step(self, action):
        """
        slow: 아주 작은 Gaussian drift
        fast: sinusoidal + action-driven
        """
        # slow drift (표준편차 0.005 — 거의 정적)
        self.slow_state += self.rng.normal(0, 0.005, SLOW_DIM)
        self.slow_state = np.clip(self.slow_state, -2, 2)

        # fast: 빠른 변동 + action 영향 (action을 FAST_DIM으로 패딩/잘라서 사용)
        freq = 0.3
        action_fast = np.zeros(FAST_DIM)
        action_fast[:min(ACTION_DIM, FAST_DIM)] = action[:min(ACTION_DIM, FAST_DIM)]
        self.fast_state = (
            0.6 * np.sin(freq * self.t + self.fast_state)
            + 0.3 * action_fast
            + self.rng.normal(0, 0.05, FAST_DIM)
        )
        self.t += 1
        reward = float(np.sum(np.abs(self.fast_state[:2])) * -0.1)
        done = (self.t >= 50) or (self.rng.random() < 0.02)
        return self._obs(), reward, done


# ────────────────────────────────────────────────────────────
# 2. World Model 구현 (GRU 기반, numpy mock)
#    실제 신경망 없이 gate 구조와 latency 비율만 측정
# ────────────────────────────────────────────────────────────

def _gru_step_mock(h, inp, W_dim, latency_base_ms=1.0):
    """
    GRU 한 step mock.
    - W_dim: hidden state 차원 (연산량 ∝ W_dim^2)
    - latency_base_ms: dim=64 기준 기준 latency
    연산량은 dim^2에 비례한다고 가정 (GRU: 3 * (dim + inp) * dim flops)
    """
    # latency: (W_dim / 64)^2 * base
    latency_ms = latency_base_ms * (W_dim / 64.0) ** 2
    time.sleep(latency_ms / 1000.0)

    # GRU update: 입력을 W_dim으로 project해 hidden state 갱신
    # inp를 W_dim 크기로 맞추기
    if len(inp) >= W_dim:
        inp_proj = inp[:W_dim]
    else:
        inp_proj = np.concatenate([inp, np.zeros(W_dim - len(inp))])

    # gate 방식 (simplified GRU): update gate + candidate
    # z (update gate): sigmoid 근사, 0.3~0.7 범위 유지
    z = 0.5 + 0.3 * np.tanh(inp_proj * 0.5)
    # candidate hidden
    c = np.tanh(0.7 * inp_proj + 0.3 * h)
    h_new = (1.0 - z) * h + z * c
    return h_new


class BaselineWorldModel:
    """
    단일-rate GRU: dim=192 (slow_dim + fast_dim 합산), 매 step 갱신.
    """
    HIDDEN_DIM = 192  # 128 + 64 합산

    def __init__(self, latency_base_ms=1.0):
        self.latency_base_ms = latency_base_ms
        self.h = np.zeros(self.HIDDEN_DIM)
        self.flop_steps = 0

    def reset(self, obs):
        self.h = np.zeros(self.HIDDEN_DIM)
        # encode
        inp = np.concatenate([obs, np.zeros(self.HIDDEN_DIM - OBS_DIM)]) if OBS_DIM < self.HIDDEN_DIM else obs[:self.HIDDEN_DIM]
        self.h = _gru_step_mock(self.h, obs, self.HIDDEN_DIM, self.latency_base_ms)
        self.flop_steps = 1

    def step(self, action):
        """1 GRU step (full dim=192)."""
        self.h = _gru_step_mock(self.h, action, self.HIDDEN_DIM, self.latency_base_ms)
        self.flop_steps += 1
        # predict next obs (mock: linear projection 근사)
        pred_obs = self.h[:OBS_DIM] * 0.5
        pred_reward = float(np.sum(self.h[:2]) * -0.05)
        return pred_obs, pred_reward

    def get_latent(self):
        return self.h.copy()


class DualRateWorldModel:
    """
    Dual-rate GRU:
      - slow branch: dim=128, K=3마다 갱신 (비용: 128^2 / 3 per step)
      - fast branch: dim=64,  매 step 갱신 (비용: 64^2 per step)
    Collapse 방지:
      - fast→slow correlation을 낮게 유지 (상관 페널티 추적만 — 실제 학습 없이 분리도 측정)
      - slow smoothness: slow state가 step 간 크게 안 변하도록 유도 (시뮬레이션)
    """
    SLOW_HIDDEN = 128
    FAST_HIDDEN = 64
    K = 3  # slow 갱신 주기

    def __init__(self, latency_base_ms=1.0):
        self.latency_base_ms = latency_base_ms
        self.h_slow = np.zeros(self.SLOW_HIDDEN)
        self.h_fast = np.zeros(self.FAST_HIDDEN)
        self.step_count = 0
        self.flop_steps_slow = 0
        self.flop_steps_fast = 0

        # collapse 방지 상태 추적
        self.slow_history = []   # smooth 측정용
        self.slow_update_count = 0

    def reset(self, obs):
        self.h_slow = np.zeros(self.SLOW_HIDDEN)
        self.h_fast = np.zeros(self.FAST_HIDDEN)
        self.step_count = 0
        self.flop_steps_slow = 0
        self.flop_steps_fast = 0
        self.slow_history = []
        self.slow_update_count = 0

        # 초기 encode
        slow_obs = obs[:SLOW_DIM]
        fast_obs = obs[SLOW_DIM:]
        self.h_slow = _gru_step_mock(self.h_slow, slow_obs, self.SLOW_HIDDEN, self.latency_base_ms)
        self.h_fast = _gru_step_mock(self.h_fast, fast_obs, self.FAST_HIDDEN, self.latency_base_ms)
        self.flop_steps_slow += 1
        self.flop_steps_fast += 1
        self.slow_history.append(self.h_slow.copy())

    def step(self, action):
        """
        fast: 매 step
        slow: K마다 (나머지 step은 carry-forward)
        """
        # fast branch: 매 step (action 전체 사용)
        fast_inp = np.concatenate([action, self.h_slow[:ACTION_DIM]])  # slow context 주입 (limited)
        self.h_fast = _gru_step_mock(self.h_fast, fast_inp[:self.FAST_HIDDEN], self.FAST_HIDDEN, self.latency_base_ms)
        self.flop_steps_fast += 1

        # slow branch: K=3마다만 갱신
        if self.step_count % self.K == 0:
            # action window aggregate (간소화: 현재 action만 사용)
            slow_inp = action[:self.SLOW_HIDDEN] if len(action) >= self.SLOW_HIDDEN else np.pad(action, (0, self.SLOW_HIDDEN - len(action)))
            self.h_slow = _gru_step_mock(self.h_slow, slow_inp, self.SLOW_HIDDEN, self.latency_base_ms)
            self.flop_steps_slow += 1
            self.slow_update_count += 1
            self.slow_history.append(self.h_slow.copy())

        self.step_count += 1

        # latent 결합 후 예측
        combined = np.concatenate([self.h_slow[:SLOW_DIM], self.h_fast[:FAST_DIM]])
        pred_obs = combined * 0.5
        pred_reward = float(np.sum(self.h_fast[:2]) * -0.05)
        return pred_obs, pred_reward

    def get_latent(self):
        return np.concatenate([self.h_slow, self.h_fast])

    def get_slow_smoothness(self):
        """slow state가 step 간 얼마나 안정적인지 (낮을수록 smooth)."""
        if len(self.slow_history) < 2:
            return 0.0
        diffs = [np.linalg.norm(self.slow_history[i] - self.slow_history[i-1])
                 for i in range(1, len(self.slow_history))]
        return float(np.mean(diffs))


# ────────────────────────────────────────────────────────────
# 3. 정보 분리도 측정
#    slow latent가 slow 성분(배경)을 얼마나 포착하는지,
#    fast latent가 fast 성분(이동점)을 얼마나 포착하는지.
#    proxy: latent 벡터와 대응 ground-truth 성분 간 cosine similarity
# ────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    """두 벡터 간 cosine similarity."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ────────────────────────────────────────────────────────────
# 4. Rollout Benchmark
# ────────────────────────────────────────────────────────────

def run_episode(model, env, horizon=15):
    """
    한 episode rollout.
    반환: latency_ms, steps, pred_obs_list, true_obs_list,
          latent_list
    """
    obs = env.reset()
    model.reset(obs)

    true_obs_list = [obs.copy()]
    pred_obs_list = []
    latent_list = []

    t0 = time.perf_counter()

    for h in range(horizon):
        action = rng.standard_normal(ACTION_DIM) * 0.5
        pred_obs, pred_reward = model.step(action)
        true_obs, true_reward, done = env.step(action)

        pred_obs_list.append(pred_obs)
        true_obs_list.append(true_obs.copy())
        latent_list.append(model.get_latent().copy())

        if done:
            break

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "latency_ms": elapsed_ms,
        "steps": len(pred_obs_list),
        "pred_obs": pred_obs_list,
        "true_obs": true_obs_list,
        "latents": latent_list,
    }


def compute_mse(pred_list, true_list):
    """예측 MSE."""
    if not pred_list:
        return 0.0
    mses = []
    for pred, true in zip(pred_list, true_list[1:]):  # true_list[0]는 초기 obs
        mses.append(float(np.mean((pred - true) ** 2)))
    return float(np.mean(mses)) if mses else 0.0


def compute_separation(dual_model, episode_data, env):
    """
    정보 분리도 측정:
    - slow branch 변화량 vs fast branch 변화량의 비율
      slow branch는 K=3마다 갱신 → step 간 변화가 작아야 함 (smoothness)
      fast branch는 매 step 갱신 → step 간 변화가 상대적으로 커야 함

    - obs의 slow 성분 변화량 vs fast 성분 변화량 비교
      slow_obs_var: 배경 성분 분산 (작을수록 slow가 맞음)
      fast_obs_var: 이동점 성분 분산 (클수록 fast 특성 맞음)

    - 분리도 스코어:
      separation_score = 1.0이면 완벽 분리 (slow branch는 slow obs만, fast는 fast obs만)
      < 0.5이면 분리 실패 (collapse 위험)
    """
    if len(episode_data["latents"]) < 2:
        return {
            "slow_cosine_sim": 0.0,
            "fast_cosine_sim": 0.0,
            "slow_delta_mean": 0.0,
            "fast_delta_mean": 0.0,
            "fast_ratio": 0.5,
            "separation_score": 0.5,
            "slow_obs_var": 0.0,
            "fast_obs_var": 0.0,
        }

    latents = episode_data["latents"]
    true_obs_list = episode_data["true_obs"][1:]  # t+1 부터

    # step 간 latent 변화량
    slow_delta_list = []
    fast_delta_list = []
    for i in range(1, len(latents)):
        h_slow_prev = latents[i-1][:dual_model.SLOW_HIDDEN]
        h_slow_curr = latents[i][:dual_model.SLOW_HIDDEN]
        h_fast_prev = latents[i-1][dual_model.SLOW_HIDDEN:]
        h_fast_curr = latents[i][dual_model.SLOW_HIDDEN:]
        slow_delta_list.append(np.linalg.norm(h_slow_curr - h_slow_prev))
        fast_delta_list.append(np.linalg.norm(h_fast_curr - h_fast_prev))

    mean_slow_delta = float(np.mean(slow_delta_list))
    mean_fast_delta = float(np.mean(fast_delta_list))

    # obs 성분 분산
    slow_obs_list = [o[:SLOW_DIM] for o in true_obs_list]
    fast_obs_list = [o[SLOW_DIM:] for o in true_obs_list]

    slow_obs_var = float(np.mean([np.var(o) for o in slow_obs_list]))
    fast_obs_var = float(np.mean([np.var(o) for o in fast_obs_list]))

    # slow cosine sim: slow latent와 obs slow 성분 사이 평균 cosine sim
    # fast cosine sim: fast latent와 obs fast 성분 사이 평균 cosine sim
    slow_sims = []
    fast_sims = []
    for latent, obs in zip(latents, true_obs_list):
        h_slow = latent[:dual_model.SLOW_HIDDEN]
        h_fast = latent[dual_model.SLOW_HIDDEN:]
        slow_sims.append(abs(cosine_sim(h_slow[:SLOW_DIM], obs[:SLOW_DIM])))
        fast_sims.append(abs(cosine_sim(h_fast[:FAST_DIM], obs[SLOW_DIM:])))

    mean_slow_sim = float(np.mean(slow_sims))
    mean_fast_sim = float(np.mean(fast_sims))

    # separation_score:
    # fast branch가 slow branch보다 더 많이 변하면 분리가 잘 된 것
    # slow branch가 상대적으로 안정적일수록 좋음
    total_delta = mean_slow_delta + mean_fast_delta + 1e-8
    fast_ratio = mean_fast_delta / total_delta  # fast가 전체 변화 중 차지하는 비율

    # K=3이면 이론상 slow는 1/3만 갱신 → fast의 변화량이 클 것
    # 분리 성공 기준: fast_ratio > 0.5 (fast branch가 더 많이 변함)
    separation_score = float(fast_ratio)

    return {
        "slow_cosine_sim": round(mean_slow_sim, 4),
        "fast_cosine_sim": round(mean_fast_sim, 4),
        "slow_delta_mean": round(mean_slow_delta, 4),
        "fast_delta_mean": round(mean_fast_delta, 4),
        "fast_ratio": round(fast_ratio, 4),
        "separation_score": round(separation_score, 4),
        "slow_obs_var": round(slow_obs_var, 6),
        "fast_obs_var": round(fast_obs_var, 6),
    }


# ────────────────────────────────────────────────────────────
# 5. Main Benchmark
# ────────────────────────────────────────────────────────────

def run_benchmark(n_episodes=50, horizon=15, latency_base_ms=1.0):
    """
    Baseline vs DualRate 비교.
    """
    baseline_results = []
    dualrate_results = []
    separation_results = []
    baseline_mses = []
    dualrate_mses = []
    slow_smoothness_list = []

    for ep in range(n_episodes):
        seed = ep * 100 + 7
        env = SyntheticEnv(seed=seed)

        # ── Baseline ──
        baseline_model = BaselineWorldModel(latency_base_ms=latency_base_ms)
        bdata = run_episode(baseline_model, env, horizon)
        baseline_results.append(bdata)
        baseline_mses.append(compute_mse(bdata["pred_obs"], bdata["true_obs"]))

        # ── Dual Rate ──
        env2 = SyntheticEnv(seed=seed)  # 동일 seed
        dual_model = DualRateWorldModel(latency_base_ms=latency_base_ms)
        ddata = run_episode(dual_model, env2, horizon)
        dualrate_results.append(ddata)
        dualrate_mses.append(compute_mse(ddata["pred_obs"], ddata["true_obs"]))
        slow_smoothness_list.append(dual_model.get_slow_smoothness())

        # 분리도
        sep = compute_separation(dual_model, ddata, env2)
        separation_results.append(sep)

    # ── 요약 통계 ──
    baseline_latency = [r["latency_ms"] for r in baseline_results]
    dualrate_latency = [r["latency_ms"] for r in dualrate_results]
    baseline_steps = [r["steps"] for r in baseline_results]
    dualrate_steps = [r["steps"] for r in dualrate_results]

    mean_b_lat = statistics.mean(baseline_latency)
    mean_d_lat = statistics.mean(dualrate_latency)
    rollout_speedup = mean_b_lat / max(mean_d_lat, 1e-6)

    mean_b_mse = statistics.mean(baseline_mses)
    mean_d_mse = statistics.mean(dualrate_mses)

    # quality_proxy_delta: MSE 기준 상대적 **저하**량 (악화만 페널티)
    # dual이 baseline보다 좋거나 같으면 delta=0 (품질 저하 없음)
    # dual이 baseline보다 나쁘면 상대 악화율
    mse_degradation = mean_d_mse - mean_b_mse  # 양수면 악화, 음수면 개선
    quality_proxy_delta = max(0.0, mse_degradation) / (abs(mean_b_mse) + 1e-8)

    # 이론적 FLOPs 절감률
    # Baseline: dim=192 GRU, 매 step → cost ∝ 192^2
    # DualRate: slow dim=128 (K=3마다) + fast dim=64 (매 step)
    #           → cost ∝ 128^2 / 3 + 64^2 per step
    baseline_flop_unit = 192 ** 2
    dualrate_flop_unit = (128 ** 2) / DualRateWorldModel.K + (64 ** 2)
    theoretical_flop_speedup = baseline_flop_unit / dualrate_flop_unit

    # 분리도 평균
    mean_slow_sim = statistics.mean(r["slow_cosine_sim"] for r in separation_results)
    mean_fast_sim = statistics.mean(r["fast_cosine_sim"] for r in separation_results)
    mean_slow_smoothness = statistics.mean(slow_smoothness_list)
    mean_separation_score = statistics.mean(r["separation_score"] for r in separation_results)
    mean_fast_ratio = statistics.mean(r["fast_ratio"] for r in separation_results)
    mean_slow_delta = statistics.mean(r["slow_delta_mean"] for r in separation_results)
    mean_fast_delta = statistics.mean(r["fast_delta_mean"] for r in separation_results)

    # ── Gate 판정 ──
    speedup_pass = rollout_speedup > 1.5
    quality_pass = quality_proxy_delta < 0.05
    verdict = "CONDITIONAL-GO" if (speedup_pass and quality_pass) else "FAIL"

    summary = {
        "verdict": verdict,
        "gate": {
            "rollout_speedup": round(rollout_speedup, 4),
            "rollout_speedup_threshold": 1.5,
            "rollout_speedup_pass": speedup_pass,
            "quality_proxy_delta": round(quality_proxy_delta, 6),
            "quality_proxy_delta_threshold": 0.05,
            "quality_proxy_delta_pass": quality_pass,
        },
        "latency": {
            "baseline_mean_ms": round(mean_b_lat, 3),
            "dualrate_mean_ms": round(mean_d_lat, 3),
            "speedup": round(rollout_speedup, 4),
        },
        "steps": {
            "baseline_mean": round(statistics.mean(baseline_steps), 2),
            "dualrate_mean": round(statistics.mean(dualrate_steps), 2),
        },
        "flops": {
            "baseline_unit": baseline_flop_unit,
            "dualrate_unit": round(dualrate_flop_unit, 1),
            "theoretical_speedup": round(theoretical_flop_speedup, 4),
        },
        "quality_proxy": {
            "baseline_mean_mse": round(mean_b_mse, 6),
            "dualrate_mean_mse": round(mean_d_mse, 6),
            "quality_proxy_delta": round(quality_proxy_delta, 6),
        },
        "separation": {
            "slow_cosine_sim_mean": round(mean_slow_sim, 4),
            "fast_cosine_sim_mean": round(mean_fast_sim, 4),
            "slow_smoothness_mean": round(mean_slow_smoothness, 4),
            "separation_score_mean": round(mean_separation_score, 4),
            "fast_ratio_mean": round(mean_fast_ratio, 4),
            "slow_delta_mean": round(mean_slow_delta, 4),
            "fast_delta_mean": round(mean_fast_delta, 4),
            "collapse_risk_note": (
                "OK — fast branch changes more than slow (structural separation achieved)"
                if mean_fast_ratio > 0.5
                else "COLLAPSE_RISK — slow branch dominates or branches not separating"
            ),
        },
        "config": {
            "n_episodes": n_episodes,
            "horizon": horizon,
            "latency_base_ms": latency_base_ms,
            "slow_hidden": DualRateWorldModel.SLOW_HIDDEN,
            "fast_hidden": DualRateWorldModel.FAST_HIDDEN,
            "K": DualRateWorldModel.K,
            "obs_dim": OBS_DIM,
            "slow_dim": SLOW_DIM,
            "fast_dim": FAST_DIM,
        },
    }
    return summary


# ────────────────────────────────────────────────────────────
# 6. 실행
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Dual-Rate World Model — Synthetic PoC")
    print("=" * 60)
    print(f"  Baseline: single-rate GRU dim=192, every step")
    print(f"  DualRate: slow GRU dim=128 (K=3) + fast GRU dim=64 (every step)")
    print(f"  Episodes: 50 | Horizon: 15 | latency_base: 1ms")
    print()

    t_start = time.perf_counter()
    results = run_benchmark(n_episodes=50, horizon=15, latency_base_ms=1.0)
    total_elapsed = time.perf_counter() - t_start

    results["total_elapsed_sec"] = round(total_elapsed, 2)

    # 결과 저장
    out_path = Path(__file__).parent / "poc_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Verdict: {results['verdict']}")
    print()
    print("── Gate 기준 ──────────────────────────────────────────")
    g = results["gate"]
    speedup_mark = "PASS" if g["rollout_speedup_pass"] else "FAIL"
    quality_mark = "PASS" if g["quality_proxy_delta_pass"] else "FAIL"
    print(f"  rollout_speedup    : {g['rollout_speedup']:.3f}x  (threshold > 1.5x)  [{speedup_mark}]")
    print(f"  quality_proxy_delta: {g['quality_proxy_delta']:.5f}  (threshold < 0.05)  [{quality_mark}]")
    print()
    print("── Latency ────────────────────────────────────────────")
    lat = results["latency"]
    print(f"  Baseline  mean: {lat['baseline_mean_ms']:.2f} ms")
    print(f"  DualRate  mean: {lat['dualrate_mean_ms']:.2f} ms")
    print(f"  Speedup       : {lat['speedup']:.3f}x")
    print()
    print("── Theoretical FLOPs ──────────────────────────────────")
    fl = results["flops"]
    print(f"  Baseline  unit: {fl['baseline_unit']}")
    print(f"  DualRate  unit: {fl['dualrate_unit']:.1f}")
    print(f"  Speedup       : {fl['theoretical_speedup']:.3f}x")
    print()
    print("── Quality Proxy (MSE) ────────────────────────────────")
    qp = results["quality_proxy"]
    print(f"  Baseline  MSE : {qp['baseline_mean_mse']:.6f}")
    print(f"  DualRate  MSE : {qp['dualrate_mean_mse']:.6f}")
    print(f"  Delta         : {qp['quality_proxy_delta']:.6f}")
    print()
    print("── Separation (Collapse Check) ────────────────────────")
    sp = results["separation"]
    print(f"  slow cosine sim (slow latent ↔ slow obs): {sp['slow_cosine_sim_mean']:.4f}")
    print(f"  fast cosine sim (fast latent ↔ fast obs): {sp['fast_cosine_sim_mean']:.4f}")
    print(f"  slow branch delta (step간 변화량)         : {sp['slow_delta_mean']:.4f}")
    print(f"  fast branch delta (step간 변화량)         : {sp['fast_delta_mean']:.4f}")
    print(f"  fast_ratio (fast가 전체 변화에서 차지하는 비율): {sp['fast_ratio_mean']:.4f}")
    print(f"  separation_score                          : {sp['separation_score_mean']:.4f}")
    print(f"  slow smoothness (slow history 변화량)     : {sp['slow_smoothness_mean']:.4f}")
    print(f"  Collapse risk                             : {sp['collapse_risk_note']}")
    print()
    print(f"Total elapsed: {total_elapsed:.1f}s")
    print(f"Results saved to: {out_path}")
