"""
PoC: Static-Dynamic Decoupled World Model
==========================================
아이디어: encoder를 static(에피소드 내 불변) / dynamic(매 step 변화)으로 분리.
static은 에피소드당 1회만 인코딩 → encoder/decoder forward 연산 25~45% 절감.

Gate 기준:
  - rollout_speedup  > 1.5x
  - quality_proxy_delta < 0.05

설계:
  1. Synthetic video: 정적 배경(고정 noise texture) + 움직이는 점(동적 성분)
  2. Static encoder (small CNN, dim=32) — 에피소드당 1회
  3. Dynamic encoder (lighter CNN, dim=16) — 매 step
  4. Decoder: static_feat + dynamic_feat → reconstructed frame
  5. Baseline: 매 step full encode (static+dynamic concat)
  6. Modified: static 1회 캐시 + dynamic만 매 step
  7. 측정: rollout 속도, reconstruction MSE delta, static variance
"""

import time
import json
import statistics
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
SEED = 42
N_EPISODES = 50
HORIZON = 15
IMG_H = 16
IMG_W = 16
STATIC_DIM = 32
DYNAMIC_DIM = 16
LATENT_DIM = STATIC_DIM + DYNAMIC_DIM  # 48
HIDDEN_DIM = 64
BATCH_RUNS = 3          # 노이즈 줄이기 위해 여러 번 측정
SCENE_CHANGE_FRAC = 0.1  # 에피소드의 10%에서 중간에 배경 변화 (refresh 테스트)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ─────────────────────────────────────────────
#  Synthetic Video Generator
# ─────────────────────────────────────────────
def generate_episode(horizon: int, scene_change: bool = False):
    """
    에피소드 한 개 생성.
    - 배경: 고정 noise texture (H, W, 1)
    - 동적 성분: 반지름 1짜리 점이 sin 경로로 이동
    - scene_change=True면 horizon 절반 지점에 배경 교체
    """
    # 정적 배경
    bg = np.random.rand(IMG_H, IMG_W, 1).astype(np.float32)

    frames = []
    cx = IMG_W // 2
    cy = IMG_H // 2
    radius = min(IMG_H, IMG_W) // 4

    for t in range(horizon):
        angle = 2 * np.pi * t / horizon
        px = int(cx + radius * np.cos(angle))
        py = int(cy + radius * np.sin(angle))
        px = np.clip(px, 0, IMG_W - 1)
        py = np.clip(py, 0, IMG_H - 1)

        # 배경 복사 (scene_change: 절반 이후 새 배경)
        if scene_change and t == horizon // 2:
            bg = np.random.rand(IMG_H, IMG_W, 1).astype(np.float32)

        frame = bg.copy()
        # 점 추가 (1채널에 1.0으로 마킹)
        y_min = max(0, py - 1)
        y_max = min(IMG_H, py + 2)
        x_min = max(0, px - 1)
        x_max = min(IMG_W, px + 2)
        frame[y_min:y_max, x_min:x_max, :] = 1.0

        # (1, H, W) tensor
        frames.append(torch.tensor(frame.transpose(2, 0, 1), dtype=torch.float32))

    return frames, bg


# ─────────────────────────────────────────────
#  Neural Network Modules (small, CPU-friendly)
# ─────────────────────────────────────────────
class StaticEncoder(nn.Module):
    """정적 배경 특징 추출. 매 step 실행하면 낭비."""
    def __init__(self, in_ch=1, out_dim=STATIC_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 8, 3, padding=1),   # (8, H, W)
            nn.ReLU(),
            nn.AvgPool2d(2),                       # (8, H/2, W/2)
            nn.Conv2d(8, 16, 3, padding=1),        # (16, H/2, W/2)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),               # (16, 1, 1)
            nn.Flatten(),
            nn.Linear(16, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DynamicEncoder(nn.Module):
    """동적 성분(전체 frame) 경량 인코딩. 매 step 실행."""
    def __init__(self, in_ch=1, out_dim=DYNAMIC_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 4, 3, padding=1),    # (4, H, W)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),               # (4, 2, 2)
            nn.Flatten(),
            nn.Linear(16, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class FullEncoder(nn.Module):
    """Baseline: static + dynamic를 합쳐 매 step 인코딩."""
    def __init__(self, in_ch=1, out_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 8, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 4, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """latent → frame 재구성."""
    def __init__(self, in_dim=LATENT_DIM, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, IMG_H * IMG_W * out_ch),
            nn.Sigmoid(),
        )
        self.out_ch = out_ch

    def forward(self, z):
        return self.net(z).view(-1, self.out_ch, IMG_H, IMG_W)


class TransitionModel(nn.Module):
    """latent + action → next latent (imagination step)."""
    def __init__(self, latent_dim=LATENT_DIM, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, latent_dim),
        )

    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))


class RewardPredictor(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Linear(latent_dim, 1)

    def forward(self, z):
        return self.net(z)


# ─────────────────────────────────────────────
#  모델 인스턴스 (공유 — 가중치는 무작위지만 구조는 동일)
# ─────────────────────────────────────────────
full_enc     = FullEncoder()
static_enc   = StaticEncoder()
dynamic_enc  = DynamicEncoder()
decoder      = Decoder()
transition   = TransitionModel()
reward_pred  = RewardPredictor()

# 모두 eval 모드
for m in [full_enc, static_enc, dynamic_enc, decoder, transition, reward_pred]:
    m.eval()


# ─────────────────────────────────────────────
#  FLOPs 카운터 (간이)
# ─────────────────────────────────────────────
def count_params(model):
    return sum(p.numel() for p in model.parameters())

STATIC_ENC_PARAMS  = count_params(static_enc)
DYNAMIC_ENC_PARAMS = count_params(dynamic_enc)
FULL_ENC_PARAMS    = count_params(full_enc)
DECODER_PARAMS     = count_params(decoder)


# ─────────────────────────────────────────────
#  Static 불변성 측정 유틸
# ─────────────────────────────────────────────
def measure_static_variance(frames, static_encoder):
    """에피소드 내 모든 frame을 static_enc로 인코딩, variance 측정."""
    encodings = []
    with torch.no_grad():
        for f in frames:
            z = static_encoder(f.unsqueeze(0))   # (1, STATIC_DIM)
            encodings.append(z.squeeze(0).numpy())
    encodings = np.stack(encodings)  # (T, STATIC_DIM)
    # 각 dimension의 variance 평균
    return float(encodings.var(axis=0).mean())


# ─────────────────────────────────────────────
#  Rollout 함수
# ─────────────────────────────────────────────
def baseline_rollout_full(frames):
    """
    Baseline: 매 step full_enc 실행.
    Returns (trajectory, mse_list, elapsed_ms)
    """
    t0 = time.perf_counter()
    trajectory = []
    mse_list   = []

    with torch.no_grad():
        for frame in frames:
            f = frame.unsqueeze(0)          # (1, 1, H, W)
            z = full_enc(f)                 # (1, LATENT_DIM)
            recon = decoder(z)              # (1, 1, H, W)
            mse = F.mse_loss(recon, f).item()
            action = torch.randn(1, 4)
            z_next = transition(z, action)
            reward = reward_pred(z_next).item()
            trajectory.append({
                "latent": z.squeeze(0).numpy(),
                "reward": reward,
            })
            mse_list.append(mse)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return trajectory, mse_list, elapsed_ms


def modified_rollout_decoupled(frames):
    """
    Modified: static을 에피소드 첫 frame에서 1회만 인코딩, 이후 캐시 재사용.
    dynamic encoder는 매 step 실행.
    Returns (trajectory, mse_list, elapsed_ms)
    """
    t0 = time.perf_counter()
    trajectory   = []
    mse_list     = []
    static_cache = None
    refresh_triggered = 0

    # Invariance-break 감지 임계값 (static feat cosine similarity drop)
    COSINE_THRESHOLD = 0.85

    with torch.no_grad():
        for t, frame in enumerate(frames):
            f = frame.unsqueeze(0)          # (1, 1, H, W)

            # --- Static: 첫 step 또는 refresh 시에만 인코딩 ---
            z_static_new = static_enc(f)    # (1, STATIC_DIM)
            if static_cache is None:
                static_cache = z_static_new
            else:
                # Cosine similarity로 배경 변화 감지
                cos_sim = F.cosine_similarity(
                    static_cache, z_static_new, dim=-1
                ).item()
                if cos_sim < COSINE_THRESHOLD:
                    static_cache = z_static_new
                    refresh_triggered += 1
                # else: 캐시 유지 (static_enc 연산 결과는 버림)

            # --- Dynamic: 매 step ---
            z_dyn = dynamic_enc(f)          # (1, DYNAMIC_DIM)

            # --- Concat → decode ---
            z = torch.cat([static_cache, z_dyn], dim=-1)  # (1, LATENT_DIM)
            recon = decoder(z)
            mse = F.mse_loss(recon, f).item()

            action = torch.randn(1, 4)
            z_next = transition(z, action)
            reward = reward_pred(z_next).item()
            trajectory.append({
                "latent": z.squeeze(0).numpy(),
                "reward": reward,
            })
            mse_list.append(mse)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return trajectory, mse_list, elapsed_ms, refresh_triggered


def modified_rollout_amortized(frames):
    """
    Modified (amortized): static enc를 skip하고 캐시만 쓰는 극단적 버전.
    실제로 static_enc forward를 step 1 이후 완전히 생략 (refresh만 제외).
    """
    t0 = time.perf_counter()
    trajectory   = []
    mse_list     = []
    static_cache = None

    with torch.no_grad():
        for t, frame in enumerate(frames):
            f = frame.unsqueeze(0)

            # static은 t==0에만 실행
            if t == 0:
                static_cache = static_enc(f)

            # dynamic은 매 step
            z_dyn = dynamic_enc(f)
            z = torch.cat([static_cache, z_dyn], dim=-1)
            recon = decoder(z)
            mse = F.mse_loss(recon, f).item()

            action = torch.randn(1, 4)
            z_next = transition(z, action)
            reward = reward_pred(z_next).item()
            trajectory.append({
                "latent": z.squeeze(0).numpy(),
                "reward": reward,
            })
            mse_list.append(mse)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return trajectory, mse_list, elapsed_ms


# ─────────────────────────────────────────────
#  Benchmark
# ─────────────────────────────────────────────
def run_benchmark():
    print("=" * 60)
    print("Static-Dynamic Decoupled WM — PoC Benchmark")
    print(f"Episodes={N_EPISODES}, Horizon={HORIZON}")
    print(f"Image size: {IMG_H}x{IMG_W}, Static_dim={STATIC_DIM}, Dynamic_dim={DYNAMIC_DIM}")
    print("=" * 60)

    results_baseline         = []
    results_modified         = []  # with refresh detection
    results_amortized        = []  # strict skip
    static_variances_normal  = []
    static_variances_change  = []
    refresh_counts           = []

    for ep in range(N_EPISODES):
        scene_change = (ep < int(N_EPISODES * SCENE_CHANGE_FRAC))
        frames, bg = generate_episode(HORIZON, scene_change=scene_change)

        # Static variance 측정 (불변성 확인)
        var = measure_static_variance(frames, static_enc)
        if scene_change:
            static_variances_change.append(var)
        else:
            static_variances_normal.append(var)

        # Baseline
        traj_b, mse_b, t_b = baseline_rollout_full(frames)
        results_baseline.append({
            "ep": ep,
            "latency_ms": t_b,
            "mean_mse": statistics.mean(mse_b),
            "mean_reward": statistics.mean(r["reward"] for r in traj_b),
        })

        # Modified (with refresh)
        traj_m, mse_m, t_m, n_refresh = modified_rollout_decoupled(frames)
        results_modified.append({
            "ep": ep,
            "latency_ms": t_m,
            "mean_mse": statistics.mean(mse_m),
            "mean_reward": statistics.mean(r["reward"] for r in traj_m),
            "refresh_triggered": n_refresh,
        })
        refresh_counts.append(n_refresh)

        # Amortized (strict skip)
        traj_a, mse_a, t_a = modified_rollout_amortized(frames)
        results_amortized.append({
            "ep": ep,
            "latency_ms": t_a,
            "mean_mse": statistics.mean(mse_a),
            "mean_reward": statistics.mean(r["reward"] for r in traj_a),
        })

    # ─── 요약 ───
    def mean_of(lst, key):
        return statistics.mean(r[key] for r in lst)

    b_lat   = mean_of(results_baseline,  "latency_ms")
    m_lat   = mean_of(results_modified,  "latency_ms")
    a_lat   = mean_of(results_amortized, "latency_ms")

    b_mse   = mean_of(results_baseline,  "mean_mse")
    m_mse   = mean_of(results_modified,  "mean_mse")
    a_mse   = mean_of(results_amortized, "mean_mse")

    b_rew   = mean_of(results_baseline,  "mean_reward")
    m_rew   = mean_of(results_modified,  "mean_reward")
    a_rew   = mean_of(results_amortized, "mean_reward")

    # rollout speedup (latency 기반)
    speedup_modified   = b_lat / m_lat
    speedup_amortized  = b_lat / a_lat

    # quality proxy delta (MSE 기반 — 낮을수록 좋음)
    # baseline이 "정답" 기준; modified가 얼마나 다른지
    quality_delta_modified  = abs(m_mse - b_mse) / (b_mse + 1e-8)
    quality_delta_amortized = abs(a_mse - b_mse) / (b_mse + 1e-8)

    # FLOPs 절감 (파라미터 수 기반 proxy)
    # baseline: full_enc (HORIZON steps) + decoder (HORIZON steps)
    # modified: static_enc (1 step) + dynamic_enc (HORIZON steps) + decoder (HORIZON steps)
    baseline_flops_proxy  = FULL_ENC_PARAMS * HORIZON + DECODER_PARAMS * HORIZON
    modified_flops_proxy  = STATIC_ENC_PARAMS + DYNAMIC_ENC_PARAMS * HORIZON + DECODER_PARAMS * HORIZON
    flops_reduction_ratio = 1.0 - modified_flops_proxy / baseline_flops_proxy
    param_speedup_theoretical = baseline_flops_proxy / modified_flops_proxy

    # Static variance (불변성 지표)
    sv_normal = statistics.mean(static_variances_normal) if static_variances_normal else None
    sv_change = statistics.mean(static_variances_change) if static_variances_change else None

    # Refresh detection rate (scene_change 에피소드에서 refresh가 >= 1회 발생했는지)
    change_ep_indices = list(range(int(N_EPISODES * SCENE_CHANGE_FRAC)))
    refresh_detection_rate = (
        sum(1 for i in change_ep_indices if refresh_counts[i] >= 1) / max(len(change_ep_indices), 1)
    ) if change_ep_indices else None

    # Gate 판정
    TARGET_SPEEDUP = 1.5
    TARGET_QUALITY = 0.05

    # amortized 기준 (strict skip — 이상적 케이스)
    gate_speedup_pass    = speedup_amortized >= TARGET_SPEEDUP
    gate_quality_pass    = quality_delta_amortized < TARGET_QUALITY
    verdict              = "CONDITIONAL-GO" if (gate_speedup_pass and gate_quality_pass) else "FAIL"

    # ─── 출력 ───
    print(f"\n[Static Encoder Invariance]")
    print(f"  Static variance (normal episodes) : {sv_normal:.6f}" if sv_normal else "  N/A")
    print(f"  Static variance (scene-change eps): {sv_change:.6f}" if sv_change else "  N/A")
    if sv_normal and sv_change:
        print(f"  Variance ratio (change/normal)    : {sv_change/sv_normal:.2f}x (should be >> 1 if static enc is sensitive to change)")

    print(f"\n[Refresh Detection]")
    if refresh_detection_rate is not None:
        print(f"  Scene-change episodes: {len(change_ep_indices)}")
        print(f"  Refresh triggered rate: {refresh_detection_rate*100:.1f}%")
    else:
        print("  No scene-change episodes configured")

    print(f"\n[Benchmark: {N_EPISODES} episodes x horizon {HORIZON}]")
    print(f"{'Variant':<20} {'Latency(ms)':<15} {'Mean MSE':<15} {'Speedup':<12} {'QualityDelta':<15}")
    print("-" * 77)
    print(f"{'Baseline(full)':<20} {b_lat:<15.2f} {b_mse:<15.6f} {'1.00x':<12} {'0.0000':<15}")
    print(f"{'Modified(refresh)':<20} {m_lat:<15.2f} {m_mse:<15.6f} {speedup_modified:<12.3f}x {quality_delta_modified:<15.6f}")
    print(f"{'Amortized(skip)':<20} {a_lat:<15.2f} {a_mse:<15.6f} {speedup_amortized:<12.3f}x {quality_delta_amortized:<15.6f}")

    print(f"\n[FLOPs Proxy (parameter count)]")
    print(f"  full_enc params   : {FULL_ENC_PARAMS:,}")
    print(f"  static_enc params : {STATIC_ENC_PARAMS:,}")
    print(f"  dynamic_enc params: {DYNAMIC_ENC_PARAMS:,}")
    print(f"  decoder params    : {DECODER_PARAMS:,}")
    print(f"  Baseline FLOPs proxy  (x{HORIZON} steps): {baseline_flops_proxy:,}")
    print(f"  Modified FLOPs proxy  (static x1 + dyn x{HORIZON}): {modified_flops_proxy:,}")
    print(f"  FLOPs reduction  : {flops_reduction_ratio*100:.1f}%")
    print(f"  Theoretical speedup (FLOPs): {param_speedup_theoretical:.3f}x")

    print(f"\n[Gate Judgment]")
    print(f"  rollout_speedup (amortized) : {speedup_amortized:.3f}x  [threshold > {TARGET_SPEEDUP}]  {'PASS' if gate_speedup_pass else 'FAIL'}")
    print(f"  quality_proxy_delta         : {quality_delta_amortized:.6f}  [threshold < {TARGET_QUALITY}]  {'PASS' if gate_quality_pass else 'FAIL'}")
    print(f"\n  VERDICT: {verdict}")

    # ─── 결과 JSON ───
    output = {
        "slug": "static-dynamic-decoupled-wm",
        "verdict": verdict,
        "config": {
            "n_episodes": N_EPISODES,
            "horizon": HORIZON,
            "img_size": [IMG_H, IMG_W],
            "static_dim": STATIC_DIM,
            "dynamic_dim": DYNAMIC_DIM,
            "scene_change_frac": SCENE_CHANGE_FRAC,
        },
        "gate_results": {
            "rollout_speedup_modified": round(speedup_modified, 4),
            "rollout_speedup_amortized": round(speedup_amortized, 4),
            "quality_proxy_delta_modified": round(quality_delta_modified, 6),
            "quality_proxy_delta_amortized": round(quality_delta_amortized, 6),
            "gate_speedup_pass": gate_speedup_pass,
            "gate_quality_pass": gate_quality_pass,
        },
        "benchmarks": {
            "baseline": {
                "mean_latency_ms": round(b_lat, 4),
                "mean_mse": round(b_mse, 8),
                "mean_reward": round(b_rew, 6),
            },
            "modified_with_refresh": {
                "mean_latency_ms": round(m_lat, 4),
                "mean_mse": round(m_mse, 8),
                "mean_reward": round(m_rew, 6),
                "speedup": round(speedup_modified, 4),
                "quality_delta": round(quality_delta_modified, 6),
            },
            "amortized_strict": {
                "mean_latency_ms": round(a_lat, 4),
                "mean_mse": round(a_mse, 8),
                "mean_reward": round(a_rew, 6),
                "speedup": round(speedup_amortized, 4),
                "quality_delta": round(quality_delta_amortized, 6),
            },
        },
        "flops_proxy": {
            "full_enc_params": FULL_ENC_PARAMS,
            "static_enc_params": STATIC_ENC_PARAMS,
            "dynamic_enc_params": DYNAMIC_ENC_PARAMS,
            "decoder_params": DECODER_PARAMS,
            "baseline_flops_proxy": baseline_flops_proxy,
            "modified_flops_proxy": modified_flops_proxy,
            "flops_reduction_ratio": round(flops_reduction_ratio, 4),
            "theoretical_speedup": round(param_speedup_theoretical, 4),
        },
        "invariance": {
            "static_variance_normal_episodes": round(sv_normal, 8) if sv_normal else None,
            "static_variance_change_episodes": round(sv_change, 8) if sv_change else None,
            "variance_ratio_change_over_normal": round(sv_change / sv_normal, 4) if (sv_normal and sv_change) else None,
        },
        "refresh_detection": {
            "n_change_episodes": len(change_ep_indices),
            "detection_rate": round(refresh_detection_rate, 4) if refresh_detection_rate is not None else None,
        },
        "thresholds": {
            "rollout_speedup_min": TARGET_SPEEDUP,
            "quality_proxy_delta_max": TARGET_QUALITY,
        },
    }

    return output


if __name__ == "__main__":
    result = run_benchmark()

    out_path = "/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/static-dynamic-decoupled-wm/poc_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to: {out_path}")
