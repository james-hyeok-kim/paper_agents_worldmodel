---
slug: latent-delta-rollout
verdict: FAIL
validated-date: 2026-06-11 KST
poc-location: experiments/wip/latent-delta-rollout/poc.py
---

## 판정: FAIL

## PoC 설정
- episodes=50, horizon=15, h_dim=256, z_dim=32
- delta_scales=[0.01, 0.05, 0.1, 0.2, 0.5] (dense→sparse regime sweep)
- tau_values=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5] (skip threshold sweep)
- predictor: 2-layer linear probe (hidden=64), 비용 speedup에 포함
- quality proxy: baseline vs modified reconstruction 간 누적 latent drift MSE
- 실행 환경: CPU
- 실행 시간: 0.57분

## Gate 기준 결과
| 기준 | 임계값 | 실측값 (최솟값) | 통과여부 |
|---|---|---|---|
| rollout_speedup | > 1.50× | 최대 37.6x (단독) | ❌ (quality와 동시 불가) |
| quality_proxy_delta | < 0.05 | 최솟값 1.74 (모든 τ) | ❌ |

게이트 통과 regime: 0/5 (기준: 2 이상) → FAIL

## 상세 결과

### Rollout 속도 vs Quality Trade-off (delta_scale=0.01, 가장 sparse regime)
| τ | skip_rate | speedup | quality_proxy_delta | gate |
|---|---|---|---|---|
| 0.005 | 68.0% | 1.55x | 1.932 | FAIL (quality) |
| 0.020 | 72.8% | 6.11x | 1.907 | FAIL (quality) |
| 0.100 | 99.3% | 37.6x | 1.758 | FAIL (quality) |
| 0.200 | 100.0% | 0.99x | 1.747 | FAIL (양쪽) |

quality_proxy_delta가 0.05 기준의 **약 35~39배** 수준으로, 어떤 τ와 regime 조합에서도 gate를 통과하지 못함.

### Delta 분포 (핵심 Risk #1)
| delta_scale | delta_mean | delta_p50 | delta_p90 |
|---|---|---|---|
| 0.01 | 0.046 | 0.016 | 0.017 |
| 0.05 | 0.379 | 0.080 | 2.248 |
| 0.1 | 0.663 | 0.162 | 2.926 |
| 0.2 | 1.431 | 0.332 | 3.370 |
| 0.5 | 3.177 | 3.152 | 3.659 |

delta_scale=0.01 (가장 sparse)에서도 평균 delta가 존재하며, skip 시 누적 drift는 방지 불가.

### Predictor 신뢰도
| delta_scale | pred_MSE | trivial_MSE | beats_trivial |
|---|---|---|---|
| 0.01 | 0.1941 | 0.1931 | false |
| 0.05 | 0.7518 | 0.7561 | true |
| 0.1 | 1.484 | 1.455 | false |
| 0.2 | 1.817 | 1.880 | true |
| 0.5 | 0.122 | 0.110 | false |

5개 regime 중 2개에서만 trivial baseline(mean predictor) 이김 → predictor 비신뢰.

### Z-Diversity (secondary diagnostic)
z_entropy_ratio ≈ 0.994~1.002 (전 조건에서 거의 변화 없음)
→ frozen h에서 z 다양성은 유지되나, 이는 h drift 문제와 무관.

## 판정 근거

**가정 위반 1 (결정적): quality_proxy_delta가 게이트 기준을 39배 초과**

skip 시 h가 carry-forward되면 h와 실제 h* 사이의 오차가 매 step마다 누적된다. 이 누적 drift는 downstream reconstruction을 크게 훼손한다. quality_proxy_delta 최솟값이 ~1.74로, 0.05 기준 대비 약 35배 수준. psnr_drop은 음수(개선처럼 보이는 수치)이지만 이는 MSE 측정 구조의 아티팩트이며 quality_proxy_delta가 핵심 지표.

**가정 위반 2: Speedup과 Quality가 구조적으로 동시 달성 불가**

- τ가 작으면 (공격적 skip 적음): speedup < 1.5x (predictor 오버헤드만 증가)
- τ가 크면 (공격적 skip 많음): speedup > 1.5x이지만 quality_proxy_delta >> 0.05

두 gate가 동시에 만족되는 τ가 존재하지 않음. 이는 단순히 τ 튜닝 문제가 아니라 h carry-forward 자체의 구조적 한계.

**가정 위반 3: Delta predictor가 trivial 이하**

2-layer linear probe가 5개 regime 중 3개에서 mean predictor보다 나쁨. predictor 비용이 speedup을 잠식하면서도 유의미한 skip 결정을 내리지 못함. 이는 h_{t-1}과 action으로부터 ||Δh_t||를 예측하는 것이 선형적으로 어렵다는 것을 시사.

## 아이디어 기각 이유 요약

RSSM h의 step-to-step delta-skip은 **h carry-forward로 인한 누적 drift**가 quality를 구조적으로 훼손한다. delta가 sparse한 구간이라도 skip 오차는 전파되며, 이를 억제할 수 없다. predictor도 비신뢰 수준으로, 비용 대비 이득이 없다.

## 다음 단계 (방향 수정 제안)

기각하되 유사 mechanism을 재설계 시 다음 방향 고려:
1. **Skip 대신 low-rank approximation**: h 전체를 freeze하는 것이 아니라 GRU 내부 gate 연산만 근사 (e.g., gate value 재사용)
2. **Hierarchical latent**: 느리게 변하는 상위 latent와 빠르게 변하는 하위 latent를 분리, 상위만 가끔 업데이트
3. **Speculative rollout**: skip한 branch와 full branch를 병렬 실행하다가 delta가 크면 full branch로 rollback (carry-forward 오차 방지)
