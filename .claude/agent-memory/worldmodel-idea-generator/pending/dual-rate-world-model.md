---
slug: dual-rate-world-model
status: literature-checked
verdict: INCREMENTAL
created: 2026-06-11 KST
category: E
venue-fit: [NeurIPS, ICLR, ICML]
blacklist-delta:
  - "BL-04 (단순 distillation teacher→student): 본 아이디어는 큰 모델을 작은 모델로 압축하는 게 아니라, 하나의 모델 안에서 slow(저빈도, 큰) dynamics와 fast(고빈도, 작은) dynamics를 분리하고, slow는 매 K step만 갱신해 평균 연산을 줄인다. 두 모델이 동시에 살아있고 시간 해상도가 다름 — 압축이 아니라 multi-rate 분해."
  - "BL-05 (horizon grid search)와 무관: horizon이 아니라 갱신 빈도(temporal stride)를 분해. slow latent는 K step마다, fast latent는 매 step 갱신."
  - "Clockwork RNN(고정 모듈별 clock rate)과 달리, 본 방법의 핵심은 (a) action을 window-aggregate해 slow branch에 주입함으로써 slow가 거친 stride로 장기 action 효과를 흡수하고 fast가 즉각 보정하는 명시적 역할 분담, (b) slow가 정보를 독점하는 trivial collapse를 fast→slow information bottleneck + slow temporal-smoothness penalty로 막는 학습 레시피다. 고정 clock 분할이 아니라 stochastic latent의 정보 분해 + collapse 방지가 메커니즘 novelty."
---

# Dual-Rate World Model: Slow/Fast Latent 시간 해상도 분해로 Dynamics 연산 절감

## 핵심 가설
시각 환경의 latent dynamics는 느리게 변하는 성분(배경, scene 구성, 전역 물리 상태)과 빠르게 변하는 성분(agent/object의 즉각 운동)으로 분해된다. slow latent를 K step마다만 갱신하는 dual-rate RSSM을 학습하면, 전체 dynamics forward 연산을 35~55% 줄여 rollout을 1.5~1.9× 가속하면서 prediction FVD/return drop을 5% 이내로 유지한다.

## 동기 (Why Now)
DreamerV3는 단일 시간 해상도로 모든 latent를 매 step 갱신한다. 그러나 imagination horizon 내에서 배경/전역 상태는 거의 정적이고 빠른 변화는 작은 부분공간에 집중된다. 이 시간 스케일 분리를 모델에 내장하면, 큰 slow branch는 드물게(K step마다), 작은 fast branch는 매 step 돌려 평균 비용을 낮출 수 있다. Director/STORM 계열은 abstraction을 다루지만 "연산 빈도"를 효율 목표로 명시하지 않았다 — 여기에 빈틈이 있다.

## 제안 방법
- latent를 `s = [s_slow, s_fast]`로 분할. 별도 transition:
  - `s_fast` : 매 step `f_fast(s_fast, s_slow, a)` 갱신 (작은 GRU)
  - `s_slow` : K step마다 `f_slow(s_slow, aggregate(a_{t:t+K}))` 갱신, 사이 step은 carry-forward (큰 GRU지만 1/K 빈도)
- 분리 유도: slow가 정보를 독점하지 않도록 fast→slow 정보 흐름에 bottleneck + slow branch에 temporal smoothness penalty. K는 학습 가능(또는 환경별 소수 후보).
- decoder/reward는 `[s_slow, s_fast]` 결합 사용. slow가 stale한 사이 step에서도 fast가 보정.

```
for t in horizon:
    s_fast = f_fast(s_fast, s_slow, a_t)        # 매 step, 저비용
    if t % K == 0:
        s_slow = f_slow(s_slow, agg(a_window))  # 1/K 빈도, 고비용
    out = decode([s_slow, s_fast])
```
평균 dynamics 비용 ≈ cost(fast) + cost(slow)/K.

## Novelty 포인트 (최소 3개)
1. (vs world model) latent를 시간 해상도로 분해하고 slow branch를 명시적으로 sub-sample해 FLOPs를 줄이는 첫 RSSM 변형. abstraction을 효율 목표로 직접 사용.
2. (vs Clockwork RNN / hierarchical RL) 고정 clock-rate 분할이 아니라, fast→slow information bottleneck + slow temporal-smoothness penalty로 stochastic latent의 정보를 능동 분해해 collapse를 막는 학습 레시피 — 이 collapse 문제는 결정론 RNN엔 없고 stochastic WM에서만 발생.
3. (vs multi-scale SSM) action을 window aggregate로 slow에 주입 → slow가 장기 action 효과를 거친 stride로 흡수, fast가 즉각 보정하는 명시적 역할 분담.

## 선행 연구 위험 요소
- Director (Hafner et al., 2022), hierarchical world models
- Clockwork RNN (Koutnik et al., 2014), Phased LSTM — multi-rate recurrence
- TWM/STORM의 temporal abstraction, multi-scale SSM (S4 hierarchies)
- Slow feature analysis 계열

## 예상 실험 Skeleton
- Base model: DreamerV3 (RSSM 분할)
- Benchmark: DMControl, Atari 100k, (가능 시) MineDojo/Crafter — 배경 정적 비율 다양
- 측정: dynamics FLOPs, slow update 빈도(1/K), rollout steps/sec, prediction FVD/PSNR, episode return
- 예상 결과: dynamics 35~55% 절감, rollout 1.5~1.9× 가속, FVD/return drop < 5%

## 예상 Contribution
- RSSM에 시간-스케일 분해를 내장하는 dual-rate 아키텍처 + collapse 방지 학습 레시피
- "연산 빈도를 abstraction의 1급 목표로" 보는 관점 제시

## 빠른 PoC 가능 여부
부분 가능. synthetic: slow/fast 분리가 학습되는지가 핵심 리스크 → 명시적 slow(상수 배경)/fast(움직이는 점) toy video에서 분리 + slow skip 후 재구성 오차 측정 2~3일. Full: DreamerV3 분할 + 안정화가 4주 한계선 — minimal PoC는 단일 task로 한정.

## Venue Fit 이유
아키텍처 novelty + efficiency/quality 곡선 + 시각 prediction(FVD)은 NeurIPS/ICLR 본 트랙 적합. dual-rate라는 명확한 메시지.

## 위험 요소
| 위험 | 가능성 | 완화 방법 |
|---|---|---|
| slow/fast가 분리 안 되고 한쪽이 정보 독점(collapse) | 높 | information bottleneck, smoothness penalty, 분리도 metric으로 모니터 |
| slow stale 구간에서 prediction 품질 저하 | 중 | fast 보정 + slow에 step-since-update 입력 |
| 4주 내 학습 안정화 실패 | 중 | minimal PoC를 단일 DMControl task + 작은 K(=2,3)로 제한 |
