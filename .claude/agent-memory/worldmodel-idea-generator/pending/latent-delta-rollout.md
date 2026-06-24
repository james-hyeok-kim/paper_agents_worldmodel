---
slug: latent-delta-rollout
status: literature-checked
verdict: INCREMENTAL
checked-date: 2026-06-11 KST
created: 2026-06-11 KST
category: B
venue-fit: [NeurIPS, ICLR, ICML]
blacklist-delta:
  - "BL-03 (픽셀 레벨 codec 압축): 본 아이디어는 픽셀이 아니라 latent deterministic state(h)의 시간축 delta에 sparsity를 적용하며, 목적은 저장이 아니라 'forward 연산 절감'이다. 픽셀 품질과 무관."
  - "BL-01 (단순 KV cache 재사용): KV cache는 attention 입력을 캐싱하지만, 본 방법은 RSSM의 GRU/recurrent state 갱신 자체를 delta-gating해 'state가 거의 안 변하면 recurrent 연산을 skip'한다. 캐싱이 아니라 연산 생략이며, stochastic z는 항상 갱신하되 deterministic h만 selective update."
  - "단순 quantization과 달리, delta의 크기(L2)가 임계 이하인 step에서 recurrent update 자체를 건너뛰고 이전 h를 carry-forward해 실제 FLOPs를 줄인다. bit-width 축소가 아니라 step 축소."
---

# Latent Delta Rollout: Deterministic/Stochastic 비대칭 갱신으로 RSSM Forward 절감

## 핵심 가설
RSSM의 deterministic recurrent state h_t는 인접 imagination step 간 변화량(delta)이 작은 구간이 많다. delta L2 norm이 임계 이하인 step에서 GRU recurrent update를 skip하고 h를 carry-forward하면, deterministic branch의 forward 연산을 30~50% 줄여 rollout을 1.4~1.7× 가속하면서 prediction PSNR drop을 5% 이내로 유지한다.

## 동기 (Why Now)
RSSM forward의 비용은 (a) deterministic GRU 갱신 + (b) stochastic prior 샘플링으로 나뉜다. 많은 환경에서 dynamics는 구간적으로 "거의 상수"인 latent drift를 보인다(예: 등속 운동, 정지). 그러나 DreamerV3는 매 step 두 branch를 모두 fully 갱신한다. h의 시간 delta가 작은 구간을 식별해 deterministic branch만 skip하면, stochastic 표현력은 유지(z는 항상 갱신)하면서 연산을 줄일 수 있다 — 이는 latent dynamics의 temporal redundancy를 직접 공략하는, world-model-specific 절감이다.

## 제안 방법
- 매 step `delta_t = || GRU(h_{t-1}, [z_{t-1}, a_{t-1}]) - h_{t-1} ||` 를 추정해야 하는데, GRU를 돌리면 절감이 안 됨 → **경량 delta-predictor** `d_psi([z_{t-1}, a_{t-1}, h_{t-1}_summary]) -> delta_hat` (linear probe, h의 저차원 sketch만 입력).
- `delta_hat < tau`이면 GRU skip, `h_t = h_{t-1}` carry-forward. 단 z(prior)는 항상 갱신 (stochastic 표현 보존).
- tau는 누적 skip이 prediction error에 미치는 영향으로 학습 — skip이 일정 step 연속되면 강제 full update(drift 누적 방지, "refresh interval").
- delta-predictor는 학습 시 실제 GRU delta를 타깃으로 회귀 (teacher forcing).

```
h_sketch = proj(h_prev)                       # cheap
delta_hat = d_psi(z_prev, a_prev, h_sketch)   # cheap
if delta_hat > tau or steps_since_refresh > K:
    h = gru(h_prev, [z_prev, a_prev]); refresh=0   # full
else:
    h = h_prev; steps_since_refresh += 1            # skip GRU
z = prior_head(h).sample()                    # 항상 갱신
```

## Novelty 포인트 (최소 3개)
1. (vs world model) RSSM의 두 branch(deterministic/stochastic)를 비대칭적으로 갱신 — deterministic만 temporal-sparse하게, stochastic은 dense하게. 기존엔 항상 동기 갱신.
2. (vs 압축 논문) delta-encoding을 저장 압축이 아니라 inference FLOPs 절감에 사용. skip 판단을 위한 delta는 GRU를 돌리지 않고 경량 predictor로 미리 추정 — "예측 후 생략" 구조.
3. drift 누적을 막는 learned refresh interval로 long-rollout 안정성 보장 — codec의 keyframe 개념을 latent recurrent state에 적용한 첫 사례.

## 선행 연구 위험 요소
- Skip-RNN (Campos et al., 2018), Adaptive computation in RNNs
- Temporal difference state compression, delta networks (Neil et al.)
- DreamerV3 RSSM 구조 논문, S4/SSM 기반 효율 WM
- Conditional computation / mixture-of-depths

## 예상 실험 Skeleton
- Base model: DreamerV3 (RSSM)
- Benchmark: DMControl (등속/정지 구간 많은 walker, cartpole), Atari 100k
- 측정: deterministic branch FLOPs 절감률, skip 비율, rollout steps/sec, 1-step/multi-step prediction PSNR & FVD
- 예상 결과: GRU update 30~50% skip, rollout 1.4~1.7× 가속, PSNR drop < 5%, return drop < 4%

## 예상 Contribution
- RSSM의 deterministic/stochastic branch 비대칭 갱신이라는 새로운 효율화 축 제시
- delta-predictor + refresh interval로 안정적 skip을 보장하는 50줄 plug-in

## 빠른 PoC 가능 여부
가능. synthetic: 구간적 상수 dynamics를 가진 toy 환경에서 GRU delta가 실제로 sparse한지 측정(즉 절감 상한 확인), delta-predictor 회귀 정확도 1~2일. Full: DreamerV3 fork 3~4일.

## Venue Fit 이유
명확한 FLOPs-quality trade-off + RSSM 구조 insight는 NeurIPS efficiency / ICLR에 적합. 측정 가능한 speedup 게이트(>1.5×)에 가까워 validator 통과 가능성 높음.

## 위험 요소
| 위험 | 가능성 | 완화 방법 |
|---|---|---|
| delta-predictor가 GRU만큼 비싸짐 | 중 | h의 저차원 sketch만 입력, predictor 파라미터 상한 명시 |
| skip 구간에서 stochastic z만 갱신해도 h가 stale → reward 오류 | 중 | refresh interval K로 강제 동기화, reward head에 step-since-refresh 입력 |
| 절감 상한이 환경 의존적 (dynamics가 dense하면 이득 없음) | 중 | dense/sparse 환경 모두 보고, 적용 가능 조건 분석을 contribution으로 |
