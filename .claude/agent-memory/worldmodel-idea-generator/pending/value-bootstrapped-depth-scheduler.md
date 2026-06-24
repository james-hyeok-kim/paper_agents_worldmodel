---
slug: value-bootstrapped-depth-scheduler
status: literature-checked
verdict: INCREMENTAL
checked-date: 2026-06-11 KST
created: 2026-06-11 KST
category: A
venue-fit: [NeurIPS, ICLR, ICML]
blacklist-delta:
  - "BL-08 (per-step entropy adaptive horizon, learned controller): BL-08은 per-step learned stopper로 실패 — reward cliff 예측 불가, label rate 3% imbalance. 본 아이디어는 per-step stopper가 아니다. rollout 시작 *전에* 초기 state의 value-landscape 곡률로 rollout depth를 결정(pre-rollout decision, not per-step gate)하고, 잘린 tail은 반드시 learned value로 bootstrap → truncation이 아니라 bootstrap. cliff 문제는 'tail을 버리는' BL-08 truncation에서 왔는데, 본 아이디어는 tail을 value로 대체하므로 cliff 자체가 없음."
  - "BL-02 (fixed threshold early stopping): 고정 임계값 아님. depth가 초기 state별로 결정되되, learned scheduler가 'value bootstrap 오차가 임계 이하가 되는 최소 depth'를 예측. task/state-adaptive."
  - "BL-05 (horizon grid search): grid search 아님. lambda-return의 bias-variance 관점에서 'bootstrap이 충분히 정확해지는 depth'를 state별로 예측하는 learned scheduler. 단순 hyperparameter가 아니라 value-landscape 곡률 기반."
---

# Value-Bootstrapped Depth Scheduler: Truncation이 아닌 Bootstrap으로 Rollout 단축

## 핵심 가설
DreamerV3의 lambda-return은 H-step imagination + value bootstrap의 가중합이다. value function이 정확한 영역(value-landscape가 평탄/선형)에서는 짧은 rollout + early bootstrap이 긴 rollout과 거의 동일한 return 추정을 준다. 초기 state의 value 곡률로 per-state rollout depth를 *rollout 전에* 예측하고 잘린 tail을 value로 bootstrap하면, 평균 imagination depth를 35~50% 줄여 rollout을 1.5~1.9× 가속하면서 policy gradient 품질을 5% 이내로 유지한다.

## 동기 (Why Now)
BL-08은 per-step entropy로 "지금 멈출까"를 학습하다 실패했다 — reward cliff가 예측 불가능하고(post-plateau에 큰 return 존재), label이 3%뿐이라 controller가 항상 continue를 골랐다. 근본 문제는 **tail을 버린다(truncation)**는 것이었다: tail에 reward가 있으면 멈추면 안 되니까.

핵심 통찰: DreamerV3는 애초에 tail을 버리지 않는다. lambda-return은 `R = Σ λ^t r_t + λ^H V(s_H)` — H에서 value로 bootstrap한다. 즉 rollout을 H'<H에서 멈춰도 `V(s_{H'})`로 bootstrap하면 tail의 기대값은 보존된다(value가 정확하다면). 따라서 질문은 "멈춰도 되는가(cliff?)"가 아니라 "**여기서 bootstrap한 value가 충분히 정확한가**"로 바뀐다 — 이건 value-landscape 곡률로 예측 가능하고 cliff 문제가 없다.

## 제안 방법
- **pre-rollout depth predictor** `D(s_0) -> H' ∈ {H_min,...,H}` : 초기 state s_0에서, value bootstrap 오차가 임계 이하가 되는 최소 depth H'를 예측. 입력은 s_0의 value gradient norm, local value 곡률(Hessian trace proxy), reward head 분산. **rollout 시작 전 1회 결정** → per-step 게이트(BL-08) 아님.
- **bootstrap 보장:** H'에서 멈추되 항상 `λ^{H'} V(s_{H'})`로 bootstrap. tail을 버리지 않음 → cliff 무관. lambda-return 공식 그대로, 합산 상한만 H'.
- **scheduler 학습:** full-depth lambda-return을 teacher로, depth H'의 truncated-bootstrapped return과의 오차가 ε 이하인 최소 H'를 라벨로 회귀. label은 "각 state의 충분 depth"라 BL-08의 3% imbalance(드문 positive)와 달리 모든 state가 dense label을 가짐.
- **value-가속 상호작용:** value가 학습 초기엔 부정확 → scheduler가 큰 H' 선택(보수적). value가 정확해질수록 H'↓ → 자연스러운 curriculum. 안전을 위해 H'의 하한과 value 신뢰도 게이트.

```
# pre-rollout: depth를 소수 level로 양자화 + batch grouping (per-step gather 없음)
H_levels = {5, 10, 15}
group = quantize(D(s_0_batch), H_levels)          # batch 전체 1회, level별 분류
for level L in H_levels:                            # m개 fixed-depth batched rollout
   B_L = batch_rows_with(group == L)
   s = B_L; R = 0
   for t in 1..L:                                  # shallow group은 적은 step만 → call-skip
      a = π(s); s = GRU(s, sample(prior(s)), a)
      R += λ^t * r_head(s)
   R += λ^L * V(s)                                 # bootstrap (tail 버리지 않음 → cliff 없음)
# 절감 = shallow group이 도는 step 수 감소 (batch 단위 call-skip, gather overhead 없음)
```

## Amdahl 체크 (gate 도달 가능성)
- 타깃: imagination rollout 단계의 GRU+prior call 총량. metric은 **imagination rollout speed(steps/sec)** — update throughput(BL-11의 37%)이 아니라 rollout 단계 isolate.
- **GPU lockstep 함정(BL-11 회피 — 핵심 설계):** DreamerV3는 imagination batch 전체를 lockstep으로 unroll한다. 단순히 trajectory별 depth를 다르게 두면 **GPU에서 call이 skip되지 않는다** — 짧게 끝난 row도 matmul을 그대로 통과하기 때문(masked rows still flow). per-step으로 batch를 compact하면 그게 정확히 BL-11을 0.49×로 만든 gather/scatter다. **따라서 depth를 소수의 이산 level로 양자화**(예: H'∈{5,10,15})하고, rollout *시작 전에* batch를 level별로 한 번에 group → m개의 fixed-depth batched rollout을 각각 실행. shallow group은 통째로 적은 step만 도므로 **실제 call-skip이 batch 단위로 발생**(few large batches, no per-step gather).
- 예: batch의 1/3이 depth 5, 1/3이 10, 1/3이 15면 평균 step 수 = (5+10+15)/3 = 10 vs full 15 → call 33% skip → 1.5×. depth 분포가 더 짧은 쪽으로 치우치면 상회.
- depth predictor 비용은 작은 MLP(s_0 1회, batch 전체 vectorized) → 무시 가능.
- **핵심:** 절감은 level-grouped batch의 shallow group이 적은 step만 도는 CALL-skip에서 발생. per-step compaction(BL-11) 아님 — pre-rollout 1회 grouping이라 gather overhead 없음.
- **BL-08과의 결정적 차이:** BL-08은 per-step 게이트라 항상 continue 선택→1.0×. 본 아이디어는 pre-rollout 결정 + dense label이라 학습 가능, bootstrap이 tail 보존이라 cliff로 인한 quality 손실 없음.

## Novelty 포인트 (최소 3개)
1. (vs BL-08 / adaptive horizon) per-step stopping이 아니라 pre-rollout value-curvature 기반 depth 예측 + value bootstrap 보장. truncation을 bootstrap으로 바꿔 cliff 문제를 구조적으로 제거.
2. (vs world model) lambda-return의 bias-variance를 state별 depth로 능동 조절하는 첫 시도. DreamerV3는 고정 H. value 정확도와 rollout depth를 명시적으로 연동.
3. (vs MBPO rollout length) MBPO는 global rollout length를 model error로 스케줄하지만, 본 아이디어는 per-state value-curvature로 미세 조절 + value bootstrap을 효율 레버로 사용. value-가속 curriculum(value 정확↑ → depth↓)이 메커니즘 novelty.

## 선행 연구 위험 요소
- MBPO (Janner 2019) rollout length scheduling — global vs per-state, model-error vs value-curvature 차별 필수
- BL-08 (자체) — per-step vs pre-rollout, truncation vs bootstrap 명시
- TD(λ)/lambda-return bias-variance — 이론적 토대이나 efficiency 적용은 새로움
- Adaptive computation time, value-aware model learning (VAML)
- Truncated rollouts in MBRL

## 예상 실험 Skeleton
- Base model: DreamerV3 (lambda-return 계산에 per-state depth 주입)
- Benchmark: DMControl, Atari 100k
- 측정: **imagination rollout steps/sec**, 평균 depth 분포, depth vs value-error 상관, policy gradient 품질(full-depth 대비), episode return, sample efficiency
- 예상 결과: 평균 depth 35~50%↓, rollout 1.5~1.9×, return drop < 5%, value 정확↑에 따라 depth↓ curriculum 관찰

## 빠른 PoC 가능 여부
가능(2일). synthetic/소규모: 학습된 DreamerV3(또는 toy MDP+learned value)에서, (1) value-curvature가 "bootstrap 충분 depth"와 실제로 상관 있는지(BL-08의 0.085 함정 회피 검증 — 이번엔 proxy가 value-error 직접 예측이라 상관 높아야 함), (2) depth predictor 학습이 dense label로 수렴하는지, (3) truncated-bootstrapped return이 full-depth와 오차 ε 이하인지 depth별 곡선. 2일.

## Validator Gate 달성 평가
- rollout_speedup > 1.5×: **유망**. 평균 depth가 직접 call 총량을 줄임. 단 환경의 value-landscape가 충분히 평탄한 영역을 가져야 평균 depth↓ 실현.
- quality_delta < 0.05: **bootstrap이 핵심 안전장치**. tail을 value로 대체하므로 value가 정확하면 보존. value 부정확 영역은 scheduler가 큰 depth 선택(보수적). BL-08의 cliff 위험이 bootstrap으로 제거됨이 핵심 주장 — PoC가 검증.

## Venue Fit 이유
lambda-return depth와 value 정확도의 연동은 model-based RL 이론+효율 → NeurIPS. value-aware 효율화 각도 → ICLR/ICML.

## 위험 요소
| 위험 | 가능성 | 완화 방법 |
|---|---|---|
| value-curvature가 bootstrap 충분 depth와 상관 약함(BL-08 재현) | 중 | proxy를 value-error 직접 회귀로(1-step latent proxy 아님), PoC 상관 선검증 |
| value 부정확 초기에 depth↓로 policy 손상 | 중 | value 신뢰도 게이트 + depth 하한, value-curriculum |
| 평탄 value 영역이 적어 평균 depth 절감 미미 | 중 | depth 분포 모니터, 평탄 영역 많은 환경 우선 |
| GPU lockstep batch에서 variable depth가 call을 안 줄임(BL-11) | 높 | depth를 소수 level 양자화 + pre-rollout batch grouping(per-step gather 금지), PoC에서 level-grouped batched rollout의 실제 wall-clock 측정 |
