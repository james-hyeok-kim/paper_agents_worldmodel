---
slug: predictive-horizon-controller
status: literature-checked
verdict: INCREMENTAL
checked-date: 2026-06-11 KST
created: 2026-06-11 KST
category: A
venue-fit: [ICLR, NeurIPS, ICML]
blacklist-delta:
  - "BL-02 (단순 early stopping, fixed threshold): 본 아이디어는 고정 임계값이 아니라, RSSM의 stochastic posterior/prior KL gap을 입력으로 받아 매 step마다 '앞으로 몇 step 더 imagine할 가치가 있는가'를 회귀(regression)로 출력하는 learned controller다. (1) 고정 N이 아닌 per-state continuous horizon 예측, (2) reward head의 예측 분산과 결합한 multi-signal gating, (3) controller 자체를 policy gradient의 variance 감소량으로 학습하는 점이 다르다."
  - "BL-05 (rollout horizon grid search): grid search로 정적 최적 horizon을 찾는 게 아니라, 학습된 함수로 state별 가변 horizon을 inference time에 결정한다. horizon은 hyperparameter가 아니라 controller의 output이다."
---

# 예측적 Horizon 컨트롤러: RSSM 내부 신호 기반 무비용 Adaptive Imagination

## 핵심 가설
RSSM의 prior/posterior divergence와 reward-head 예측 분산을 입력으로 받는 경량 horizon-controller를 학습하면, 평균 imagination rollout 길이를 절반 이하로 줄여 rollout 연산을 1.6~2.0× 가속하면서도 DreamerV3의 episode return drop을 3% 이내로 유지한다.

## 동기 (Why Now)
DreamerV3는 모든 imagined trajectory를 고정 H=15 step으로 전개한다. 그러나 dynamics가 거의 deterministic해진 구간(예: 물리적으로 안정된 자세 유지)에서는 추가 rollout의 정보 이득이 0에 수렴하고, 반대로 분기점(접촉/충돌 직전)에서는 짧은 horizon이 손해다. 즉 "한 trajectory 안에서도" 필요한 깊이가 state마다 다르다. 기존 uncertainty 기반 truncation(MBPO/M2AC)은 ensemble disagreement라는 비싼 신호에 의존하지만, RSSM은 이미 prior/posterior라는 두 분포를 내부에 들고 있어 추가 모델 없이 신호를 공짜로 얻을 수 있다 — 이 구조적 기회가 DreamerV3 계열에만 존재한다.

## 제안 방법
- 신호 (모두 RSSM 내부에서 추가 forward 없이 추출):
  - `s_kl[t] = KL(posterior_t || prior_t)` 의 prior-only rollout 추정치 (imagination 중에는 posterior가 없으므로 prior entropy 변화율 `ΔH(prior_t)`를 proxy로 사용)
  - `r_var[t] = Var` of reward head over the stochastic state samples
- Controller: `g_phi(ΔH, r_var, t/H) -> p_continue ∈ [0,1]`, 2-layer MLP (<5k params).
- 학습 신호: imagination을 끝까지(H_max) 전개한 reference에서, step t 이후의 advantage 기여분 `|A_{t:H}|`를 타깃으로 `p_continue`를 회귀. 기여분이 임계 이하인 첫 step에서 stop하도록 학습 (policy gradient variance 감소를 직접 reward로).
- Inference: `p_continue < 0.5`이면 그 trajectory의 rollout 조기 종료, 남은 step은 terminal value bootstrap.

```
for t in range(H_max):
    h, z = rssm.img_step(h, z, a)          # 기존 forward 1회
    dH   = prior_entropy(z) - prev_entropy  # 공짜 신호
    rv   = reward_head.sample_var(h, z)
    if g_phi(dH, rv, t/H_max) < 0.5:
        bootstrap_with_value(h, z); break
```

## Novelty 포인트 (최소 3개)
1. (vs world model 논문) DreamerV3의 고정 horizon을 per-state continuous horizon으로 대체하되, ensemble 없이 RSSM 내부 prior entropy 변화율을 신호로 쓴다 — 추가 모델/forward 0.
2. (vs RL efficiency 논문) MBPO/M2AC의 truncation은 model rollout을 "얼마나 신뢰할지"를 보지만, 본 방법은 "추가 step이 policy gradient에 기여하는가(advantage 잔차)"를 직접 타깃으로 controller를 학습한다 — 신뢰도가 아니라 정보 이득 기준.
3. controller가 reward-head 예측 분산과 dynamics 신호를 함께 본다는 점에서, reward-sparse 구간(정보 이득 낮음)과 reward-cliff 구간(정보 이득 높음)을 구분한다.

## 선행 연구 위험 요소
- MBPO (Janner et al., 2019), M2AC (Pan et al., 2020) — model rollout length truncation by uncertainty
- "When to Trust Your Model" 계열, Dyna-style adaptive rollout
- DreamerV3 horizon ablation, TD-MPC2 planning horizon
- Adaptive computation time (Graves, 2016) — 일반 ACT와의 구분 필요

## 예상 실험 Skeleton
- Base model: DreamerV3 (danijar/dreamerv3)
- Benchmark: DMControl (cheetah, walker, quadruped), Atari 100k 일부
- 측정: rollout speed (imagine steps/sec), 평균 종료 horizon, episode return, sample efficiency (env steps to target return)
- 예상 결과: 평균 horizon 15→6, rollout 1.6~2.0× 가속, return drop < 3%

## 예상 Contribution
- RSSM 내부 신호만으로 무비용 adaptive horizon이 가능함을 보임 (architecture-level insight)
- imagination 연산을 task별로 절반 절감하는 plug-in controller (DreamerV3에 50줄 추가)

## 빠른 PoC 가능 여부
가능. synthetic: prior entropy가 plateau에 드는 toy RSSM에서 controller가 plateau 직후 stop하는지, advantage 잔차 회귀가 수렴하는지 1~2일 내 확인. Full: DreamerV3 fork에 controller 삽입, 단일 DMControl task 3일.

## Venue Fit 이유
"공짜 내부 신호로 imagination 연산을 줄인다"는 메커니즘 insight + 명확한 efficiency/quality trade-off curve는 ICLR efficiency track에 적합. DreamerV3 재현 실험으로 ICML/NeurIPS도 가능.

## 위험 요소
| 위험 | 가능성 | 완화 방법 |
|---|---|---|
| prior entropy proxy가 posterior KL을 잘 못 대체 | 중 | 학습 중 posterior가 있는 구간에서 proxy 보정(calibration) term 추가 |
| 조기 종료가 long-horizon credit assignment 해침 | 중 | terminal value bootstrap + horizon floor(최소 3 step) |
| controller 학습이 policy 학습과 불안정하게 coupling | 중 | controller는 stop-gradient된 reference rollout로만 학습 |
