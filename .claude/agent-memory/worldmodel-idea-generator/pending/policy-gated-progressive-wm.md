---
slug: policy-gated-progressive-wm
status: literature-checked
verdict: INCREMENTAL
checked-date: 2026-06-11 KST
created: 2026-06-11 KST
category: F
venue-fit: [NeurIPS, ICLR, CoRL]
blacklist-delta:
  - "BL-04 (teacher→student distillation): 큰 WM을 작은 WM 하나로 압축하는 게 아님. heterogeneous-cost 2-tier WM(cheap fast model + expensive accurate model)을 두고, policy-visitation 기반 gating network가 매 imagination step에서 어느 모델을 호출할지 결정 → policy가 자주 가는 영역은 cheap, 드물거나 high-stakes 영역만 expensive. 정적 student 하나로의 압축이 아니라 conditional compute routing."
  - "BL-04 (co-distillation과 차별): teacher-student가 아니라, policy와 gating을 co-train해 'cheap model로 충분한 영역'을 policy의 실제 방문 분포에서 학습. gating은 policy improvement에 미치는 영향(value gap)으로 라우팅 — pure distillation loss가 아님."
  - "BL-08 (per-step controller): BL-08은 stop/continue 이진 controller로 reward-cliff에 실패. 본 아이디어의 gating은 stop이 아니라 model-tier 선택(둘 다 끝까지 rollout) → truncation/cliff 없음. 또한 라우팅 오류 시 expensive fallback이 안전망."
---

# Policy-Gated Progressive World Model: 방문 분포 기반 Conditional Compute Routing

## 핵심 가설
학습된 policy는 state space의 좁은 manifold만 방문하며, 그 영역의 dynamics는 cheap model로 충분히 정확하고 expensive model이 필요한 곳은 드문 high-stakes/OOD 영역뿐이다. cheap(fast)·expensive(accurate) 2-tier world model을 두고, policy-visitation과 value-gap으로 학습된 gating이 매 imagination step에서 cheap을 기본 호출하고 필요할 때만 expensive로 escalate하면, 평균 imagination 비용을 40~55% 줄여 model-based update를 1.5~1.9× 가속하면서 policy return drop을 4% 이내로 유지한다.

## 동기 (Why Now)
DreamerV3/TD-MPC2의 WM은 모든 state를 동일한 full-capacity로 예측한다. 그러나 학습이 진행되면 policy는 일부 영역만 방문하고, 그 on-policy manifold의 dynamics는 매우 예측 가능해진다(국소적이고 반복적). full model을 항상 쓰는 것은 낭비다. BL-04의 정적 distillation은 영역 구분 없이 하나로 압축해 어려운 영역에서 무너진다. 핵심 통찰: 압축 여부를 state별로 conditional하게 결정하고, 그 결정을 policy 방문 분포에서 학습하면 평균 비용만 낮추고 worst-case 품질은 expensive로 보장한다.

## 제안 방법
- 2-tier dynamics: `f_cheap`(작은 GRU/MLP, full의 ~20% FLOPs), `f_exp`(표준 RSSM). 둘 다 같은 latent space 공유(f_cheap은 f_exp의 시간-amortized 근사로 학습).
- gating `g(h_t) -> {cheap, exp}`: 입력은 latent + policy entropy + cheap/exp 예측 불일치 추정. 학습 신호 = "이 step에서 cheap을 썼을 때 value gap이 임계 이하인가"(policy improvement에 미치는 영향). on-policy rollout에서 value-gap 라벨 self-supervised 수집.
- escalation 안전: cheap 사용 후 reward/value head 잔차가 크면 즉시 expensive 재계산(routing 오류 fallback). cliff 영역(value 급변)은 자연히 high value-gap → expensive로 라우팅.
- policy와 co-train: gating이 cheap을 더 자주 쓰도록 하는 압력(compute budget loss) + policy가 cheap-friendly 영역에 머물도록 하는 정규화(선택적) → 효율-품질 공동 최적화.

```
for t in imagine:
   if g(h_t) == cheap:  h_t1 = f_cheap(h_t, z, a)       # 기본, 저비용
   else:                h_t1 = f_exp(h_t, z, a)         # escalate
   if ||reward_residual|| > τ: h_t1 = f_exp(...)        # fallback
```

## Amdahl 체크 (gate 도달 가능성)
- 타깃: dynamics transition = DreamerV3 imagination hot-path(latent-only). **precondition:** transition이 imagination 비용의 ~70%라는 가정은 given이 아니라 **PoC가 먼저 FLOP 분해로 측정**해야 하는 전제(BL-09가 가정된 bottleneck이 틀려 실패한 교훈).
- cheap 모델 = full의 ~20% FLOPs. policy가 수렴하면 cheap 라우팅 비율 60~75% 기대(on-policy manifold).
- 평균 비용: 0.7×0.2 + 0.3×1.0 + gating(~3%) ≈ 0.47× → transition 53%↓.
- 추정 speedup: transition hot-path 70%면 1/(0.3 + 0.7×0.47) ≈ 1.6×. cheap 비율↑·cheap FLOPs↓ 시 1.9×까지. → gate 1.5× 상회.

## Novelty 포인트 (최소 3개)
1. (vs BL-04 distillation) 단일 student 압축이 아니라 heterogeneous-cost mixture + conditional routing. 어려운 영역은 expensive로 보존 → distillation의 worst-case 붕괴를 회피.
2. (vs value-equivalence/IterVAML) value-equivalent model은 단일 모델을 value 보존하게 학습할 뿐 compute는 일정. 본 아이디어는 value-gap을 라우팅 신호로 써서 step별 compute를 가변화 — 효율이 1차 목표. (선행연구 위험에 명시)
3. (vs Mixture-of-Experts) MoE는 capacity 확장이 목표지만, 본 아이디어는 capacity 축소(평균 compute↓)가 목표이고 gating이 policy-visitation/value-gap에 종속 — RL-specific routing objective.

## 선행 연구 위험 요소
- Value-equivalence principle (Grimm 2020), IterVAML, value-aware model learning — **가장 가까운 위험**. "policy 방문 영역만 정확히" = on-distribution VAML로 읽힐 수 있음. 차별점: compute routing(가변 비용)이 핵심이지 value 보존 학습 자체가 아님.
- Mixture-of-Experts, conditional computation, early-exit
- Progressive networks, anytime prediction
- TD-MPC2(value-equivalent latent model) 자체
- Dyna with model error / model-based rollout under uncertainty

## 예상 실험 Skeleton
- Base model: DreamerV3 (2-tier dynamics fork), 보조 TD-MPC2
- Benchmark: DMControl, Atari 100k, (CoRL 각도용) robot manipulation sim
- 측정: cheap/exp 라우팅 비율, escalation/fallback 빈도, transition FLOPs, model update steps/sec, value-gap 분포, episode return
- 예상 결과: transition 40~55%↓, update 1.5~1.9×, return drop < 4%, 학습 후반 cheap 비율 상승 곡선

## 빠른 PoC 가능 여부
가능(2~3일). synthetic: toy MDP에서 학습된 policy의 방문 분포를 만들고, on-policy 영역에서 cheap model의 value-gap이 작은지, gating이 high-gap(OOD/cliff) 영역만 expensive로 보내는지, cheap-비율 vs return 곡선. value-gap 라우팅이 reward-cliff를 expensive로 자동 분류하는지(BL-08 회피 검증) 2~3일.

## Validator Gate 달성 평가
- rollout_speedup > 1.5×: **도달 가능**. cheap FLOPs와 라우팅 비율이 레버. policy 수렴 후 cheap 비율 60%+면 1.5~1.9×.
- quality_delta < 0.05: **유망하나 gating 정확도 의존**. value-gap 라우팅이 어려운 영역을 놓치면 drift. fallback(reward residual escalation)이 안전망. 핵심 검증: cheap 영역에서 value-gap < 0.05 유지 + gating이 high-gap 영역을 expensive로 보내는 정확도.

## Venue Fit 이유
conditional compute + model-based RL 효율 → NeurIPS. routing/anytime 이론은 ICLR. robot sim에서 실시간 planning 비용 절감은 CoRL 매력.

## 위험 요소
| 위험 | 가능성 | 완화 방법 |
|---|---|---|
| value-equivalence/IterVAML와 novelty 충돌 | 중 | compute-routing(가변 비용)을 1차 contribution으로 framing, value-gap은 신호로만 |
| gating 오류로 어려운 영역에 cheap 사용 → drift | 중 | reward-residual fallback escalation, value-gap 보수적 임계 |
| f_cheap이 f_exp latent와 정합 안 됨 | 중 | shared latent + f_cheap을 f_exp residual로 학습(stop-grad) |
| policy 미수렴 초반 cheap 비율 낮아 이득 적음 | 낮 | 초반 expensive 위주 → 수렴하며 cheap 비율 상승(학습 곡선으로 보고) |
