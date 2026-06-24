---
slug: amortized-rollout-operator
status: literature-checked
verdict: INCREMENTAL
created: 2026-06-11 KST
category: D
venue-fit: [NeurIPS, ICLR, ICML]
blacklist-delta:
  - "BL-10 (rank-r cheap-per-step GRU 교체): cheaper-per-call이 아니다. sequential H-step GRU 호출(H번의 작은 kernel launch)을 단일 transformer/SSM forward pass 1회로 대체 → call 횟수 자체를 H→1로 줄임. GPU kernel launch overhead가 BL-10을 죽인 원인인데, 본 아이디어는 정확히 그 launch 횟수를 제거하는 방향(many-small→one-large, GPU가 선호하는 방향)."
  - "BL-06 (carry-forward delta-skip): 정보를 carry하거나 skip하지 않는다. 모든 H step의 latent를 명시적으로 예측하되, recurrent dependency를 amortized operator로 한 번에 푼다. drift corrector가 아니라 trajectory-level direct prediction."
  - "dual-rate(성공 패턴)와의 차별: dual-rate는 slow branch를 1/K 빈도로 sub-sample(여전히 sequential). 본 아이디어는 시간축 sequential dependency 자체를 제거 — H개 GRU step을 1개 parallel operator로 fold. 빈도 절감이 아니라 recurrence 제거."
---

# Amortized Rollout Operator: Closed-Loop Imagination을 단일 Parallel Pass로 Fold

## 핵심 가설
DreamerV3 imagination의 비용은 H-step sequential GRU 호출(H번의 작은 GPU kernel launch, 각 step이 직전 step에 의존)에 지배된다. policy와 prior를 함께 흡수한 amortized rollout operator가 초기 latent로부터 H-step closed-loop trajectory 전체를 단일 parallel forward로 예측하면, imagination을 2.5~4× 가속하면서 return drop을 5% 이내로 유지한다.

## 동기 (Why Now)
GPU에서 sequential recurrence는 치명적이다 — H=15 step rollout은 15번의 직렬 kernel launch이고, 각 GRU는 작아서(D=512) launch overhead가 연산을 압도한다(BL-10의 핵심 교훈). dual-rate가 1.93× 나온 이유도 결국 "큰 call을 덜 호출"했기 때문. 그러나 dual-rate조차 fast branch는 매 step sequential이다. 진짜 ceiling은 sequential recurrence 자체를 제거하는 것이다.

**핵심 blocker(정면 돌파):** DreamerV3 imagination은 closed-loop다 — `a_t = π(s_t)`라서 미래 action을 모르면 미래 state를 unroll할 수 없다. naive parallel unroll은 불가능. 따라서 amortized operator는 **policy를 내부에 흡수**해, 초기 state `s_0`만으로 closed-loop 하의 `(s_1,...,s_H)` 분포를 직접 예측해야 한다. 이것이 novelty이자 핵심 리스크.

## 제안 방법
- amortized operator `Φ(s_0; π) -> (ŝ_1, ..., ŝ_H)` : 작은 causal transformer 또는 parallel SSM(S5/Mamba 류)가 H개 latent를 한 번의 forward로 생성. positional/step embedding으로 시간축 표현.
- **closed-loop 일관성:** operator는 teacher인 full sequential rollout(GRU+π closed-loop)의 trajectory를 라벨로 distillation. 즉 `Φ`는 "현재 policy 하에서 GRU를 H번 돌린 결과"를 한 번에 근사. policy가 갱신되면 operator도 재학습(online distillation, stop-grad teacher).
- **stochasticity:** operator는 H-step latent의 결합 prior(autoregressive-in-parallel via masked attention, 또는 step별 prior head)를 예측. imagination 시 한 번의 parallel sample.
- **value bootstrap:** operator 출력 latent에 기존 value/reward head 그대로 적용. operator는 dynamics만 amortize, head는 불변.
- **policy-chase(make-or-break, 부차 리스크 아님):** DreamerV3는 imagination batch마다 actor를 갱신하므로 operator가 추적하는 teacher(=현재 π 하의 GRU rollout)가 매 step 움직이는 moving target이다. operator의 online distillation 비용이 절감분을 상쇄하지 않으려면 (a) operator fine-tune이 매우 적은 step으로 수렴(soft target/EMA로 target 안정화), (b) teacher full-GRU rollout은 sparse하게만 호출(예: M batch마다 1회)해 distillation 라벨 비용 자체를 amortize해야 한다. 이 chase 비용이 본 아이디어의 go/no-go이며 PoC에서 직접 측정.
- **policy-drift 안전:** operator 예측 trajectory와 sparse 호출한 full GRU trajectory의 KL을 모니터해 divergence 크면 teacher rollout 비중 상향(self-correcting trust region).

```
# teacher (sparse, for distillation only):
s = s_0; traj_T = []
for t in 1..H: a=π(s); s=GRU(s, sample(prior(s)), a); traj_T.append(s)

# student (used at imagination time, 1 parallel pass):
traj_S = Φ(s_0)                      # single forward, H latents at once
loss = KL(traj_S || stopgrad(traj_T)) # online distillation under current π
# imagination uses traj_S only → H sequential GRU calls collapse to 1 Φ call
```

## Amdahl 체크 (gate 도달 가능성)
- 타깃: imagination rollout 단계의 **sequential GRU+prior 연산**. BL-11이 지적한 "imagination이 update의 37%"는 *model-based update 전체* 기준 — 본 아이디어의 타깃 metric은 **imagination rollout speed(steps/sec)**이지 update throughput이 아님. 파일에 명시: PoC는 imagination 단계만 isolate해 측정.
- sequential H call → 1 parallel call: kernel launch가 H배에서 1배로. 연산량은 비슷해도 GPU에서 launch+직렬 의존성 제거가 실질 speedup의 원천(many-small→one-large).
- operator 자체 비용: transformer 1 pass ≈ GRU 2~3 step 수준이면, H=15에서 latency 15→3 등가 → 5× 이론, 실측 2.5~4× 보수 추정.
- **핵심:** 절감은 sequential dependency 제거(병렬화)에서 나옴. cheaper-per-step(BL-10) 아님 — call 횟수와 직렬 latency를 동시에 줄임.

## Novelty 포인트 (최소 3개)
1. (vs world model) RSSM의 closed-loop H-step recurrence를 policy-흡수 amortized operator로 fold하는 첫 시도. DreamerV3/dual-rate 모두 sequential을 유지 — 본 아이디어는 recurrence 자체를 제거.
2. (vs parallel SSM / 선형 RNN) S4/Mamba는 fixed dynamics를 병렬화하지만 action이 policy로부터 closed-loop로 들어오는 경우를 다루지 않음. 본 아이디어의 novelty는 policy-conditioned closed-loop trajectory를 amortize하는 것 — open-loop 병렬화와 근본적으로 다름.
3. (vs amortized inference / neural ODE) 정적 dynamics solver가 아니라, online으로 갱신되는 policy에 종속된 trajectory operator를 distillation으로 추적 + trust-region 안전장치. policy-drift 하에서의 amortization stability가 메커니즘 핵심.

## 선행 연구 위험 요소
- Parallel-in-time / parallel scan for RNN (S5, parallel RNN unrolling) — open-loop라 다름 강조
- Amortized planning / learned simulators (Dreamer의 actor-critic 자체)
- Diffusion/flow trajectory models (Diffuser, Decision Diffuser) — trajectory를 한 번에 생성하지만 policy-distillation/효율 목표 아님
- Trajectory transformers (Tт, Trajectory Transformer) — autoregressive라 sequential, 효율이 아닌 planning 목표
- Speculative/parallel decoding (정확도 무손실 목표)

## 예상 실험 Skeleton
- Base model: DreamerV3 (imagination loop를 amortized operator로 교체)
- Benchmark: DMControl proprio (smooth dynamics, amortization 학습 쉬움), Atari 100k
- 측정: **imagination rollout steps/sec(isolated)**, operator vs teacher trajectory KL, sequential GRU call 횟수, episode return, sample efficiency(env steps to target), imagined-vs-real latent MSE
- 예상 결과: imagination 2.5~4× 가속, return drop < 5%, operator-teacher KL bounded

## 빠른 PoC 가능 여부
가능(3일). synthetic: 학습된 toy RSSM+random policy에서, 작은 causal transformer가 H=10 closed-loop GRU trajectory를 single pass로 distill 가능한지 — trajectory MSE vs H, wall-clock(sequential 10 GRU vs 1 transformer)을 GPU에서 직접 측정. 핵심 go/no-go: (1) operator 1-pass latency < H×GRU latency가 GPU에서 실재하는지(launch overhead 검증), (2) closed-loop trajectory가 amortize 가능할 만큼 policy-smooth한지.

## Validator Gate 달성 평가
- rollout_speedup > 1.5×: **유망**. sequential→parallel은 GPU에서 가장 확실한 speedup 패턴(launch 제거). 단 operator 1-pass 비용이 충분히 작아야 함.
- quality_delta < 0.05: **핵심 리스크**. operator가 closed-loop trajectory를 얼마나 충실히 재현하느냐. policy update마다 operator가 따라가는 distillation stability가 go/no-go. 발산 시 trust-region으로 teacher 비중 상향.

## Venue Fit 이유
recurrence-free world model rollout은 model-based RL 효율의 최전선 + parallel-in-time이라는 분명한 메시지 → NeurIPS. amortization/distillation 이론 각도는 ICLR.

## 위험 요소
| 위험 | 가능성 | 완화 방법 |
|---|---|---|
| operator가 closed-loop trajectory를 못 따라가 quality 붕괴 | 중 | online distillation + trust-region, divergence 시 teacher rollout 비중 상향 |
| policy 급변 시 operator 재학습 비용이 절감분 상쇄 | 중 | operator fine-tune은 소수 step, teacher는 sparse 호출 |
| operator 1-pass 비용이 예상보다 커 ceiling 하락 | 중 | 작은 transformer/SSM, PoC에서 latency 선측정 |
