---
slug: adaptive-mppi-budget
status: pending
base: TD-MPC2 (baselines/tdmpc2)
axis: efficiency (planning, test-time)
created: 2026-06-19 KST
venue-fit: [CoRL, ICLR, NeurIPS]
amdahl: SATISFIED for planning hot-path (코드 검증); 단 "후반 iteration 잉여" 전제는 EMPIRICAL-ONLY(미검증, die-point #1)
blacklist-delta:
  - "BL-05 (rollout horizon grid search): grid sweep이 아니다. TD-MPC paper가 이미 iterations/horizon static sweep을 했음 → 본 방법은 *per-state, in-call로 학습/관측된 수렴 신호*에 따라 iteration/sample을 동적 종료. static 최적값 탐색이 아니라 state별 adaptive 종료."
  - "BL-08 (prior-entropy adaptive horizon, learned controller 실패): BL-08은 *관측 불가능한 미래 reward-cliff*를 사전에 알아야 해서 실패. 본 방법의 수렴 신호(elite-value variance, elite-action std, mean 변화량)는 **MPPI iteration 내부에서 직접 관측 가능**(L190-194에서 이미 계산되는 양) — 미래를 예측할 필요 없이 '지금 분포가 수렴했나'만 봄. 이 observability가 BL-08과의 핵심 차이."
  - "BL-07 (branch merge): merge/dedup 아님. 전체 iteration/sample을 *통째로 skip*(call-skip, dual-rate 성공패턴). 겹치는 trajectory를 찾지 않으므로 continuous-action 발산과 무관."
---

# Adaptive MPPI Budget: 수렴 관측 신호로 iteration/sample을 per-state 동적 종료

## base & axis
**TD-MPC2, efficiency(test-time planning).** MPPI의 dynamics-rollout term(`_estimate_value`의 horizon×512 반복)을 per-state로 줄임.

## Amdahl & 전제 (정직)
- **Hot-path: SATISFIED(코드)** — planning은 TD-MPC2 test-time hot-path. dynamics term(horizon=3 × 512 sample × [`next`+`reward`], 매 iteration)이 per-sample 비용의 ~50%(나머지 ~50%는 terminal Q-ensemble — 그쪽은 [[elite-staged-value-planning]]이 침). 본 idea는 *iteration 수*와 *sample 수*를 줄여 dynamics term을 직접 절감.
- **전제(EMPIRICAL-ONLY, die-point #1)**: "iterations=6 중 후반 iteration이 수렴해 잉여"라는 전제는 **코드로 읽을 수 없고 실행해야 확인**(Round 4b의 posterior-existence처럼 code-readable이 아님). 따라서 die-point로 사전등록: 학습된 checkpoint 1개에서 instrumented planning call로 **iteration별 elite-value variance / elite-action std / mean Δ를 로깅** → 후반 iteration이 실제로 수렴하는지(잉여 존재) 선확인. 잉여 없으면 idea 기각. (terminal-Q anchor가 Amdahl-verified라 deliverable이 이 전제에 *의존하지 않음* — 안전.)

## 제안 메커니즘
- MPPI 루프(L174-198)에서 매 iteration 후 **수렴 신호** 계산(대부분 이미 L190-194에서 계산됨):
  - elite-value variance(elite_value의 분산), elite-action std(L194의 std), mean 변화량 ‖mean_k − mean_{k-1}‖.
- **Iteration 종료**: 신호가 임계 아래로 수렴하면 남은 iteration skip(예: 6→3). 임계는 fixed가 아니라 *학습된 stopping rule* 또는 분포 통계 기반 적응(BL-02/BL-05 회피).
- **Sample 감축(선택)**: 초기 iteration은 적은 sample로 거친 탐색, 수렴할수록 sample 유지/축소 — 또는 elite 주변만 재샘플.
- 핵심: 절감은 *통째로 skip된 iteration/sample*에서 나옴(call-skip, GPU-friendly), micro-dedup 아님.

```
for k in range(max_iterations):           # 기존 6
    value = _estimate_value(z, actions)   # dynamics + terminal
    elites = topk(value, num_elites)
    mean, std = update_mppi(elites)        # L190-194
    sig = convergence_signal(elite_value, std, mean_delta)  # 대부분 재사용
    if learned_stop(sig): break            # 후반 iteration call-skip
```

## 왜 효과 (무엇을, 얼마나)
- 후반 iteration이 잉여면(전제), iteration 6→평균 3~4로 줄여 dynamics+terminal 양쪽 forward를 통째로 절감 → planning 1.3~1.8× (전제 성립 시).
- return 보존: 수렴한 뒤 멈추므로(아직 개선 중이면 계속), 품질 손실 최소가 가설.

## 측정 metric + 예상 (실물 TD-MPC2)
- 측정: **(die-point #1) iteration별 수렴 신호 로그**(잉여 존재 확인), planning wall-clock/step, episode return(drop), 평균 사용 iteration 수, return vs iteration-budget Pareto.
- 예상: 전제 성립 시 planning 1.3~1.8×, return drop < 3%. (gate: 잉여 확인 AND wall-clock 단축 AND return 비유의 차.)

## 핵심 ablation + baseline
1. **baseline: fixed iteration 감축**(6→3 무조건) vs adaptive. adaptive가 같은 평균 비용에서 return 더 보존하는가(핵심 — BL-05 static sweep과의 차별).
2. **수렴 신호 종류**: elite-value var vs action std vs mean Δ vs 학습된 stop. 어느 것이 return-safe 종료를 가장 잘 예측.
3. **per-state 가치 입증**: 종료 iteration이 state마다 다른가(다르면 adaptive가 의미; 항상 같으면 static으로 충분 → 약화).
4. easy/hard state에서 사용 iteration 분포(adaptive가 hard에 더 쓰는가).

## 선행 연구 위험 (literature self-check 완료)
- **classical adaptive-MPPI**(dsMPPI, uncertainty-aware MPPI, Biased-MPPI): robotics MPC에서 adaptive sampling 존재하나 **learned latent WM(TD-MPC2) context 아님**, 대부분 분포 scaling이지 *iteration 조기 종료* 아님. 차별점: latent-WM value 기반 수렴 + per-state stop.
- **TD-MPC paper**: iterations/horizon **static** 비용-성능 sweep 함 → 본 방법은 *adaptive/learned*(BL-05 차별).
- **TD-M(PC)²(ICLR 2025)**: policy constraint로 planning *의존도*를 학습-시 낮춤. test-time iteration 동적 종료 아님(다른 메커니즘). 단 "TD-MPC2 성능의 상당부분이 online planning"이라 했으므로 **planning 과도 절감은 return 위험** → adaptive(수렴 시만 멈춤)가 정당화.
- 메인 리스크: "adaptive iteration termination for sampling-based planning"이 classical MPC에 있는지 + TD-MPC2 특화 novelty가 충분한지(literature-checker).

## venue
CoRL/ICLR. 단 classical adaptive-MPPI와의 차별이 [[elite-staged-value-planning]]보다 미묘 → 그쪽이 anchor, 본 idea가 보완(orthogonal: 50/50 split의 다른 절반인 dynamics term을 침).

## 위험 요소
| 위험 | 가능성 | 완화 |
|---|---|---|
| 후반 iteration 잉여 전제 불성립(6 다 필요) | 중 | die-point #1로 선확인. 불성립이면 기각(terminal-Q anchor가 deliverable 보전) |
| classical adaptive-MPPI와 novelty 충돌 | 중-높 | latent-WM + per-state learned stop + TD-MPC2 특화로 3점 차별, literature-checker 정밀 확인 |
| planning 절감이 return 훼손(TD-M(PC)²: 성능이 planning 의존) | 중 | 수렴 시만 멈춤(아직 개선 중이면 full), return-safe stop을 ablation 2로 검증 |
| wall-clock 이득 < FLOPs (vectorization) | 중 | wall-clock gate |
