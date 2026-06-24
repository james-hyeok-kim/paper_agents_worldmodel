---
slug: adaptive-mppi-budget
verdict: INCREMENTAL (thin)
date: 2026-06-19 KST
checked-by: team-lead (직접 WebSearch)
base: TD-MPC2 · axis: planning efficiency
---

# Novelty Verdict — adaptive-mppi-budget

## 판정: INCREMENTAL (thin — 선점 강함)
per-state 수렴신호 기반 MPPI iteration 조기종료의 **핵심 메커니즘이 선점**됨.

## 선점 (핵심 우려)
- **Adaptive Online Planning (arXiv:1912.01188)**: "deciding number of planning iterations on a per-timestep basis... later iterations less useful... measures improvement of new trajectories vs previous, terminate when improvement below threshold (stochastic)" — adaptive-mppi-budget의 핵심(개선신호 기반 per-step iteration 종료)과 거의 동일.
- **Learning to Plan, Planning to Learn (arXiv:2512.17091)**: adaptive RL-MPPI, value-uncertainty로 MPPI 탐색 가변.
- real-robot MPC는 실시간성 위해 iteration 조기종료가 관행.

## 남은 차별 (얇음)
- gen-r4 주장: "static sweep(BL-05) 아닌 *learned per-state stop*" + "latent-WM MPC 특화" + "BL-08과 달리 in-call 관측가능 신호". 그러나 1912.01188도 관측가능 개선신호로 per-step 종료 → 차별 margin 작음.

## 권고
**보류/후순위.** anchor(elite-staged, terminal term) PoC에서 dynamics term 보완이 꼭 필요하다고 나오면 재고하되, novelty가 얇아 단독 paper 가치 낮음. anchor 우선.
