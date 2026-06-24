---
slug: elite-staged-value-planning
verdict: INCREMENTAL
date: 2026-06-19 KST
checked-by: team-lead (직접 WebSearch)
base: TD-MPC2 · axis: planning efficiency
---

# Novelty Verdict — elite-staged-value-planning

## 판정: INCREMENTAL (margin 합리적, funnel 진행)
"cheap-rank 512 sample → full 5-head Q는 elite 64에만" 2-stage planning-value 평가의 직접 선점 미발견.

## 선행 + 차별
- **REDQ / DC-MPC critic subsampling**: critic 부분집합이나 *learning overestimation-bias* 목적. 본 방법은 *planning 비용* + elite-ranking 보존 2-stage. 목적·구조 다름.
- **AERO-MPPI (2509.17340) 2-stage MPPI**: drone collision-cost 조기폐기/guiding traj용. latent-WM value-ensemble 아님.
- **EfficientTDMPC (2605.16692)**: MPC objective sample-efficiency. planning 비용 절감 아님.
- 검색 결론: "specific combination may represent a novel research direction" — 직접 중복 없음.
- 메인 방어: cheap-rank(Q-subset vs distilled scalar)로 elite recall 보존하는 staged value의 novelty. CoRL 적합.

## 권고
funnel 진행. die-point(PoC): (1) terminal Q-term의 실제 planning 비용 비중(Amdahl, forward-count ~50% 추정이나 MLP 크기·vectorization 미반영 → 실측), (2) cheap-rank elite-recall@64 → return 보존. 실물 TD-MPC2 측정.
