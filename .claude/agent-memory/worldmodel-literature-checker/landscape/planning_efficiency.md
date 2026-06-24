# Landscape: Planning Efficiency

MCTS/MPPI 계획 효율화, selective planning 관련 논문 목록.

## 핵심 논문

| 논문 | venue | 핵심 기여 | 관련성 |
|---|---|---|---|
| MuZero (Schrittwieser et al., 2020) | Nature 2020 | MCTS + learned model | planning baseline |
| TD-MPC2 (Hansen et al., 2024) | ICLR 2024 | MPPI in latent space | latent planning |
| EfficientZero (Ye et al., 2021) | NeurIPS 2021 | Data-efficient MuZero | sample efficiency |
| Sampled MuZero (Hubert et al., 2021) | ICML 2021 | Continuous action MCTS | continuous planning |
| Monte-Carlo Graph Search (Leurent & Maillard, 2020) | ACML 2020 | exact-identity state merge in planning tree → regret bound 개선 | branch-shared-imagination 핵심 선행연구. tabular/symbolic, value estimation 목적. neural WM forward 절감 아님 |
| State Aggregation in MCTS (Hostetler, Fern & Dietterich, 2014) | AAAI 2014 | hand-crafted state partition으로 MCTS branching factor 축소 | approximate merge 선행, but pre-defined partition required |
| Bisimulation Metric for MPC (Shimizu & Tomizuka, 2025) | ICLR 2025 | bisimulation loss로 MPC encoder 학습 | training-time only, planning inference merge 없음 |
| Path Integral Control with Rollout Clustering (2024) | arxiv 2403.18066 | MPPI rollout clustering (안전성 목적) | 계산 절감 아님, latent WM 없음 |

## branch-shared-imagination 검증 결과 (2026-06-11)
- MCGS가 가장 가까운 선행연구: exact-identity state merge in MCTS tree
- 차별점: (1) KL-based approximate merge on stochastic latent z, (2) MPPI flat-sampling 구조 (structural prefix 없음), (3) neural WM forward count reduction as primary goal, (4) reward/value head sharing
- 판정: INCREMENTAL

## 검색 키워드 (검증 완료)
- "uncertainty-based planning depth" — 확인 필요
- "selective imagination model-based RL" — 확인 필요
- "MCTS world model efficiency" — 확인 필요

<!-- 새 논문 발견 시 추가 -->
