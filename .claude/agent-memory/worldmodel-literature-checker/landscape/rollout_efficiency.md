# Landscape: Rollout Efficiency

imagination rollout 속도, adaptive rollout depth, early termination 관련 논문 목록.

## 핵심 논문

| 논문 | venue | 핵심 기여 | 관련성 |
|---|---|---|---|
| DreamerV3 (Hafner et al., 2023) | ICML 2023 | Universal RSSM, categorical latent | baseline |
| TD-MPC2 (Hansen et al., 2024) | ICLR 2024 | Latent MPC, MPPI planning | planning efficiency |
| EfficientZero (Ye et al., 2021) | NeurIPS 2021 | MuZero + data efficiency | sample efficiency |

## 검색 키워드 (검증 완료)
- "adaptive rollout horizon world model" — 2022-2025 논문 없음 확인 필요
- "uncertainty-based planning depth MBRL" — 확인 필요
- "early exit world model rollout" — 확인 필요

## latent-delta-rollout 검증 중 발견 논문

| 논문 | venue | 핵심 기여 | 관련성 |
|---|---|---|---|
| Skip-RNN (Campos et al., 2018) | ICLR 2018 | binary gate로 RNN state update skip + carry-forward. 경량 gate, 연속 skip 시 확률 누적 | RSSM에 미적용. carry-forward 패턴의 원조 |
| DISK (arXiv:2602.00440, 2026) | arXiv preprint | diffusion WM에서 확산 스텝 단위 skip. 2×/1.6× speedup. consecutive skip cap + warm-up guard | diffusion WM 적용. RSSM GRU skip과 다름 |
| S5WM (arXiv:2502.20168, 2025) | arXiv preprint | S5 parallel scan으로 RSSM 대체. 10× training speedup, 4× overall MBRL speedup | SSM 기반 WM 가속. GRU skip이 아닌 병렬화로 가속 |

## 확인된 gap
- RSSM(DreamerV3) imagination rollout에서 GRU update를 step 단위로 skip해 FLOPs를 절감하는 논문: 미발견 (2026-06-11 기준)
- "Skip Dreamer", "temporally sparse RSSM", "early-exit RSSM GRU" 등 키워드로 검색 결과 없음

## predictive-horizon-controller 검증 중 발견 논문 (2026-06-11)

| 논문 | venue | 핵심 기여 | 관련성 |
|---|---|---|---|
| MACURA (Frauenknecht et al., 2024) | ICML 2024 | per-state rollout 종료, GJS divergence over ensemble members, threshold heuristic | per-state truncation — ensemble 필요, heuristic threshold, MBPO 계열 |
| M2AC (Pan et al., 2020) | NeurIPS 2020 | 불확실성 상위 25% 상태 step masking, fixed threshold | rollout masking — ensemble 필요, fixed threshold |
| AutoMBPO (Lai et al., 2021) | NeurIPS 2021 | hyper-MDP로 rollout length 포함 MBPO 하이퍼파라미터 자동 스케줄링 | learned schedule — global (per-state 아님), MBPO 전용 |
| Metacontrol (Hamrick et al., 2017) | ICLR 2017 | model-free RL metacontroller로 imagination 횟수 결정, reliability+compute cost 최적화 | value-of-computation 개념 — pre-RSSM, cost 기반 (advantage 잔차 아님) |
| ELVIS (arXiv 2026) | arXiv 2026 | RSSM 내 critic ensemble UCB → time-varying λₜ로 soft horizon modulation | RSSM adaptive horizon — soft modulation (hard stop 아님), ensemble 필요 |
| COPlanner (Li et al., 2024) | ICLR 2024 | DreamerV3 + ensemble uncertainty penalty, 불확실한 영역 회피 | DreamerV3 uncertainty-aware — action 선택 기반 penalty (per-state stop 아님) |
| When in Doubt, Think Slow (2024) | arXiv 2024 | DreamerV3 기반 iterative inference로 latent 개선 (고정 λ) | 저자들이 adaptive RSSM gating을 명시적 미래 연구로 지목 — gap 공인 |

## predictive-horizon-controller 검증 후 확인된 gap
- RSSM prior entropy 변화율(ΔH)을 무비용 신호로 쓴 per-state hard stop: 미개척
- advantage 잔차(|A_{t:H}|)를 직접 회귀 타깃으로 학습된 stopping controller: 미개척
- ensemble 없이 RSSM 내부 신호만으로 hard early termination + terminal value bootstrap: 미개척

<!-- 새 논문 발견 시 추가 -->
