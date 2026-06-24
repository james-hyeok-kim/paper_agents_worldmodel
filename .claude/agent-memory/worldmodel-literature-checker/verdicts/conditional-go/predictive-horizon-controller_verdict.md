---
slug: predictive-horizon-controller
verdict: INCREMENTAL
checked-date: 2026-06-11 KST
papers-reviewed: 9
---

## 판정: INCREMENTAL

## 검색 요약

| 검색어 | 관련 논문 |
|---|---|
| adaptive imagination horizon Dreamer RSSM variable rollout length | ELVIS, COPlanner, Horizon Imagination |
| learned rollout length latent world model per-state horizon controller | AutoMBPO, MACURA |
| M2AC when to trust your model adaptive rollout truncation uncertainty | MACURA (2024 후속), M2AC (2020) |
| AutoMBPO adaptive rollout length schedule NeurIPS 2021 | AutoMBPO (Lai et al., 2021) |
| Hamrick metacontrol adaptive imagination value of computation | Hamrick et al. (ICLR 2017) |
| advantage residual rollout truncation model-based RL learned stop | 해당 논문 없음 확인 |
| DreamerV3 adaptive horizon variable horizon world model 2024 2025 | 해당 논문 없음 확인 |
| predictive horizon controller model-based RL imagination learned | 해당 논문 없음 확인 |
| Dreamer RSSM prior entropy rollout adaptive truncation | 해당 논문 없음 확인 |

## 관련 논문 목록 (9개 확인)

1. **Trust the Model Where It Trusts Itself (MACURA)** (Frauenknecht et al., ICML 2024) — per-state adaptive rollout 종료. GJS divergence over ensemble members를 threshold와 비교하여 step마다 continue/stop 결정. 관련성: per-state stop 메커니즘이 유사하나, heuristic threshold (학습된 controller 아님), ensemble 필수 (free internal signal 아님), MBPO 계열 (RSSM 아님).

2. **ELVIS: Ensemble-Calibrated Latent Imagination** (arXiv 2026) — RSSM 내부에서 critic ensemble UCB를 이용해 time-varying λₜ를 조정하여 adaptive horizon 효과를 냄. 관련성: RSSM 내 adaptive horizon이 유사하나, soft λ-modulation (hard stop 아님), ensemble critics 필수 (free signal 아님), deterministic 규칙 (learned controller 아님).

3. **Metacontrol for Adaptive Imagination-Based Optimization** (Hamrick et al., ICLR 2017) — model-free RL metacontroller가 imagination 횟수와 전문가 모델 선택을 결정. reliability + computational resource cost를 최적화하는 MDP. 관련성: 가치 기반 imagination 깊이 결정 개념이 유사하나, "model reliability + compute cost" 최적화 (advantage 잔차 기준 아님), pre-RSSM 아키텍처, per-state가 아닌 episode-level.

4. **On Effective Scheduling of Model-based RL (AutoMBPO)** (Lai et al., NeurIPS 2021) — hyper-MDP로 rollout length를 포함한 MBPO 하이퍼파라미터를 자동 스케줄링. model-free RL로 hyper-controller 학습. 관련성: learned controller로 rollout length 결정이 유사하나, global schedule (per-state 아님), MBPO 전용, 내부 신호 아닌 학습 진행 지표 사용.

5. **Trust the Model When It Is Confident: M2AC** (Pan et al., NeurIPS 2020) — 불확실성이 높은 상태의 rollout step을 마스킹. 관련성: rollout truncation 아이디어가 유사하나, fixed top-25% uncertainty threshold, ensemble 필요, Dyna-style MBPO (RSSM 아님).

6. **Dynamic Horizon Value Estimation (DMVE)** (arXiv 2020) — reconstruction 기반 이미지 유사도로 adaptive rollout horizon 결정. 관련성: per-state adaptive horizon이 유사하나, visual reconstruction 기반 (latent entropy 아님), learned controller 없음.

7. **When in Doubt, Think Slow** (arXiv 2024) — DreamerV3 기반, iterative inference로 latent representation 개선. 관련성: RSSM 내 imagination depth 관련이나, 고정 λ 사용 (adaptive gating 없음). 저자 스스로 "adaptive gating은 미래 연구"라 명시 — 제안 아이디어의 gap을 확인.

8. **COPlanner** (Li et al., ICLR 2024) — DreamerV3에 ensemble uncertainty penalty를 적용하여 불확실한 영역을 피하는 보수적 rollout. 관련성: DreamerV3 내 uncertainty-aware rollout이 유사하나, penalty 기반 action 선택 (per-state 조기 종료 아님), ensemble 필수.

9. **MBPO** (Janner et al., NeurIPS 2019) — model error schedule에 따라 rollout length를 전역적으로 조절. 관련성: rollout length adaptation의 원조이나, global schedule, heuristic, ensemble model.

## Novelty 분석

### 제안 방법과 유사한 점
- per-state rollout 조기 종료 개념: MACURA가 정확히 이를 구현 (step마다 GJS 비교)
- RSSM 내 adaptive horizon: ELVIS가 RSSM 안에서 time-varying λ로 구현
- imagination 깊이를 가치 기반으로 결정: Hamrick et al.의 metacontrol 개념

### 명확히 다른 점 (차별점)

**차별점 1: 무비용 내부 신호 — ensemble 없음**
기존 모든 per-state 종료 방법(MACURA, M2AC, ELVIS, COPlanner)은 ensemble disagreement/UCB를 신호로 사용. 제안 방법은 RSSM이 이미 계산하는 prior entropy 변화율(ΔH)을 재활용 — 추가 forward pass 0, ensemble 불필요. 이는 DreamerV3 계열에만 가능한 구조적 기회로, 기존 논문 중 이를 신호로 쓴 사례 없음.

**차별점 2: advantage 잔차를 타깃으로 학습된 controller**
MACURA/M2AC: heuristic threshold (학습 없음). AutoMBPO: hyper-MDP reward (task performance, global). Hamrick et al.: reliability + compute cost 최적화. 제안 방법은 "이후 step들의 advantage 기여분 |A_{t:H}|"를 직접 회귀 타깃으로 학습 — 신뢰도/비용이 아닌 "policy gradient에 대한 정보 이득" 기준. 이 정확한 학습 목표는 기존 논문에 없음.

**차별점 3: RSSM 계열에 hard per-state stop**
ELVIS는 RSSM 내 soft modulation (λ 조정), COPlanner는 action 선택 시 penalty. 제안 방법은 RSSM 내에서 hard early termination + terminal value bootstrap — 실제 computation 절감을 직접 달성.

## 판정 근거

제안 아이디어의 핵심 intersection — (1) RSSM 내부 무비용 신호 + (2) learned per-state controller + (3) advantage 잔차 학습 목표 — 을 동시에 충족하는 논문 없음.

그러나 구성 요소들은 개별적으로 존재함:
- per-state truncation (MACURA): ✓ 메커니즘 존재, 신호/아키텍처 다름
- RSSM 내 adaptive horizon (ELVIS): ✓ 존재, soft/ensemble 기반
- learned controller for rollout (AutoMBPO, Hamrick): ✓ 존재, global/cost 기반

"When in Doubt, Think Slow" (2024) 저자들이 adaptive RSSM gating을 명시적으로 미래 연구로 지목한 것은 이 gap이 실재함을 공인된 peer review를 통해 확인된 것.

따라서 판정은 **INCREMENTAL** — 방향은 이미 알려진 영역이나, 세 차별점이 동시에 미개척 상태. 단, 차별점 2(advantage 잔차 학습)의 강도가 Hamrick et al.의 value-of-computation 프레임워크와 "다른 적용"임을 논문에서 명확히 구분해야 함.

## 권고 사항

- **다음 단계**: worldmodel-idea-validator로 synthetic PoC gate 진행 가능
- **논문 작성 시 포지셔닝**: MACURA 대비 "ensemble-free + RSSM-native", ELVIS 대비 "hard stop + zero overhead", Hamrick et al. 대비 "advantage-residual target (not reliability+cost)"로 contribution table 구성
- **위험 클레임 완화**: 차별점 2는 "wholly novel"보다 "differently motivated and applied" 프레임이 심사 설득력 높음
- **추가 확인 권고**: Hamrick et al. 전문에서 학습 목표가 정확히 "value improvement per imagination step"인지 확인 후, 제안 방법과의 gap을 논문에서 명확히 서술
