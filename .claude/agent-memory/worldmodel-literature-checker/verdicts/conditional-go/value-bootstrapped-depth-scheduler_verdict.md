---
slug: value-bootstrapped-depth-scheduler
verdict: INCREMENTAL
checked-date: 2026-06-11 KST
papers-reviewed: 10
---

## 판정: INCREMENTAL

## 검색 요약
| 검색어 | 결과 수 | 관련 논문 |
|---|---|---|
| adaptive rollout depth world model MBRL per-state value curvature | 7 | WIMLE, Double Horizon MBPO, DMVE |
| variable horizon imagination rollout length scheduling DreamerV3 | 8 | COPlanner, DreamerV3-XP, DMVE |
| MBPO rollout length adaptive per-state value bootstrap truncated return | 8 | MBPO, MACURA, To bootstrap or rollout? |
| anytime value estimation truncated lambda return bootstrap world model | 8 | Bootstrap Off-policy WM, BMPC, Averaging n-step returns |
| STEVE stochastic ensemble value expansion horizon adaptive per-state | 8 | STEVE, URE, DMVE |
| adaptive horizon value function bootstrap model-based imagination | 6 | Conservative Bayesian MBRL, DMVE, Long-Horizon Offline MBRL |
| batch grouping level grouping imagination rollout depth GPU efficiency | 8 | 없음 (해당 메커니즘 미발견) |
| Trust Model-Based Actor-Critic uncertainty rollout adaptation | 6 | MACURA (ICML 2024) |

## 관련 논문 목록 (10개)

1. **Trust the Model Where It Trusts Itself — MACURA** (Frauenknecht et al., ICML 2024, arXiv:2405.19014) — 관련성: per-step model uncertainty(GJS divergence)로 rollout 종료 결정. per-state 적응형 rollout length. 그러나 결정 시점이 per-step(rollout 중)이고, value function accuracy/curvature가 아닌 model ensemble disagreement 사용. value bootstrap 보상 메커니즘 없음. batch grouping 없음.

2. **Dynamic Horizon Value Estimation (DMVE)** (2020, arXiv:2009.09593) — 관련성: world model rollout horizon을 동적으로 조절. reconstruction-based novelty detection으로 horizon 선택. value curvature/accuracy 아님. per-state adaptive하지만 value bootstrap을 효율 레버로 사용하지 않음.

3. **Sample-Efficient RL with Stochastic Ensemble Value Expansion (STEVE)** (NeurIPS 2018, arXiv:1807.01675) — 관련성: ensemble uncertainty로 여러 horizon을 가중 혼합하여 value 추정. per-sample horizon 적응형. 그러나 GPU 효율화(batch grouping) 아님, value curvature 아님.

4. **When to Trust Your Model — MBPO** (Janner et al., NeurIPS 2019, arXiv:1906.08253) — 관련성: model error 기반 global rollout length 스케줄링. per-state 아님, value curvature 아님, bootstrap compensation 없음.

5. **Double Horizon Model-Based Policy Optimization** (arXiv:2512.15439, 2024) — 관련성: rollout을 distribution rollout + training rollout 두 단계로 분리. horizon tradeoff 처리. 그러나 per-state value curvature 기반이 아니고, lambda-return bootstrap을 효율 레버로 사용하지 않음.

6. **Conservative Bayesian Model-Based Offline RL** (ICLR 2023, arXiv:2210.03802) — 관련성: model-based vs model-free value prediction의 신뢰도를 학습 중 자동 조절하며 effective horizon이 변동. global 수준, value curvature 아님.

7. **Bootstrap Off-policy with World Model (BOOM)** (arXiv:2511.00423, 2024) — 관련성: world model + bootstrap loop 구조. planner가 policy를 bootstrap. rollout depth 조절 메커니즘이 주 기여가 아님.

8. **On Rollouts in Model-Based Reinforcement Learning (Infoprop)** (arXiv:2501.16918, 2025) — 관련성: aleatoric/epistemic uncertainty 분리로 rollout 종료 기준 제시. per-step model uncertainty 기반. value bootstrap compensation 없음.

9. **Diminishing Return of Value Expansion Methods** (arXiv:2303.03955 / 2412.20537) — 관련성: rollout horizon 증가의 diminishing return 분석. horizon과 value accuracy의 상호작용 논의. 이론 분석이며 효율화 방법 아님.

10. **Uncertainty-based Value Expansion (URE)** (arXiv:1912.05328) — 관련성: model uncertainty로 per-state value expansion 가중치 조절. ensemble disagreement 사용. lambda-return depth 명시적 조절 아님.

## Novelty 분석

### 제안 방법과 유사한 점
- per-state 또는 per-sample adaptive rollout horizon: MACURA, STEVE, DMVE 모두 구현
- rollout을 짧게 끊고 value function으로 보완: STEVE에서 부분적으로 존재
- model-based RL에서 rollout horizon을 학습 중 조절: MBPO, Conservative Bayesian MBRL

### 명확히 다른 점 (차별점)

1. **pre-rollout 결정 + value-curvature 신호**: MACURA는 per-step(rollout 중) model uncertainty 기반. 본 아이디어는 rollout 시작 전 초기 state의 value gradient norm, Hessian trace proxy로 depth를 1회 결정. 결정 시점과 신호 소스 모두 다름. 기존 논문들은 model uncertainty(ensemble disagreement, reconstruction error)를 사용하지, value function의 curvature(Hessian trace)를 bootstrap 충분 조건으로 사용하지 않음.

2. **lambda-return bootstrap을 효율 레버로 명시적 활용**: 기존 방법들은 rollout을 truncate하고 별도 value bootstrap을 덧붙이지만, 이를 "tail 보존 + cliff 제거 + 효율 레버"로 명시적으로 프레이밍한 논문은 없음. MBPO, MACURA 등은 truncation 후 value로 보완하지만 이것이 per-state depth 결정의 핵심 안전장치로 설계된 것은 아님.

3. **이산 level 양자화 + pre-rollout batch grouping으로 GPU lockstep 유지**: DreamerV3 imagination의 GPU lockstep 함정을 인식하고, depth를 소수의 이산 level로 양자화한 뒤 batch 전체를 pre-rollout에서 1회 grouping하여 fixed-depth batched rollout을 병렬 실행하는 메커니즘. MACURA의 per-step gather/scatter 방식(BL-11과 동일 문제)과 달리 per-step overhead 없음. 이 batch-grouping 설계를 논문 기여로 내세운 선행 연구 없음.

## 판정 근거

**INCREMENTAL**로 판정. 이유:

- 핵심 구성요소(adaptive rollout horizon, value bootstrap compensation, per-state scheduling)는 MACURA(ICML 2024), STEVE, DMVE 등에서 각각 선행 연구가 존재함.
- 그러나 세 가지 구성요소의 구체적 조합이 이 방식으로 구현된 선행 연구는 없음: (a) value curvature를 bootstrap 충분 조건으로 사용, (b) pre-rollout 1회 결정(not per-step), (c) 이산 level batch grouping으로 GPU lockstep 유지.
- 특히 (b)+(c)의 조합 — pre-rollout depth 결정 + GPU-friendly batch grouping — 은 MACURA가 명시적으로 해결하지 못한 GPU overhead 문제를 타겟으로 하며, DreamerV3 구조에 특화된 기여임.
- 단, value-curvature(Hessian trace proxy)가 실제로 bootstrap 충분 depth와 상관관계가 있는지는 PoC 검증이 필요하며, 이것이 실패하면 아이디어의 핵심 premise가 무너짐.

## 권고 사항
- worldmodel-idea-validator로 진행 권고 (CONDITIONAL-GO)
- PoC 최우선 검증: value Hessian trace proxy와 "bootstrap 충분 depth" 간 상관관계 측정 (BL-08의 0.085 실패 재현 방지)
- 논문 framing에서 MACURA와의 차별점 명시 필수: per-step vs pre-rollout, model uncertainty vs value curvature, no batch grouping vs batch grouping
- DMVE와의 비교 실험 추가 권고 (baseline으로 사용)
