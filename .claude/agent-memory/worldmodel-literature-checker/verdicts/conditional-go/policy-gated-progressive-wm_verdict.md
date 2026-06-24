---
slug: policy-gated-progressive-wm
verdict: INCREMENTAL
checked-date: 2026-06-11 KST
papers-reviewed: 16
---

## 판정: INCREMENTAL

## 검색 요약

| 검색어 | 관련 논문 |
|---|---|
| mixture of experts world model MBRL conditional routing 2023-2025 | Mixtures of Experts for Deep RL (parameter scaling, RL policy), T2MIR, PA-MoE |
| adaptive computation world model MBRL dynamic compute allocation | Metacontrol (2017), MACURA (2024), Acting upon Imagination (2021) |
| value-equivalent model conditional compute policy visitation | Value Equivalence (Grimm 2020), IterVAML (Farahmand 2018), λ-IterVAML |
| two-tier world model cheap expensive model selection | DMWM (2025, NeurIPS spotlight), Mixture-of-World-Models (2026) |
| DynaQ adaptive model selection policy switching | Switch-based Active Deep Dyna-Q (2018) |
| conditional computation early exit world model rollout | S-GRPO, Acting upon Imagination (2021) |
| progressive world model coarse-to-fine MBRL compute | AHEAD (spatial-temporal adaptive compute, 2026) |
| when and how much to imagine adaptive test-time scaling | AVIC framework (2026) |
| sparse imagination world model planning | Sparse Imagination for Efficient Visual World Model Planning (2025) |
| DDP-WM disentangled dynamics efficient world model | DDP-WM (2026, 9× speedup via foreground/background split) |
| heterogeneous model ensemble world model policy-dependent | 명확한 exact match 없음 |
| policy visitation visitation distribution world model routing | 명확한 exact match 없음 |
| model-based RL fast slow dynamics heterogeneous cost routing | Acting upon Imagination (2021), MACURA (2024) |
| value gap routing gating cheap expensive RL step selection | 명확한 exact match 없음 |
| mixture of world models policy routing compute reduction | Mixture-of-World-Models (2026): multi-task capacity, not compute reduction |
| anytime world model adaptive complexity state-dependent capacity | MACURA (2024), state-space WM acceleration |

## 관련 논문 목록 (16개 확인)

1. **Metacontrol for Adaptive Imagination-Based Optimization** (Hamrick et al., ICLR 2017, arXiv:1705.02670) — 관련성: **가장 가까운 선행연구**. metacontroller가 compute-heterogeneous expert들 중 어떤 모델을 선택할지 학습. routing 신호 = expert reliability + computational cost. 단, action optimization iteration 수준의 선택(planning loop 내 N회 반복), per-step latent dynamics routing 아님. RSSM/Dreamer 이전 시대 (2017). policy-visitation 기반 신호 없음, value-gap gating 없음.

2. **The Value Equivalence Principle for Model-Based RL** (Grimm et al., NeurIPS 2020, arXiv:2011.03506) — 관련성: "policy 방문 영역에서 정확한 모델" 개념의 이론적 뿌리. compute routing(가변 비용)은 다루지 않음. 본 아이디어의 가장 가까운 이론적 선행연구이지만 efficiency가 목적이 아님.

3. **IterVAML: Iterative Value-Aware Model Learning** (Farahmand, NeurIPS 2018) — 관련성: value-aware model learning. 단일 모델을 value 보존하게 학습하는 것이 목적, compute routing 없음.

4. **Acting upon Imagination: When to Trust Imagined Trajectories** (arXiv:2105.05716, 2021) — 관련성: trajectory-level uncertainty로 replanning 시점 결정. 단일 모델, trajectory-level 이진 결정(skip/not), policy-visitation 기반 아님, 2-tier model 없음.

5. **MACURA: Model-Based Actor-Critic with Uncertainty-Aware Rollout Adaption** (arXiv:2405.19014, 2024) — 관련성: model uncertainty로 rollout 길이를 adaptive하게 조정. 단일 모델, rollout length adaption이지 model-tier routing이 아님. cheap/expensive 2-tier 없음.

6. **DMWM: Dual-Mind World Model with Long-Term Imagination** (arXiv:2502.07591, NeurIPS 2025 Spotlight) — 관련성: 2-tier 구조(RSSM-S1 + LINN-S2). 그러나 목적은 System 1(직관) + System 2(논리 추론) 조합으로 long-term imagination 품질 향상. compute reduction이 목표가 아니고, policy-visitation/value-gap gating 없음.

7. **Mixture-of-World-Models: Scaling Multi-Task RL with Modular Latent Dynamics** (arXiv:2602.01270, 2026) — 관련성: task-conditioned expert mixture WM. routing = task clustering(gradient-based), compute reduction이 아니라 multi-task capacity 확장이 목표. policy-visitation/value-gap routing 없음.

8. **Mixtures of Experts Unlock Parameter Scaling for Deep RL** (2024) — 관련성: MoE를 RL policy network에 적용. WM imagination routing 아니고 parameter scaling이 목표.

9. **DDP-WM: Disentangled Dynamics Prediction for Efficient World Models** (arXiv:2602.01780, 2026) — 관련성: foreground(primary dynamics) + background(LRM 저비용 업데이트)로 9× speedup. 가장 가까운 현대적 유사작 중 하나. 그러나 분해 기준이 spatial(foreground/background)이지 policy-visitation이 아님. value-gap gating 없음. 고정된 2-stage 분해, conditional routing 없음.

10. **Sparse Imagination for Efficient Visual World Model Planning** (arXiv:2506.01392, 2025) — 관련성: transformer token sparsity로 planning 가속. 단일 모델 내 token 수 줄이기, model-tier routing 아님. policy-visitation/value-gap 무관.

11. **Switch-based Active Deep Dyna-Q** (arXiv:1811.07550, 2018) — 관련성: real vs simulated experience 전환. model quality 기반 binary switch. 2-tier model 아님, value-gap gating 아님.

12. **Value Equivalence: Deciding What to Model** (arXiv:2206.02072, 2022) — 관련성: rate-distortion으로 value-equivalent lossy compression. 단일 compressed model, compute routing 없음.

13. **Model-Value Inconsistency as a Signal for Epistemic Uncertainty** (arXiv:2112.04153) — 관련성: value inconsistency를 exploration/planning robustness에 사용. compute routing에 사용되지 않음. "value gap" 신호 자체는 존재하나 routing 목적이 아님.

14. **Metacontrol for RL: When and How Much to Imagine (AVIC)** (arXiv:2602.08236, 2026) — 관련성: visual spatial reasoning에서 imagination 양/시점 결정. VQA 도메인, latent RL WM이 아님. reward 기반 policy로 imagination call 여부 결정.

15. **On Rollouts in Model-Based RL (Infoprop)** (arXiv:2501.16918, 2025) — 관련성: aleatoric/epistemic uncertainty 분리로 rollout termination. 단일 model, termination 기준만 다룸.

16. **λ-IterVAML / λ-models** (arXiv:2306.17366, 2023) — 관련성: value-aware model learning 안정화. 단일 model, compute routing 없음.

---

## Novelty 분석

### 제안 방법과 유사한 점

**Metacontrol (2017) — 핵심 유사성:**
- compute-heterogeneous model들 중 선택을 학습 → 계산 효율화가 목표
- 여러 "expert" 모델을 상황에 따라 다르게 호출
- metacontroller가 RL로 학습됨

**DMWM (2025) — 구조적 유사성:**
- 2-tier WM (fast System 1 + slow System 2)
- imagination rollout에서 두 컴포넌트를 함께 사용

**DDP-WM (2026) — 목표 유사성:**
- single forward pass를 cheap/expensive 두 경로로 분해
- FLOPs 절감을 직접 측정

**MACURA (2024) — 아이디어 유사성:**
- model uncertainty를 compute 할당 신호로 사용
- adaptive rollout 기반 compute 절감

### 명확히 다른 점 (차별점 3개)

**차별점 1: Policy-visitation 기반 on-policy routing signal (vs. 모든 선행연구)**
- Metacontrol: cost + reliability (task difficulty, EVOC)
- MACURA: epistemic model uncertainty
- DDP-WM: spatial foreground/background
- DMWM: 고정 2-tier (fast intuition / slow logic), gating 없음
- **본 아이디어**: policy의 실제 방문 분포에서 학습한 gating — "현재 policy가 자주 가는 영역은 cheap으로 충분하다"는 RL-specific 신호. value-gap(cheap model과 expensive model의 value prediction 차이)을 routing 트리거로 사용.

**차별점 2: Per-step latent dynamics tier routing with reward-residual fallback (vs. Metacontrol)**
- Metacontrol: action optimization iteration 수준의 expert 선택 (planning loop의 N회 반복 중 어떤 모델을 쓸지)
- **본 아이디어**: 현대 RSSM/Dreamer 스타일의 latent imagination에서 매 transition step에서 f_cheap / f_exp 중 선택. 둘 다 끝까지 rollout(truncation 없음) + reward-residual fallback escalation 안전망. 2017년 Metacontrol은 pre-Dreamer 시대로 latent dynamics per-step routing 아님.

**차별점 3: Compute reduction as primary RL-specific objective (vs. MoE 계열)**
- MoE 계열: capacity 확장이 목표 (더 많은 parameter를 stable하게)
- Mixture-of-World-Models: multi-task specialization
- **본 아이디어**: 평균 transition FLOPs 40~55% 절감, policy update 1.5~1.9× 가속이 1차 목표. gating이 RL policy의 visitation 분포와 co-train됨.

---

## 판정 근거

### INCREMENTAL 판정 이유

**NOVEL이 아닌 이유:**
Metacontrol(Hamrick et al., ICLR 2017)이 "compute-heterogeneous model들 중 어떤 것을 imagination에 쓸지 learned routing으로 결정한다"는 핵심 원리를 이미 확립했다. 이 원리가 세부 구현(2017년 action optimization loop vs 2026년 per-step latent dynamics)은 달라도 카테고리 수준에서 선점한다. NOVEL 기준("동일 메커니즘을 world model efficiency/quality에 적용한 논문 없음")을 충족하지 못한다.

또한 DDP-WM(2026)이 WM을 cheap/expensive 두 경로로 분해해 FLOPs를 직접 절감하는 방향을 최신 문헌에서 이미 선점하고 있다. 분해 기준(foreground/background spatial)이 다르지만 "WM 내부에서 비용 비대칭 경로를 conditional하게 사용한다"는 방향성이 겹친다.

**NO-GO가 아닌 이유:**
검색한 16개 논문 중 **policy-visitation 분포와 value-gap을 per-step latent dynamics routing signal로 co-train하는 논문은 발견되지 않았다**. 이것이 가장 강한 novelty 핵심이다:
- Metacontrol의 routing signal은 cost/reliability (policy distribution 무관)
- DDP-WM의 분해는 공간적(foreground/background), 고정적
- MACURA의 uncertainty는 epistemic uncertainty (value-gap 아님)
- DMWM의 2-tier는 gating 없는 고정 조합

본 아이디어의 기여 각도는 "policy가 실제로 방문하는 on-policy manifold에서 학습된 value-gap 신호로 per-step dynamics tier를 선택한다"이며 이 조합은 선행연구에 없다.

**INCREMENTAL 판정:** 기존 문헌이 원리는 선점했으나 구체적 구현(per-step latent routing + policy-visitation/value-gap signal + Dreamer/TD-MPC2 통합 + reward-residual fallback)은 아직 없다. 명확한 차별점 3개가 존재하며 framing을 compute-routing efficiency + RL-specific routing signal로 잡으면 투고 가능한 contribution이 된다.

---

## 권고 사항

1. **다음 단계: validator(PoC) 진행 권고**. INCREMENTAL은 PoC gate 통과 기준인 rollout_speedup > 1.5× AND quality_delta < 0.05를 만족할 가능성이 충분하다.

2. **가장 강한 framing**: "Metacontrol의 원리를 현대 latent WM(DreamerV3/TD-MPC2)에 통합하면서 routing signal을 policy-visitation/value-gap으로 RL-specialization한다"는 각도. Metacontrol을 직접 citation하고 "what's new" 명확화.

3. **DDP-WM과의 차별화**: DDP-WM은 spatial foreground/background 고정 분해, 본 아이디어는 policy-visitation 기반 동적 routing. 둘을 Related Work에서 명시적으로 비교.

4. **Value-gap routing의 claim 보수화**: model-value inconsistency(arXiv:2112.04153)가 value 불일치를 다루지만 routing 목적이 아님 → routing 목적으로의 전용이 contribution이나 실험에서 BICHO(Acting upon Imagination의 reward divergence)와 비교해야 함.

5. **주의할 비교 대상**: DDP-WM(2026)이 매우 최근. 이 논문과 동일 venue/conference 제출 시 direct comparison 요구될 가능성 높음.
