---
slug: amortized-rollout-operator
verdict: INCREMENTAL
checked-date: 2026-06-11 KST
papers-reviewed: 9
---

## 판정: INCREMENTAL

## 검색 요약
| 검색어 | 결과 수 | 관련 논문 |
|---|---|---|
| parallel prefix scan RNN world model MBRL | 10 | Accelerating MBRL with SSM World Models (2502.20168), S4WM (2307.02064) |
| parallel SSM S5 Mamba RL world model | 10 | HIEROS (2310.05167), Drama (2410.08893), S4WM |
| amortized trajectory prediction policy conditioned latent MBRL | 7 | PolyGRAD (2312.08533), Latent Geometry Beyond Search (2605.08732) |
| S4WM RSSM parallel imagination rollout closed-loop | 8 | S4WM, TransDreamerV3 (2506.17103) |
| trajectory distillation rollout single pass MBRL | 8 | TWIST (2311.03622), PolyGRAD (2312.08533) |
| closed-loop policy absorbed trajectory single forward pass | 6 | PolyGRAD (2312.08533) |
| DRAMA Mamba closed-loop imagination | 3 | Drama (2410.08893) |
| Horizon Imagination diffusion parallel denoising | 5 | Horizon Imagination (2602.08032) |

## 관련 논문 목록 (최소 5개)

1. **PolyGRAD: World Models via Policy-Guided Trajectory Diffusion** (2312.08533, NeurIPS 2024) — 관련성: **가장 가까운 선행 연구.** "단일 diffusion pass로 on-policy trajectory 전체를 생성"하며, policy gradient로 action을 trajectory에 흡수하여 closed-loop 비자기회귀적 trajectory 생성을 구현. 아이디어의 핵심 blocker("미래 action을 모르면 미래 state를 unroll할 수 없다")를 이미 해결. 단, full state+action+reward 공간에서 동작하며 iterative diffusion denoising 방식.

2. **S4WM: Facing Off World Model Backbones: RNNs, Transformers, and S4** (2307.02064, ICLR 2024) — 관련성: RSSM을 parallelizable SSM(S4 계열)으로 교체한 첫 world model. 단, 병렬화는 **학습 시**에 국한되며 imagination rollout은 여전히 sequential.

3. **Accelerating Model-Based RL with State-Space World Models** (2502.20168, Feb 2025) — 관련성: SSM으로 world model dynamics 병렬화. training 10× 가속, overall MBRL 4×. imagination rollout은 병렬화되지 않음.

4. **HIEROS: Hierarchical Imagination on Structured State Space Sequence World Models** (2310.05167, ICLR 2024) — 관련성: S5 기반 계층적 world model. S5의 parallel scan으로 학습 효율화, 그러나 imagination은 iterative (recurrent S5WM은 sequential 의존성 재도입). closed-loop policy conditioning 없음.

5. **Drama: Mamba-Enabled Model-Based RL** (2410.08893, 2024) — 관련성: Mamba 기반 world model로 O(n) 복잡도 달성. imagination 시작점에서 현재 policy action 사용 후 반복적으로 다음 latent 생성 — sequential 유지.

6. **TransDreamerV3: Implanting Transformer In DreamerV3** (2506.17103, 2025) — 관련성: TSSM이 학습 시 parallel update, **inference 시 sequential rollout 유지** (논문 명시: "roll out sequentially for trajectory imagination at test time"). policy 흡수 없음. 아이디어의 gap이 직접 확인됨.

7. **Horizon Imagination** (2602.08032, 2025) — 관련성: diffusion world model에서 multiple future 관측을 **denoising iteration 내에서 병렬** 처리. 단, H-step을 single forward pass로 압축하는 것이 아니라 iterative denoising 반복. closed-loop 안정화 메커니즘 포함.

8. **TWIST: Teacher-Student World Model Distillation** (2311.03622, 2023) — 관련성: world model distillation 패턴. teacher WM에서 privileged information으로 student WM 학습. trajectory-level distillation이 아닌 latent state alignment.

9. **Latent Geometry Beyond Search: Amortizing Planning in World Models** (2605.08732, May 2026) — 관련성: "amortizing planning"을 표방하나 H-step rollout이 아닌 single-step action 예측(GC-IDM)으로 online search 대체. rollout 자체를 amortize하지 않음. 100-130× per-decision cost 절감.

## Novelty 분석

### 제안 방법과 유사한 점

**PolyGRAD (2312.08533) — 핵심 선행 연구:**
- closed-loop policy 하에서 non-autoregressive trajectory 생성: 아이디어 Novelty #2와 직접 충돌
- policy gradient guidance로 action을 trajectory 안에 흡수: "policy를 내부에 흡수"와 동일 개념
- 전체 trajectory를 sequential하지 않게 한 번에 생성: 아이디어의 핵심 blocker 해결 방식

**SSM/Transformer WM 계열 (S4WM, HIEROS, Drama, 2502.20168):**
- RSSM sequential recurrence를 병렬 구조로 교체하는 방향 동일
- 학습 단계 병렬화는 기존 연구가 충분히 탐구

### 명확히 다른 점 (차별점)

1. **Latent space vs full state space:** PolyGRAD는 full state+action+reward 공간에서 동작. 제안 방법은 DreamerV3의 compact latent space(RSSM latent)에서 동작 → stochastic latent 구조, reward/value head 재사용 등 RSSM-specific challenge가 별도 존재.

2. **Single parallel forward pass vs iterative denoising:** PolyGRAD는 diffusion denoising을 반복 (B 번의 denoising iteration). 제안 방법은 단 1회의 transformer/SSM forward pass — GPU kernel launch 횟수를 H→1로 줄이는 것이 핵심. PolyGRAD는 launch overhead를 해소하지 못함. Horizon Imagination도 동일 한계.

3. **Online distillation from teacher GRU rollout:** 제안 방법은 현재 policy 하의 sequential GRU rollout을 teacher로 삼아 online distillation. policy update에 따른 moving target 추적 + trust-region 안전장치. PolyGRAD는 distillation 구조 없음.

4. **Imagination rollout speedup as primary metric:** 기존 SSM 기반 WM들(S4WM, HIEROS, Drama)은 모두 training speedup을 목표. imagination rollout speedup을 직접 타깃으로 삼아 sequential inference bottleneck을 제거하는 시도는 없음. TransDreamerV3도 imagination은 sequential 유지를 명시.

## 판정 근거

**NOVEL이 아닌 이유:** PolyGRAD (2312.08533)가 "policy를 흡수한 closed-loop non-autoregressive trajectory 생성" 메커니즘을 이미 구현. 아이디어 파일이 이 방향을 "첫 시도"로 주장한 Novelty #2는 유지되지 않음. 아이디어 파일 자체가 PolyGRAD 계열을 "return-conditioned, 효율 목표 아님"으로 선제 기각했으나, PolyGRAD는 명시적으로 policy-guided이며 closed-loop efficiency를 다룸.

**NO-GO가 아닌 이유:** PolyGRAD와의 결정적 차별점이 2개 이상 성립함 — (1) RSSM latent space에서 stochastic latent를 다루는 latent-domain amortization, (2) iterative denoising이 아닌 진정한 single-pass (launch overhead 제거가 핵심 메커니즘). 또한 모든 SSM/Transformer WM 계열이 imagination rollout은 sequential로 유지하고 있어, **inference-time imagination speedup이라는 gap은 실재**.

**INCREMENTAL로 판정:** 인접 방향(PolyGRAD)이 존재하지만, (latent-domain + single-pass + online-distillation + imagination-specific)의 조합은 선행 연구에 없음. NeurIPS/ICLR 투고 시 PolyGRAD를 정면으로 비교하고 latent efficiency + single-pass 우위를 실험으로 증명해야 함.

## 권고 사항

1. **다음 단계: worldmodel-idea-validator로 feasibility PoC 진행.** policy-chase 비용(moving target 추적)과 GPU에서의 실제 single-pass speedup(launch overhead 측정)이 go/no-go 핵심.
2. **PolyGRAD를 핵심 비교 baseline으로 설정.** 논문 작성 시 "PolyGRAD는 full state 공간 + iterative denoising; 우리는 latent space + single-pass"를 핵심 contribution 표현으로 사용.
3. **아이디어 파일 Novelty #2 수정 필요.** "첫 시도" 주장을 "PolyGRAD의 latent-domain/single-pass 확장"으로 재표현.
4. BLACKLIST 추가 불필요 — INCREMENTAL 수준의 차별점이 충분히 존재.
