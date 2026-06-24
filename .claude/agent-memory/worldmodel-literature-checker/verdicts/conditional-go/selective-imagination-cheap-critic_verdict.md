---
slug: selective-imagination-cheap-critic
verdict: INCREMENTAL
checked-date: 2026-06-11 KST
papers-reviewed: 9
---

## 판정: INCREMENTAL

## 검색 요약
| 검색어 | 관련 논문 |
|---|---|
| "selective imagination" world model rollout start state | 없음 (직접 히트 없음) |
| prioritized model rollouts uncertainty value variance start state MBRL | 2502.07825, DreamerV3-XP |
| M2AC masked model-based actor-critic rollout selection | M2AC (NeurIPS 2020) — 판정 (b) |
| "imagination budget" allocation DreamerV3 latent | 없음 |
| MACURA uncertainty-aware rollout adaptation | MACURA (2405.19014) — 판정: within-rollout |
| "prioritized imagination" TD loss world model | 2502.07825 (핵심 위협) |
| "cheap critic" lightweight critic imagination selection latent | 없음 |
| "value variance" imagination rollout selection DreamerV3 | 없음 |
| "information gain" imagination rollout actor-critic latent | 없음 |

## 관련 논문 목록

1. **"Pre-Trained Video Generative Models as World Simulators"** (arXiv 2502.07825, 2025) — 관련성: **가장 가깝다.** "Prioritized Imagination" 섹션에서 TD-loss 크기(학습 진도 proxy)로 imagination 시작 state를 우선순위 선택. 핵심 (c) 메커니즘이 일부 겹침.

2. **"DreamerV3-XP: Optimizing exploration through uncertainty estimation"** (arXiv 2510.21418, 2025) — 관련성: replay buffer를 return + reconstruction loss + value error로 점수화해 우선순위 샘플링. imagination에 영향을 주나 compute 절감 목표 아님.

3. **"Trust the Model When It Is Confident: Masked Model-based Actor-Critic (M2AC)"** (arXiv 2010.04893, NeurIPS 2020) — 관련성: model uncertainty로 masking. 판정 (b)류: rollout 후 활용 여부 결정, pre-selection으로 rollout count를 줄이지 않음.

4. **"Trust the Model Where It Trusts Itself: MACURA"** (arXiv 2405.19014, 2024) — 관련성: rollout 내부에서 매 step GJS 발산으로 early termination. rollout 개수 축소 아님 — 시간축 내부 조절. advisor 분류상 within-rollout adaptation.

5. **"Acting upon Imagination: when to trust imagined trajectories"** (arXiv 2105.05716, 2021) — 관련성: imagined trajectory의 신뢰 여부 결정으로 replanning 계산 20~80% 절감. 그러나 shooting MBRL(시간축 내 plan 신뢰도) — DreamerV3 latent batch-axis 선택과 다른 문제.

6. **"On Rollouts in Model-Based RL (Infoprop)"** (arXiv 2501.16918, ICLR 2025) — 관련성: epistemic uncertainty 분리 + rollout 내 오류 누적 추적 → termination 기준 제공. 역시 within-rollout mechanism, start-state 선택이 아님.

7. **"Sparse Imagination for Efficient Visual World Model Planning"** (arXiv 2506.01392, 2025) — 관련성: token sparsity로 imagination forward 경량화. transformer 토큰 축소 문제 — latent batch-axis 선택과 다름.

8. **"MBPO: When to Trust Your Model"** (Janner et al., NeurIPS 2019) — 관련성: replay buffer에서 uniform 샘플링으로 model rollout 시작. 선택 없음. 기준선으로 확인.

9. **"DreamerV3: Mastering Diverse Domains through World Models"** (Hafner et al., 2023) — 관련성: 모든 replay state를 동일하게 H-step imagine. 선택 없음. 이 논문이 겨냥하는 비효율성의 출발점.

## Novelty 분석

### 제안 방법과 유사한 점
- **2502.07825 Prioritized Imagination**: start state를 rollout 전에 선택한다는 (c) 계층 메커니즘이 겹침. "다른 state보다 특정 state를 더 자주 imagine해야 한다"는 핵심 직관을 공유.
- **DreamerV3-XP**: value error 기반 prioritization이 "가치 관련 신호로 imagination 자원을 배분"이라는 방향성 공유.
- **M2AC, MACURA**: uncertainty 기반으로 model rollout 활용을 조절한다는 넓은 방향.

### 명확히 다른 점 (차별점 3개)

**차별점 1: 선택 기준 — value-variance (epistemic) vs TD-loss (learning-progress)**
2502.07825는 TD-loss 크기(예측 오류, 학습 진도)로 우선순위를 정한다. 이는 PER(Prioritized Experience Replay)의 imagination 버전이다. 제안 방법의 기준은 **value-variance** — 앙상블 value head의 disagreement 또는 stochastic prior에서의 K-sample 분산으로 측정한 **policy-gradient signal의 정보량**. 동일 state라도 TD-loss는 높고 value-variance는 낮을 수 있고 그 반대도 가능하다. TD-loss는 "모델이 얼마나 틀렸는가"를 보지만, value-variance는 "이 state를 imagine하면 policy gradient에 얼마나 기여하는가"를 직접 측정한다.

**차별점 2: 목표 — hard compute budget (M<B rollout) vs soft reweighting at constant compute**
2502.07825는 sampling frequency를 조절하지만 총 rollout 수를 명시적으로 줄이지 않는다. 제안 방법은 **budget M < B**를 설정해 imagination forward 호출 수 자체를 40~55% 절감하는 compute reduction이 명시적 목표다. 이는 speedup(1.5~1.9×)이 직접 측정 가능한 engineering 목표.

**차별점 3: 경량 proxy critic 아키텍처 — 5% 비용의 latent-only 1-step spread**
기존 어떤 논문도 "full H-step rollout의 ~5% 비용으로 value-variance를 추정하는 latent-only cheap critic"을 제안하지 않았다. 제안의 핵심 architectural innovation은 critic g(s_i) 자체 — K z-samples의 1-step value spread로 H-step rollout을 대리하는 proxy. 이 critic의 설계·학습·calibration이 PoC가 검증해야 할 핵심.

## Claim-by-Claim 비교 테이블

| 제안 클레임 | 선점 여부 | 가장 가까운 논문 | 판정 |
|---|---|---|---|
| start state를 rollout 전에 선택 | 부분 선점 | 2502.07825 | INCREMENTAL |
| 선택 기준 = value-variance (ensemble disagreement) | 미선점 | 없음 | NOVEL |
| hard budget M<B로 rollout count 절감 | 미선점 | 없음 | NOVEL |
| latent-only 1-step proxy critic (~5% 비용) | 미선점 | 없음 | NOVEL |
| DreamerV3 latent actor-critic imagination에 적용 | 미선점 | 없음 | NOVEL |
| trajectory-space (batch축) sparsity | 부분 선점 | 2502.07825 | INCREMENTAL |
| cliff-free (전체 선택 trajectory는 full horizon) | 미선점 | 없음 | NOVEL |

## 판정 근거

(c) 계층 메커니즘 — "rollout 전에 start state를 선택해 rollout compute 절감" — 의 개념적 조상이 **2502.07825**에 존재한다. 이 논문의 Prioritized Imagination은 TD-loss 기반 sampling prioritization을 implement하며, "다른 state보다 중요한 state를 더 자주 imagine해야 한다"는 직관을 공유한다.

그러나 2502.07825와의 차이가 충분히 구체적이고 검증 가능하여 NO-GO에 해당하지 않는다:
- 기준 신호가 다르다 (value-variance ≠ TD-loss)
- 목표가 다르다 (hard compute budget ≠ soft reweighting)
- proxy critic 아키텍처가 없다 (2502.07825는 cheap proxy 설계를 제안하지 않음)
- 적용 설정이 다르다 (video world simulator ≠ DreamerV3 RSSM latent imagination)

아이디어 파일의 Novelty 포인트 #2 ("vs prioritized replay")는 2502.07825에 의해 부분적으로 무효화되었다 — 2502.07825는 이미 "replay가 아니라 imagination 자체를 선택"한다. 그러나 value-variance criterion + hard budget이 살아 있는 차별점이다.

**잔류 위험**: "value-disagreement-based imagination allocation"을 정확히 구현한 논문을 검색했으나 발견하지 못함. 이 부재가 INCREMENTAL 판정의 핵심 가정이다. Validator 단계에서 관련 논문 추가 발굴 가능성을 열어둬야 한다.

## 권고 사항
- **다음 단계**: worldmodel-idea-validator로 이동. PoC의 핵심 검증점: (a) latent-only value-variance proxy가 full return-variance와 상관 > 0.7, (b) 동일 env-step에서 return drop < 5%, (c) imagination 호출 수 실측 절감 비율.
- **논문 서술 주의**: Related Work에서 2502.07825의 Prioritized Imagination과의 차이를 criterion(TD-loss vs value-variance)과 compute-budget 목표 차이로 명확히 서술해야 심사위원 공격을 막을 수 있다.
- **아이디어 파일 갱신 필요**: Novelty #2의 프레이밍을 "vs prioritized replay" → "vs TD-loss-based prioritized imagination (2502.07825)"로 교체할 것.
