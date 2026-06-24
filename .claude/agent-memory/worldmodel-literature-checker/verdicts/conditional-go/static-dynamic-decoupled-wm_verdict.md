---
slug: static-dynamic-decoupled-wm
verdict: INCREMENTAL
checked-date: 2026-06-11 KST
papers-reviewed: 11
---

## 판정: INCREMENTAL

## 검색 요약
| 검색어 | 관련 논문 |
|---|---|
| static dynamic decoupled encoder world model MBRL efficiency | DDP-WM, VDFD, Iso-Dream++ |
| content motion disentanglement world model RL efficiency | DDP-WM, DisWM, VDFD |
| temporal invariant static background encoder model-based RL | ReCoRe, TACO, invariant representation 계열 |
| denoised MDP task-relevant representation world model | Denoised MDPs (ICML 2022) |
| object centric world model efficiency SLATE SAVi | SOLD, OC-STORM |
| static dynamic disentanglement VLA 2602.03983 | DySta (arXiv 2602.03983) |
| background foreground separation encode once visual MBRL | SMG (arXiv 2410.10834), DDP-WM |
| temporal redundancy encode once perception world model | R2-Dreamer, DySta |
| temporal invariance contrastive loss world model separation | ReCoRe (2312.09056), 2209.14932 |

## 관련 논문 목록 (11개)

1. **DDP-WM: Disentangled Dynamics Prediction for Efficient World Models** (Yin et al., 2026, arXiv 2602.01780) — foreground/background을 분리해 sparse primary dynamics + low-rank background update로 9× 추론 가속. 단, DINOv2로 매 step 전체 재인코딩. 절감 지점이 dynamics predictor 단계이며 imagination-based RL이 아닌 MPC/플래닝 도메인.
2. **DySta: Efficient Long-Horizon VLA via Static-Dynamic Disentanglement** (arXiv 2602.03983, 2026) — "static tokens는 episode 내 단일 복사본 유지, recache gate로 필요할 때만 갱신"을 VLA 모델에 적용. 2× 추론 가속. encoder amortize + 동적 refresh 게이트 개념이 제안 아이디어와 직접 겹침. 단, imagination loop/latent dynamics/reward head가 없는 VLA 정책.
3. **Iso-Dream: Isolating and Leveraging Noncontrollable Visual Dynamics in World Models** (Pan et al., NeurIPS 2022, arXiv 2205.13817) — controllable/non-controllable 동적 성분을 action-conditioned·action-free 두 브랜치로 분리. MBRL에서 dynamics 분해의 원조 격. 단, static(에피소드 불변) 성분을 별도로 식별하거나 encoder를 amortize하지 않음.
4. **VDFD: Multi-Agent Value Decomposition Framework with Disentangled World Model** (arXiv 2309.04615, 2023) — action-conditioned / action-free / static 세 브랜치로 world model 분해. multi-agent RL에서 적용. static 브랜치 존재하나, 에피소드당 1회 인코딩·broadcast 메커니즘 없음.
5. **Denoised MDPs: Learning World Models Better Than the World Itself** (Wang et al., ICML 2022, arXiv 2206.15477) — 정보를 controllability × reward-relevance 2축으로 분류해 noise 제거. task-irrelevant/static 분리 개념 선구. 단, encoder amortization 또는 broadcast 메커니즘 없음.
6. **DisWM: Disentangled World Models for Visual RL with Distracting Videos** (arXiv 2503.08751, ICCV 2025) — offline 사전학습 action-free 모델에서 disentanglement를 world model로 distill. 배경 변동 robustness 초점. efficiency amortize 없음.
7. **SMG: Focus On What Matters: Separated Models For Visual-Based RL Generalization** (arXiv 2410.10834, 2024) — foreground/background encoder 두 브랜치 분리 + cooperatively reconstruction + 일관성 loss. 목표가 generalization이며 background encoding amortize 없음.
8. **ReCoRe: Regularized Contrastive Representation Learning of World Model** (arXiv 2312.09056, 2023) — world model에서 contrastive invariance loss로 task-irrelevant 특징 제거. temporal invariance contrastive 개념 선점.
9. **Contrastive Unsupervised Learning of World Model with Invariant Causal Features** (arXiv 2209.14932, 2022) — augmentation-invariant contrastive learning으로 인과적 특징 분리. intervention-invariant 보조 task.
10. **Iso-Dream++ / Model-Based RL with Isolated Imaginations** (arXiv 2303.14889, 2023) — Iso-Dream 후속. 복수 imagination rollout에서 controllable/noncontrollable 분리.
11. **R2-Dreamer: Redundancy-Reduced World Models** (arXiv 2603.18202, 2026) — Barlow Twins 기반 redundancy-reduction objective. decoder-free MBRL, DreamerV3 대비 1.59× 가속. temporal redundancy 감소 목표 동일하나 static/dynamic 분해 방식 아님.

## Novelty 분석

### 제안 방법과 유사한 점

- **메커니즘 가족이 기존재**: "static/temporal-invariant 정보는 드물게 계산하고, scene 변화 시 refresh" 패턴은 DDP-WM(dynamics 단), DySta(VLA token 단)에서 이미 출판됨
- **static branch in WM**: VDFD가 이미 world model에서 static 브랜치를 사용
- **Temporal invariance contrastive**: ReCoRe, 2209.14932가 world model에서 이미 사용
- **Non-object-centric disentanglement**: DySta는 객체 분할 없이 token-level로 static/dynamic 분리

### 명확히 다른 점 (유효 차별점)

1. **Imagination loop 내 encoder amortize**: DySta는 VLA(KV-cache 기반 정책), DDP-WM은 MPC/플래닝용. Dreamer 계열의 imagination rollout + latent dynamics + reward head 내에서 static encoder를 에피소드당 1회로 amortize하는 구조는 선행 논문에 없음
2. **Decoder까지 savings 전파**: imagination 중 decoder 호출 시 c를 broadcast해 decoder forward도 절감하는 설계는 기존 논문에서 다루지 않음
3. **Invariance-break refresh가 RSSM 내에 통합**: RSSM/SSM 기반 world model dynamics 루프 안에서 refresh 신호를 통합하는 구조는 선행 연구에 없음

## 판정 근거

**Claim-by-Claim 비교 (가장 가까운 논문 3개)**

| Claim | DDP-WM | DySta | Iso-Dream |
|---|---|---|---|
| static을 에피소드당 1회만 인코딩 (encoder amortize) | X — dynamics 단 절감, 매 step 전체 재인코딩 | O (VLA) — static token 1회 유지, recache gate | X — 분리하나 amortize 없음 |
| imagination rollout + latent dynamics + reward head에 적용 | X — MPC/플래닝만 | X — VLA 정책, imagination loop 없음 | 부분 O — WM 내 dynamics 분리는 있음 |
| decoder broadcast로 decoder forward 절감 | X | X | X |
| invariance-break refresh | X — 없음 | O (recache gate) — 개념 유사 | X |
| temporal invariance contrastive | X | X | X (ReCoRe/2209.14932가 선점) |
| 목적이 efficiency (not interpretability) | O | O | X — 성능/interpretability |

**결론**: "static을 encoder 단에서 에피소드당 1회 amortize하는 imagination-based MBRL WM" 구조 자체가 존재하지 않는 것은 사실이다. 그러나 핵심 메커니즘 가족(static/dynamic 분리로 efficiency 획득 + refresh gate)은 DysTA·DDP-WM이 이미 인접 도메인에서 출판했고, temporal invariance contrastive도 선점됐다. 제안 방법이 새로운 locus(Dreamer imagination loop 내 encoder/decoder amortize)에 기존 메커니즘을 적용하는 것이므로 **INCREMENTAL** 판정이 적절하다.

## 권고 사항

1. **다음 단계**: worldmodel-idea-validator로 이동해 PoC feasibility 검증
2. **핵심 차별화 포인트 집중**: "Dreamer 계열 imagination loop 내 encoder/decoder amortize" 하나로 contribution을 좁히고, DDP-WM·DySta를 명시적으로 비교하는 실험 설계 필요
3. **주의할 reviewer 공격**: temporal invariance contrastive 자체는 이미 선점됐으므로 "이 loss를 imagination WM에 처음 적용"이 아닌 "이 loss를 encoder amortize trigger로 활용"하는 각도로 포지셔닝
4. **DySta와의 명확한 차이 수치화 필요**: VLA의 KV-cache 절감 vs MBRL의 encoder/decoder amortize가 다른 병목을 공략함을 실험으로 보여야 함
5. **BLACKLIST 추가 불필요**: 동일 메커니즘이 imagination WM에 적용된 논문은 없으므로 기각 권고하지 않음
