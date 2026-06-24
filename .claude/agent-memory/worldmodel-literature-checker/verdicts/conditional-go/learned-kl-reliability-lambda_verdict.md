---
slug: learned-kl-reliability-lambda
verdict: INCREMENTAL
date: 2026-06-18 KST
checked-by: team-lead (literature-checker agent가 spawn 후 3회 연속 idle/무응답 → fallback 직접 검증)
axis: quality/sample-efficiency (DreamerV3 RSSM)
---

# Novelty Verdict — learned-reliability-weighted-λ-return

## 판정: INCREMENTAL (funnel 통과 — NO-GO 아님)

핵심 메커니즘(**ensemble-free, RSSM-internal 학습 KL-predictor가 prior-only imagined step별 reliability weight를 산출해 DreamerV3 λ-return target을 reweight**)을 직접 선점한 논문은 발견되지 않음. 단, 상위 개념("불확실/부정확한 imagined value target을 downweight")은 STEVE(2018) 이래 잘 다져진 영역이라 NOVEL이 아닌 INCREMENTAL.

## 가장 가까운 선행연구 + 차별성

| 선행연구 | 메커니즘 | 본 아이디어와의 차별 |
|---|---|---|
| **STEVE** (arXiv:1807.01675, NeurIPS 2018) | Q/reward/model **ensemble**의 inverse-variance weighting으로 horizon별 value target 보간 | ensemble 필요. 본 방법은 **ensemble-free** 단일 RSSM 내부 학습 head. + horizon-length 보간이 아니라 **per-step λ-return 기여 reweight** |
| **Uncertainty-based Value Expansion / UMVE** (arXiv:1912.05328) | 모델 불확실성 기반 value expansion 가중 | 여전히 uncertainty-from-ensemble 계열. KL-supervised single-head proxy 아님 |
| **Acting upon Imagination** (arXiv:2105.05716) | imagined trajectory 신뢰 online 평가 → **MPC re-plan/truncation**, probabilistic NN ensemble | ensemble + truncation + random-shooting MPC 대상. 본 방법은 ensemble-free + soft weighting + Dreamer/RSSM imagination 대상(cliff-free) |
| **DreamerV3-XP** (arXiv:2510.21418) | trajectory **replay 우선순위**(task relevance+recon+critic error) + 성능추세 기반 **global scalar** adaptive λ | replay prioritization과 global λ. 본 방법은 **per-horizon-step** 신뢰를 RSSM 내부 모델오차로 가중(다른 hook, 다른 신호) |
| MACURA (ICML 2024), M2AC (NeurIPS 2020) | per-state rollout **truncation/masking**, ensemble, fixed threshold | weighting(연속) vs truncation(이산), ensemble-free, 학습된 연속 신뢰 |

## 추가 확인 (WebSearch 2026-06)
- "learned **per-step** λ-return weighting to address compounding model error during imagination"은 문헌에 확립돼 있지 않음 — Dreamer 표준은 **fixed λ**. (per-step learned weighting은 fresh angle)
- DreamerV3의 기존 "imagined vs replay critic loss scale"은 **global scale**이지 per-step reliability가 아님 — 충돌 아님.

## novelty 기여의 무게중심 (논문 framing)
- 주장 가능한 핵심: (1) **ensemble-free** (STEVE 계열의 ensemble 비용 제거), (2) prior-only imagination의 구조적 한계(posterior-KL 부재)를 **학습된 KL-proxy head**로 메움, (3) truncation이 아닌 **soft per-step λ weighting** (cliff-free, BL-08 회피).
- 약점(reviewer 공격 지점): 상위 개념이 STEVE/UMVE와 같음 → "ensemble-free가 충분한 기여인가"가 핵심 방어 포인트. NeurIPS/ICLR full track엔 ablation(특히 ρ vs prior-entropy, weighting vs truncation)이 강해야 함.

## 권고
**funnel 진행 (→ validator PoC).** 단 PoC die point는 novelty가 아니라 **신호 validity**(ρ/critic-spread가 실제 multi-step drift와 상관하는가)임을 검증자에게 명확히 전달. 이게 fail하면 아이디어 자체가 무너짐(novelty와 무관).
