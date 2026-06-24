---
slug: decision-fidelity-atlas
verdict: INCREMENTAL (thin margin)
date: 2026-06-19 KST
checked-by: team-lead (lit-checker agent 무응답 → 직접 WebSearch)
axis: evaluation-diagnostic
---

# Novelty Verdict — decision-fidelity-atlas

## 판정: INCREMENTAL (margin 얇음, novelty 우려 명시)

NO-GO는 아님(diagnostic atlas + DF_dyn/DF_rew 분해는 직접 복제 아님). 그러나 **핵심 개념이 선점**되어 margin이 얇다.

## 선점 (핵심 우려)
- **value-equivalence / value-aware / decision-aware model learning**이 "model은 state-정확이 아니라 *decision/value 보존*이 중요"라는 본 아이디어의 개념적 핵심을 정확히 정립:
  - Value Equivalence Principle (Grimm 2020, arXiv:2011.03506), **Proper Value Equivalence** (2106.10316)
  - VAML / **IterVAML**, Decision-Aware Model Learning, λ-models (2306.17366)
  - **Model-Advantage and Value-Aware Models** (arXiv:2106.14080) — advantage 보존 직접 다룸 (가장 가까움)
  - Calibrated Value-Aware Model Learning (2505.22772)
- 기존 diagnostic도 일부 존재: imagined-vs-real을 **action/input-state/reward 레벨에서 비교해 deviation 추적**(WorldLens 등), "Action Simulation Fidelity" 메트릭.

## 차별점 (남은 novelty)
- value-equivalence는 **training objective(loss)**, 본 아이디어는 표준 DreamerV3를 안 바꾸고 **측정/진단**(diagnostic).
- **DF_dyn vs DF_rew 분해**(decision-infidelity를 dynamics/reward로 귀속) — 미발견.
- horizon×modality×in/OOD **격자 atlas** + state-fidelity vs decision-fidelity 괴리 map.
- → 이게 충분한 기여인지가 핵심 방어 포인트. NeurIPS D&B(benchmark)면 가능하나, 본트랙이면 reviewer가 value-equivalence 선점을 지적할 위험 큼.

## 권고
funnel 진행 가능(INCREMENTAL)이나 **margin이 얇아** 실험 투자 전 team-lead 판단 권장. die point(PoC): low-DF 셀이 실제 학습-return/gradient 손상과 대응하는가(proxy validity) — 이게 fail하면 진단 자체가 무의미.
