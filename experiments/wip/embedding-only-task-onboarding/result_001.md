# Result 001 — embedding-only-task-onboarding PoC

## 판정: FAIL ✗ (2026-06-22 KST)

## 수치 결과

| mass | base | embed | fullFT | gap (abs) | gap/embed (rel) | s2 |
|------|------|-------|--------|-----------|-----------------|-----|
| 0.1 | 0.001418 | 0.001041 | 0.000443 | 0.000598 | 57% | 0.2101 |
| 0.3 | 0.000958 | 0.000713 | 0.000311 | 0.000401 | 56% | 0.2430 |
| 0.5 | 0.000521 | 0.000399 | 0.000166 | 0.000233 | 58% | 0.3329 |
| 1.0 | 0.000569 | 0.000395 | 0.000133 | 0.000262 | 66% | 0.3719 |
| 2.0 | 0.000278 | 0.000168 | 0.000023 | 0.000145 | 86% | 0.4184 |
| 4.0 | 0.000919 | 0.000299 | 0.000015 | 0.000284 | 95% | 0.4548 |
| 8.0 | 0.000422 | 0.000238 | 0.000003 | 0.000236 | 99% | 0.5378 |

**평가 metric**: collapse-proof (z_true = frozen base encoder + live embedding) — 공정한 비교.

## Gate 결과

| Gate | 기준 | 결과 | 판정 |
|------|------|------|------|
| Q1 | gap < 20% of embed-only @ mass=1.0 | gap=0.000262, thresh=0.000079 (66%) | FAIL ✗ |
| Q2 | Spearman(s2, gap) > 0.5 | rho=-0.536, p=0.215 | FAIL ✗ |

## 실패 원인 분석

### Q1 실패
- mass=1.0 (in-manifold)에서도 embed-only가 full FT보다 3× 높은 eval loss
- **진짜 원인**: frozen dynamics backbone은 질량이 다른 task에서 embedding만으로 dynamics 변화를 보상 못함
- 질량 변형 = dynamics function 자체를 바꿈 (힘=질량×가속도). 96-dim embedding은 "어떤 task인지"를 알려주지만, dynamics network 자체 가중치를 바꾸지 않으면 다른 physics를 모델링 불가
- **핵심 교훈**: in-manifold (mass=1.0)조차 embed-only가 full FT보다 훨씬 나쁨 → "96-dim ≈ backbone" 가설 기각

### Q2 실패
- s2(embedding 이동 거리)와 absolute gap의 Spearman 상관 = -0.536 (음의 상관)
- **이유**: base loss가 mass scale에서 비단조 (random-action rollout의 state distribution 편향)
  - 가벼운 robot(mass=0.1): 작은 action → 큰 state 변화 → random rollout이 wide state distribution → 기본 loss 높음 → absolute gap 큼
  - 무거운 robot(mass=8.0): 작은 action → 작은 state 변화 → narrow distribution → 기본 loss 낮음 → absolute gap 작음
- **상대 gap**(gap/embed)은 단조증가 (57%→99%), s2와 양의 상관 가능하지만 사후 선택이므로 invalid

## v3 PoC 설계 개선사항 (이전 버전 대비)

1. **Collapse-proof target**: z_true를 frozen base encoder로 고정 (v1/v2의 full FT latent contraction 문제 해결)
2. **독립적 Q2 predictor**: s1(embed-only loss) 대신 s2(embedding space L2 distance) 사용 → circular correlation 제거
3. **500 steps** 적용

→ 개선 후에도 genuine FAIL. 이전 Q2 PASS (rho=0.964)는 circular artifact였음이 확인됨.

## 사전등록 NULL 조건 적용

> "Q1: ANY mass_scale에서 embed-only가 full FT와 comparable하지 않으면 → idea 死"

mass=1.0에서도 comparable하지 않음 (66% gap). **→ idea 死 확정.**

## 다음 액션

- BLACKLIST에 BL-19로 추가
- R14 생성: b1/b2/b3 방향에서 새 아이디어
