# Experiment 001 — embedding-only-task-onboarding PoC

## 가설
1. **Q1**: in-manifold held-out task에서 embedding-only(96 params, frozen backbone) ≈ full FT (WM prediction loss 기준)
2. **Q2**: dynamics-similarity s1 (frozen dynamics residual) が embedding-only vs full FT gap를 예측함 (Spearman > 0.5)

## 설정

**Base**: TD-MPC2 mt30-1M checkpoint (7.6MB, Hugging Face)
- latent_dim=128, task_dim=96, obs_dim=24 (walker), action_dim=6

**Held-out tasks**: walker-walk 물리 변형 variants (mass_scale ∈ {0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0})
- mass_scale=1.0 = training 태스크 그 자체 (in-manifold, s1~0 예상)
- mass_scale=8.0 = 매우 OOD (s1 높음 예상)
- 같은 obs_dim/action_dim → encoder 구조 그대로 사용 가능

**Adaptation**: 각 variant에 대해
- A. embedding-only: row 30(새 태스크)만 학습, backbone 전부 freeze (96 params)
- B. full FT: 모든 params 학습 (separate copy)
- 최적화: Adam lr=1e-3, 500 steps (WM prediction loss = latent MSE)

**데이터**: 각 variant당 500 step random rollout → train 400 / test 100

**측정**:
- s1 = embed-only test loss (frozen dynamics + optimized embedding의 최소 latent MSE)
- gap = embed-only test loss − full FT test loss
- Q1 check: mass_scale=1.0에서 gap < 10% of embed-only loss
- Q2 check: Spearman(s1, gap) > 0.5

## 예상 결과
- mass_scale=1.0: s1 낮음, gap 작음 → Q1 PASS
- mass_scale=8.0: s1 높음, gap 큼 → OOD에서 full FT 우위
- Spearman > 0.5 → Q2 PASS

## NULL 사전등록
- Q1: ANY mass_scale에서 embed-only가 full FT와 comparable하지 않으면 → idea 死
- Q2: Spearman < 0.3 이면 → boundary prediction claim 死 (축소 fallback)

## 실행 스크립트
`poc_r13.py` — 2026-06-22 KST 작성
