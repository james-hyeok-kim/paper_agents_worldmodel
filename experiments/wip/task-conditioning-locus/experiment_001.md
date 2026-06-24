# Experiment 001 — task-conditioning-locus PoC (R14)

## 가설
동일 embodiment family (walker-{stand,walk,run}) 내에서:
- **Q1**: reward head가 wrong task embedding에 FRAGILE (sensitivity_ratio(rew/dyn) > 3)
- **Q2**: dynamics head가 wrong task embedding에 ROBUST (mean dyn_sensitivity < 0.5)

## 설정

**Base**: TD-MPC2 mt30-1M checkpoint (walker-stand=0, walker-walk=1, walker-run=2)
- latent_dim=128, task_dim=96, obs_dim=24 (walker), action_dim=6

**Same-family pairs**: (stand↔walk), (stand↔run), (walk↔run)

**Sensitivity metric**: `(loss_wrong - loss_correct) / (|loss_correct| + 1e-8)`

**Data strategies (v1→v3)**:
- v1: random rollout for both arms — reward variation near-zero
- v2: reward-filtered rollout — walk/run don't get upright states via task reward filter
- v3: random rollout for dynamics; velocity sweep (qvel[1] ∈ [0, 0.5, 1, 2, 4, 8, 12]) on stand-reward-proxy upright states for reward

## NULL 사전등록
- Q1 FAIL: mean ratio ≤ 3 → idea 死
- Q2 FAIL: mean dyn_sensitivity ≥ 0.5 → "dynamics confused" (Q2 was expected to pass trivially)

## 실행 스크립트
`poc_r14.py` — 2026-06-22 KST 작성
