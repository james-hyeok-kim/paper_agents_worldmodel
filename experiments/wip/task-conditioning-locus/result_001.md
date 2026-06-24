# Result 001 — task-conditioning-locus PoC (R14)

## 판정: FAIL ✗ (2026-06-22 KST)

## 수치 결과 (v3 최종 실행)

| pair | dyn_correct | dyn_wrong | dyn_Δ | rew_correct | rew_wrong | rew_Δ | ratio |
|------|-------------|-----------|-------|-------------|-----------|-------|-------|
| stand→walk | 0.0140 | 0.0116 | -0.17 | 0.0495 | 0.0573 | +0.16 | -0.97 |
| stand→run  | 0.0140 | 0.0120 | -0.14 | 0.0495 | 0.0601 | +0.21 | -1.51 |
| walk→run   | 0.0140 | 0.0117 | -0.17 | 0.1100 | 0.0570 | -0.48 | +2.84 |

**Q1 mean(rew_Δ/dyn_Δ) > 3: -0.48 → FAIL ✗** (2/3 pairs have negative ratio)
**Q2 mean(dyn_sensitivity) < 0.5: ~-0.16 → PASS ✓** (dynamics extremely robust)

## 실패 원인 분석

### 실패 A: 근본적 Tautology (치명)

walker-stand / walker-walk / walker-run은 **동일한 physics XML**을 공유함.
- 같은 body mass, inertia, joint structure (2D free-joint + 6 hinge joints)
- 유일한 차이: reward function 파라미터 (`move_speed` = 0 / walk_speed / run_speed)
- 따라서 **dynamics function 자체가 세 task에서 identical**

→ "dynamics robust to within-family embedding swap"은 발견이 아니라 **설계상 당연한 결과(tautology)**.
  same-physics → same dynamics → embedding이 달라도 dynamics prediction 불변.
  Q2 PASS는 model이 올바르게 동작하는 것을 확인할 뿐이지 novel finding이 아님.

### 실패 B: reward arm — OOD velocity sweep (기법적 실패)

velocity sweep states (qvel[1] 수동 설정)는 모델 훈련 분포 외부(OOD):
- 모델은 expert policy rollout 데이터로 훈련됨 (자연스러운 walking gait)
- velocity sweep은 qvel[1]만 설정하고 관절 속도는 0 → "몸통만 이동하는 물리적 비정상 상태"
- 결과: reward head가 잘못된 예측 (v=1에서 pred_walk=0.39 vs actual=0.83)
- wrong embedding이 우연히 더 낮은 MSE를 주는 경우 발생 → 비율 음수

debug 예시 (walk source, walk→run swap):
```
vel=0.0:  pred_walk=0.414, pred_run=0.306, actual=0.138
vel=1.0:  pred_walk=0.390, pred_run=0.310, actual=0.829  ← model 완전 틀림
vel=8.0:  pred_walk=0.867, pred_run=0.713, actual=0.829
vel=12.0: pred_walk=0.879, pred_run=0.945, actual=0.829
```

올바른 embedding으로도 중간 속도(v=1-4)에서 reward를 크게 틀림 → correct/wrong 차이 noise 수준.

## 핵심 교훈

1. **Same-embodiment family로는 dynamics robustness를 증명할 수 없음**: dynamics가 identical하기 때문
2. **Policy-generated data 없이 reward sensitivity를 테스트할 수 없음**: synthetic states는 OOD
3. **"분리된다"는 흥미로운 inversion 가능성**: reward head가 state features에 의존하고 task embedding을 underuse할 수 있음 — 단 in-distribution rollout 없이는 테스트 불가

## 다음 액션

- BLACKLIST BL-20 추가 (tautology가 주 kill reason)
- R15 생성: b1 (latent compression) / b2 (planning efficiency) / b3 (rollout efficiency) 방향
