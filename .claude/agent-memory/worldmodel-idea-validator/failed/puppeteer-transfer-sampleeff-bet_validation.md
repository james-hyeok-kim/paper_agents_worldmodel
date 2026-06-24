---
slug: puppeteer-transfer-sampleeff-bet (Round 8 open-space pointer)
verdict: FAIL (pre-registration gate, 0 compute)
date: 2026-06-19 KST
validated-by: team-lead (advisor-ordered pre-registration gate, harness 확인 후)
base: Puppeteer · type: diagnostic open-space bet (de-risk 불가, gen-r4 flag)
---

# PoC — Puppeteer transfer sample-efficiency bet → FAIL (pre-registration gate)

## 요약
R8 gen-r4의 단 하나 open-space bet("frozen 유능 low-level 하 high-level whole-body task 획득의 sample-eff frontier·한계요인"). harness 작동 확정(train.py task=walk low_level_fp=tracking.pt, B200서 학습됨). advisor 지시로 **학습 전에 non-obvious 가설 사전등록 게이트**(무료)를 먼저 침.

## 게이트: 4-bar 사전등록 (advisor)
non-obvious 가설 한 문장이 (1)plausible (2)Puppeteer 아는 이에게 surprising (3)decision-changing (4)실험으로 obvious와 구별가능 을 다 통과해야 함. 못 쓰면 = FAIL.

가장 유망한 후보(advisor 제안 = structural reachability ceiling): "frozen low-level의 reachable behavior manifold 밖 task는 high-level 학습 무관하게 capped, cheap reachability probe로 사전 예측":
- plausible ✓
- surprising ✗ — **Puppeteer(ICLR25)가 frozen low-level로 hurdles/gaps/stairs/walls/corridor + walk/stand/run 8개를 *이미 다 성공*(체크포인트 16개 보유 확인).** obstacle task가 manifold 안임을 이미 입증. tracker가 학습분포에 제한된다는 것도 자명.
- decision-changing ✗ — Puppeteer가 어느 task가 되는지 이미 맵핑. probe가 예측할 새 failure 없음.

대안 각도(good vs degraded low-level→acquisition: obvious monotone / negative-transfer / command-interface ceiling)도 전부 training 필요 + 4-bar 미통과.

## 판정: FAIL (compute 0)
4-bar 통과하는 cheap 가설 작성 불가 → 사전등록 게이트 FAIL. primary-source(Puppeteer 8-task 성공)가 핵심 ceiling 가설을 반증. advisor 예측대로 종이 위 cheap FAIL, gen-r4 "non-obvious gate에서 죽음" flag와 일치.

## 교훈 (재사용)
diagnostic-shaped idea(decision-fidelity-atlas, 이 bet)는 **학습 전에 non-obvious 가설 4-bar 사전등록**으로 무료 판정. base 논문이 이미 푼 task suite 위에서 "한계요인/reachability"를 묻는 건 surprising/decision-changing 안 됨.
harness artifacts: `baselines/puppeteer` (train.py high-level 작동), deps: sb3+matplotlib 추가.
