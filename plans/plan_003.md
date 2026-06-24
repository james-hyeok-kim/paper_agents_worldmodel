# Plan 003 — Round 5: 새 axis(robustness/OOD/eval-diagnostic) funnel (실물 실험 가능)

## 목표
efficiency(BL-01~13)·quality/sample-eff(BL-14) 고갈 확인 후, **미탐색 axis**로 전환해
새 idea를 funnel(novelty → synthetic PoC → 실물 DreamerV3 실험)로 흘려보낸다. 전 게이트 통과만 push.

## 배경 (왜 axis 전환)
- Round 1~3 efficiency: BL-01~13, pass 0 (dual-rate만 예외적 GO)
- Round 4 family-pivot: 7/7 문헌 포화
- Round 4b quality/sample-eff: clean anchor 1개(learned-kl-reliability-lambda) → synthetic FAIL + **실물 DreamerV3 FAIL** (BL-14)
- **새 capability**: GL infra 수리(LD_LIBRARY_PATH=/home/jovyan/egl_libs) → 실물 DMControl 실험 가능

## 사용자 결정 (2026-06-19 KST)
(b) 새 idea 생성, 다른 axis(robustness/OOD/eval-diagnostic 등), funnel.

## Pivot 방향 (generator에 지시)
- efficiency·pure quality 회피. 후보 axis: WM **robustness**(distractor/perturbation/sensor noise), **OOD generalization**(train-test 환경 shift, unseen dynamics), **evaluation-diagnostic**(WM 실패 모드 probe/새 metric), 또는 WM-adjacent 미탐색 문제
- literature-fresh self-check 먼저(pending≠literature), 측정가능 precondition, BL-01~14 회피, collision manufacturing 금지
- 이제 실물 DreamerV3 검증 가능 → 이득이 실물 학습에서만 보이는 idea도 OK

## 게이트 (success criteria)
| Gate | 기준 | Fail |
|---|---|---|
| 1 Novelty | NOVEL/INCREMENTAL (NO-GO 탈락) | drop+log |
| 2 synthetic PoC | axis별 measurable + 사전등록 기준(통제 포함) | drop+log |
| 3 실물 DreamerV3 실험 | analytic + 품질/robustness/OOD metric (사전등록, baseline+통제) | drop+log |

- PASS = Gate 3까지 통과 → PushNotification + 기록
- agent(lit-checker/validator) 무응답 시 team-lead 직접 수행

## 진행 로그
- 2026-06-19 KST: plan 작성, gen-r4에 Round 5(새 axis) 생성 지시.
- 2026-06-19 KST: gen-r4 결과 — clean anchor 1개 `decision-fidelity-atlas`(eval-diagnostic). novelty INCREMENTAL이나 **thin**: 핵심 개념("state-wrong이어도 decision-right")을 value-equivalence(Grimm/VAML/IterVAML/**Model-Advantage 2106.14080**)가 선점. diagnostic atlas+DF_dyn/rew 분해만 차별.
- 2026-06-19 KST: gen-r4 **구조적 전략 노트** — "단일 base model × 포화 문헌 = 라운드당 ~1 clean"(4라운드 연속). throughput 레버: (a) 2nd base model, (b) novelty bar↓, (c) 현행.
- 2026-06-19 KST: **사용자 결정** — (1) decision-fidelity-atlas **drop**(thin novelty), (2) throughput = **2nd base model 추가** = **TD-MPC2**.

## Round 6 준비 — TD-MPC2 2nd base 통합 (2026-06-19 KST)
- TD-MPC2 = decoder-free latent + MPPI planning + Q-ensemble(5) — DreamerV3와 근본 다른 bottleneck(planning 512×H3×6iter, Q-ensemble, SimNorm latent). 새 문헌 cell 열림.
- vendored: `baselines/tdmpc2/` (공식 repo). 실행: hydra `python tdmpc2/train.py task=walker-walk obs=state steps=...`.
- **격리 conda env**: `/home/jovyan/envs/tdmpc2` (python 3.9 + torch 2.6.0 + tensordict/torchrl 0.7.2 + dm-control 1.0.16 + numpy 1.24.4) — 메인 env(torch 2.11, DreamerV3) 오염 방지.
- smoke test 전략: GL fix(LD_LIBRARY_PATH=/home/jovyan/egl_libs) + walker-walk 짧은 학습 → metrics/return 기록·무크래시 확인.
- 통과 후: gen-r4가 TD-MPC2 bottleneck 겨냥 idea 생성 → funnel(novelty → PoC → 실물 TD-MPC2 실험).
