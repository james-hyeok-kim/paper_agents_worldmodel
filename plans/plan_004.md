# Plan 004 — Round 6: TD-MPC2 / Puppeteer funnel (새 base, 실물 실험)

## 목표
새로 셋업·검증한 base 2종(TD-MPC2, Puppeteer)을 겨냥해 idea 생성 → funnel(novelty → PoC → 실물 실험) → 전 게이트 통과만 push.

## 배경
- DreamerV3 축(efficiency BL-01~13, quality/sample-eff BL-14) 고갈, 실물 FAIL 확정.
- 사용자 결정: 2nd base 추가(TD-MPC2) → 셋업 완료 → 후속 Puppeteer도 셋업 완료. DreamerV4는 toy 구현뿐이라 base 보류.
- **새 bottleneck = fresh 문헌 cell**: TD-MPC2 MPPI planning은 *진짜 hot-path*(BL-11/Amdahl 함정의 반대) → planning-efficiency 재개봉 가능.

## base 인프라 (작동 확인됨, [[base-models-infra]])
- TD-MPC2: `baselines/tdmpc2`, env `/home/jovyan/envs/tdmpc2`(torch 2.7.1+cu128, B200), smoke R 27→83.
- Puppeteer: `baselines/puppeteer`, 동일 env + gym, 체크포인트 16개, eval corridor R 11.7 확인. mocap 454M·h5py 해결됨.
- GL: `LD_LIBRARY_PATH=/home/jovyan/egl_libs MUJOCO_GL=egl`. GPU 1 사용(GPU 0 외부 작업).

## Pivot 방향 (gen-r4 지시)
TD-MPC2(MPPI planning 비용, Q-ensemble, decoder-free latent, multitask) / Puppeteer(hierarchical, humanoid) 의 bottleneck 겨냥. efficiency/quality/robustness/hierarchical 다 열림. literature-fresh 먼저, BL-01~14 회피, 실물 측정 가능, collision 금지.

## 게이트 (success criteria)
| Gate | 기준 | Fail |
|---|---|---|
| 1 Novelty | NOVEL/INCREMENTAL (NO-GO 탈락) | drop+log |
| 2 PoC | axis별 measurable + 사전등록 기준(통제 포함). planning-eff면 FLOPs/wall-clock + quality | drop+log |
| 3 실물 실험 | 실제 TD-MPC2/Puppeteer에서 baseline+통제, 사전등록 metric | drop+log |

- PASS = Gate 3 통과 → PushNotification + 기록. agent 무응답 시 team-lead 직접 수행.

## 진행 로그
- 2026-06-19 KST: TD-MPC2/Puppeteer 셋업·smoke 완료, DreamerV4 보류 확정. gen-r4에 Round 6 생성 지시.
- 2026-06-19 KST: gen-r4 결과 — clean 2개(elite-staged-value-planning[anchor, terminal Q term], adaptive-mppi-budget[보완, dynamics/iteration term]). Amdahl을 코드로 검증. Puppeteer 보류(interface 메커니즘 부재).
- 2026-06-19 KST: novelty — elite-staged INCREMENTAL, adaptive-mppi-budget **INCREMENTAL-thin**(Adaptive Online Planning 1912.01188 선점).
- 2026-06-19 KST: **elite-staged PoC = FAIL** (advisor-ordered wall-clock microbench 맨 먼저). GPU 0.94×/CPU 1.011× — terminal Q는 vmap vectorized라 wall-clock bottleneck 아님(dynamics rollout이 지배). BL-15 등재. gen-r4 "terminal ~50%"는 FLOPs O/wall-clock X.
- 2026-06-19 KST: **Round 6 통과 0/push 0.** anchor FAIL, 보완(adaptive-mppi-budget) novelty thin이라 보류 → 방향 결정 대기.

## 누적(세션 전체) 결과
- base 3종 셋업(DreamerV3/TD-MPC2/Puppeteer 작동, DreamerV4 toy라 보류) + GL/B200 infra 수리.
- funnel 완주: learned-kl-reliability-lambda(FAIL synth+실물), elite-staged-value-planning(FAIL microbench). decision-fidelity-atlas(drop thin). adaptive-mppi-budget(보류 thin).
- **통과 idea 0 / push 0** (이 세션). funnel은 나쁜 idea를 싸게 죽이는 데 효과적(특히 wall-clock microbench·oracle 통제·코드 Amdahl). 단 idea 공간이 hard.
- 2026-06-19 KST: **R7 = 0 clean** (named-paper 선점: iQRL/TD-M(PC)²/RePo 등). gen-r4 cross-round 메타분석 — survival=0(R4~R7), "fresh base=throughput 레버" falsified, bottleneck은 survival(wall-clock + literature saturation), 더 생성 비추천. dual-rate 논문화(#1 권고) 확인: Plan 001 실험 11 run 전부 6-8k/100k step 멈춤(이제 infra로 완주 가능).
- 2026-06-19 KST: 사용자에 3-option 제시(dual-rate 논문 완성 권고). **사용자 = "그래도 계속 생성" 선택.** 존중하고 R8 dispatch — 단 efficiency 금지·포화 cell 금지, 가장 덜 worked된 새 problem-setting/capability 조준.
- 2026-06-19 KST: **R8 = 0 clean + 구조적 reframe + 1 bet.** gen-r4: freshness-check는 novelty를 반증만 가능(입증 불가) → genuinely-new setting은 도구상 de-risk 불가. 8라운드 0의 근본 원인. 단 하나 bet = Puppeteer transfer sample-eff(harness 존재).
- 2026-06-19 KST: 사용자 = "bet PoC bounded 먼저". Puppeteer HL 학습 harness 작동 확정(sb3+matplotlib 설치, train.py task=walk frozen low-level 위 학습됨, B200).
- 2026-06-19 KST: **bet = FAIL (advisor: 학습 전 사전등록 게이트, compute 0).** non-obvious 4-bar(plausible/surprising/decision-changing/distinguishable) 통과 가설 작성 불가 — **Puppeteer가 이미 8-task 전부 성공**(primary-source)이라 reachability-ceiling/한계요인이 surprising·decision-changing 안 됨. advisor 예측 "cheap FAIL" 적중.
- **누적 최종: 8라운드 survival=0.** 심지어 R8 bet도 가장 싼 게이트(사전등록)에서 사망. generation 완전 소진 확인. 남은 high-EV = dual-rate 논문(미완 실험 완주)뿐.
