# Plan 005 — New idea loop (R9~), fresh start, 모든 게이트 장착

## 목표
사용자 요청으로 새 idea loop 시작. 8라운드 survival=0 + gen-r4 구조적 진단(generation 소진, freshness-check는 novelty 입증 불가)을 인지하고도, 누적 교훈을 전부 장착해 fresh start. 통과한 것만 push.

## 누적 게이트 (이번 loop 전부 적용)
- novelty: NOVEL/INCREMENTAL(NO-GO 탈락). literature-fresh 최엄격, named-paper 정확일치 즉시 제외.
- PoC: efficiency면 **GPU+CPU wall-clock microbench 맨 먼저**(BL-10/12/15). diagnostic이면 **학습 전 4-bar 사전등록**(plausible/surprising/decision-changing/distinguishable). feasibility는 oracle/agnostic 통제 + tune/test split.
- 실험: 실물 base(DreamerV3/TD-MPC2/Puppeteer) 사전등록 metric + baseline+통제.
- BL-01~15 회피. collision manufacturing 금지(0이면 0+open-space 지목).

## 방향 (gen-r4 R8 진단 반영)
literature-clean 재채굴은 mapped→死. 따라서 **3-part test(harness 존재/싸게 가능 · face-value 가치 · 기존 WM objective 위 mechanism 아님)**로 평가되는 *genuinely-new problem-setting/capability*를 gen-r4가 제안. base 3종(2 패러다임) + 작동 infra를 활용. 측정 harness 존재 필수(R7 robustness 교훈).

## 에이전트
generator(gen-r4) 정상. lit-checker/validator는 wedge → team-lead 직접 수행.

## 진행 로그
- 2026-06-20 KST: 세션 정리(docs/session_summary_2026-06.md) 완료. plan 작성, gen-r4에 R9 dispatch(3-part-test 기반 새 problem-setting, 전 게이트 장착).
- 2026-06-20 KST: **R9 = 0. cross-paradigm lever 구조적으로 닫힘.** lead 후보(Dreamer vs TD-MPC2 per-state competence complementarity)가 self-4-bar의 **bar-4(distinguishable)에서 FAIL** — paradigm 차이를 implementation(codebase/network/budget)과 분리 불가(uncontrolled confound). 이 confound가 cross-paradigm 전체(comparison/combine/hierarchy-vs-flat) 닫음.
- **누적 lever 3개 전부 닫힘**: 새 axis(R4b/R5/R7)·새 base(R6)·cross-paradigm(R9). 방법론 작동(4-bar가 validator 전 kill). 9라운드 = 고신뢰 negative map + 미완 자산 1개(dual-rate).
- **결론: agent-driven generation 구조적 소진 확정.** 유일 미테스트 lever = *human이 통제된 형태로 정의하는 새 problem-setting*(독립구현 비교로는 confound 불가피). → 사용자 결정 대기(human-defined 새 방향 / dual-rate / 중단).
- 2026-06-20 KST: **사용자가 첫 human-defined lever 제공** — DreamerV4 + Puppeteer, 두 상반 패러다임의 **inference efficiency**(통합 이상). reality-check: DreamerV4 efficiency는 논문 자체 헤드라인(shortcut 16×)+공식코드 없음, Puppeteer=MPPI graveyard(BL-15), 통합은 bottleneck 달라(denoising NFE vs planning) confound 위험. → gen-r4 R10 dispatch, 전 게이트(wall-clock microbench 먼저, literature-fresh 최엄격) 강제, "0이면 왜 닫혔는지".
- 2026-06-20 KST: **R10 = 0, 두 arm 닫힘.** Arm A(Puppeteer 2-level planning): 코드확인상 매 step MPPI 2개 → high-level adaptive skip은 *실제 wall-clock 절감*(BL-15 함정 아님)이나 = **event-triggered/self-triggered MPC**(정의된 control 서브필드)로 novelty-선점. Arm B(DreamerV4): 저자 자기선점(shortcut 16×)+코드부재→analytic-only→wall-clock gate 불가. **핵심 finding: lever 형식(human-defined)은 옳았으나 *efficiency framing*이 graveyard 원인. 10라운드 패턴 = efficiency 死 / quality·sample-eff 生(dual-rate proof).** → 사용자 재조준 제안(같은 모델 관심, framing을 quality/sample-eff로). (gen-r4 권고한 R8 bet은 team-lead가 이미 4-bar FAIL 판정.)
- 2026-06-20 KST: 사용자 lever 수정(DreamerV4→DreamerV3). gen-r4 R11 dispatch: DreamerV3+Puppeteer inference efficiency, 둘 다 runnable, wall-clock microbench die-point 강제.
- 2026-06-20 KST: **R11 = 0 by construction (search 불필요, 코드+문서로 4벽 확정).** DreamerV3 arm: `dreamer.py:94 _policy` 코드확인 — inference=encoder+RSSM+actor 1 forward/step, **decoder 없음·planning 없음** → inference hot-path 자체가 *구조적으로 거의 없음*. BL 벽에 닿기 전에 비어있음("또 채굴됨"보다 깨끗한 진술). Puppeteer arm: event-triggered/self-triggered MPC 선점(R10 재확인). 통합: DreamerV3에 공유할 inference hot-path 없음→collision trap. wall-clock microbench: 후보 0이라 미도달.
- **gen-r4 escalation 명확화(R11)**: gen-r4가 "R8 bet validator 미실행"으로 escalate했으나 **R8 bet는 이미 team-lead가 4-bar FAIL 판정 완료** (2026-06-19 KST, `failed/puppeteer-transfer-sampleeff-bet_validation.md`). Puppeteer가 8-task 전부 이미 성공(primary-source)이라 surprising/decision-changing 안 됨 → compute 0. gen-r4의 "validator wedge" 정보는 구버전. testable asset 0 확정.
- **11라운드 누적 최종 결론**: efficiency framing = 구조적 死(BL-01~15+event-triggered+DreamerV3-near-empty). testable asset = 0. 유일 미완 high-EV = **dual-rate 논문**(Plan 001, quality/sample-eff로 이긴 유일 survivor, 11 run 6-8k/100k step 멈춤, infra 수리됨). → 사용자 결정 대기: (A) dual-rate 논문 완주 / (B) quality/sample-eff 재조준 human lever / (C) 중단.
