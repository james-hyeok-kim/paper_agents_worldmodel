# Plan 007 — Efficiency idea loop 재개 (R12~), PoC 통과까지 반복

## 목표
사용자 선택: efficiency framing으로 추가 라운드. 새 아이디어 → 인터넷 novelty 검증 →
experiment PoC를 PoC 통과할 때까지 반복 → 통과 시 사용자 알람.

## 배경 / 인지된 제약
- 11라운드 survival=0, dual-rate(유일 자산)도 parity로 종결.
- 세 벽: (1) GPU wall-clock kernel-launch-bound (작은 연산은 GPU에서 안 빨라짐, call 완전 skip만 유효) (2) 문헌 포화 (3) genuinely-new setting은 search로 검증 불가.
- 사용자가 EV 알고도 efficiency 라운드 선택 → 존중하되 게이트로 싸게 검증.

## 워크플로우 (iteration당)
1. **생성**: worldmodel-idea-generator(gen-r4) dispatch. BLACKLIST(BL-01~15) 회피, 실제 hot-path 타겟, 3개 이상 차별점.
2. **novelty (인터넷 필수)**: team-lead가 WebSearch로 arXiv/Semantic Scholar/named-paper 충돌 확인. (literature-checker 에이전트는 wedge → 직접 수행). NO-GO면 다음 아이디어.
3. **PoC 3a — wall-clock microbench 맨 먼저**: GPU+CPU 양쪽 speedup 측정. < 1.5×면 즉시 FAIL → BLACKLIST 등재 → 다음. (BL-10/12/15 교훈: FLOPs 아닌 wall-clock, kernel-launch-bound 확인)
4. **PoC 3b — quality proxy**: speedup 통과 시에만. quality_proxy_delta < 0.05 확인. oracle/agnostic 통제 + tune/test split (BL-14 교훈).
5. **통과 = 3a AND 3b** → PushNotification 알람 + 기록. 실패 → BLACKLIST + 다음 iteration.

## PoC 게이트 (success criteria)
- rollout_speedup > 1.5× (GPU+CPU wall-clock microbench, FLOPs 아님)
- quality_proxy_delta < 0.05
- 둘 다 통과해야 PASS.

## Edge cases / 실패 모드 (사전 인지)
- GPU kernel-launch-bound: 소형 연산 교체는 이론 speedup ≠ wall-clock. 반드시 양 device 측정.
- Amdahl precondition: 가정된 bottleneck이 hot-path 아닐 수 있음 → 코드 프로파일 먼저.
- 문헌 충돌: PoC 전 인터넷 검색으로 named-paper 선점 확인.
- vacuous quality pass: oracle/state-agnostic baseline으로 통제.

## 진행 로그
- 2026-06-22 KST: 사용자가 efficiency 추가 라운드 선택. plan 작성, R12 시작. 게이트 전부 장착.
- 2026-06-22 KST: generator R12 = "0 + 적체 5개 게이트로" 진단. 5개 backlog CONDITIONAL-GO를 user 워크플로우(novelty→PoC)로 처리하기로. G4 게이트 평가 후 우선순위: action-quantized-planning-cache(1순위, TD-MPC2 planning Amdahl-verified) > value-bootstrapped-depth(2순위) > 나머지(base 미셋업 or dual-rate 변형).
- 2026-06-22 KST: **#1 action-quantized-planning-cache: novelty PASS(인터넷 검증 — DC-MPC/BiC-MPPI/NN-memoization 모두 각도 다름) → PoC Stage 1 FAIL.** oracle best-case collision(rho=0.9)도 GPU 0.397×/CPU 1.209×. BL-16 등재. 다음 후보로.

## 후보 큐 (CONDITIONAL-GO backlog, 게이트 처리 중)
| # | 후보 | base | 상태 |
|---|---|---|---|
| 1 | action-quantized-planning-cache | TD-MPC2 | ❌ FAIL GPU 0.397× (BL-16) |
| 2 | value-bootstrapped-depth-scheduler | DreamerV3 | ❌ FAIL GPU 1.069× best (BL-17) |
| 3 | spectral-multilevel-rssm | DreamerV3 | ❌ FAIL GPU 0.710× (BL-18) |
| 4 | subtree-reuse-muzero | MuZero(미셋업) | 보류 — base 빌드 사인오프 필요 |
| 5 | iris-token-budget-rollout | IRIS(미셋업) | 보류 — base 빌드 + BL-13 모순 |

## R12 종결 (2026-06-22 KST)
**runnable base(DreamerV3/TD-MPC2)의 3개 fresh efficiency 후보 전부 wall-clock microbench FAIL — 동일한 GPU launch-bound 벽.** (BL-10/11/12/15 + dual-rate 0.926× + 이번 3개 = ~6 measurement deep). 남은 2개(subtree/iris)는 base 미셋업(MuZero/IRIS, 수일 빌드) + 동일 결함 공유(MCTS node forward는 더 작고 sequential) 또는 measured BL-13(tokenizer 76%) 모순. **efficiency-on-runnable-bases 구조적 종결 empirically 재확인.** → 사용자 결정 대기: (a) efficiency 중단 / (b) base 빌드(低EV) / (c) non-efficiency lever.

## R12b 재확인 (2026-06-22 KST) — 코드 레벨 검증으로 완전 종결
generator-r12b가 base 코드 직접 walk로 3조건 전부 닫힘 확인:
- **조건 1 (큰 serial op skip)**: TD-MPC2=큰 serial op 자체 없음(전부 소형 MLP); DreamerV3=CNN encoder/decoder가 유일하나 decoder-skip=R7 선점, encoder-skip=gradient sever(`models.py:153` embed→`:191` model_opt 전체 backprop). skip하면 WM 학습 gradient 끊김.
- **조건 2 (episode amortize)**: encoder latent 캐싱 = gradient sever 동일 + stale latent + BL-09(1.2×) 선점.
- **조건 3 (train-step 감소)**: error-prioritized replay=MaPER/UPER 선점; UTD scheduling=generic RL(WM-specific 아님); curriculum=quality scope drift.
**결론**: GPU launch-bound 벽을 피하려면 큰 serial op를 gradient sever 없이 skip해야 하는데, 두 base 모두 그런 hot-path 부재. efficiency framing 코드레벨로 구조적 종결. **사용자 결정 필요**.
