# Plan 002 — Round 4 World Model 아이디어 자동 funnel 루프

## 목표
새 WM family/bottleneck로 **pivot**하여 Round 4 아이디어 ~5개를 생성하고,
idea → novelty → PoC → 실험 PoC → 검증 funnel을 자동으로 흘려보낸다.
**전 게이트 통과한 것만 push 알람.** 중간 탈락은 파일 로그만.

## 배경 (왜 pivot)
- 기존 카테고리 A~G(call-reduction 효율화): 15 ideas / 13 blacklist(BL-01~13) / pass 1개(dual-rate)뿐 → base rate ≈0
- 유일 성공 dual-rate조차 wall-clock 0.926×, GO 근거는 **FLOPs 1.78× + 품질 -1.3% + sample-eff +12%**
- 교훈: GPU에서 실질 이득은 "call 통째 skip"일 때만. 같은 공간 재탐색은 소모적.

## Pivot 방향 (generator에 지시)
DreamerV3와 **bottleneck이 다른** WM family에서 효율/품질 아이디어:
- IRIS/STORM/TWM 등 transformer/autoregressive WM — imagination이 진짜 hot-path (BL-11 대비)
- diffusion-based WM, JEPA/non-generative WM
- BLACKLIST BL-01~13 회피, 최소 3개 차별점 필수

## 사용자 확정 결정 (2026-06-18 KST)
1. Direction: **Pivot — 새 WM family/bottleneck**
2. Exp scale: **Quick PoC** (rollout/FLOPs bench + short training, ~30분~2h/idea, per-idea wall-clock budget 2h)
3. Loop bound: **Fixed batch ~5 ideas 후 보고**

## Funnel 게이트 (success criteria)
| Gate | Agent | Pass 기준 | Fail |
|---|---|---|---|
| 1 Novelty | literature-checker | NOVEL/INCREMENTAL (NO-GO 탈락) | drop+log |
| 2 Feasibility PoC | idea-validator | **hot-path 먼저 프로파일** 후 speedup>1.5× AND quality_delta<0.05 → CONDITIONAL-GO | drop+log |
| 3 Experiment PoC | planner+runner | **analytic FLOPs/param 감소 + 품질 threshold 내(<15% drop) + sample-eff** (wall-clock 금지) → GO/WEAK-GO | drop+log |

- **PASS = Gate 3까지 전부 통과** → PushNotification + idea_status.md 기록
- 모든 idea 처리 후 → 요약 보고

## Edge case / 실패 처리
- GPU kernel overhead → wall-clock 금지, analytic FLOPs 사용 (BL-10/12)
- Amdahl precondition → 가정 bottleneck이 실제 hot-path인지 먼저 프로파일 (BL-09/11)
- proxy quality → proxy-실제 상관 검증 후 신뢰 (BL-11, Pearson 0.085 전례)
- 게이트 탈락 → idea drop, idea_status.md "Round 4" 섹션에 기록, 다음 idea 계속
- 실험 단계 per-idea 2h 초과 → 중단, WEAK 또는 inconclusive 기록

## 상태 관리
- idea_status.md: **idea 처리할 때마다 즉시** 갱신 (context reset 대비)
- 루프 로그: 이 파일 하단 "진행 로그"에 append

## 진행 로그
- 2026-06-18 KST: plan 작성, Round 4 generator 호출 시작
- 2026-06-18 KST: generator 결과 — family pivot(transformer/diffusion/JEPA efficiency) **7/7 포화**. STORM/DINO-WM은 hot-path 검증서 탈락. 5개 생성 불가(정직). saturation map 반환. open lane = DreamerV3 quality/sample-eff 축(BL 미적용). → 방향 재확정 위해 사용자에 push+질문.
- 2026-06-18 KST: **사용자 재확정 = DreamerV3 quality/sample-eff 축.** gen-r4에 재지시 (literature-fresh 먼저, MaPER/UPER·JEPA-2602.18639 회피, measurable precondition, collision manufacturing 금지). 아이디어 생성 대기.
- 2026-06-18 KST: gen-r4 결과 — clean anchor **1개**(learned-kl-reliability-lambda) + 후보 4개 literature-blocked로 미생성(정직). quality 축도 5셀 중 4셀 선점.
- 2026-06-18 KST: **lit-checker/validator agent 모두 spawn 후 무응답(gen-r4만 정상)** → team-lead가 novelty(WebSearch)·PoC(직접 구현) 수행.
- 2026-06-18 KST: novelty = **INCREMENTAL**(STEVE/Acting-upon-Imagination/DreamerV3-XP 대비 ensemble-free per-step KL-proxy로 차별, NO-GO 아님) → 게이트 통과.
- 2026-06-18 KST: PoC — DMControl은 EGL/libglvnd 부재로 학습 불가 → validator 정의대로 synthetic gate. advisor 검토로 t-통제·decision-relevant drift·Spearman·reweighting feasibility 반영. Run1 reweighting은 음수보상 confound로 무효(advisor 수학 확인) → Run2 sign-unbiased selective bootstrap + oracle/agnostic 통제 + tune/test split으로 사전등록 후 1회 실행.
- 2026-06-18 KST: **PoC FAIL (robust).** signal validity PASS(0.535)지만 oracle 완벽신호조차 trivial agnostic m=1 못 이김 → reliability reweighting 무이득. BL-14 등재.

## 배치 결과 요약 (fixed batch ~5, 종료)
- 생성 시도: family-pivot(7/7 saturated) → axis-pivot(quality, 4/5 cells blocked) → **clean anchor 1개**.
- funnel 통과 현황: novelty INCREMENTAL 1 → PoC **FAIL** 1 → **실험 GO 0 → push 0.**
- 정직한 결론: efficiency·quality 두 축 모두 2025-26 문헌+자체 BL(이제 BL-01~14)로 상당히 mined-out. collision manufacturing 없이 얻은 1 candidate가 PoC에서 robust FAIL.
- 부산물(가치): critic-distribution spread가 prior-only imagination의 reward/value drift를 state별로 예측(BL-08 entropy 실패 지점에서) — uncertainty-aware imagination 진단 등 별도 방향 가능성.
- infra blocker: DMControl headless 학습 불가(libglvnd 부재). 실물 실험 단계 진입하려면 GL backend 수리(시스템 패키지, 사용자 승인 필요) 선행.

## 실물 실험 단계 (2026-06-19 KST, 사용자 (b) 선택)
- **GL infra 수리**: 설치 불필요 — `/home/jovyan/egl_libs/`에 이미 libglvnd dispatcher 존재. `LD_LIBRARY_PATH=/home/jovyan/egl_libs`만으로 dm_control 작동. 메모리 기록(dmcontrol-headless-egl-fix).
- **실물 실험**: vanilla DreamerV3를 dmc_proprio walker_walk에 학습(50k 설정, train_return~621). `experiment_real.py`로 synthetic과 동일 사전등록 분석(N=512, H=15) 재현.
- **결과 FAIL 재확정**: signal validity 붕괴(critic_spread t-ctrl -0.058 vs synthetic 0.535), feasibility full=agnostic(m=15)=oracle=5.096(imagination이 H=15에서 정확→reweight 대상 없음). synthetic("신호 valid·critic 지배·m=1")과 정반대 regime이나 둘 다 아이디어 전제 반박. **synthetic caveat 해소, idea 종결.**
- push: 없음 (pass 0, 요청대로 passes-only). 실물 실험 단계까지 0 pass로 종결.
