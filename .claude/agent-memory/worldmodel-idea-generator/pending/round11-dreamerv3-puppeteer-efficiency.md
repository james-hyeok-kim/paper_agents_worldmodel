---
slug: round11-dreamerv3-puppeteer-efficiency
status: closed-by-construction (0 clean)
created: 2026-06-20 KST
note: "R11 = R10 lever 수정(DreamerV4→DreamerV3, 둘 다 runnable). inference efficiency. 결과 0 — DreamerV3 arm은 saturation이 아니라 *construction*으로 0(inference에 hot-path가 거의 없음, 코드확인). 4-wall statement. 진짜 binding constraint는 validator wedge(R8 bet이 gate에 못 감)."
---

# Round 11 — DreamerV3 + Puppeteer inference efficiency: 0 by construction + validator wedge

사용자가 R10 lever 수정: DreamerV4(코드없음)→DreamerV3(runnable). 둘 다 실물 microbench 가능. inference efficiency(이상적 통합). 결과 **0** — search 없이 코드+문서로 4개 벽 전부 확정. (efficiency=wall-clock microbench 먼저 게이트는, *살아남은 후보가 없어* 도달 안 함.)

## 핵심 kill: DreamerV3 arm은 saturation이 아니라 *construction*으로 0
코드확인(`dreamer.py:94 _policy`): DreamerV3 **inference(act-time) = encoder + RSSM + actor 1 forward/step. decoder 없음, planning 없음.** 이미 싸다. DreamerV3의 비용 story는 전부 *training*(imagination rollout, WM update, decoder reconstruction). decoder(`models.py`)는 `_train`과 `video_pred`(eval 로깅)에만, act path엔 없음.
→ TD-MPC2/Puppeteer는 정반대(planning이 무거운 per-step inference). **두 base는 *공유* hot-path가 없는 정도가 아니라, 한쪽(DreamerV3)은 inference hot-path가 *거의 없다*.** DreamerV3 arm은 어떤 BL 벽에 닿기도 전에 비어있다(empty by construction).

## drift 차단: "decoder training cost"는 쫓지 않는다
검토 중 "BL-09이 안 친 decoder training cost 각도"로 drift할 뻔 → 2중 kill: (1) **off-lever** — 그건 *training* efficiency지 사용자가 준 *inference*가 아님. (2) **pre-empted** — "decoder 제거/축소"=reconstruction-free WM(MuDreamer/DreamerPro/iQRL), R7에서 닫음. search하면 R7 재유도. 안 함.
→ **양 해석 모두 0**: 사용자가 혹시 *training* efficiency를 의도했어도 BL-06~13 graveyard + reconstruction-free라 0. 보고가 두 해석 다 막음.

## 4-wall statement (team-lead 3 sub-question 답, search 불필요)
1. **DreamerV3 inference**: near-empty surface(싼 forward 1개). 존재하는 비용은 training이고 그건 BL-06~13 + reconstruction-free(R7). → 0.
2. **Puppeteer**: event-triggered/self-triggered MPC로 선점(R10, 제어이론). 유일 non-event-triggered 각도(frozen low-level의 online MPPI를 policy-prior로 ablate)는 "state-dependent value of planning"=worked(R9/TD-M(PC)²). → 0.
3. **통합(unification)**: substrate 없음 — DreamerV3가 공유할 inference hot-path 자체가 없음. 공통 메커니즘은 collision trap(통합 함정). → 0.
4. wall-clock microbench: 살아남은 후보 0이라 도달 안 함.

## R11의 진짜 binding constraint: validator wedge (escalation)
- R9·R10에서 "R8 Puppeteer acquisition-curve bet을 validator로"를 권고했으나 **미실행** — plan note상 validator가 wedge되어 team-lead가 수동 수행 중.
- 따라서 binding constraint는 더 이상 "이번 라운드가 0인가"가 아니라 **유일한 testable asset(R8 bet)이 gate에 도달 못 한다는 것**이다.
- generator인 나는 validation을 떠안지 않는다(역할 밖). 대신 **wedge를 고쳐야 할 대상으로 escalate**: validator unblock → R8 bet 테스트, 아니면 loop는 construction상 rigorous 0만 계속 반환.

## 메타교훈 (다음 generator)
- **"inference efficiency"를 요구받으면 base의 *inference* 경로를 코드로 먼저 봐라** — DreamerV3는 inference가 싸서(decoder/planning 없음) inference-efficiency 대상이 *구조적으로* 아니다. cost는 training. base마다 어느 phase가 비싼지 다름.
- saturation("또 채굴됨")보다 **construction("그 hot-path가 애초에 없음")이 더 깨끗한 0 진술**. 가능하면 후자로.
- 10+라운드: lever 5개(새 axis·새 base·cross-paradigm·human-efficiency·DreamerV3+Puppeteer-eff) 전부 닫힘. 패턴 불변: efficiency framing 死, 진짜 미테스트는 (a) validator wedge 해소 후 R8 bet, (b) human-defined + quality/sample-eff framing.
