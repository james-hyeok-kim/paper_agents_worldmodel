---
slug: round10-human-lever-efficiency
status: closed-both-arms (0 clean)
created: 2026-06-20 KST
note: "R10 = 첫 human-defined lever(사용자 직접): DreamerV4 + Puppeteer inference efficiency. 결과 0 — 두 arm 모두 닫힘. Puppeteer arm은 event-triggered/self-triggered MPC(제어이론)로 선점(code-level R6 확인), DreamerV4 arm은 저자 자기선점+코드부재로 wall-clock gate 불가. 중요: human의 *efficiency* framing이 graveyard에 떨어짐 — dual-rate(survivor)는 efficiency 아닌 quality로 이김. human 재조준 필요."
---

# Round 10 — Human Lever (DreamerV4 + Puppeteer inference efficiency): 두 arm 모두 닫힘

9라운드 만에 첫 human-defined lever. 사용자가 DreamerV4 + Puppeteer 두 상반 패러다임의 **inference efficiency**(가능하면 통합)를 직접 지시. 내 R8 진단상 human-defined가 유일한 productive lever라 진지하게 평가. arm을 분리(통합은 "통합 함정"이라 강제 안 함) 후 각각 판정 → **둘 다 닫힘.**

## Arm A — Puppeteer 2-level planning efficiency → event-triggered MPC로 선점 (CLOSED)
**코드 발견(실재 확인)**: Puppeteer는 진짜 **2개 MPPI를 매 env step 구동**한다. high-level `agent.act()`(outer) + low-level은 별도 full TD-MPC2 agent(`transfer.py:86 self.low_level_policy = TDMPC2(...)`)이고 `low_level_policy.act()`=full MPPI가 `TransferWrapper.step` 안에서 매 step 호출(transfer.py:130). → high-level은 느리게 변하는 reference(appendages_pos offset)를 내는데 둘 다 같은 빈도로 full MPPI. **"high-level MPPI를 adaptive하게 skip"하면 wall-clock 절감 실재**(call-skip이라 BL-15의 within-call 함정 아님).
**그러나 이건 novelty가 막힌다(Amdahl이 아니라)** — advisor 지적: R6에서 이 cell("high-level 저빈도 replan/command persistence")은 *novelty*로 기각됐지 Amdahl로가 아니었다. code finding은 Amdahl(절감 실재)만 확립, novelty는 안 건드림.
**제어이론 freshness check(RL이 아니라 control 키워드로) → 직접 선점**:
- fixed skip-every-K = **multi-rate / hierarchical MPC**(수십 년).
- adaptive skip-until-trigger = **event-triggered / self-triggered MPC** — "성능기준 위반 시에만 재계산해 연산 절감"으로 *정의된* 서브필드.
- 내가 watch하라던 narrow opening(learned value-of-replanning trigger)조차 닫힘: **Learning Event-Triggered MPC with Hierarchical RL(Ren 2026)**, **Event-Triggered MPC + Deep RL(2208.10302)**, MPC+learned-value+event-triggered replanning(2025). learned-trigger 변형 occupied.
→ R6 finding을 code-level + named-field로 확정. **CLOSED.**

## Arm B — DreamerV4 inference efficiency → 저자 자기선점 + 코드부재 (CLOSED)
- DreamerV4(2509.24527) abstract: "real-time interactive inference on a single GPU through a **shortcut forcing** objective and an **efficient transformer architecture**." → inference efficiency가 *논문 헤드라인*. 저자 홈그라운드에서 입찰하는 것.
- **공식 코드 없음** → analytic-only → 이 프로젝트 필수 게이트(wall-clock)를 확립 불가. BL-15 전례: FLOPs-analytic "승리"는 wall-clock에서 死. fresh angle이어도 validator 통과 불가.
- video-diffusion efficiency(Efficient-vDiT 등) 별도 crowded 라인.
→ 특정 non-shortcut 메커니즘을 *이미* 손에 쥐지 않은 한 PDF 사냥 무의미(안 쥠). **CLOSED.**

## 통합(unification) — 강제 안 함
두 bottleneck이 다름(denoising NFE vs planning samples/calls). 가로지르는 단일 메커니즘은 "통합 함정"(자명/억지 collision). arm 분리 유지가 정답.

## R10 finding (human에게 — 중요, 재조준용)
**human의 첫 lever가 productive lever 형식(human-defined)은 맞았으나, *efficiency* framing이 Puppeteer를 graveyard에 떨어뜨렸다.** 핵심 인접 사실: **dual-rate(9라운드 유일 survivor)는 efficiency가 아니라 sample-eff/quality로 이겼다**(+12% best-checkpoint, -25% params에서 drop 없음). 즉 패턴은 "efficiency framing은 이 base들에서 계속 死, quality/sample-eff framing이 살아남는다"이다.
- **나는 이 arm을 quality로 몰래 바꾸지 않는다**(human이 efficiency를 명시했으므로). 대신 human이 재조준하도록 표면화한다.
- **재조준 제안**: 같은 자산(Puppeteer hierarchical)에서 *efficiency*가 아니라 *capability/sample-eff* 질문으로 — 단 그건 R8 Puppeteer acquisition-curve bet이 이미 지목한 공간이고, validator 4-bar 미통과 상태. **R11 생성보다 그 bet을 validator로 보내는 게 우선**(R9 권고 반복).

## 메타교훈 (다음 generator)
- **efficiency 후보는 RL 키워드가 아니라 인접 *제어이론* 키워드(event-triggered/self-triggered/multi-rate MPC)로 freshness check하라** — "local-fresh, globally-crowded"의 crowded-field가 control이었다.
- **code finding이 Amdahl을 확립해도 novelty 차단은 안 풀린다** — R6에서 막힌 게 무엇이었는지(novelty vs Amdahl) 구분.
- DreamerV4류 코드-부재 base는 wall-clock gate 불가 → efficiency idea의 validator 통과 구조적 불가(analytic만).
- 누적: lever 4개(새 axis·새 base·cross-paradigm·human-efficiency) 테스트. human-defined는 형식은 맞으나 *efficiency* 내용이 graveyard. 남은 건 human-defined + *quality/sample-eff 또는 capability* framing.
