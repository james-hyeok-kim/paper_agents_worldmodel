---
slug: round9-crossparadigm-confound
status: closed-lever-finding (0 clean)
created: 2026-06-20 KST
note: "R9: team-lead가 cross-paradigm 대비를 새 lever로 제시. 결과 0 clean — cross-paradigm *comparison*은 attribution confound(bar 4)로 구조적으로 막힘. 이 confound가 lever 전체를 닫는다. 4-bar self-kill을 validator 전에 적용해 死."
---

# Round 9 — Cross-Paradigm Lever는 Attribution Confound로 닫힘 (0 clean)

team-lead가 R8 방법론(3-part test로 genuinely-new setting 사냥)을 받아, **3 작동 base의 대비(RSSM imagination vs MPPI planning vs hierarchical)에서만 보이는 setting**을 새 lever로 제시. 가장 유망 후보를 freshness + 내 4-bar로 자가검증 → 死, 그리고 그 死 이유가 lever 전체에 일반화됨.

## 검토한 lead 후보: cross-paradigm per-state competence complementarity
"Dreamer(RSSM imagination)와 TD-MPC2(MPPI planning)가 동일 task의 *같은* state에서 실패하는가, *다른* state에서 실패하는가" — disjoint면 paradigm 상보성(router/ensemble 시사), 같으면 task-intrinsic 한계.
- **freshness**: per-state cross-paradigm failure 비교는 미발견(searcher가 "available results에 없음, 신규/특화일 수 있음"이라 명시). 단 "state-dependent value of planning(언제 planning이 policy를 이기나)"은 TD-M(PC)²·"Learning to Plan, Planning to Learn"(2512.17091)·KL-reg learning-to-plan(2510.04280)이 *TD-MPC2 내부에서* 선점. genuinely-fresh slice는 *cross-substrate* per-state 비교로 좁음.

## 4-bar 자가판정 (validator 전, team-lead 강제 게이트) → FAIL
| bar | 판정 |
|---|---|
| plausible | PASS (생성/imagination vs latent local opt은 inductive bias 달라 실패 영역 다를 法) |
| surprising | WEAK (가장 그럴듯한 결과는 "둘 다 같은 hard state에서 실패=task-intrinsic"으로 unsurprising) |
| decision-changing | WEAK (disjoint여도 test-time에 두 full agent를 동시 구동 불가→router 비현실, 결정 안 바뀜) |
| **distinguishable** | **FAIL (decisive)** — Dreamer vs TD-MPC2 competence 차이를 *paradigm*에 귀속 불가. 두 agent는 codebase·network 크기·training budget·구현선택 수십 개가 다름. paradigm/implementation 교란 = 통제 불가 변수 2개. 측정을 더 sharpen해도 안 사라짐. |

→ bar 4(distinguishable)에서 구조적 死. 작성 안 함.

## 왜 이 死가 lever *전체*를 닫는가 (R9의 진짜 finding)
bar 4의 attribution confound는 per-state 비교에 국한되지 않는다 — **모든 cross-paradigm *comparison*을 막는다**: 두 paradigm = 두 codebase = paradigm/implementation 교란. 따라서 team-lead의 cross-paradigm lever는:
- **comparison-shaped** → confound로 死 (위).
- **combine-paradigms(router/hybrid/distill)** → 기존 WM objective 위 mechanism → mapped/BL (R6 planning-hybrid, TD-M(PC)², RL-MPC 선점).
- 유일하게 confound가 *작은* 경우 = Puppeteer(hierarchical) vs flat TD-MPC2 (공유 code lineage) → **Puppeteer 자기 ICLR2025 논문이 이미** hierarchy가 flat TD-MPC2 baseline을 이긴다고 보임(선점).

세 경로 모두 닫힘. cross-paradigm lever는 R8 bind의 또 다른 인스턴스(comparison⟹confound, combine⟹mechanism-mapped).

## R9 정직 산출
- **0 clean.** 새 후보 더 사냥 안 함 — bar-4 일반화가 lever를 이미 닫음(추가 search는 base rate 0).
- **방법론이 작동한 것이 이번 win**: 4-bar가 validator *전에* 후보를 죽임(설계대로). 이 게이트는 track record 있음 — decision-fidelity-atlas·R8 bet의 死를 예측, BL-14/15도 flag대로 死. 9라운드는 9개 실패가 아니라 **고신뢰 negative map + 미완 자산 1개(dual-rate)** 를 생산.

## forward action (R10 생성보다 우선)
- **R8 Puppeteer acquisition-curve bet을 validator로** — R8/R9에서 pointer만 냈지 validator의 4-bar pre-registration을 한 번도 안 통과시킴. 9번째 후보 생성 전에 8번째를 테스트하는 게 맞다(싸고 결정적). 4-bar에서 死하면 그것도 documented.
- EV 순위 불변: dual-rate 1편(검증 survivor) > 죽을 새 idea. (재론 아님, 한 줄.)

## 메타교훈 (다음 generator)
- **cross-paradigm comparison은 attribution confound(다른 codebase)로 구조적으로 막힌다.** 두 시스템 비교로 paradigm 결론을 내려면 *통제된* 변형(같은 codebase, 한 축만 변경)이 필요 — 우리 자산(독립 구현 3종)으로는 불가.
- diagnostic-shaped 후보는 *내가 먼저* 4-bar 자가판정 → bar 4(distinguishable)가 가장 자주 죽이는 bar(교란/귀속).
- 9라운드 누적: survival=0, lever 3개(새 axis·새 base·cross-paradigm) 전부 테스트되어 닫힘. 미테스트 lever는 *human이 정의하는 통제된 새 problem-setting*뿐.
