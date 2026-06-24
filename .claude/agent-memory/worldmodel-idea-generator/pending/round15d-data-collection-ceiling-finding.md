---
slug: round15d-data-collection-ceiling-finding
status: closed-finding (없음)
created: 2026-06-23 KST
category: what-data-enters-buffer (data collection / exploration)
result: 0 — 모든 named 후보가 ceiling+pincer로 닫힘. 세 번째이자 마지막 구조 카테고리 종결.
---

# R15d — what-data-enters-buffer: dense-reward base에서 collection ceiling 도달 → 없음

## 요청
coordinator: efficiency(b1/b2/b3)+quality/sample-eff 모두 닫힘. **구조적으로 유일 열린 경로 =
what-data-enters-buffer.** (WM은 buffer에 없는 데이터로 학습 불가 → 무엇을/언제 수집하는지 바꾸면
기존 WM+planner가 흡수 불가.) 5각도 탐색: (1)task/goal-conditioned (2)model-disagreement
(3)contrastive/diverse novelty (4)offline→online (5)pred-error curriculum. un-absorbability gate +
literature check 필수. 없으면 "없음" 명확히(어느 gate/선행연구에서 死).

## 결론: 없음. un-absorbability는 통과하나 **return ceiling**에서 死. 닫힘은 efficiency와 동형 pincer.

---

## 결정적 blade — occupancy가 아니라 **ceiling** (advisor)

내 literature search가 직접 반환한 사실:
> **Plan2Explore(Sekar 2020)는 task-reward에 접근하는 supervised oracle(Dreamer)을 DMC zero-shot에서
> *거의 따라잡거나 일부 능가*한다** (예: Hopper-Hop P2E 432 vs Dreamer 336; Cheetah-Run P2E 784 vs Curiosity 495).

내 runnable base는 정확히 **dense-reward continuous control**:
- DreamerV3 = dmc walker_walk / cheetah_run (dense locomotion)
- TD-MPC2 = walker-walk (dense)
- Puppeteer = RunThroughCorridor(target_velocity=6.0, dense) / Stand·Walk·Run (dense locomotion)
- 코드 확인: sparse-reward / hard-exploration task **부재**.

따라서 matched baseline은 "WM + random"이 아니라 **WM + Plan2Explore collection**. 이 baseline 대비
새 collection mechanism은:
- un-absorbability **통과** (다른 데이터를 수집하므로 WM이 흡수 불가) ✓
- 그러나 **capture할 return headroom이 없음** — dense-reward에서 Plan2Explore-data ≈ oracle-data가
  이미 downstream return ceiling에 도달. → un-absorbable AND novel AND **여전히 무이득**.

"angle X → occupant Y"보다 강하고 구체적인 kill: 흡수 불가여도 **천장에 막혀** 못 이김.

## efficiency를 닫은 것과 *동일한* pincer (이게 finding을 airtight하게 함)
| 카테고리 | branch 1 | branch 2 |
|---|---|---|
| efficiency | vectorized base → launch-bound 死 | non-vectorized base(MuZero) → runnable 아님 死 |
| data-collection | dense-reward base → **ceiling**(headroom 없음) 死 | hard-exploration base(headroom 존재) → **runnable infra 부재** → PoC 불가 → validator runnability gate 死(DreamerV4/MuZero 기각 패턴) |

양 branch 모두 닫힘. 구조적으로 efficiency closure와 동형.

## 5각도 1-line kill (추가 search 불필요)
- (2) model-disagreement = **Plan2Explore 그 자체**(coordinator 본인이 overlap 지목).
- (1) reward-aware/task-conditioned = **dominated** — Plan2Explore가 *이미 이 비교를 실행*해 reward-oracle을
  matching. dense-reward에서 reward-aware가 추가로 줄 headroom 없음.
- (3) novelty/diversity (novelty≠uncertainty) = RND / count-based / state-marginal-matching 선점.
- (5) pred-error curriculum = ICM / Pathak 2017 선점.
- (4) offline→online transition = 유일하게 구조적으로 distinct("explore better"가 아님). 그러나 named area
  (offline-to-online MBRL) + matched baseline(offline-pretrain + standard online)이 "무엇을 add할지"
  결정을 흡수. + 우리 infra에 offline 데이터셋 파이프라인 부재(추가 runnability gate).
- (value-aware/decision-relevant model error로 collect) = objective-mismatch / value-equivalence(VAML,
  2310.06253) 선점 — 이미 decision-fidelity-atlas(R5)를 죽인 그 literature.

## 메타교훈
- **data-collection의 binding kill = return ceiling(dense-reward에서 Plan2Explore≈oracle)**, occupancy가
  아님. un-absorbability를 통과해도 천장이 있으면 死 — un-absorbability는 *necessary not sufficient*.
- closure 형태를 결정하는 단 하나의 check = "runnable infra에 sparse-reward/hard-exploration task가 있나?"
  → **없음**(30초 코드확인). 있었으면 occupancy-only closure(여전히 fatal, 5각도 전부 named occupant).
  없으므로 full pincer.
- efficiency(framing-dead) + quality(un-absorbable) + data-collection(ceiling-capped) = **세 구조 카테고리
  전부 종결.** generation lever가 현 runnable base에서 소진.

## forward (escalation 아님 — 정확한 statement)
이번 라운드가 세 구조 카테고리의 *마지막*을 닫음. 이건 "validator wedge 6번째 escalation"이 아니라 더
깨끗한 statement: **generation lever가 현 runnable base 3종에서 세 카테고리 모두 소진됐다.** 그림을
바꾸는 유일한 것:
1. **headroom 있는 regime의 새 *runnable* base** — hard-exploration task(sparse reward)에서 data-collection이
   ceiling-capped가 아닌 곳. 단 그 base가 wall-clock microbench 가능해야(runnability gate).
2. **human-defined problem-setting** (R8 bet 공간) — generator로는 certify 불가, validator/human taste 영역.
manufacturing a data-collection candidate는 ceiling에서 死 — 없음이 정확한 결과(coordinator 승인됨).
