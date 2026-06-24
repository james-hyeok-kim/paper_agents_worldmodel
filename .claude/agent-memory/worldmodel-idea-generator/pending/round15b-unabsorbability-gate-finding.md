---
slug: round15b-unabsorbability-gate-finding
status: closed-gate-finding
created: 2026-06-22 KST
category: quality / sample-eff / capability (direction i pivot)
result: 0 clean (2 candidates killed on un-absorbability gate, "none passed on paper" branch)
---

# R15b — quality/sample-eff/capability pivot: the binding gate is un-absorbability, not literature. Two candidates die on it.

## 요청 (direction i)
R15 efficiency 종결 후 사용자가 direction (i) 선택: **quality / sample-eff / capability pivot.**
efficiency b1/b2/b3 완전 포기. 탐색축: WM quality / sample-eff / capability acquisition /
Puppeteer acquisition-curve(R8 bet). "Puppeteer-only 국한 금지, 더 넓게."

## 결론: 0 clean. 이 space의 binding gate는 literature가 아니라 **un-absorbability**.
## 두 후보(controllability-whitening, low-level-internals)가 같은 gate에서 死. "none-passed-on-paper" 분기.

---

## 핵심 reframe — gate가 바뀌었다 (advisor)

efficiency space에서 binding gate = literature breadth + wall-clock(GPU vmap/launch).
**quality/sample-eff space에서 binding gate = PoC-vs-resource-matched-trivial-baseline.**

지난 4개 死를 다시 보면 *하나의 死가 네 번*:
- BL-14: 신호 valid + literature 통과 → **state-agnostic trivial baseline**(oracle 포함)을 못 이김.
- BL-19: novel + literature 통과 → **full-FT baseline**을 못 이김.
- BL-20: PoC에서 死(tautology + OOD).
- dual-rate: **size-matched vanilla baseline** → parity.

= 동일 死: **잘 튜닝된 MBRL의 WM+planner가 네가 주입하려는 구조를 *같은 데이터로* 이미 흡수한다.**

**un-absorbability test (이 space의 결정 게이트):**
> resource-matched baseline(같은 reward-free 데이터로 pretrain된 WM + return-driven planner)이
> 이걸 *같은 데이터/연산으로* 흡수할 수 있는가? YES → 死.

이 test가 즉시 죽이는 것: reward-free dynamics 데이터에서 유도 가능한 것(WM-pretrain 흡수) /
나쁜 방향을 downweight하는 것(planner 흡수) / 이미 계산하는 target reweight(BL-14).

survivor는 baseline이 *흡수 못 하는* 것에서 이득을 얻어야 함:
(a) **buffer에 무슨 데이터가 들어가는가**(WM은 수집 안 한 데이터로 학습 불가) — 단 crowded(Plan2Explore/curiosity, replay-side=MaPER/UPER 선점).
(b) baseline 입력에 *아예 없는* 신호.
(c) task reward가 필요한 정보를 RL이 발견하는 것보다 싸게 공급.

## 후보 1 — controllability-whitening (death: absorbed 양쪽)
frozen low-level의 command-controllability spectrum을 reward-free로 추출해 high-level action space를
whiten. **死**: matched baseline은 TD-MPC2-from-scratch가 아니라 *같은 reward-free rollout으로
pretrain된 WM*임 — 그 dynamics model이 command→consequence 매핑 전체를 흡수(whitening보다 strictly
more) + WM-pretrain-on-reward-free는 그 자체로 WPT(2502.19544)/XTRA 선점. 양쪽에서 dominated.

## 후보 2 — low-level internals as privileged signal (death: code가 반증)
advisor option (b): frozen low-level(TD-MPC2 tracker)의 internal value V_low / latent을 high-level
WM에 privileged 입력으로. "standard HRL은 low-level을 action executor로 black-box → 신호가 입력에
구조적으로 부재" 주장. **死 — Puppeteer 코드가 반증:**
- `transfer.py:100` `high_level_obs.update(low_level_obs)` → **high_level_obs ⊇ low_level_obs**.
  high-level이 low-level obs 전체를 이미 봄(+target/camera). vision task에서도 proprioception 포함(`:100`).
- command = high-level 자신의 action (`:126` reference = appendages_pos + action).
- TD-MPC2는 non-recurrent + eval deterministic → **private hidden state 없음**.
- ∴ **V_low = f(low_level_obs, command) = f(high-level이 완전관측하는 입력)**. privileged 아니라
  *computable feature*. matched baseline(같은 reward-free rollout으로 WM pretrain하며 V_low를 0
  추가 env-cost로 logging → auxiliary target으로 distill)이 흡수. asymptotic + low-sample 둘 다.
  = BL-14/dual-rate/whitening 네 번째. + asymmetric-AC/Informed-Dreamer/PIGDreamer 영역.
- "현재 안 들어감 ≠ un-absorbable." test는 *derive 가능한가*를 묻지 *현재 wiring*을 묻지 않음.

**salvage-check 실패**: 이 fully-observed proprioceptive base(Puppeteer)에서 (high_level_obs, command)의
deterministic 함수가 *아닌* 신호가 있는가? 없음. proprioception이 vision task에서도 obs에 있음.
"read the low-level's internals" framing에 un-absorbable channel 부재.

## 메타교훈 (다음 generator — 중요)
- **quality/sample-eff space의 binding gate = un-absorbability**(PoC-vs-matched-baseline)지 literature
  아님. 이 cell들은 이미 mapped(R4b 4/5 선점) — fresh cell 못 찾음. R13/R14가 literature 통과한 건
  *fresh mechanism이 아니라 mapped cell 안의 specific falsifiable claim*이어서였음. lit check는 후보당
  1회만, gate는 un-absorbability.
- **"inject structure X" 아이디어는 fully-observed base에서 대부분 死**: 잘 튜닝된 WM+planner가
  같은 데이터로 흡수. 특히 reward-free dynamics에서 유도 가능 / 나쁜 방향 downweight / 이미 계산하는
  target reweight = 자동 死.
- **"현재 안 wiring됨 ≠ un-absorbable."** privileged-info 주장은 *코드로* base의 obs superset 관계를
  먼저 확인할 것 — Puppeteer는 high_level_obs ⊇ low_level_obs라 low-level internals가 privileged 아님.
- **dual-rate-as-fallback은 사라짐**(2026-06-22 parity 확정, abandon). 그 fallback("죽을 새 idea 대신
  dual-rate ship")은 더 이상 유효하지 않음 → survivor landing이 load-bearing.
- "dual-rate가 sample-eff로 이겼다"는 **거짓**(coordinator framing 오류). result_003.md = parity,
  confound(deter 384 vs 512). 진짜 교훈은 kill-control(size-matched baseline) = un-absorbability test #1.

## forward (escalation, 5번째)
- un-absorbability gate가 "inject structure WM already has" family를 fully-observed base에서 닫음
  (whitening absorbed / internals absorbed). 이건 publishable-internally finding이지 死 idea 아님.
- 유일 structurally-open category = (a) what-data-enters-buffer (WM은 수집 안 한 데이터로 학습 불가) —
  단 crowded(Plan2Explore/curiosity, replay=MaPER/UPER 선점), 새 pre-mortem+lit check 필요, EV가
  escalation을 못 넘는다는 게 advisor+내 판단.
- **validator wedge 해소가 새 생성보다 EV 높음**(pending 24+, conditional-go 18, survival=0).
  R9·R10·R11·R12·R15 = 5번째 escalation. team-lead 결정 필요: (1)validator unblock 우선, 또는
  (2)candidate (a) buffer-data 한 swing(crowded, pre-mortem 미통과 시 死).
