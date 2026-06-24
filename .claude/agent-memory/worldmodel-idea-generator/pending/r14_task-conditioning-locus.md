---
slug: task-conditioning-locus
status: literature-checked
verdict: INCREMENTAL
created: 2026-06-22 KST
direction: b1/b2 (capability/representation + sample-efficiency/transfer; claim-led, NOT diagnostic)
base: TD-MPC2 multitask (cwd; mt30-1M.pt 공개 checkpoint, 30 tasks 전부 DMControl, runnable today)
axis: representation-locus / transfer (NOT efficiency-compute, NOT a sensitivity-table diagnostic)
venue-fit: [NeurIPS, ICLR, CoRL]
blacklist-delta:
  - "BL-19 (embedding-only onboarding): 정반대. BL-19은 frozen backbone에 task vector를 *학습*해 dynamics 변화를 따라잡으려다 死. 본 idea는 *학습 안 함* — 이미 학습된 frozen model에서 task 정보가 *어디에 사는지*를 측정(swap intervention). BL-19의 사후 설명까지 제공한다."
  - "BL-04 (WM distillation): 압축/student 없음. 모델 1개, 학습 0, intervention만."
  - "efficiency BL(06~18): wall-clock/GPU-launch 무관. forward 횟수 불변, compute 절감 주장 없음(trimming은 명시적으로 trap으로 차단). 측정은 return(quality)과 transfer sample-eff."
  - "BL-14 (reliability-reweighting): signal로 target 고치는 메커니즘 아님. intervention(wrong-embedding swap)의 *return 효과*가 deliverable. oracle/state-agnostic baseline 통제는 BL-14 교훈 그대로 적용(아래 PoC)."
collision-flag-for-checker:
  - "TD-MPC2 (ICLR2024): task embedding을 5개 module에 concat하나, per-module 필요성/redundancy를 보고하지 않음. → 확인 요청."
  - "feature attribution / input ablation (general): 본 idea를 '단순 input ablation study'로 보이게 할 위험. firewall은 (a) swap(not zero)으로 OOD confound 제거, (b) deliverable이 sensitivity-table이 아니라 adaptation-locus decision rule, (c) BL-19 실패의 mechanistic 설명."
  - "mixture-of-experts / conditional computation routing: '어느 module이 task에 의존' = routing처럼 보일 위험. 차별: 우리는 학습된 단일 dense model의 정보 locus를 측정, 새 routing 학습 아님."
---

## 핵심 가설 (claim-led, falsifiable, null-direction이 surprise)

**학습 완료된 frozen TD-MPC2(mt30)에서, *같은 embodiment 안의 다른 task* embedding을 dynamics 모듈에 먹여도 return이 거의 안 떨어진다(R(dynamics) 높음 = task label 무시) — 즉 frozen latent dynamics는 reward만 다른 task variation 사이를 task label 없이도 일반화한다 — 반면 reward 모듈에 같은 swap을 하면 return이 무너진다(R(reward) 낮음 = task label 필수).** 핵심 분리는 *같은 embodiment*(physics 동일, reward만 다름)에서만 측정된다. 예측: physics가 같으면 dynamics는 task label이 redundant(physics는 이미 z·동역학에 박혀 있음), 그러나 reward는 stand vs run을 구별해야 하므로 label이 load-bearing.

이 분리는 BL-19을 정확히 설명한다: BL-19은 *physics/질량 자체를 바꾸는*(=embodiment를 넘는) task에 frozen dynamics + embedding으로 적응하려다 死 → physics가 바뀌면 dynamics weight가 바뀌어야 하므로 96-dim으로 불가. 반대로 *embodiment 안*(physics 고정, reward만 변)에서는 frozen dynamics + (reward쪽) embedding이 충분해야 한다. 따라서 처방: **"frozen dynamics는 reward-variation은 transfer하지만 physics-variation은 못 한다"** — onboarding 시 새 task가 reward-only variation이면 dynamics를 풀 필요가 없고, physics가 바뀌면 dynamics를 풀어야 한다.

**Surprise의 방향(중요):** clean surprise는 *null-drop* — "학습 때 task label을 받았는데 *같은 embodiment 안에서는* 안 쓴다(dynamics)"이다. 반대 방향("X 없으면 망가진다")은 confound(학습 때 받은 입력을 빼면 OOD라 망가지는 게 당연)이라 claim 안 함. **cross-embodiment swap(walker dynamics emb→cheetah)은 confound 영역이라 main claim에서 제외**: z가 이미 walker obs에서 왔으므로 label이 trivially redundant("당연하지"). surprise는 오직 same-embodiment(physics 동일, reward만 변)에 산다.

## 동기 (Why Now)

TD-MPC2의 distinctive 설계: task embedding(96-dim)이 **5개 모듈 전부**(encoder, dynamics, reward, pi, Q)에 concat된다(`world_model.py:88-101`, 각 모듈이 `self.task_emb(z, task)` 호출). 표준 가정은 "모든 모듈이 task를 알아야 한다"이다. 이 가정은 검증된 적이 없다. 그리고 BL-19(embedding-only onboarding)가 정확히 이 가정 위에서 죽었다: embedding을 *모든 모듈*에 학습 주입하려 했고, dynamics function이 task마다 바뀌는 경우 96-dim으로 dynamics weight를 못 움직여 실패. 만약 dynamics가 사실 task 정보를 *거의 안 쓴다*면(encoder가 z에 task identity를 이미 라우팅), BL-19의 실패는 "잘못된 locus에 적응을 시도"한 것으로 깔끔히 설명되고, onboarding은 dynamics가 아니라 encoder/입력 표현 쪽에서 일어나야 한다는 처방이 나온다. 지금 칠 수 있는 이유: mt30-1M.pt(공개, 30 task 전부 DMControl, 학습 불필요) — frozen model에 intervention만 하면 됨.

## 제안 방법

세팅: frozen mt30 model. **학습 없음.** `task_emb(x, task)`가 각 모듈 입력에 붙이는 **96-dim embedding 벡터만** monkeypatch(intervention)한다.

**[필수 #3] task ID가 아니라 *embedding 벡터*만 교체한다.** `pi()`의 action masking은 `mean * self._action_masks[task]`로 *task ID*에 indexing된다(`world_model.py:159-162`). 만약 intervention이 task ID 자체를 바꾸면 잘못된 action dimension으로 plan하게 되어 return이 task-conditioning과 무관한 이유로 붕괴한다(false premise-confirmation). 따라서 intervention은 `task_emb()`가 append하는 96-dim 벡터만 다른 task의 것으로 바꾸고, `_action_masks[task]`는 진짜 task로 유지한다. (구현: `_task_emb(task)` 호출만 가로채 다른 row 반환, masking 경로는 건드리지 않음.)

**[필수 #1] swap arm을 same-embodiment / cross-embodiment로 분리한다.** mt30은 embodiment마다 obs/action space가 다르다. cross-embodiment swap(walker dynamics emb→cheetah)은 z가 이미 walker obs에서 왔으므로 label이 trivially redundant("당연하지", confound 영역) → **main claim에서 제외**. surprise는 **same-embodiment family** 안에서만:
- walker-{stand, walk, run, run-backwards}
- cheetah-{run, jump, run-backwards, run-front, run-back}
- cartpole-{balance, balance-sparse, swingup, swingup-sparse}
- hopper-{stand, hop, hop-backwards}
- finger-{spin, turn-easy, turn-hard}, pendulum-{swingup, spin}, cup-{catch, spin}
- (주의: reacher-{easy,hard}는 same physics지만 reacher-three-{easy,hard}는 3-link → physics 다름 → same-embodiment에서 *제외*.)

각 family 안에서 dynamics/reward 모듈에 *같은 family의 다른 task* embedding을 swap. 예측: **R(dynamics) 높음**(physics 동일 → label redundant), **R(reward) 낮음**(reward 다름 → label 필수).

Intervention 종류(모두 swap-based, zero 아님 — OOD confound 제거):
- **swap-within-family:** 모듈 M에만, family 내 다른 task t'의 embedding. (main surprise 측정.)
- **swap-mean-family:** 모듈 M에만 family 평균 embedding(가장 in-distribution한 "task-agnostic" 입력 → swap-wrong과 교차확인).
- **all-swapped (premise check):** 5개 모듈 *전부* swap → return 붕괴해야 함(task 정보가 어딘가 load-bearing). 안 붕괴하면 전제 무효 → 死.

per-module redundancy score (family 내):
```
R(M) = 1 - (return_clean - return_swap(M)) / (return_clean - return_all_swapped)
```
R(M)≈1 = 모듈 M이 family 내 task label 무시(redundant), R(M)≈0 = 필수.

**[필수 #2] R(M)을 "task 무시"로 읽으려면 그 모듈의 *총 영향력*이 비자명해야 한다.** horizon=3(config 확인)에서 dynamics는 3 step만 굴러 plan에 영향이 작고 terminal Q가 value를 지배할 수 있다 → 높은 R(dynamics)이 "task 무시"가 아니라 "H=3에서 dynamics 영향 자체가 작음"일 수 있다. pi도 MPPI가 나쁜 prior를 보정하므로 redundant처럼 보일 수 있다(planning이 보상). 따라서 swap과 *별도로* 각 모듈의 **전체 입출력을 corrupt**(task slice만이 아니라)해 그 모듈의 total influence를 측정한다. R(M)은 total influence가 비자명한 모듈에서만 "task 무시"로 해석한다. (이 control이 H=3 dynamics-influence 함정과 pi-MPPI-보정 함정을 분리.)

**핵심 mechanistic 확인 (왜 redundant한가):** dynamics가 same-family에서 redundant로 나오면, encoder embedding을 같은 family 내 swap했을 때 z→task linear-probe accuracy가 *떨어지는지* 측정한다. (단순 z→task probe는 obs가 task-distinctive해 trivially 높게 나오므로, *embedding을 swap했을 때 probe가 떨어지느냐*가 "embedding이 task를 라우팅"의 진짜 테스트. 안 떨어지면 narrative는 "obs가 task를 구별, label redundant"이고 그게 바로 #1 same-embodiment framing이 load-bearing인 이유.)

**Onboarding 처방 검증(decision-change, b2) — [#4 별도 후속, 학습 필요, 2-3h PoC 밖]:** held-out task에서, locus map이 가리키는 모듈만 풀어 적응 vs 모든 모듈 적응(BL-19식) vs full FT 비교 — locus가 맞으면 "지목된 모듈만 적응"이 full FT에 근접하면서 BL-19식을 이긴다. **이 arm은 *학습*을 요구하므로 2-3h frozen PoC에 포함되지 않는다.** 명확히: cheap PoC는 (a) premise check, (b) same-family locus map(surprise), (c) mechanism probe를 제공. b2/onboarding payoff는 더 큰 follow-up 실험.

## Novelty 포인트 (최소 3개)

1. **TD-MPC2 대비**: 그들은 task embedding을 5개 모듈에 주되 per-module 필요성을 검증 안 함. "어느 모듈이 task를 실제로 쓰는가"의 locus map은 미보고.
2. **BL-19 대비(self-graveyard)**: 정반대 방법론(학습 0, intervention). 게다가 BL-19의 *실패 원인을 설명*하는 mechanistic narrative를 제공 — generator가 자기 실패를 진단하는 follow-up.
3. **input-ablation/attribution 일반 대비**: (a) swap(not zero)으로 OOD-zero confound 제거, (b) deliverable이 sensitivity-table이 아니라 *adaptation-locus decision rule* + encoder-routing mechanism, (c) null-direction만 주장해 confound-free.
4. **결정-변경(bar3)**: "새 task가 오면 dynamics가 아니라 encoder/표현 쪽에 적응 capacity를 넣어라"는 실무 처방. BL-19(embedding-only)와 naive full-FT 사이의 *어디를 풀지*를 바꿈.

## 선행 연구 위험 요소 (literature checker가 확인할 항목)

- TD-MPC2 후속/citing 중 multitask conditioning ablation을 한 것(2024~2026).
- "where does task/context information live in a conditioned policy/dynamics" 류 representation-locus 논문(meta-RL, contextual MDP, RL probing).
- Causal/intervention-based attribution in RL(activation patching, embedding swap) — LLM mechanistic interp의 activation patching이 RL/WM로 포팅된 사례.
- conditional computation / MoE routing과의 구분 확인.

## 예상 실험 Skeleton

- **Base model**: mt30-1M.pt (공개, `evaluate.py task=mt30 ...`로 로드 검증). 학습 불필요.
- **Benchmark**: mt30의 30 DMControl task(state obs, 8GB GPU OK).
- **측정 (cheap PoC, 학습 0, 2-3h)**:
  1. (premise) all-swapped return 붕괴 확인.
  2. (main, same-family) per-module R(M) — dynamics/reward/pi/Q/encoder, family별. + total-influence control(#2).
  3. (mechanism) encoder embedding swap 시 z→task probe accuracy *하락* 여부.
- **측정 (follow-up, 학습 필요, PoC 밖)**:
  4. (decision, b2) held-out reward-variation task vs physics-variation task에서 locus-guided 적응 vs embedding-only(BL-19) vs full-FT.
- **예상 결과**: same-family에서 **R(dynamics) 높음**(physics 동일→label redundant, total-influence는 비자명), **R(reward) 낮음**(reward 다름→label 필수). cross-family swap은 trivially redundant(claim 제외). follow-up: reward-variation onboarding은 dynamics 안 풀어도 됨, physics-variation은 풀어야 함(BL-19 설명).
- **NULL 사전등록(반드시)**:
  - all-swapped가 안 붕괴 → 전제 무효 → 死.
  - same-family에서 R(dynamics)과 R(reward)이 *구별 안 됨*(둘 다 높거나 둘 다 낮음) → physics/reward locus 분리 없음 → surprise 死.
  - R(dynamics)이 높지만 total-influence control에서 dynamics 영향이 자명하게 작음(H=3 함정) → "task 무시"로 해석 불가 → claim 축소(influence 보정 후 재판정).
  - (follow-up) locus-guided 적응이 random-module/full-FT 대비 이점 없음 → decision-change 死(축소 fallback: locus map+mechanism만 기여).

## BL-14 교훈 적용 (valid ≠ beneficial)

intervention의 *return 효과*가 직접 deliverable이라 "신호로 target 고치기" 구조가 아니지만, decision-change arm(locus-guided 적응)은 BL-14처럼 무이득일 수 있다. 그래서:
- **trivial baseline**: random-module 적응(아무 모듈이나 1개 풀기) — locus-guided가 이걸 이겨야 함.
- **oracle 통제**: locus map이 가리키는 모듈 = post-hoc best 모듈이 아니라 *swap-PoC로 사전 결정*된 모듈이어야 함(circular 금지).

## Venue Fit 이유

NeurIPS/ICLR: model-based RL representation의 surprising empirical claim(학습된 dynamics가 task conditioning을 무시) + mechanism(encoder routing) + decision rule(onboarding locus). 자기 실패(BL-19) 진단까지 포함해 narrative가 강함. CoRL: 이종 task onboarding의 실무 처방.

## 위험 요소

| 위험 | 가능성 | 완화 |
|---|---|---|
| **모든 모듈이 task 필요로 나옴 (surprise 死, NULL)** | MED | mt30은 task가 30개로 많아 encoder가 z에 라우팅할 압력이 큼 → dynamics redundancy 가능성 ↑. 그래도 PoC가 결정. NULL 사전등록으로 양방향 publishable(단 null-drop이 없으면 약함). |
| all-swapped가 안 붕괴(전제 무효) | LOW | task embedding이 명시적으로 5개 모듈에 들어가고 action_mask까지 task별이라 load-bearing 가능성 매우 높음. premise check가 1순위. |
| "단순 ablation study"로 INCREMENTAL 공격 | MED-HIGH | firewall 3종(swap-not-zero / decision-rule-not-table / mechanism+BL-19 진단). headline을 *locus + onboarding 처방*으로, 절대 sensitivity-table로 안 함. |
| decision-arm(locus-guided 적응) 무이득(BL-14형) | MED | trivial(random-module) baseline + 사전결정 locus. 무이득이면 locus map 자체로 축소 fallback(여전히 surprise+mechanism 기여). |
| swap이 여전히 mild-OOD(다른 task emb도 학습 때 그 모듈이 본 분포지만 그 state엔 안 맞음) | LOW-MED | mean-embedding arm으로 교차확인(mean은 hull 내부라 가장 in-distribution). swap-wrong과 swap-mean이 일치하면 robust. |

## 내 4-bar self-judgment (validator 전, 정직)

| bar | 판정 | 근거 |
|---|---|---|
| plausible | PASS | task embedding이 5개 모듈에 들어가고(코드 확인), 30 task면 encoder가 z에 task를 심을 압력 큼. dynamics가 z만으로 충분할 개연성 실재. |
| surprising | **CONDITIONAL** | null-drop(dynamics가 task 무시)이 나오면 surprising("5곳에 줬는데 안 쓴다"). 전부 필수면 unsurprising. → PoC가 결정. NULL 사전등록. |
| decision-changing | PASS | onboarding locus 처방(어디를 풀지) + BL-19 실패 진단. 실무 behavior 변경. |
| distinguishable | **PASS (R9 confound 회피)** | 같은 frozen model, intervention 한 축(어느 모듈에 wrong-emb)만 변경 → attribution 완전 통제. cross-codebase 없음(단일 mt30). swap-vs-clean은 오직 그 모듈 입력만 차이. |

→ **bar2(surprise)가 유일 risk, PoC로만 풀림(NULL 사전등록).** bar4는 single-frozen-model intervention이라 깨끗이 통과(R9 lesson). validator로 (1) all-swapped 붕괴? (2) 어떤 모듈 R(M)이 높나? (3) encoder probe acc? — 학습 0이라 2-3h 내 가능.
