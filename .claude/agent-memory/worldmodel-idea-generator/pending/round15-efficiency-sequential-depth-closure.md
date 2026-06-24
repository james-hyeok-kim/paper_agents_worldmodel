---
slug: round15-efficiency-sequential-depth-closure
status: closed-lever-finding
created: 2026-06-22 KST
category: efficiency (b1/b2/b3)
result: 0 clean
---

# R15 — b1/b2/b3 efficiency: the one live target (planning sequential-depth) closes to literature

## 요청
team-lead: R14 PoC FAIL(BL-20) 확정 후 b1(latent compression)/b2(planning efficiency)/b3(rollout
efficiency) 중 가장 유망한 방향에서 R15 아이디어 1개 생성 + 문헌 검증.

## 결론: 0 clean. efficiency framing은 R12/R12b에서 이미 code-level로 닫혔고,
## 유일하게 남아있던 narrow 타깃(planning sequential-depth)이 R15 문헌검증에서 두 named-paper에 선점됨.

---

## Step 1 — discriminating check (advisor 권고): wall-clock가 실제로 떨어지는 메커니즘을 이름붙여라

내 누적 BLACKLIST가 닫은 efficiency 메커니즘 5종:
1. **call-skip** — BL-18(spectral 0.710×) + dual-rate parity 종결
2. **batch-reduction / dedup / sub-batch** — BL-11(0.49×) / BL-16(0.397×) / BL-17(0.541×)
3. **small-op / low-rank replacement** — BL-10(1.41× 천장미달) / BL-12(0.71×)
4. **adaptive horizon/iteration stop** — BL-08(controller 1.0×) + AOP(1912.01188) 선점
5. **gradient-path skip** — R12b gradient-sever(encoder/decoder가 매 step backprop path)

b1/b2/b3 후보가 이 5종 중 하나로 speedup을 얻으면 validator GPU+CPU microbench에서
지난 6개 BL과 동일하게 dead-on-arrival. 6번째 메커니즘이 필요.

## Step 2 — 유일한 예외: planning sequential-depth (launch-bound 벽이 적용 안 되는 곳)

BL-15가 직접 지명한 진짜 bottleneck = TD-MPC2/Puppeteer MPPI의 **horizon-sequential dynamics
rollout**. 이건 kernel-launch-bound가 아니라 **sequential-latency-bound** → sequential step을
제거하면 batch-reduction과 달리 실제 GPU wall-clock이 줄어듦. 이론상 유일한 live efficiency 타깃.

단 이 타깃의 알려진 attack 2개는 이미 occupied:
- adaptive-horizon stop = AOP(1912.01188) 선점 (BL-15 명시)
- parallel amortization(전체 rollout을 single feedforward로 fold) = 내 pending **amortized-rollout-operator**
  = closed-loop blocker + INCREMENTAL verdict

→ 살려면 sequential-depth에 대한 **genuinely 세 번째 attack**이 필요 + jumpy/temporal-abstraction
  WM 대비 lit-check 생존. advisor가 "high bar"로 명시.

## Step 3 — 세 번째 attack 후보 문헌검증 (WebSearch, 2026-06-22)

후보: **parallel-in-time iterative solve** (Parareal / DEQ / Picard) — 원래 dynamics function은
유지하되, H개 latent를 K<<H번의 parallel sweep으로 동시에 수렴시켜 sequential depth H를 줄임.
AOP(single-shot feedforward distill)와 구조적으로 다름.

**두 named-paper에 정면 선점됨:**

1. **PaMoRL** — "Parallelizing Model-based RL Over the Sequence Length" (NeurIPS 2024,
   OpenReview R6N9AGyz13, code: github.com/Wongziseoi/PaMoRL). Parallel World Model(PWM) +
   Parallelized Eligibility Trace Estimation(PETE)로 MBRL을 sequence length 축으로 병렬화.
   = parallel-in-time/parallel-scan attack를 직접 점유.
   **결정적**: PaMoRL 자료가 closed-loop blocker를 자기 입으로 명시 — *"the imaginations cannot
   be parallelized over the sequence length because a non-linear actor network is required for
   action sampling. (open-loop model output + TD-λ만 parallel scan 가능)"* → 정책-조건부 rollout의
   sequential depth는 PaMoRL도 못 푼다고 인정한 부분이고, 그걸 푸는 게 바로 AOP의 closed-loop
   흡수(=INCREMENTAL). 즉 parallel-scan 갈래는 PaMoRL이 먹고, 남은 closed-loop 갈래는 AOP가 먹음.

2. **Looped World Models (LoopWM)** — arXiv:2606.18208v1 (2026-06-16, **R15 6일 전**).
   parameter-shared transformer block이 latent representation을 iterative refine(=DEQ/looped),
   contractive state-retention + **adaptive early-exit**, 빠른 transition은 inner loop 조기종료,
   FLOPs 최대 2 orders-of-magnitude 절감. "Deferred Decoding"(LoopWM-DD)으로 latent-space만
   rollout 후 terminal step에서만 decode. = DEQ/fixed-point + adaptive-stop + latent-only rollout
   attack을 통째로 점유. "iterative latent depth as a new scaling axis"를 명시적으로 선점.

→ 세 번째 attack의 두 자연스러운 형태(parallel-scan, DEQ-iterate)가 각각 PaMoRL / LoopWM에
  선점됨. advisor의 "high bar"(genuinely third + lit-survive) 미충족. **clean efficiency idea 없음.**

## 4-wall 종합

| 갈래 | 차단 |
|---|---|
| b1 latent compression | 압축으로 얻는 wall-clock = call-skip(BL-18)/small-op(BL-10/12)/batch(BL-11/16/17) 5종 회귀. dual-rate parity가 latent-skip의 quality 천장도 종결 |
| b2 planning efficiency (sequential-depth) | adaptive-stop=AOP선점 / parallel-amortize=AOP(INCREMENTAL,closed-loop) / parallel-scan=PaMoRL선점 / DEQ-iterate=LoopWM선점(6일전) |
| b2 planning efficiency (terminal Q / dedup) | BL-15(vmap vectorized) / BL-16(gather-scatter launch). vectorized op batch 축소 무이득 |
| b3 rollout efficiency | RSSM call-skip 전체 종결(BL-18). imagination이 hot-path 아님(37%, BL-11). batch 축소 launch-bound(BL-11/17) |

## 메타교훈
- **efficiency는 메커니즘이 아니라 framing 단위로 죽는다(R12 재확인).** 어떤 신규 메커니즘도
  "wall-clock가 실제 떨어지는 5+1 통로" 밖으로 못 나감. 유일 통로(sequential-depth)가 R15에서
  3-attack 모두 occupied로 확정.
- **literature는 6일 전에도 움직인다.** LoopWM(2606.18208)이 R15 6일 전 등장해 DEQ-iterate 갈래를
  닫음 — "pending-fresh ≠ literature-fresh"의 또 다른 사례. 생성 전 *최신* arXiv를 반드시 칠 것.
- **PaMoRL의 closed-loop 자백이 핵심 증거**: 정책-조건부 sequential rollout은 "병렬화 불가"가
  literature consensus. 그걸 우회하는 유일 길(policy 흡수)은 AOP가 이미 점유 + INCREMENTAL.
- R15는 efficiency lever를 *코드*가 아니라 *literature*로 한 번 더 못박음(R12=code-walk closure,
  R15=literature closure of the one code-level survivor spot). lever 6개째 닫힘.

## forward (team-lead 결정 필요)
efficiency(b1/b2/b3)는 runnable base에서 code-level + literature-level 양쪽으로 닫힘.
미테스트로 남은 유일 lever = **human-defined quality / sample-eff / capability problem-setting**
(R8 acquisition-curve bet 공간). efficiency로 rebrand 금지(R10 rule). validator wedge 해소가
새 idea 생성보다 EV 높음(R9·R10·R11·R12 4번 escalation, pending 24+ 적체 survival=0).
