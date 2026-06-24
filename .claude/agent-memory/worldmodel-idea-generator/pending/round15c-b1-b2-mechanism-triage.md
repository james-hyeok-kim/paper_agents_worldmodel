---
slug: round15c-b1-b2-mechanism-triage
status: closed-finding (없음)
created: 2026-06-23 KST
category: b1 latent compression + b2 planning efficiency (efficiency 재개)
result: 0 — 모든 named 후보가 gate-1(wall-clock)에서 mechanism-mapping 레벨로 닫힘. gate-3(literature) 미도달.
---

# R15c — b1/b2 efficiency 재개: named 후보 전부 gate-1에서 닫힘 (없음)

## 요청
coordinator: direction (i)(quality/sample-eff) 중단, b1(latent compression)+b2(planning efficiency)
재개. 각 방향 "아직 탐색 안 된 가능성" 명시. 필수 gate: ①wall-clock(launch-bound 死) ②Amdahl
(hot-path=horizon-sequential dynamics) ③literature. b1/b2 각 1+ 후보 탐색, 없으면 "없음" 명확히.

## 핵심 원칙 (이번 triage의 논리)
**efficiency는 mechanism이 아니라 framing 레벨에서 죽었다(R12/R15).** framing-level kill의 원인 =
base 구조(작은 vectorized op → GPU launch-bound). mechanism은 base 구조를 바꾸지 않는 한 framing-level
kill을 탈출 못 함. 따라서 named "untested" 후보들은 *mechanism*이고, 전부 닫힌 6-lever 집합으로 환원됨:
**{①call-skip ②batch-reduction ③small-op ④adaptive-stop ⑤gradient-skip ⑥sequential-depth 제거}.**
gate-1(wall-clock)을 *순서대로 먼저* 적용 → literature 도달 전에 전부 닫힘(coordinator의 gate 순서와 일치).

## Triage 표 — 각 named 후보 → 어느 닫힌 lever로 routing되나

### b1 (latent compression)
| 후보 | routing | 死 근거 |
|---|---|---|
| latent dimension reduction | #3 small-op | BL-10이 D=128 vs 512 GRU를 측정: 16× 이론 천장 대비 1.41×(launch-bound). 작은 latent ≠ 빠른 wall-clock. + 압축→quality cost |
| structured/factored latent | #1 call-skip / #3 | **BL-18이 정확히 frequency-factored 시도**(spectral multi-level RSSM, 0.710×). 이미 종결 |
| quantized latent | — | categorical latent은 이미 DreamerV2 baseline. INT quantization은 compute/bandwidth-bound 큰 op만 도움, launch-bound 작은 op엔 wall-clock 이득 0 |
| information bottleneck | (miscategorized) | efficiency mechanism이 아님 — representation-quality objective라 wall-clock 안 줄임. R15b(quality, closed) 소속 |

### b2 (planning efficiency)
| 후보 | routing | 死 근거 |
|---|---|---|
| action sampling strategy | #2 batch-reduction / #4 adaptive-stop | fewer samples=BL-15/16(launch-bound) / fewer iterations=AOP(1912.01188) 선점 |
| elite selection 방법 | #2 | **BL-15가 정확히 cheap-rank elite**(0.94× GPU). 死 |
| planning algorithm 변경(CEM 등) | #6 sequential-depth | optimizer 교체해도 dynamics는 여전히 horizon-sequential rollout(coordinator 본인이 지목한 Amdahl-dominant cost). sequential depth 제거 아님 → PaMoRL/LoopWM/AOP 선점 영역 그대로 |
| MuZero-family base | (launch-bound 미적용 — 다른 死) | 아래 별도 |

## MuZero/MCTS — launch-bound로 죽지 않음. *다른* 근거 2개로 死 (mislabel 금지)
MCTS는 sequential, non-vectorized node expansion → TD-MPC2 vectorized MPPI에 없는 per-call headroom이
*있을 수 있음*. launch-bound kill 자동 적용 안 됨. 그러나 2개 다른 근거로 死:
1. **runnable infra 부재**: base-models-infra = DreamerV3/TD-MPC2/Puppeteer만. MuZero/EfficientZero 코드 없음
   → wall-clock microbench 불가 → validator wall-clock gate를 *구조적으로* 통과 못 함(R10/R11 DreamerV4 기각과 동형).
2. **이미 probed**: `pending/subtree-reuse-muzero.md` 존재, **verdict INCREMENTAL**(literature-checked).
   inter-step subtree carry-over는 BL-07 redirect로 시도했으나 INCREMENTAL.

## 결론: 없음 (crisp closure)
launch-bound vectorized base에서 남은 wall-clock headroom = **generic systems optimization**(kernel
fusion, CUDA graphs, mixed precision, CUDA streams)뿐 — 이건 novel WM research contribution이 아니고
어차피 obvious baseline. **모든 WM-novel lever는 닫힌 6-lever를 경유해야 함** → efficiency는
mechanism-dead가 아니라 **framing-dead**. b1/b2 양쪽 0.

**caveat (선제 차단)**: launch-bound는 이 base들의 *실제 operating batch size에서 측정된* 것이지 가정이
아님. 다른 batch regime을 전제한 아이디어는 실제 base의 wall-clock을 못 도와 gate 통과 못 함.

## 메타교훈
- mechanism-level "untested" 리스트는 framing-level kill을 탈출 못 함 — base 구조(vectorized→launch-bound)를
  바꾸지 않으면 모든 mechanism이 닫힌 6-lever로 환원. gate-1을 순서대로 먼저 적용하면 literature 도달 전 닫힘.
- **MuZero류는 launch-bound가 아니라 (runnable-infra 부재 + already-probed)로 죽인다** — MCTS가
  vectorized가 아님을 모르면 live한 줄 착각. base 교체 제안은 runnability를 먼저 확인.
- information bottleneck처럼 quality objective를 efficiency로 miscategorize한 후보 주의 — wall-clock 안 줄임.
- R12=code-walk closure, R15=literature closure(sequential-depth), R15c=mechanism-triage closure(b1/b2).
  efficiency lever 7번째 확인. framing-dead 재확정.
