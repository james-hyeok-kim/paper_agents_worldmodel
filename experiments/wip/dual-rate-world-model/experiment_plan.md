## Experiment Plan: Dual-Rate World Model (slow/fast latent 시간 해상도 분해)

작성: 2026-06-11 KST · slug: `dual-rate-world-model` · category E (Hierarchical)
선행 검증: literature=INCREMENTAL, PoC=CONDITIONAL-GO (rollout_speedup 1.93×, quality_proxy_delta 0.0)

---

### Core Claim

> Dual-rate RSSM(slow latent을 K step마다, fast latent을 매 step 갱신)은 collapse 방지 레시피(fast→slow IB + slow smoothness penalty) 하에서, DreamerV3 대비 **matched 100k env-step budget에서 episode return drop <5%**를 유지하면서 **imagination rollout throughput을 1.3~1.6× 가속**(dynamics FLOPs 35~55% 절감)한다.

핵심 메시지 두 가지 (literature verdict가 강제):
1. **imagination rollout FLOPs를 1급 효율 목표로 명시한 최초의 dual-rate stochastic RSSM** — CW-VAE/MTS3/THICK 중 누구도 rollout speedup을 수치로 보고하지 않음.
2. **stochastic latent의 slow/fast collapse 방지 학습 레시피** — fast→slow information bottleneck + slow temporal-smoothness penalty.

> 주의(verdict 권고): "action window-aggregate를 slow에 주입"은 MTS3(NeurIPS 2023)에 선점됨. 이를 novelty로 내세우지 말고, MTS3와의 차이(stochastic vs deterministic Kalman, RL rollout efficiency vs system-ID RMSE)를 related-work에서 서술. 본 실험은 위 2개 차별점을 직접 측정하는 데 집중한다.

---

### PoC 결과 요약 (이미 완료, synthetic CPU mock)

| 지표 | 값 | gate | 판정 |
|---|---|---|---|
| rollout_speedup | 1.93× | >1.5× | PASS |
| quality_proxy_delta (MSE) | 0.0 | <0.05 | PASS |
| slow_delta / fast_delta | 0.088 / 0.327 | fast>slow (분리) | PASS (collapse 없음) |
| theoretical FLOPs speedup | 3.86× | — | (상한선) |

PoC 파일: `experiments/wip/dual-rate-world-model/poc.py`. K=3, slow_hidden=128, fast_hidden=64.
**한계**: numpy GRU mock + synthetic slow(상수배경)/fast(이동점) video. 실제 RSSM에서 분리가 학습되는지는 미검증 → Week 1의 핵심 리스크.

---

### 환경 현황 (실측, 2026-06-11)

| 항목 | 상태 | 영향 |
|---|---|---|
| torch 2.9.1 + CUDA, 2× B200 (각 ~40GB free, 138GB는 타 작업 점유) | OK | DreamerV3 DMC vision은 <10GB → 단일 GPU로 충분. 공유 사용 존중. |
| mujoco 3.6.0, gymnasium-robotics 1.4.2 | 설치됨 | dm_control의 hard dependency 충족 |
| `dm_control` | **미설치** | Day-1 `pip install dm_control` (one-line, mujoco 존재하므로 blocker 아님) |
| JAX | **없음** | 공식 DreamerV3(JAX) 사용 불가 → **PyTorch port를 fork** |
| `ale_py` | 미설치 | Atari는 Week 3+에서만 필요, Day-1 불필요 |

**Fork 타깃 (확정)**: [`NM512/dreamerv3-torch`](https://github.com/NM512/dreamerv3-torch) — 표준 충실 PyTorch port.
- `networks.py`: `class RSSM`, per-step recurrent 업데이트는 `RSSM.img_step()` / `obs_step()` 내부의 `self._cell(x, [deter])` (`class GRUCell`). config 필드: `deter`(deterministic dim), `stoch`, `hidden`.
- `models.py`: `_imagine()`이 horizon 동안 `dynamics.img_step(state, action)`만 호출 — **encoder/decoder 미사용, 순수 latent 공간**. 여기가 dual-rate가 이기는 정확한 지점.

---

## 1주 Mini-Experiment (가장 빠른 검증) — M0

목표: "실제 RSSM에서 dual-rate가 (a) 안정적으로 학습되고, (b) 분리가 유지되며(collapse 없음), (c) matched 100k budget에서 return drop이 작고, (d) imagination throughput이 실제로 빨라진다"를 **단일 task**로 proof-of-life.

> Week 1은 full claim gate(return drop<5% AND speedup>1.5×)에 바인딩하지 않는다. 그건 Week 3+ 바.
> Week 1 바: **compute breakdown 확보 + 학습 안정 + 분리 유지 + matched-budget return 비교 + imagination speedup 측정값 확보.**

#### M0-Day0: Compute Breakdown 프로파일링 (split 구현 전 1급 선행 task)

> **이걸 먼저 한다. dual-rate 구현 전에.** PoC의 1.93×/3.86×는 GRU가 비용을 지배한다는 가정 위의 dynamics-only 수치다. 실제 DreamerV3에서는 두 phase에서 GRU 비중이 다르다:
> - **World-model training step**: CNN encoder/decoder + heads가 비용을 지배. RSSM dynamics는 일부분 → dynamics를 3× 빠르게 해도 training step wall-clock은 거의 안 움직임.
> - **Imagination rollout** (`_imagine`): CNN 없음, latent-only → RSSM + actor/critic MLP가 지배. **여기서 dual-rate가 실제로 이긴다.**
>
> **측정 대상 (vanilla, NM512 port, dmc_vision walker-walk, 실제 architecture dims로):**
> - model-learning vs behavior-learning 각각에서 GRU(dynamics) / encoder / decoder / heads(reward·cont) / actor·critic의 wall-clock·FLOPs 분해 (`torch.profiler` 또는 구간 cuda-synced timer + `thop`).
> - 산출: dynamics가 imagination에서 차지하는 비율 → **realized imagination speedup의 상한(ceiling) 확정.**
>
> **이 숫자가 하는 일**: (a) Core Claim을 "imagination-rollout speedup"(방어 가능)으로 고정할지, "training speedup"(희석되어 방어 어려움)을 피할지 확정. (b) GPU-week를 쓰기 전에 헤드라인을 재조정. 만약 dynamics가 imagination 비용의 일부에 불과하면 speedup 타깃을 미리 하향(예: 1.3×)하고 deter_slow 비중·K를 그에 맞춰 설계.
> 산출물: `experiments/wip/dual-rate-world-model/breakdown.json` + 1줄 결론(헤드라인 claim 확정).

#### 코드 변경 포인트 (구체)

기준 파일: `dreamerv3-torch/networks.py`, `models.py`, `configs.yaml`

1. **`RSSM.__init__` (networks.py)** — deter를 slow/fast로 분할.
   - 새 config: `deter_slow`(예: 256), `deter_fast`(예: 128), `slow_K`(예: 3). 기존 단일 `deter`(=512 등)를 두 GRUCell로 대체.
   - `self._cell_fast = GRUCell(..., deter_fast)`, `self._cell_slow = GRUCell(..., deter_slow)`.
   - feature(decoder/reward 입력)는 `[deter_slow, deter_fast, stoch]` concat.

2. **`RSSM.img_step()` (networks.py)** — dual-rate 갱신 + carry-forward.
   ```python
   # 매 step: fast 갱신 (slow는 입력으로 read-only 참조)
   x_fast = cat([prev_stoch, prev_deter_slow, action]); _, deter_fast = self._cell_fast(x_fast, [prev_deter_fast])
   # K step마다만 slow 갱신, 사이 step은 carry-forward
   if step_idx % self.slow_K == 0:
       x_slow = cat([prev_deter_fast, agg_action_window]); _, deter_slow = self._cell_slow(x_slow, [prev_deter_slow])
   else:
       deter_slow = prev_deter_slow   # carry-forward, 연산 0
   ```
   - `step_idx`를 state dict에 carry (또는 `step_since_slow_update` 카운터를 feature에 추가 → slow stale 보정 단서).
   - `obs_step()`도 동일 분할(관측 시엔 매 step 둘 다 갱신해도 무방, 핵심은 imagination).

3. **Collapse 방지 손실 (models.py, `WorldModel._train` loss 부분) — Week 1에 필수 포함(novelty 핵심)**
   - **fast→slow information bottleneck**: slow 갱신 입력의 fast 성분에 작은 KL/L2 penalty (`λ_ib · ||proj(deter_fast)||²` 또는 fast→slow 채널에 variational bottleneck). 시작값 `λ_ib=1e-3`.
   - **slow temporal-smoothness**: `λ_sm · mean(||deter_slow_t − deter_slow_{t-1}||²)` over imagination horizon. 시작값 `λ_sm=1e-2`.
   - 두 penalty를 기존 model loss에 가산. (변경량 ~20줄)

4. **분리도 metric 로깅 (models.py 또는 별도 hook)** — 매 eval마다:
   - `slow_delta = mean ||Δdeter_slow||`, `fast_delta = mean ||Δdeter_fast||` (목표 fast>slow).
   - `separation_score = fast_delta / (slow_delta + fast_delta)` (PoC에선 0.78).
   - slow/fast cosine sim. collapse 경보: slow_delta ≈ fast_delta 또는 한쪽 ≈ 0.

5. **imagination throughput 계측 (models.py `_imagine`)** — `torch.cuda.synchronize()` 감싼 timer로 imagined steps/sec 측정. baseline(vanilla)과 동일 horizon/batch에서 비교. FLOPs는 분석적으로 계산(fast 매step + slow/K) + `fvcore`/`thop`로 교차검증.

#### 실행 명령

```bash
# Day 0: 셋업 + compute breakdown (split 구현 전 선행)
pip install dm_control
cd /data/jameskimh/worldmodel/dual-rate-world-model && git clone https://github.com/NM512/dreamerv3-torch.git
# 코드는 git tracked 위치로 복사: experiments/wip/dual-rate-world-model/dreamerv3-torch/
# vanilla short-run에서 model-learning / imagination 각 구간의 dynamics vs encoder/decoder/heads/MLP wall-clock·FLOPs 분해
CUDA_VISIBLE_DEVICES=0 python experiments/wip/dual-rate-world-model/profile_breakdown.py \
    --configs dmc_vision --task dmc_walker_walk --out breakdown.json
# → imagination 비용 중 dynamics 비율 확인 → realized speedup ceiling 확정 → 헤드라인 claim 고정

# Day 1: vanilla baseline (matched budget)
CUDA_VISIBLE_DEVICES=0 python dreamer.py --configs dmc_vision --task dmc_walker_walk \
    --logdir /data/jameskimh/worldmodel/dual-rate-world-model/baseline_walker \
    --steps 100000 --seed 0    # vanilla DreamerV3 baseline

# Day 3-4: dual-rate 구현 후
CUDA_VISIBLE_DEVICES=1 python dreamer.py --configs dmc_vision dual_rate --task dmc_walker_walk \
    --logdir /data/jameskimh/worldmodel/dual-rate-world-model/dualrate_K3_walker \
    --steps 100000 --seed 0
# dual_rate config block: deter_slow=256, deter_fast=128, slow_K=3, lambda_ib=1e-3, lambda_sm=1e-2

# Day 5: imagination throughput + 분리도 + return 비교 → results.json
python experiments/wip/dual-rate-world-model/measure_rollout.py \
    --baseline_ckpt .../baseline_walker --dualrate_ckpt .../dualrate_K3_walker
```

#### M0 성공 기준 (proof-of-life)

| 체크 | 기준 |
|---|---|
| Compute breakdown 확보 | imagination 내 dynamics 비율 측정 완료 → headline claim(imagination-rollout speedup) + realized ceiling 확정 (`breakdown.json`) |
| 학습 안정 | dual-rate loss 발산/NaN 없이 100k 완주 |
| 분리 유지 | `fast_delta > slow_delta` (separation_score > 0.6), collapse 경보 없음 |
| Return (matched 100k) | dual-rate return ≥ vanilla return − 15% (Week1 느슨한 바; full 바는 5%) |
| Imagination speedup | measured imagination steps/sec > 1.2× (실측 > 0이면 진행, 1.5×는 Week3 바) |

M0 실패 시 → fallback (아래) 적용 후 K=2 또는 deter 비율 재조정으로 1회 재시도.

---

## 4주 논문용 Full Experiment

> 핵심 metric은 **imagination rollout throughput**(primary). 전체 training step wall-clock은 encoder/decoder(dual-rate가 안 건드림) 포함이라 speedup이 희석됨 → **secondary로 정직하게 보고**. PoC의 1.93×는 순수 dynamics loop였고, 실제 imagination엔 prior/posterior/reward MLP가 안 줄어드는 부분이 있어 realized speedup은 1.93× 미만(1.3~1.6× 타깃)으로 정직하게 명시.

### Week 1–2: M0 + Factor Sweep
- M0 (위) 완료.
- **K sweep**: K ∈ {2, 3, 4, 6} — quality-efficiency tradeoff curve. walker-walk + cheetah-run 2개 task.
- **deter 분할 비율 sweep**: (slow,fast) ∈ {(256,128),(384,128),(256,256)}.
- 산출: tradeoff curve (x=imagination steps/sec 또는 dynamics FLOPs, y=episode return). sweet spot K 확정.

### Week 3–4: Main Experiments + Ablation (논문용)
- **Benchmark 2종 (최소)**:
  - DMControl (vision): walker-walk, cheetah-run, cartpole-swingup, finger-spin, reacher-easy, quadruped-walk (배경 정적 비율 다양 — dual-rate 가설에 직접 부합).
  - Atari 100k (`pip install ale_py`): 4 games (Pong, Breakout, Krull, MsPacman) — sample efficiency 벤치마크.
- **Seed 3개** (0,1,2), mean ± std. 동일 env 설정(action_repeat=2, image 64×64; Atari sticky_actions=True, 100k budget).
- **Ablation (collapse 방지 레시피 = novelty 증거, 1급 실험)**:
  | variant | 설명 | 측정 목적 |
  |---|---|---|
  | full | dual-rate + IB + smoothness | 제안 방법 |
  | − IB | bottleneck 제거 | IB 기여 (collapse로 분리 붕괴 예상) |
  | − smoothness | smoothness 제거 | smoothness 기여 |
  | − both | 둘 다 제거 (naive dual-rate) | **핵심: 레시피 없으면 collapse → separation_score 붕괴 입증** |
  - 이 ablation이 "CW-VAE + actions가 아니다"의 증거. separation_score가 −both에서 fast≈slow로 무너지면 novelty 입증.

---

### Baseline 설정 (반드시 포함)

| Baseline | 설명 | 포함 이유 |
|---|---|---|
| DreamerV3 (vanilla, NM512 port) | dual-rate 없는 원본, 동일 deter 총합 | floor/ceiling line 필수, fair compute |
| DreamerV3 deter-축소 | deter를 dual-rate와 동일 평균-FLOPs로 단순 축소 | "그냥 작게 만든 것"과 dual-rate 구분 (효율은 같고 품질은 dual-rate가 높아야 함) |
| naive dual-rate (−both) | IB/smoothness 없는 dual-rate | collapse 방지 레시피 기여 증명 |
| **Dual-Rate (full)** | 제안 방법 | — |

선택적 비교(related-work 서술용, 재구현 부담 크면 인용만): MTS3(deterministic, system-ID), CW-VAE(action 없음). 직접 재구현은 4주 범위 밖 → 정성 비교로 처리.

---

### Metrics

**효율 (필수)**
- **Imagination steps/sec** (primary, `_imagine` 구간 cuda-synced timer)
- Dynamics FLOPs per imagined step (분석적 + thop 교차검증) — fast + slow/K
- Total training step wall-clock (secondary, 희석됨 명시)
- GPU memory peak (GB)

**품질 (필수)**
- Episode return (normalized, mean ± std over 3 seeds)
- Sample efficiency: vanilla 최종 성능의 90% 도달까지 env steps
- Open-loop prediction error: latent rollout 후 decoder 복원 PSNR/SSIM (slow stale 구간 품질 직접 측정)

**분리도 (novelty 증거, 필수)**
- separation_score = fast_delta/(slow_delta+fast_delta), slow/fast cosine sim, slow smoothness

**논문 필수 Figure**
- Quality-Efficiency Tradeoff: episode_return (y) vs imagination steps/sec 또는 dynamics FLOPs (x), K sweep 점들
- Learning curve: return vs env_steps (dual-rate vs vanilla, sample efficiency)
- Ablation bar chart: separation_score across {full, −IB, −sm, −both} → 레시피 없으면 collapse 시각화
- (선택) slow stale 구간 PSNR vs step-since-slow-update

---

### Compute 예산

| 단계 | 구성 | GPU | 예상 | GPU-hours |
|---|---|---|---|---|
| M0 Smoke | vanilla + dualrate, walker, 100k, seed0 | 1× B200 | 2×3h | 6 |
| Factor Sweep | K(4)×deter(3) ≈ 7 config × 2 task, 100k | 1× B200 | 14×3h | 42 |
| Main DMC | 6 task × 4 baseline × 3 seed | 1× B200 | 72×3h | 216 |
| Main Atari | 4 game × 4 baseline × 3 seed | 1× B200 | 48×2h | 96 |
| Ablation | 4 variant × 2 task × 3 seed | 1× B200 | 24×3h | 72 |
| **Total** | — | — | — | **~432 GPU-hours** |

> 2× B200 활용 시 wall-clock ~1주 압축 가능. Main DMC가 가장 큼 → task를 4개로 줄이면 ~290 GPU-hours로 축소 가능(예산 타이트 시 quadruped/reacher 제외).

---

### 디스크 위치 (MANDATORY)

| 아티팩트 | 위치 |
|---|---|
| 스크립트, results.json, plots, README.md, run.log, dreamerv3-torch fork(코드) | `/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/dual-rate-world-model/` |
| 모델 체크포인트, replay buffer, 대용량 캐시 | `/data/jameskimh/worldmodel/dual-rate-world-model/` |

필수 아티팩트: `README.md`(한국어 자기완결), `results.json`, `run.log`, `run_experiment.py`(또는 fork의 dreamer.py 래퍼), `profile_breakdown.py` + `breakdown.json`(Day-0 compute 분해), `measure_rollout.py`.
CLAUDE.md 규약: `experiment_{index}.md` + `result_{index}.md` 쌍, plan은 `plan_{index}.md`. 시각은 KST.

---

### 주요 위험과 Fallback

| 위험 | 가능성 | 완화 / Fallback |
|---|---|---|
| **실제 RSSM에서 slow/fast collapse** (한쪽 정보 독점) | 높음 | λ_ib·λ_sm 강화 sweep; slow에 step-since-update 입력; 그래도 안 되면 slow를 stop-gradient로 fast와 분리 학습. 분리도 metric으로 매 eval 모니터. |
| **imagination은 빨라도 total training time 증가** | 중간 | training wall-clock 동시 측정, primary는 imagination throughput으로 정직하게 framing. encoder/decoder 미변경 명시. |
| **realized speedup이 1.5× 미만** (MLP 부분 미축소) | 중간 | deter_slow를 더 크게(연산 집중) + K↑로 평균비용↓. FLOPs breakdown(어느 부분이 안 줄었나) 투명 보고. 1.3×라도 품질 동등이면 publishable. |
| **NM512 port API가 fetch와 다름** | 중간 | Day-1에 코드 직접 확인(img_step/_cell/GRUCell 실존 검증). 다르면 동일 구조 port(pydreamer 등)로 전환, 핵심 분할 지점은 동일. |
| **return drop이 5% 초과** | 중간 | factor sweep에서 sweet spot K; slow stale 보정 신호 추가; deter 분할 비율 조정. |
| **dm_control/ALE 재현 오차** | 낮음 | seed 3개 + 동일 env 설정(action_repeat, sticky_actions). |
| **2× B200 공유 점유로 OOM** | 낮음 | DMC vision <10GB로 40GB free 충분; 단일 GPU per job, 공유 사용 존중. |
