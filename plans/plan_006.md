# Plan 006 — Dual-Rate World Model 논문 확장 (DreamerV3 → MWM + STORM)

## 목표
Dual-rate 아이디어를 DreamerV3 단일 모델에서 **3-model 논문**으로 확장.
"Temporal Rate Separation in World Model Dynamics Improves Sample Efficiency"

MWM (RSSM + ViT encoder, Meta-World) + STORM (Transformer dynamics, Atari) 에 적용해
"RSSM 계열 + Transformer 계열, 3개 벤치마크에서 sample efficiency 개선"을 claim.

---

## 코드 실증: dual-rate가 실제로 skip하는 것

`networks.py:260-268` 코드 확인 (2026-06-20 KST):

```python
# img_step 내부
if step_flat[0].item() % self._slow_K == 0:
    # slow GRU 실행 (K step마다만)
    new_deter_slow = _cell_slow(...)
else:
    # carry-forward: slow GRU 완전 skip
    new_deter_slow = prev_deter_slow  # NO computation
```

**결론**: 
- **Skip되는 것**: `_cell_slow` (느린 GRU, K-1 step마다 skip)
- **항상 실행**: `_cell_fast` (빠른 GRU, 매 step), encoder (매 step)
- **핵심 win**: slow GRU skip = 계산 절감이 아니라 temporal rate 분리 → sample efficiency 정규화 효과

**중요 confound (Plan 001 수정 필요)**: 현 dual_rate config에서 total deter = 256+128 = **384** vs baseline **512** (75% 용량). 크기 차이가 효과를 confound. size-matched baseline 필수.

---

## 현재 상태 (Plan 001 진행 중)

원래 11개 run 모두 6-8k/100k step에서 stall → infra 수리 후 재가동 필요.
GPU 현재 점유 (LLaMA LoRA, PID 441362/441363): 여유 없음 → GPU 해제 후 즉시 relaunch.

---

## Phase 1: DreamerV3 기반 확립 (Plan 001 완주 + confound 수정)

**최우선 (confound 수정)**:
| Run | Config | deter | 목적 |
|---|---|---|---|
| baseline_deter384_seed0 | dmc_vision dyn_deter=384 | 384 | size-matched baseline |
| baseline_deter384_seed1 | dmc_vision dyn_deter=384 | 384 | size-matched multi-seed |

**원래 Plan 001 runs 재가동** (11개):
- Batch A: baseline_seed1/2, dualrate_K3_seed1/2 (walker_walk)
- Batch B: K sweep K=2,4,6 (seed=0)
- Batch C: cheetah_run + hopper_hop (seed=0)

**성공 기준**:
- size-matched baseline 대비 dual-rate ≥0.90× at late-stage (100k), 3 seed mean
- OR size-matched baseline 대비 dual-rate early-phase (+12.4% at 95k) 재현 (≥1.05×)
- 둘 다 실패 시 → MWM/STORM 진행 중단, 논문 재설계 필요

**launch 명령 (GPU 해제 후)**:
```bash
cd experiments/wip/dual-rate-world-model/dreamerv3-torch
# size-matched baseline
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/jovyan/egl_libs MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python dreamer.py --configs dmc_vision --task dmc_walker_walk \
  --logdir /data/jameskimh/worldmodel/dual-rate-paper/baseline_deter384_seed0 \
  --steps 100000 --seed 0 --dyn_deter 384 &

CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/jovyan/egl_libs MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python dreamer.py --configs dmc_vision --task dmc_walker_walk \
  --logdir /data/jameskimh/worldmodel/dual-rate-paper/baseline_deter384_seed1 \
  --steps 100000 --seed 1 --dyn_deter 384 &
```

---

## Phase 2: MWM 확장 (RSSM 계열 generalization)

**Architecture 확인 (younggyoseo/MWM)**:
- CNN stem (3 conv, stride 2) → 4-layer ViT encoder
- RSSM dynamics (DreamerV3와 동일 클래스 구조)
- Decoder: 3-layer ViT
- 주 도메인: Meta-World (robotic manipulation), RLBench
- DMControl: known bugs (June 2023), 사용 금지

**Dual-rate 적용 방식**:
DreamerV3와 동일한 slow/fast GRU split을 MWM의 RSSM에 적용.
ViT encoder는 여전히 매 step 실행 (encoder skip이 아님).
"동일 메커니즘 + 다른 encoder 아키텍처 + 다른 벤치마크"로 generalization 주장 가능.

**구현 난이도**: DreamerV3 dual-rate patch와 거의 동일 (RSSM 구조가 같음).

**Benchmark**: Meta-World MT10 or MT50 (manipulation tasks).

**리스크**: MWM codebase 재현성 (known bugs outside Meta-World).

**Phase 2 gate**: Phase 1에서 size-matched baseline 대비 유의미한 효과 확인 후 진행.

---

## Phase 3: STORM 확장 (Transformer dynamics generalization)

**Architecture 확인 (weipu-zhang/STORM)**:
- Categorical VAE encoder → discrete tokens
- GPT-like causal Transformer dynamics (no GRU)
- Domain: Atari (ALE), Atari 100k benchmark
- Status: no longer maintained (→ OC-STORM), but NeurIPS numbers reproduce

**Dual-rate 적용 방식 (다름)**:
GRU split 대신, Transformer call을 K step마다만 실행:
- Slow path: 전체 Transformer forward (K step마다)
- Fast path: 마지막 hidden state carry-forward (K-1 step)
메커니즘은 같지만 구현이 다름 → "temporal rate separation의 Transformer 버전"으로 기술.

**구현 난이도**: DreamerV3보다 높음 (다른 dynamics 아키텍처).

**Benchmark**: Atari 100k (Atari 표준 benchmark).

**리스크**: 
- 다른 벤치마크 + 다른 메커니즘 → "별도 논문" 같아 보일 수 있음
- 유지보수 중단으로 repro 어려울 수 있음

**Phase 3 gate**: Phase 1 + Phase 2 긍정 결과 후 진행.

---

## 논문 structure (3-model case)

```
Title: "Temporal Rate Separation in Latent World Models Improves Sample Efficiency"

1. Introduction
   - World models: RSSM (DreamerV3, MWM) + Transformer (STORM)
   - Slow/fast temporal processing in biology → apply to world models

2. Method: Dual-Rate World Model Dynamics
   - Slow GRU (every K steps) + Fast GRU (every step)
   - Temporal compression regularization → better sample efficiency
   - IB bottleneck + smoothness loss for collapse prevention

3. Experiments
   - DreamerV3 on DMControl (walker_walk, cheetah_run, hopper_hop)
   - MWM on Meta-World MT10
   - STORM on Atari 100k (Pong, Breakout, Asterix, ...)

4. Analysis
   - K sweep (efficiency-quality Pareto)
   - Ablation (±IB, ±smoothness)
   - Separation score trajectory
```

---

## 즉시 할 일 (GPU 해제 전)

1. [x] plan_006.md 작성
2. [x] MWM 코드베이스 조사: TF 기반 확인, RSSM 동일 구조 확인, framework mismatch 발견
3. [x] STORM 코드베이스 조사: PyTorch ✅, Transformer carry-forward 구현 위치 파악
4. [x] Launch script 작성: `scripts/launch_plan001_all.sh` (13개 run)
5. [x] GPU Monitor 시작 (LLaMA 파인튜닝 종료 감시)
6. [ ] GPU 해제 감지 → `bash scripts/launch_plan001_all.sh` 실행

## GPU 해제 후 할 일

7. [ ] Plan 001 runs 재가동 (11개 + 2개 size-matched = 13개 total)
8. [ ] Phase 1 결과 판정 (multi-seed, size-matched)
9. [ ] Phase 1 PASS → MWM dual-rate TF 포팅 구현 (baselines/MWM/mwm/common/nets.py RSSM 수정)
10. [ ] Phase 2 설계 → STORM dual-rate Transformer 구현

---

## STORM 아키텍처 (코드 확인)

`baselines/STORM/sub_models/world_models.py`:

```python
def encode_obs(obs):            # CNN → categorical tokens (posterior)
def calc_last_dist_feat(latent, action):  # full Transformer → prior logits
def predict_next(last_flattened_sample, action):  # Transformer KV-cache → next token
```

**Dual-rate STORM**: imagination loop에서 `storm_transformer.forward_with_kv_cache(latent, action)`을 K step마다만 실행, 나머지 K-1 step은 dist_feat carry-forward.

**Framework**: PyTorch ✅  **Benchmark**: Atari 100k

## MWM 아키텍처 (코드 확인)

`baselines/MWM/mwm/common/nets.py`:
- `class RSSM` (DreamerV3와 동일한 obs_step/img_step/GRUCell 구조)
- **Framework**: TensorFlow ❌ — PyTorch dual-rate 직접 포팅 불가
- **결정**: **(A) dual-rate를 TF로 포팅** — 공식 MWM 코드베이스, Meta-World 실험 (2026-06-20 사용자 확정)

---

## ⛔ 종료 (2026-06-21 KST): Phase 1 FAIL → dual-rate 아이디어 死

experiment_002 + experiment_003 결과로 dual-rate 종료. MWM/STORM 확장 무의미.

- **exp002 (Phase 1)**: size-matched baseline 추가하니 dual-rate(256/128) 375 vs deter384 755 = 49.6%. 원래 +12.4%는 confound(deter 384 vs 512)였음.
- **exp003 (ratio 재튜닝 kill-test)**: 최선 config(128/256, K=3) = 634 vs bar 755 = 83.9%. 최고 seed(685) < deter384 최저 seed(729), robust FAIL.
- **3축 모두 음성**: wall-clock 0.926×(느림) + quality 열등 + sample-eff 열등. 살아남은 leg 0.
- frozen mass 진단은 정확(375→634 회복)했으나 2-GRU 분할 자체가 same-capacity 단일 GRU에 열등.

→ MWM TF 포팅(옵션 A) 보류, STORM 보류. 방향 재설정 사용자 결정 대기.

## 변경 이력

- 2026-06-20 KST: 초기 작성. Dual-rate 코드 확인 완료 (slow GRU skip, NOT encoder skip). Confound 발견 (deter=384 vs 512). GPU 점유 (LLaMA 파인튜닝) 확인.
- 2026-06-20 KST: MWM=TF 확인, STORM=PyTorch 확인. STORM Transformer carry-forward 방식 파악. Launch script 작성. GPU Monitor 가동.
- 2026-06-21 KST: Phase 1 FAIL 확정 (exp002+003). dual-rate 종료. MWM/STORM 확장 중단.
