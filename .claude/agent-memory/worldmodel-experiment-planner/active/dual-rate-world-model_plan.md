---
name: dual-rate-world-model-plan
description: Dual-rate RSSM (slow K-step/fast every-step) 실험 계획 — imagination rollout 효율 + collapse 방지 레시피
metadata:
  type: project
---

# Dual-Rate World Model 실험 계획 (active)

계획 파일: `experiments/wip/dual-rate-world-model/experiment_plan.md`

**Core claim**: dual-rate RSSM이 collapse 방지 레시피(fast→slow IB + slow smoothness) 하에서 matched 100k budget return drop <5%로 imagination throughput 1.3~1.6× 가속.

**상태**: active, 2026-06-11 KST 작성. PoC=CONDITIONAL-GO(1.93× synthetic), literature=INCREMENTAL.

**핵심 설계 결정 (검증된 제약)**:
- literature verdict가 INCREMENTAL → 두 차별점만 novelty: (1) imagination rollout FLOPs를 1급 목표로, (2) stochastic collapse 방지 레시피. "action window-aggregate→slow"는 MTS3에 선점됨, novelty로 쓰지 말 것.
- Fork 타깃: `NM512/dreamerv3-torch` (JAX 없음 → 공식 DreamerV3 불가). 변경 지점: `RSSM.img_step()`의 `self._cell`(GRUCell)을 slow/fast 분할. `_imagine()`은 encoder/decoder 없이 순수 latent → imagination speedup 격리 가능.
- 환경: torch+CUDA 2×B200(~40GB free 각, 공유), mujoco 3.6 설치됨, dm_control/ale_py/jax 미설치. dm_control은 Day-1 one-line install.
- speedup은 imagination throughput이 primary(1.3~1.6× 정직 타깃), total training wall-clock은 encoder/decoder 미변경으로 희석됨 → secondary.
- **Day-0 선행 gate (split 구현 전 필수)**: vanilla DreamerV3 compute breakdown 프로파일링(imagination 내 dynamics-GRU vs encoder/decoder/heads/MLP 비율) → realized imagination-speedup ceiling 확정 + headline claim 고정. PoC 1.93×는 dynamics-only 가정이라 end-to-end와 다름. 산출물 `breakdown.json`. (NM512 RSSM API는 WebFetch로 실존 검증됨: img_step/obs_step/GRUCell/deter·hidden config.)
- collapse 방지 ablation(−IB/−sm/−both, separation_score 붕괴)은 Week3 1급 실험 + Week1에 minimal 포함 (novelty 증거).

**Compute**: ~432 GPU-hours (DMC 6 + Atari 4 task, 4 baseline, 3 seed). 타이트 시 ~290으로 축소.

[[관련: experiments/wip/dual-rate-world-model/poc.py]]
