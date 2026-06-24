---
name: feedback_b200_launch_bound
description: B200 GPU에서 작은 GRU matmul은 kernel launch-bound; FLOPs 이점이 wall-clock으로 직결되지 않음
metadata:
  type: feedback
---

B200 GPU에서 DreamerV3 규모의 GRUCell (hidden=512, deter=256-512)은 kernel launch-bound 영역에 있다. FLOPs 감소가 wall-clock speedup으로 직결되지 않는다.

**Why:** dual-rate-world-model M0에서 FLOPs 1.78x 감소임에도 batch=512에서 wall-clock speedup < 1.0. fused Python loop (K-chunked) 시도도 0.97x — Python overhead가 병목이 아님. 실제 병목은 두 개의 순차적 GRU kernel launch. 공유 GPU 측정에서 baseline sps가 30% 변동.

**How to apply:**
- FLOPs/step을 primary efficiency metric으로 보고한다. Wall-clock은 "also measured" 항목.
- Wall-clock은 반드시 idle GPU에서 단독 측정. 공유 GPU 측정은 std > mean이면 noise.
- K-chunked loop 구현으로 speedup 시도는 dead end (병목이 Python loop가 아님).
- FLOPs 이점이 실용적으로 나타나는 조건: (1) contention-free GPU + (2) FP16/TF32 활성화 시 큰 행렬이 유리한 regime. 학습 시 effective batch가 작으면 unrealized.
