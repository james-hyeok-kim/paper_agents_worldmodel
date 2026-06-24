---
name: feedback_gpu_contention
description: 공유 GPU(다른 학습 동시 실행)에서 wall-clock 측정 시 ratio가 noise가 됨
metadata:
  type: feedback
---

공유 GPU에서 두 모델의 throughput을 비교 측정할 때, 다른 프로세스가 GPU를 사용 중이면 절대값이 크게 변동해 비율도 신뢰할 수 없다.

**Why:** dual-rate-world-model M0에서 학습 프로세스(143GB 사용)가 동일 GPU 1에서 실행 중인 상태로 측정. baseline sps가 175k → 128k (28% 하락) 변동. 이 상태에서 나온 1.32x, 0.73x 등의 비율은 모두 noise.

**How to apply:**
- wall-clock timing 전에 `nvidia-smi`로 GPU 사용률과 다른 프로세스 확인. 다른 학습이 실행 중이면 결과를 신뢰하지 않는다.
- FLOPs (분석적) 또는 idle-GPU 전용 측정만 publication-grade 수치로 보고.
- std_latency > mean_latency * 0.2 이면 timing distribution이 outlier-dominated → 더 많은 trial 또는 idle GPU 필요.
- 학습 중 throughput 측정이 필요하면, 별도 GPU (CUDA_VISIBLE_DEVICES)를 할당해 경합 방지.
