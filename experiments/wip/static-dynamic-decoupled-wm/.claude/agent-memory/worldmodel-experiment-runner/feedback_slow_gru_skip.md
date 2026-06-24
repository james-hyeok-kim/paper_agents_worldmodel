---
name: feedback_slow_gru_skip
description: slow GRU를 mask-blend하면 연산은 여전히 실행됨; scalar branch로 실제 skip 필수
metadata:
  type: feedback
---

carry-forward를 mask로 구현하면 GRU 연산은 항상 실행된다:

```python
# WRONG: GRU always runs, just blends output
new_slow = mask * gru_slow(x) + (1-mask) * prev_slow  # gru_slow() called every step
```

FLOPs 절감을 위해서는 조건에 따라 GRU call 자체를 skip해야 한다:

```python
# CORRECT: actually skip the GRU call
if step_idx % K == 0:
    new_slow = gru_slow(ib_proj(fast), prev_slow)
else:
    new_slow = prev_slow  # zero compute
```

**Why:** dual-rate-world-model에서 masking 구현으로 오히려 baseline보다 FLOPs가 더 많았음 (fast GRU + slow GRU 항상 실행). scalar branch 수정 후에도 Python call overhead 문제 있었지만 FLOPs 자체는 올바르게 감소.

**How to apply:** "carry-forward" 패턴이 필요한 곳에서는 반드시 실제 computation skip을 구현. imagination rollout 시 step_idx는 batch 전체가 uniform하므로 scalar Python if-else로 safely skip 가능. obs_step (non-uniform step_idx)에서는 mask 필요하지만 그것은 training이므로 FLOPs 절감보다 gradient correctness가 중요.
