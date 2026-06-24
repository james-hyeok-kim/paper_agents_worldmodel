# 실험 001 — Dual-Rate World Model (M0 Week-1)

## 메타데이터
- 날짜: 2026-06-11 KST
- 단계: M0 Week-1
- 상태: PARTIAL (FLOPs + training-separation 확인, episode return 대기)
- 판정: PARTIAL (return 도착 후 WEAK GO 예상)
- 관련 아이디어 slug: dual-rate-world-model
- 모델: DreamerV3 (NM512/dreamerv3-torch, PyTorch)
- Benchmark: DMControl Walker-walk 100k steps + analytical FLOPs
- 연결된 실험: result_001.md (상세 결과)

---

## 검증한 가설

DreamerV3 RSSM을 **slow** (256-dim, K=3 step마다 갱신) + **fast** (128-dim, 매 step 갱신)로 분리하면:
- Imagination rollout FLOPs 1.5x+ 감소
- fast latent가 slow latent보다 빠르게 변화 (separation_score > 0.5)
- 에피소드 리턴 baseline 대비 -15% 이내

---

## 방법

### DualRateRSSM 핵심 구조

```
img_step(state, action):
  # Fast: 매 step 갱신
  x_fast = fast_inp([stoch, action, deter_slow])
  deter_fast = GRU_fast(x_fast, prev_deter_fast)

  # Slow: step_idx % K == 0 일 때만 실제 갱신
  if step_idx % K == 0:
    deter_slow = GRU_slow(slow_inp(ib_proj(deter_fast)), prev_deter_slow)
  else:
    deter_slow = prev_deter_slow  # carry-forward, zero compute

  deter = cat([deter_slow, deter_fast])  # 384-dim
  stoch = sample_stoch(img_out(deter))
```

**Collapse 방지**: IB penalty (fast→slow bottleneck) + slow smoothness penalty

---

## 핵심 결과

### FLOPs 감소 (분석, K=3): **1.783x**

| 모델 | FLOPs/step | 비율 |
|---|---|---|
| Baseline RSSM (deter=512) | 4,724,736 | 1.0x |
| DualRateRSSM K=3 (amortized) | 2,649,429 | **1.783x 감소** |

FLOPs는 matmul count 기준 분석값. Wall-clock은 공유 GPU 경합으로 측정 불신뢰 (별도 deferred).

### Training-time Separation (step 5000): **0.717**

| 지표 | 학습 중 (step 5k) | 기준 | 판정 |
|---|---|---|---|
| separation_score | **0.717** | > 0.5 | PASS |
| fast/slow delta ratio | **2.7x** | > 1.0 | PASS |
| collapse_flag | False | — | PASS |
| ib_penalty | 0.000497 | 비폭발적 | PASS |

**계획의 중심 위험 해소**: real training gradient 하에서 fast/slow 분리가 유지됨.

### 모델 파라미터
| 모델 | 파라미터 수 | 감소율 |
|---|---|---|
| Baseline RSSM (deter=512) | 15.69M | — |
| DualRateRSSM (deter=384) | 14.38M | **-8.4%** |

---

## 중요 발견

1. **FLOPs 이점은 분석적으로 확정 (1.78x)**: matmul count 기준. Wall-clock은 공유 GPU 경합으로 noise; idle GPU 재측정 필요.

2. **Separation은 학습 중에도 유지**: random init 0.756뿐 아니라 학습 step 5000에서도 0.717. ib_penalty/sm_penalty 모두 안정적.

3. **fused loop는 speedup 없음**: Python loop overhead 제거해도 wall-clock 동일. 실제 병목은 두 개의 순차 GRU kernel launch (소형 행렬 launch-bound 영역). K-chunked loop는 dead end.

4. **mask ≠ skip**: slow GRU를 mask로 blend하면 연산은 항상 실행. `if step_idx % K == 0:` scalar branch로 진짜 skip 필요.

---

## 한계 / 주의사항

- **Wall-clock speedup 미확정**: 모든 측정이 경합 GPU에서 수행됨. FLOPs가 primary metric.
- **Episode return 미측정**: 학습 100k steps 진행 중 (~10% 완료).
- **checkpoint loading 미검증**: `latest.pt` 저장 전이라 `_orig_mod.` prefix 처리 실제 확인 필요.

---

## 방향성 (Week-2+)
1. 학습 완료 후 episode return 비교 → WEAK GO 확정
2. K sweep (K=2,4,6): FLOPs-quality trade-off
3. Idle GPU에서 wall-clock 재측정
4. 성공 시 1M steps 풀 트레이닝

---

## 파일 목록
| 파일 | 설명 |
|---|---|
| `experiment_001.md` | 실험 설계 문서 |
| `result_001.md` | 상세 결과 (이 파일의 장문 버전) |
| `results.json` | 머신-readable 결과 |
| `breakdown.json` | Day-0 compute profiling |
| `rollout_chunked_results.json` | fused loop 비교 |
| `profile_breakdown.py` | compute breakdown 스크립트 |
| `measure_rollout.py` | throughput + separation 측정 |
| `measure_rollout_chunked.py` | fused loop 측정 |
| `dreamerv3-torch/` | 수정된 DreamerV3 구현 |
| `experiment_plan.md` | M0 실험 계획서 |
