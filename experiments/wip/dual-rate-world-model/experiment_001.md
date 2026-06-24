# Experiment 001 — Dual-Rate RSSM M0 Week-1

## 메타데이터
- 날짜: 2026-06-11 KST
- 단계: M0 (1-week mini-experiment)
- 관련 아이디어 slug: dual-rate-world-model
- 모델: DreamerV3 (NM512/dreamerv3-torch, PyTorch)
- GPU: B200 (80GB)
- CUDA: 13.0 / PyTorch 2.9.1

---

## 검증할 가설

DreamerV3의 RSSM을 slow (256-dim, K=3 step마다 갱신) + fast (128-dim, 매 step 갱신)로 분리하면:

1. **FLOPs 감소**: imagination rollout당 분석적 FLOPs가 1.7x 이상 감소한다
2. **Separation**: fast latent가 slow latent보다 빠르게 변한다 (separation_score > 0.5)
3. **Wall-clock speedup**: imagination rollout이 baseline 대비 빠르다 (Week-1 bar: > 1.0x)
4. **Quality preservation**: episode return 하락 < 15%

---

## 실험 설계

### Day-0: Compute Breakdown Profiling
- vanilla DreamerV3 GRU vs. dynamics 비율 측정
- 스크립트: `profile_breakdown.py`
- 출력: `breakdown.json`

### Day-1~3: DualRateRSSM 구현
- `networks.py`에 `DualRateRSSM` 클래스 추가
- `models.py` `WorldModel._train`에 collapse loss + separation logging 추가
- `configs.yaml`에 `dual_rate` config 블록 추가

핵심 구현 포인트:
- `img_step`: fast GRU 매 step, slow GRU는 `step_idx % K == 0`에만 실행
- scalar branch: `(step_flat == step_flat[0]).all()` → Python if로 slow GRU 완전 skip
- collapse 방지: IB penalty (fast→slow L2 norm) + smoothness penalty (slow temporal L2)
- `kl_loss`: (B,T) shape 반환 (RSSM 인터페이스 호환)
- `ImagBehavior.feat_size`: `world_model.dynamics._deter` 사용 (384 not 512)

### Day-3~5: 100k 학습 실험
- 환경: DMControl Walker-walk
- baseline: `defaults + dmc_vision` → deter=512
- dual-rate: `defaults + dmc_vision + dual_rate` → deter=384 (256+128), K=3
- 로그 위치: `/data/jameskimh/worldmodel/dual-rate-world-model/`
- GPU: GPU 0 (두 실험 순차, 각 ~3시간)

### 측정 지표
| 지표 | 측정 방법 | Week-1 기준 |
|---|---|---|
| FLOPs 감소 | 분석적 계산 (matmul count) | > 1.5x |
| Wall-clock speedup | CUDA-synced timer, batch×horizon / latency | > 1.0x (절대값은 Week-3+ 타깃) |
| Separation score | fast_delta / (slow_delta + fast_delta) | > 0.5 |
| Episode return | Walker-walk normalized return | > 0.85 × baseline |

---

## 예상 결과
- FLOPs: K=3, 256+128 split → 이론 ceiling 1.78x 감소
- Wall-clock: 소형 batch에서는 kernel launch overhead가 문제, 대형 batch에서 이점 실현
- Separation: 랜덤 초기화에서도 fast-slow 구조로 인해 > 0.5 예상

---

## 위험 요소
1. slow GRU skip 구현 오류 → 실제로는 mask만 적용, 연산 수행 (발생 및 수정됨)
2. checkpoint loading prefix mismatch (`_orig_mod.`) → 발생 및 수정됨
3. batch=512에서 kernel overhead가 FLOPs 이득 상회 → 발생, 대형 batch로 대응
4. DualRateRSSM collapse → separation_score로 모니터링

---

## 파일 목록
- `profile_breakdown.py`: Day-0 compute profiling
- `breakdown.json`: profiling 결과
- `dreamerv3-torch/networks.py`: DualRateRSSM 구현
- `dreamerv3-torch/models.py`: WorldModel, ImagBehavior 수정
- `dreamerv3-torch/configs.yaml`: dual_rate config 블록
- `measure_rollout.py`: throughput + separation 측정
- `results.json`: 실험 결과
