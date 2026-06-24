# Plan 009: Partial Tracker Quality Ablation (Option A)

## 목표
Tracker 훈련 step 수(quality level)와 high-level corridor 정책 획득 속도(AUC_100k) 간의 관계를 실증적으로 측정한다.
- 핵심 질문: threshold가 있는가? (비선형) vs 선형 비례?
- 가설: 특정 tracker quality 미만에서는 high-level 학습이 거의 불가능하고, threshold 이상에서 급격히 가능해진다

## 실험 설계

### Tracker Quality Levels
| Level | Tracker 훈련 steps | checkpoint 파일 |
|-------|-------------------|----------------|
| 0 | 0 (random) | random_tracker.pt (기존 완료) |
| 1 | 100k | tracker_100k.pt |
| 2 | 500k | tracker_500k.pt |
| 3 | 1M | tracker_1m.pt |
| 4 | 3M | tracker_3m.pt |
| 5 | 10M | tracking.pt (기존 완료) |

### High-level 훈련 설정
- Task: corridor (RunThroughCorridor, target_velocity=6.0)
- Steps: 200,000 per run
- Seeds: 1, 2 (2 seeds each)
- Eval: every 20,000 steps
- Metric: AUC_100k (np.trapz over [0,20k,40k,60k,80k,100k])

### Success Criteria
- PASS (비자명): AUC_100k가 tracker quality에 비선형 응답 (threshold 발견)
- FAIL (자명): AUC_100k가 tracker steps에 선형 비례

## 단계별 작업

### Phase 1: 환경 세팅 (예상 시간: 30분)
- [x] plan_009.md 작성
- [ ] MoCapAct 패키지 설치 (GitHub clone → pip install -e .)
- [ ] MoCap data 다운로드 → /data/jameskimh/mocapact/ (~50GB, background)

### Phase 2: Tracker 훈련 (예상 시간: ~75h)
- [ ] tracker 훈련 스크립트 작성 (steps=3M, save_freq=100k)
- [ ] 훈련 시작 (GPU 0 또는 1)
- [ ] 체크포인트 100k, 500k, 1M, 3M 저장 확인

### Phase 3: High-level 훈련 (예상 시간: ~48h, tracker 완료 후)
- [ ] tracker_100k: corridor 200k, seed=1,2 (GPU 0,1)
- [ ] tracker_500k: corridor 200k, seed=1,2 (GPU 0,1)
- [ ] tracker_1m: corridor 200k, seed=1,2 (GPU 0,1)
- [ ] tracker_3m: corridor 200k, seed=1,2 (GPU 0,1)

### Phase 4: 분석
- [ ] AUC_100k 계산 (모든 6 quality levels)
- [ ] Quality-AUC curve 플롯
- [ ] Threshold 위치 및 메커니즘 분석
- [ ] result_002.md 작성

## 파일 경로
- MoCap 데이터: /data/jameskimh/mocapact/
- Tracker 체크포인트: /data/jameskimh/worldmodel/tracker_checkpoints/
- 실험 결과: experiments/wip/puppeteer-acquisition-curve/

## 예상 타임라인
- Phase 1: 2026-06-24 (오늘)
- Phase 2: 2026-06-24 ~ 2026-06-27 (75h)
- Phase 3: 2026-06-27 ~ 2026-06-29 (48h)
- Phase 4: 2026-06-29 (분석)

## 위험 요소
- Tracker 훈련 속도가 high-level보다 다를 수 있음 (offline+online 혼합)
- 100k step tracker가 너무 약해서 corridor에서 즉시 넘어질 수 있음
- 3M step tracker가 10M과 비슷하면 새 레벨 추가 필요

## History
- 2026-06-24: plan 작성, Phase 1 시작
