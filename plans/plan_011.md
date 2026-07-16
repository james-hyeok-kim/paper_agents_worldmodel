# Plan 011: 10M tracker seed 확장 (n=2 → n=5) + 후속 실험 리뷰

## 목표
R8 실험 2(tracker quality ablation)의 10M(완전학습) 지점을 n=2 → n=5로 확장해서,
3M vs 10M 차이가 통계적으로 유의한지("10M에서 다시 튄다"가 진짜인지, 노이즈인지) 확인한다.
동시에 "이 연구를 논문으로 완성하려면 추가로 뭐가 더 필요한지" 리뷰한다.

## 배경
- 실험 2의 최종(n=5) 결과에서 500k/1M/3M 세 구간은 서로 통계적으로 구분 안 됨(ANOVA p=0.62)이
  확인됐으나, 10M(9.536)은 여전히 실험 1(strawman)에서 가져온 n=2 값이라 3M(8.155)과의
  차이를 통계 검정할 방법이 없었음.
- 사용자가 "10M도 seed 3~4개 더 돌려서 확인해달라"고 명시적으로 요청 (2026-07-08 KST).

## 테스트 전략 (CLAUDE.md 요구사항)
- **검증할 것**: 3M(n=5, mean=8.155) vs 10M(n=5로 확장 후) Welch t-test
- **방법**: 기존 500k/1M/3M seed 확장과 동일한 인프라(`run_hl_resumable.sh` 자동재개) 재사용.
  10M tracker(`tracking.pt`)를 `tracker_checkpoints/tracker_10000000.pt`로 symlink해서
  기존 스크립트가 그대로 동작하게 함. seed 3,4,5 실행 (기존 seed 1,2와 합쳐 n=5).
- **성공 기준**: p<0.05 → "10M에서 진짜로 다시 좋아진다"(3단계 구조로 재해석 필요).
  p≥0.05 → 기존 "500k부터 포화" 결론이 10M까지 확정.
- **병렬화**: GPU 0, 1 둘 다 유휴 상태 확인 후, seed=3(GPU0)+seed=4(GPU1) 동시 실행 →
  seed=5(GPU0) 순차 실행. 순수 순차 대비 절반 가까이 단축(예상 ~14시간 vs ~20시간).
- **엣지 케이스**: cgroup 메모리 한도(과거 2회 OOM 이력) → mem/gpu watchdog 재가동.
  경로가 예전 `experiments/wip/puppeteer-acquisition-curve/`에서 새 구조
  `puppeteer-acquisition-curve/results/`로 바뀐 걸 스크립트에 반영(EXPDIR 수정) 완료.
- **실패 시**: `run_hl_resumable.sh`가 크래시 시 최신 checkpoint에서 최대 20회 자동 재시도.

## 실행 파일
- `src/run_10m_seed_expansion.sh` (신규) — seed 3,4,5 큐, GPU0/1 병렬
- `src/run_hl_resumable.sh` (EXPDIR을 `results/`로 수정)
- `src/mem_watchdog.sh`, `src/gpu_mem_watchdog.sh` (EXPDIR 수정, gpu watchdog는 GPU0+1 둘 다 감시하도록 확장)
- 심볼릭 링크: `/data/jameskimh/worldmodel/tracker_checkpoints/tracker_10000000.pt` →
  `baselines/puppeteer/checkpoints/tracking.pt`

## 시작 시각
2026-07-08 07:56 KST — seed=3(GPU0), seed=4(GPU1) 동시 시작

## 예상 소요 시간
런 1개당 약 6~7시간(3M seed=5 실측 기준). seed3+4 병렬(~7시간) 후 seed5(~7시간) = 총 ~14시간
예상. 완료 예정: 2026-07-08 22시경 KST (변동 가능, 재시도 발생 시 늘어남).

## 위험 요소
- 2개 프로세스가 동시에 도는 건 이번이 처음(기존 큐는 항상 GPU1 하나로 순차 실행) — GPU/CPU
  자원 경합으로 개별 run이 더 느려질 가능성. GPU 사용률 낮음(35%/24%)으로 병목은 CPU/env
  시뮬레이션 쪽일 가능성 높음 — 모니터링하면서 필요시 순차 실행으로 전환 고려.
- 동시 2-run이 cgroup 메모리를 더 빨리 소진할 수 있음 → watchdog으로 감시.

## 완료 후 할 일
1. 10M seed 1~5 AUC_100k 재계산 (seed 1,2는 condA_s1/condA_s2 eval.csv 재사용,
   seed 3,4,5는 신규 ablation_10000000_s{3,4,5} eval.csv)
2. 3M vs 10M Welch t-test 실행, `experiment_summary.md` / `paper_idea.md` 갱신
3. "논문 완성까지 추가로 필요한 실험" 리뷰 (완료, 아래 추가 섹션 참조)

---

## 추가: 리뷰 결과 반영 — seed 확장(n=5→11) + task 일반화 검증 (2026-07-08 08:07 KST 시작)

### 배경
10M 학습 진행 중 리뷰를 수행한 결과, 두 가지 중요한 gap 발견:
1. **통계적 검정력 부족**: 500k vs 3M(Cohen's d=1.18, "큰 효과") 비교의 n=5 기준 검정력이
   시뮬레이션 결과 약 35%에 불과함. 80% 검정력에는 그룹당 약 11 seed 필요.
2. **task 일반화 미검증**: 지금까지 모든 실험이 `corridor` task 하나에만 의존. 코드 확인 결과
   `gaps-corridor`/`walls-corridor`/`stairs-corridor`/`hurdles-corridor`가 이미 구현되어
   있어(`envs/transfer.py`) 추가 구현 없이 바로 검증 가능함이 확인됨.

사용자 확인 후 진행 범위 확정: (1) 500k/1M/3M을 seed 6~11까지 확장(n=5→11, 18 run),
(2) gaps-corridor + walls-corridor에서 0-step/500k 조건 × seed 1~3 (12 run) — 총 30 run.

### 자원 여유 확인
- cgroup CPU quota: 16 코어. 학습 1개당 실측 CPU 사용량 ~1.3 코어 → 이론상 최대 동시 ~12개.
- GPU 메모리: 183GB 중 학습 1개당 ~4GB만 사용 (여유 큼, GPU utilization도 4~35%로 낮아
  병목이 GPU가 아니라 CPU/env 시뮬레이션 쪽임을 확인).
- cgroup 메모리: 440GB 한도 중 현재 ~28GB만 사용.
- → 이미 도는 10M 큐(2개 동시) + 신규 6개 동시 = 총 8개 동시 실행으로 설정
  (8×1.3≈10.4코어, 16코어 한도 내 안전 마진 확보).

### 실행 파일 (신규)
- `src/run_generic_resumable.sh` — task/tracker_path/exp_name을 인자로 받는 범용 버전
  (기존 `run_hl_resumable.sh`는 corridor+tracker_step 고정 형식이라 gaps/walls task와
  random_tracker.pt(0-step)를 표현 못해서 새로 작성)
- `src/run_expansion_and_generalization.sh` — 30-job 큐, `wait -n` 기반 동시성 6-cap
  디스패처, GPU0/1 round-robin 배정

### 시작 시각
2026-07-08 08:07 KST, 30 jobs, 최대 6개 동시 실행 (10M 큐 2개와 합쳐 총 8개 동시)

### 예상 소요 시간
런 1개당 ~6~7시간. 30 jobs / 6 동시 ≈ 5 wave × ~6~7시간 ≈ 30~35시간. 10M 큐(3개 남음,
~14시간)와 리소스를 공유하며 병행 진행. 전체 완료 예상: 2026-07-09 낮~저녁 KST 경
(재시도 발생 시 늘어날 수 있음).

### 완료 후 할 일
1. 500k/1M/3M n=11 재계산 + ANOVA/pairwise t-test 재실행, 검정력 재평가
2. gaps-corridor/walls-corridor 0-step vs 500k AUC_100k 비교 — corridor에서 본 "급전환"이
   재현되는지 확인
3. 모든 결과를 `experiment_summary.md`/`paper_idea.md`에 반영, 통계적 검정력 한계 명시

---

## 완료 결과 (2026-07-10 07:21 KST 큐 완료 → 08:xx KST 분석 완료)

### 실행 요약
- 10M seed 확장(3 run) + 500k/1M/3M seed 6-11 확장(18 run) + gaps/walls 일반화(12 run) =
  **총 33 run 전부 성공**. 크래시/FAILED/OOM 이력 전무 (`mem_watchdog.log`: oom_kill 0 유지).
  10M 큐: seed=4 완료 07-08 17:17, seed=3 완료 07-08 17:51, seed=5 완료 07-09 04:16 KST.
  확장+일반화 30-job 큐: `expansion_generalization_queue.log` 최종 라인 —
  `[2026-07-10 07:21:06 KST] Expansion+generalization queue COMPLETE (30 jobs).`
  전체 파이프라인 wall-clock: 2026-07-08 07:56 KST 시작 → 2026-07-10 07:21 KST 종료 (약 47시간,
  다른 큐와 자원 공유하며 병행 진행).

### 분석 스크립트
`src/analyze_results_final.py` — 기존 `experiment_summary.md`와 동일한 방법론(0~100k을
6개 지점으로 보간 후 trapezoid AUC/100000)으로 재검증. seed1,2 값이 기존 발표치와 3자리까지
일치함을 확인해 방법론 재사용의 정합성을 검증함. 결과는
`results/final_analysis_results.json`에 저장.

### 최종 결과 (n=11 for 500k/1M/3M, n=5 for 10M, n=3 for gaps/walls)

| 조건 | n | mean | std |
|---|---|---|---|
| corridor 0-step | 2 | 2.139 | 0.336 |
| corridor 500k | 11 | 9.362 | 0.814 |
| corridor 1M | 11 | 8.935 | 2.362 |
| corridor 3M | 11 | 7.890 | 0.925 |
| corridor 10M | 5 | 8.411 | 1.287 |
| gaps-corridor 0-step | 3 | 1.811 | 0.024 |
| gaps-corridor 500k | 3 | 9.617 | 1.036 |
| walls-corridor 0-step | 3 | 2.375 | 0.243 |
| walls-corridor 500k | 3 | 8.976 | 1.093 |

### 통계 재검정 (n=5 → n=11로 검정력 확보 후 결론이 바뀐 부분 있음 — 정직하게 기록)
- 등분산 가정 ANOVA(500k/1M/3M): F=2.666, p=0.0859 (유의 아님)
- **Welch's ANOVA(이분산 보정, 1M의 표준편차가 500k/3M보다 훨씬 커서 등분산 가정이 깨짐)**:
  F=7.614, df=(2.0, 18.4), **p=0.0039 (유의함)** — 세 조건 사이 진짜 이질성 존재.
- Pairwise Welch t-test (Bonferroni α=0.05/3=0.0167 기준):
  - 500k vs 1M: p=0.5809, d=0.242 — 유의 아님
  - 1M vs 3M: p=0.1951, d=0.583 — 유의 아님
  - **500k vs 3M: p=0.0008, d=1.689 (큰 효과, Bonferroni 생존)** — n=5에서는 못 봤던 유의한 차이
  - 검정력(n=11, 시뮬레이션): 96.3%
- 3M(n=11) vs 10M(n=5): p=0.4471, d=-0.464 — 유의 아님
- 500k(n=11) vs 10M(n=5): p=0.1833, d=0.884 — 유의 아님이지만 중간 크기 효과 (10M은 n=5라
  검정력 부족, 확정 불가)
- gaps-corridor 500k vs 0-step (n=3): **p=0.0058, d=10.653** (매우 큰 효과)
- walls-corridor 500k vs 0-step (n=3): **p=0.0068, d=8.337** (매우 큰 효과)

### 결론 수정 — "완전 plateau"는 부정확했음
n=5 시점 결론("500k~10M 전 구간 통계적으로 구분 안 됨, 완전 평탄")은 검정력 부족(500k vs 3M
검정력 ~35%)에 의한 것이었음. n=11로 확장하니:
- **0-step → 500k의 급격한 전환은 더 강하게 재확인됨** — 이제 corridor뿐 아니라 gaps-corridor,
  walls-corridor 두 개의 새로운 task에서도 동일 패턴이 극단적으로 큰 효과크기(d=10.7, d=8.3)로
  재현됨. 이게 가장 견고한 핵심 주장.
- 500k~3M 구간은 완전한 평탄이 아니라 **500k에서 3M으로 가면서 통계적으로 유의한 하락**이 있음
  (Welch's ANOVA 유의, 500k vs 3M 유의). 다만 1M은 분산이 매우 커서 500k/3M 어느 쪽과도
  유의하게 구분되지 않고, 10M(n=5)도 3M/500k 어느 쪽과도 유의하게 구분되지 않아 — "500k가 peak
  이고 이후 단조 감소한다"고 확정할 근거는 아직 부족함 (10M 평균이 3M보다 높아 단조 감소와도
  안 맞음). 정직한 서술: "0.5M~10M 구간은 전반적으로 좁은 범위(7.9~9.4)에 포화되어 있고, 그 안에서
  500k와 3M 사이에만 통계적으로 유의한 국소적 차이가 관측됨 — 10M을 포함한 전체 형태를 확정하려면
  10M/1M 쪽 seed를 더 늘려야 함."
- 다음 문서 업데이트(`experiment_summary.md`, `paper_idea.md`)에서 이 수정된 서사를 반영.
