# Experiments Index

World Model Efficiency & Quality 논문 실험 목록.

## 실험 상태별 분류

### 진행 중 (wip/)
<!-- 새 실험 시작 시 추가 -->
<!-- | slug | 단계 | 시작일 | 예상 완료 | GPU-hours | verdict |
|---|---|---|---|---|---| -->

### 완료
<!-- 실험 완료 시 이동 -->

### 중단
<!-- 중단된 실험 이동 -->

---

## 슬롯 가이드

```
experiments/
├── wip/<slug>/          # 진행 중 실험
│   ├── poc.py           # PoC 스크립트 (validator 산출물)
│   ├── run_experiment.py
│   ├── results.json
│   ├── run.log
│   └── README.md
└── INDEX.md             # 이 파일
```

**대용량 파일(체크포인트, 캐시)은 `/data/jameskimh/worldmodel/<slug>/`에 저장.**
