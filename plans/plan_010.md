# Plan 010: puppeteer-acquisition-curve 디렉토리 재구성

## 목표
`experiments/wip/puppeteer-acquisition-curve/`(플랫 구조, 96개 파일)를 워크스페이스 루트의
독립 폴더 `puppeteer-acquisition-curve/{docs,results,src}/`로 재구성한다.

## 배경
- plan_009(R8 tracker quality ablation)의 9-run seed expansion 큐(500k/1M/3M × seed 3,4,5)가
  8/9 완료, 마지막 3M seed=5가 진행 중인 상태(2026-07-08 01:5x KST)에 사용자가 재구성을 요청.
- 이전에(2026-07-06) 한 차례 동일한 재구성을 시도했다가 VS Code에서 새 폴더가 보이지 않는 문제로
  롤백하고 "실험 다 끝나면" 재시도하기로 유예한 바 있음.
- 이번에는 사용자가 큐 완료를 기다리지 않고 지금 진행해달라고 명시적으로 요청.

## 전략: 정적 파일은 지금 복사, 라이브 파일은 완료 후 이동
마지막 3M seed=5 run과 watchdog들이 여전히 구 경로(`experiments/wip/puppeteer-acquisition-curve/`)의
특정 파일에 계속 append하고 있으므로, 그 경로를 지금 옮기면 실행 중인 프로세스가 깨질 위험이 있다.
따라서:

1. **지금 실행**: 완료되어 더 이상 쓰기가 없는 파일(로그 58개 + 스크립트 29개 + 문서 3개 = 90개)을
   새 구조로 **복사**(원본은 그대로 둠). 복사 후 전체 파일 byte-for-byte checksum 비교로 무결성 확인.
2. **지금 보류(원본 위치 유지, 손대지 않음)**: 아래 6개 — 현재 실행 중인 프로세스가 계속 append 중:
   - `ablation_3000000_s5.log`, `ablation_3000000_s5_supervisor.log` (마지막 run, train.py 출력)
   - `mem_watchdog.log`, `gpu_mem_watchdog.log` (watchdog 루프가 각각 30s/60s마다 append)
   - `seed_expansion_queue.log`, `seed_expansion_queue_stdout3.log` (큐 supervisor가 "COMPLETE" 라인을 추가할 예정)
3. **9-run 큐 완료 후**(plan_009 참조, task #1):
   - 위 6개 파일을 새 구조(`results/`)로 이동
   - 구 경로(`experiments/wip/puppeteer-acquisition-curve/`)의 원본 정리(삭제) 여부는
     사용자에게 재확인 후 진행 (CLAUDE.md 파일 삭제 규칙)

## 새 구조
```
/home/jovyan/workspace/paper_agents_worldmodel/puppeteer-acquisition-curve/
├── docs/     # experiment_001.md, result_001.md, result_002.md (실험 설계·결과 서술)
├── src/      # 29개 스크립트 (.sh, .py) — watchdog, supervisor, sanity, 옛 run 스크립트 포함
└── results/  # 58개 완료된 로그·체크포인트(random_tracker.pt)·pid 파일
```

참고: 저장소 루트 `docs/idea_status.md`에 R8 아이디어의 상위 서술이 있으나, 이는 여러 아이디어를
포괄하는 공용 로그이므로 이번 재구성에서는 이동하지 않고 그대로 둔다.

## 검증
- 파일 카운트: DOCS(3) + SRC(29) + RESULTS(58) + LIVE(6) = 96 == 원본 전체 파일 수(96) ✅
- 복사된 90개 파일 전부 `cmp` byte-for-byte 일치 확인 ✅
- 원본은 삭제하지 않음 — 마지막 run이 끝날 때까지 구 경로도 그대로 유지되어 이중 안전판 역할

## 예상 시간
파일 복사(즉시) + 마지막 3M seed=5 완료 대기(수 시간, plan_009 진행 상황에 따름) + 최종 이동/정리(수 분)

## 위험 요소
- 라이브 파일을 성급히 이동/삭제하면 진행 중인 학습이 로그를 잃거나(watchdog 실패) 최악의 경우
  supervisor의 체크포인트 탐색 경로가 깨질 수 있음 → 완료 후로 명확히 분리하여 회피
