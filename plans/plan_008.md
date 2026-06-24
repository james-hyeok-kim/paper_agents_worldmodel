# Plan 008 — Non-efficiency lever 탐색 (R13~), PoC 통과까지 반복

## 목표
efficiency framing 종결(R12b, plan_007) 후 사용자 선택: non-efficiency lever로 전환.
3방향 전부 열어두고 generator가 가장 강한 아이디어를 제안 → novelty → PoC 통과까지 반복.

## 배경
- R12+R12b로 efficiency framing 코드레벨 완전 종결 (plan_007).
- 남은 lever: (b1) quality 향상 / (b2) sample efficiency / (b3) new problem-setting.
- 세 방향 전부 열고 가장 강한 아이디어를 산출하도록 지시.

## 탐색 방향

| ID | 방향 | 설명 |
|---|---|---|
| b1 | Quality | WM 예측 정확도 / representation 품질 향상 → downstream return 개선 |
| b2 | Sample efficiency | 같은 asymptotic 성능을 더 적은 env step으로 달성 |
| b3 | New problem-setting | non-stationarity, multi-task, offline RL 통합 등 human-defined 새 framing |

## 워크플로우 (iteration당)
1. **생성**: worldmodel-idea-generator. BLACKLIST(BL-01~18) 회피. 3방향 전부 탐색 후 가장 강한 아이디어 1개 산출.
2. **novelty (인터넷 필수)**: team-lead가 WebSearch로 arXiv/Semantic Scholar/named-paper 충돌 확인. NO-GO면 다음.
3. **PoC 게이트** (방향별):
   - b1 quality: quality_proxy_delta > +0.05 vs size-matched baseline
   - b2 sample-eff: 동일 return milestone 도달 step 수 ≤ 70%
   - b3 new problem-setting: 아이디어 나온 뒤 게이트 사전등록 후 측정
4. **2nd metric**: quality 방향이면 sample-eff 역전 없는지도 확인
5. **통과** → PushNotification 알람 + 기록. 실패 → BLACKLIST + 다음 iteration.

## PoC 게이트 (success criteria)
- b1: quality_proxy_delta > +0.05 (oracle/agnostic 통제, BL-14 교훈 적용)
- b2: step_to_milestone ≤ 0.70×baseline
- b3: 아이디어별 사전등록
- wall-clock speedup은 요구하지 않음 (efficiency 게이트 해제)

## Edge cases / 실패 모드
- vacuous quality pass: oracle/state-agnostic baseline으로 통제 (BL-14 교훈)
- n=2 lucky-baseline trap: 핵심 결과는 n≥5로 확인 (dual-rate 교훈)
- new problem-setting은 게이트 미정 → 아이디어 먼저 보고 후 게이트 공동 결정

## 진행 로그
- 2026-06-22 KST: 사용자가 non-efficiency lever 선택. plan 작성, R13 시작.
