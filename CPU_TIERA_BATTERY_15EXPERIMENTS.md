# CPU TIER-A BATTERY — 전선별 15개 실험 검증 결과 (2026-07-04)

검증 환경: train.jsonl 70,000 전체, 조건부 분포 분석 (학습 없음)
목적: 0.8 미만 3개 전선(inspect/communicate/execute)에 대해 각 5개 가설을
직접 검증하고, serialize v2.2 번들과 결정 단계에 반영할 카드를 확정한다.
판정 기준: 타깃 클래스 확률이 기저 대비 상대 30%+ 변동 = 통과.
"관측가능" 표기 = 신호는 실존하나 재료가 이미 직렬화에 노출됨 (INTENT의
교훈: 모델이 이미 보는 재료의 태그화는 한계효용이 작다).

---

## 전선 1. INSPECT (기저: grep 0.34 / read 0.32 / glob 0.18 / list 0.15)

| ID | 가설 | 핵심 결과 | 판정 |
|---|---|---|---|
| I-1 | 프롬프트 경로 깊이(슬래시 수)가 list/glob을 가른다 | slash0/1/2+ 전부 기저와 동일 (purity 0.344) | ❌ 기각 |
| I-2 | 직전 list/glob의 **파일 수**로 분기한다 | list가 1-3개 반환 → read 0.43 / glob이 16+ 반환 → glob 재시도 0.31 (기저 0.18) | ⭕ 약통과 |
| I-3 | inspect **연속 길이(streak)**가 길수록 전략이 바뀐다 | streak 0 → list 0.22 / streak 1 → read·grep 각 0.40 / **streak 3-4 → glob 0.37~0.38 (기저 2배)** — 탐색 장기화 시 glob 에스컬레이션 | ✅ 통과 |
| I-4 | **open_files 개수**가 탐색 방식을 정한다 | open 0개 → list 0.26·read 0.35 / open 1개 → grep 0.43 / **open 2+ → grep 0.51**. 단일 조건 purity 0.396 (last_action 조건과 동급) | ✅ 통과 (반관측가능) |
| I-5 | 백틱/따옴표 코드 심볼 → grep | 해당 샘플 274개뿐, 이동 미미 | ❌ 기각 |

전선 1 해석: 프롬프트 표면(I-1, I-5)은 재차 사망, **세션 상태(I-3 streak,
I-4 open 개수, I-2 결과 수치)**가 살아있다. inspect의 남은 신호는 전부
"에이전트가 어디까지 뒤졌는가"라는 탐색 진행도에 있다.

## 전선 2. COMMUNICATE (기저: respond 0.44 / ask 0.23 / plan 0.23 / web 0.11)

| ID | 가설 | 핵심 결과 | 판정 |
|---|---|---|---|
| C-1 | 콜드스타트(t1)는 계획으로 시작한다 | **t1 → plan_task 0.58 (기저 2.5배)**, ask 0.32. respond는 t1에서 희귀 | ✅ 통과 (관측가능) |
| C-2 | communicate가 연쇄된다 (대화 체인) | 직전도 comm → plan 0.40 / **web 0.20 (기저 1.8배)**. 직전이 작업 action → respond 0.56 | ✅ 통과 (관측가능) |
| C-3 | 프롬프트 길이가 유형을 가른다 | **<40자 → respond_only 0.92 (준결정적)** / 100자+ → plan 0.40·ask 0.38·web 0.17 | ⭕ 약통과* |
| C-4 | 다중 요구 나열("그리고/먼저") → plan | multi일수록 오히려 respond 상승 — 가설 반대 방향 | ❌ 기각 |
| C-5 | CI failed가 대화 유형을 바꾼다 | **failed 시 respond 0.55→0.24 붕괴**, ask 0.32·plan 0.30으로 이동 — "문제 있는 상황에선 답만 하지 않는다" | ✅ 통과 (관측가능) |

*C-3 주석: short→respond 0.92는 대어처럼 보이지만 respond_only F1이 이미
0.996 — 모델이 완전 학습한 구간. 실익은 long 버킷의 trio 분리인데 길이만으론
0.40/0.38/0.17로 안 갈림. len_bucket 태그는 v2.2에 저비용 편승만.

전선 2 해석: trio(ask/plan/web)를 가르는 신호는 존재하되 전부 조합형이고
재료([META] turn·ci, [SEQ], 프롬프트 자체)가 이미 노출돼 있다 — 남은
레버는 태그가 아니라 **bias(희소 클래스 경계 이동)와 모델 용량**이라는
기존 결론이 재확인됨.

## 전선 3. EXECUTE (기저: bash 0.43 / tests 0.38 / lint 0.19)

| ID | 가설 | 핵심 결과 | 판정 |
|---|---|---|---|
| E-1 | **test/lint 상태 분리** (기검증) | purity 0.654 → 0.693 (+0.04). "수정 직후 테스트 이력 없음 → run_tests 0.67", "세션 첫 execute → run_bash 0.85" 등 강상태 12개 | ✅ **통과 (강) — v2.2 확정** |
| E-2 | 직전 **수정 파일 확장자**가 검증 방식을 정한다 | **tsx 수정 → lint 0.37 (기저 2배)**, ts → lint 0.29 / py → tests 0.56 / 수정 이력 없음 → bash 0.57. purity 0.557 | ✅ 통과 — v2.2 편입 |
| E-3 | CI 상태가 execute를 가른다 | passed/failed 간 이동 ±0.05 수준 | ❌ 기각 (약) |
| E-4 | execute는 자기반복한다 | **bash→bash 0.75, tests→tests 0.58~0.63, lint→lint 0.44~0.57**. purity 0.664. + tests fail → lint 0.23 소폭 상승 | ✅ 통과 (관측가능) |
| E-5 | 세션 지배 언어 → lint 비율 | ts에서 lint 0.22로 소폭. E-2(수정 파일)와 중복되며 더 약함 | ⭕ 약함 (E-2로 대체) |

전선 3 해석: execute의 문법이 거의 다 풀렸다 — ① 검증 채널 2개(test/lint)를
분리 추적하고(E-1), ② 방금 뭘 고쳤는지가 어느 채널인지 정하고(E-2: TS는
린트, PY는 테스트), ③ 같은 execute를 반복하는 관성(E-4). tests↔lint
고마진 오류의 상당 부분이 E-1+E-2 미주입 때문일 가능성 — R-체크 예측
R1(tests↔lint = 표현 결함)의 근거 강화.

---

## 종합: serialize v2.2 번들 확정안

채택 카드 (전부 결정적 계산 가능, 한 번의 직렬화 개정으로 통합):

```text
[STATE] 개정:
  test={never|pass|fail} lint={never|pass|fail}          ← E-1 (확정)
  edits_after_test={0|1|2+} edits_after_lint={0|1|2+}    ← E-1
  insp_streak={0|1|2|3|4+}                                ← I-3 (신규)
  last_mod_ext={py|ts|tsx|js|other|none}                  ← E-2 (신규)
  open_cnt={0|1|2+}                                       ← I-4 (신규)
[LAST] result_bucket 확장:
  list/glob 결과의 파일 수 버킷 {0|1-3|4-15|16+} 추가     ← I-2
[META]: len_bucket={s|m|l} 편승                            ← C-3 (저비용)
```

기각·보류 목록 (재론 방지):
I-1 경로깊이, I-5 심볼, C-4 나열, E-3 CI(execute용), E-5 언어(E-2로 대체),
budget(I8), 아키타입(I2, 보류), INTENT(기사망).

관측가능 확인군 (태그 불필요, 모델 사용 여부만 추후 점검):
C-1/C-2/C-5(재료가 [META]·[SEQ]에 노출), E-4(자기반복 — [LAST] 노출.
단, bias 튜닝 시 execute 자기반복 관성이 prior로 반영되는지 참고).

## 실행 지시

```text
1) serialize.py를 위 v2.2로 개정 + golden 재생성 (사람 육안 1회)
2) v2.2는 단독 GPU 슬롯 배정 금지 — 다음 재학습(사이클3 분기에 따라
   SupCon A-런 또는 자기증류 student)에 동승. 동승 시 3ep 시점에서
   v2 기준(0.7017)과 비교해 번들 기여 분리 기록.
3) 본선 자산: 본 문서의 15행 판정표 자체가 "가설-검증-채택" 프로세스
   증거물 — 발표 슬라이드 1장으로 직행.
```

## 예측 등록 (v2.2 효과, 결과 대조용)

```text
V1. v2.2 번들이 execute 그룹 F1 +0.015~0.03 (E-1+E-2 합산 — 가장 큰 수혜)
V2. inspect 그룹 F1 +0.005~0.015 (I-3/I-4 — purity 이득이 v2 대비 증분이라 소폭)
V3. communicate는 ±0.005 (신규 주입이 len_bucket뿐)
V4. 전체 Macro-F1 +0.005~0.012
```
