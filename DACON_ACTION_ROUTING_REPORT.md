# DACON AI Agent Action Routing 실험 결과 보고서

작성일: 2026-07-02  
작업 디렉터리: `C:\Users\kiros\Desktop\데이콘라워`

## 1. 문제 개요

본 과제는 AI 코딩 에이전트 세션의 특정 시점 상태를 입력으로 받아, 에이전트가 다음에 수행할 action을 14개 클래스 중 하나로 예측하는 분류 문제이다.

예측 대상 action은 다음과 같다.

```text
read_file, grep_search, list_directory, glob_pattern,
edit_file, write_file, apply_patch,
run_bash, run_tests, lint_or_typecheck,
ask_user, plan_task, web_search, respond_only
```

평가지표는 14개 클래스에 대한 Macro-F1이다. 따라서 자주 등장하는 action만 잘 맞히는 모델보다, 희소 action까지 균형 있게 맞히는 모델이 중요하다.

## 2. 입력 데이터 구조와 판단 포인트

각 sample은 크게 네 부분으로 구성된다.

| 필드 | 의미 | 모델 판단에서의 역할 |
|---|---|---|
| `current_prompt` | 현재 사용자 발화 | 직접적인 의도 신호 |
| `history` | 이전 user/action/result 흐름 | 다음 action을 결정하는 가장 강한 상태 전이 신호 |
| `session_meta` | 요금제, 언어 선호, 토큰 예산, turn index 등 | 약한 보조 신호 |
| `workspace` | 언어 구성, open files, git dirty, CI 상태 등 | 파일 탐색/수정/실행 여부 판단 보조 |

핵심 발견은 이 문제가 단순 intent classification이라기보다 **agent workflow state transition** 문제에 가깝다는 점이다.

예를 들어 현재 prompt가 `한번 돌려봐`일 때, 직전 action이 `edit_file`이면 `run_tests`, `lint_or_typecheck`, `run_bash` 가능성이 커진다. 반대로 직전 action이 `grep_search`이고 사용자가 `그 파일 열어봐`라고 하면 `read_file`이 강해진다.

## 3. 최종 접근 전략

최종 모델은 14개 action을 직접 한 번에 맞히는 대신, 먼저 action을 4개 coarse group으로 나누고, 그 안에서 fine action을 예측하는 hierarchical routing 구조를 사용했다.

```text
input sample
  -> compact_flags text serialization
  -> TF-IDF word + char FeatureUnion
  -> coarse LinearSVC: 4개 group 예측
  -> group별 fine LogisticRegression: group 내부 action 예측
  -> final action
```

4개 coarse group은 다음과 같다.

| Group | Actions |
|---|---|
| `inspect` | `read_file`, `grep_search`, `list_directory`, `glob_pattern` |
| `modify` | `edit_file`, `write_file`, `apply_patch` |
| `execute` | `run_bash`, `run_tests`, `lint_or_typecheck` |
| `communicate` | `ask_user`, `plan_task`, `web_search`, `respond_only` |

최종 inference에서는 margin threshold를 두지 않고, 항상 coarse group을 선택한 뒤 해당 group의 fine model을 사용한다. 실험상 `threshold=0.0`이 가장 좋았고, `0.4` 같은 보수적 threshold는 점수를 낮췄다.

## 4. 주요 실험 결과

| 단계 | 방법 | Validation Macro-F1 | 판단 |
|---|---:|---:|---|
| Baseline | `current_prompt` TF-IDF + LogReg | `0.436882` | 매우 기본적인 기준선 |
| Linear feature engineering | compact history/meta + linear model | `0.633169` | history action sequence가 큰 폭으로 기여 |
| Embedding 추가 | TF-IDF + MiniLM embedding + numeric | `0.639156` | embedding 단독은 약하지만 보조 신호 있음 |
| Model zoo | Voting ensemble | `0.658509` | 앙상블은 개선되나 제출 복잡도 증가 |
| Compact score router | compact flags + transition/group/rule prior | `0.666351` | 제출 가능한 강한 단일 artifact |
| Coarse/Fine v1 | coarse SVC C=0.7 + fine LogReg | `0.694036` | hierarchical routing이 큰 폭으로 개선 |
| Coarse/Fine final | coarse SVC C=2.0 + fine LogReg | `0.695150` | 최종 선택 |

## 5. GroupShuffleSplit 검증

동일 session의 여러 step이 train/validation에 함께 들어가면 validation 점수가 과대평가될 수 있다. 이를 확인하기 위해 `session_id = id.rsplit("-step_", 1)[0]` 기준 GroupShuffleSplit도 수행했다.

| 모델 | Stratified Macro-F1 | GroupShuffle Macro-F1 |
|---|---:|---:|
| Coarse/Fine v1, C=0.7 | `0.694036` | `0.686921` |
| Coarse/Fine final, C=2.0 | `0.695150` | `0.687763` |

GroupShuffleSplit에서도 동일한 구조가 유지되므로, 개선이 단순 session leakage 착시만은 아니라고 판단했다.

## 6. Margin Threshold 실험 결론

초기 문서에서는 coarse model의 margin이 `0.4` 이상일 때만 fine model을 쓰는 전략을 제안했다. 하지만 실제 검증 결과는 반대였다.

Stratified split 기준:

| Threshold | Macro-F1 |
|---:|---:|
| `0.0` | `0.695150` |
| `0.1` | `0.694994` |
| `0.2` | `0.694531` |
| `0.3` | `0.694425` |
| `0.4` | `0.693461` |
| `0.5` | `0.693005` |
| `0.6` | `0.692006` |
| `0.8` | `0.690209` |

coarse group accuracy가 약 `99%` 수준으로 매우 높았기 때문에, 일부 샘플만 gating하는 방식보다 모든 샘플에 coarse/fine routing을 적용하는 것이 더 좋았다.

## 7. 추가 문서 기반 실험 결과

`codex_next_steps_margin04.md`의 제안에 따라 다음 실험을 수행했다.

1. doc-style rule serializer
2. rule hint token 추가
3. fine model을 LogisticRegression에서 LinearSVC로 변경
4. compact text와 doc-style text 결합
5. margin threshold sweep

결론은 다음과 같다.

| 실험 | GroupShuffle Macro-F1 | 판단 |
|---|---:|---|
| `doc_rule8` 단독 | `0.659354` | 기존 compact flags보다 낮음 |
| `doc_rule12` 단독 | `0.647508` | history를 길게 raw text로 넣으면 잡음 증가 |
| compact + fine LinearSVC | `0.682792` | fine LogReg보다 낮음 |
| compact + doc_rule8 | `0.686315` | oracle fine은 높지만 deployable은 낮음 |
| compact + coarse C=2.0 + fine LogReg | `0.687763` | 최종 선택 |

즉, rule hint 자체는 가능성이 있지만 현재 구현에서는 coarse group 정확도를 낮춰 최종 deployable 점수에는 도움이 되지 않았다.

## 8. 최종 모델 상세

최종 artifact는 `model/routing_margin_router.pkl` 하나에 필요한 객체를 모두 담는다.

구성:

```text
kind: routing_margin_router
vectorizer: TF-IDF word + char FeatureUnion
coarse_svc: LinearSVC(C=2.0, class_weight="balanced")
fine_models:
  inspect: LogisticRegression(C=2.0, class_weight="balanced")
  modify: LogisticRegression(C=2.0, class_weight="balanced")
  execute: LogisticRegression(C=2.0, class_weight="balanced")
  communicate: LogisticRegression(C=2.0, class_weight="balanced")
strategy: coarse_svc_then_fine_logreg_always
```

최종 validation metadata:

```text
Stratified Macro-F1: 0.695150
GroupShuffle Macro-F1: 0.687763
Oracle group Macro-F1: 0.699362
```

Oracle group 점수는 coarse group이 항상 정답이라고 가정했을 때의 상한이다. 최종 deployable 점수와 oracle 점수의 차이가 크지 않으므로, 현재 병목은 coarse routing보다 group 내부 fine action 분류에 있다.

## 9. 최종 제출 산출물

최종 제출 zip:

```text
C:\Users\kiros\Desktop\데이콘라워\submit_routing_margin_router.zip
```

zip 내부 구조:

```text
model/routing_margin_router.pkl
requirements.txt
script.py
```

파일 크기:

```text
routing_margin_router.pkl: 약 16.3 MB
submit_routing_margin_router.zip: 약 16.3 MB
```

검증:

```text
python script.py
-> ./output/submission.csv 생성 성공
```

제출 환경 제약 대비:

| 조건 | 상태 |
|---|---|
| 추론 시간 10분 이하 | 충족 예상 |
| 제출 용량 1GB 이하 | 충족 |
| 오프라인 실행 | 충족 |
| GPU 필요 여부 | 불필요 |
| 외부 인터넷 필요 여부 | 불필요 |
| requirements | `scikit-learn==1.8.0`, `joblib==1.5.3` |

## 10. 현재 모델의 해석

이번 문제에서 효과가 컸던 신호는 다음 순서로 정리된다.

1. `current_prompt`
2. 최근 assistant action sequence
3. 직전 action의 result/args에서 추출한 flag
4. workspace open files/path/ext
5. CI 상태, git dirty, turn index
6. coarse group structure

특히 `history`는 단순 보조 정보가 아니라 다음 action을 결정하는 핵심 상태 정보였다. 모델 성능이 크게 오른 지점도 current prompt 중심 모델에서 벗어나, 최근 action sequence와 workflow 상태를 반영한 이후였다.

## 11. 남은 개선 방향

현재 1위권 점수가 약 `0.76`이라고 가정하면, 아직 약 `0.06~0.07`의 gap이 있다. 다음 개선은 전체 구조를 바꾸기보다 group 내부 혼동을 줄이는 방향이 유리하다.

우선순위:

1. `inspect` 내부 분류 강화  
   `read_file`, `grep_search`, `list_directory`, `glob_pattern` 혼동이 가장 큰 병목일 가능성이 높다.

2. fine model별 feature specialization  
   모든 group에 같은 serializer를 쓰지 말고, inspect용 path/pattern feature, execute용 result/test/lint feature를 별도로 강화한다.

3. OOF stacking  
   flat14, coarse/fine, compact score router, voting ensemble의 OOF prediction을 meta model로 결합한다.

4. high precision override 검증  
   `web_search`, `lint_or_typecheck`, `write_file`, `respond_only` 같은 희소 class에 대해 precision 0.90 이상 rule만 override 후보로 검토한다.

5. class별 confusion 분석  
   다음 pair를 우선 확인한다.

```text
read_file vs grep_search
grep_search vs glob_pattern
list_directory vs glob_pattern
edit_file vs apply_patch
edit_file vs write_file
run_bash vs run_tests
run_tests vs lint_or_typecheck
ask_user vs plan_task
plan_task vs web_search
respond_only vs plan_task
```

## 12. 최종 결론

본 실험의 최종 선택은 `compact_flags + coarse LinearSVC(C=2.0) + group-specific fine LogisticRegression` 구조이다.

이 구조는 baseline `0.436882`에서 최종 stratified `0.695150`까지 성능을 끌어올렸고, GroupShuffleSplit에서도 `0.687763`을 유지했다. 또한 모델 크기가 작고 GPU가 필요 없으며, 오프라인 코드 제출 환경에 적합하다.

현재 제출 권장 파일은 다음이다.

```text
C:\Users\kiros\Desktop\데이콘라워\submit_routing_margin_router.zip
```

## 13. 추가 실험 업데이트: Advanced Router

2026-07-02에 `codex_experiment_design_action_routing.md`와 `NEXT_EXPERIMENT_v2.md` 기반 추가 실험을 수행했다.

핵심 결과는 다음과 같다.

| 모델/실험 | GroupShuffle Macro-F1 | 비고 |
|---|---:|---|
| 이전 최종 `routing_margin_router` | `0.687763` | coarse SVC + shared fine LogReg |
| group-specific same text | `0.688180` | group별 vectorizer만 분리 |
| group-specific specialized | `0.699546` | group별 feature token 추가 |
| group-specific specialized_x2 | `0.706314` | feature token 반복 강화 |
| specialized_x2 + last2 transition prior | `0.710302` | `alpha=0.3` |
| specialized_x2 + prior + pair resolver | `0.711324` | 최종 advanced 선택 |

최종 advanced 제출 파일:

```text
C:\Users\kiros\Desktop\데이콘라워\submit_advanced_router.zip
```

최종 advanced artifact:

```text
C:\Users\kiros\Desktop\데이콘라워\model\advanced_router.pkl
```

구성:

```text
coarse: compact_flags TF-IDF + LinearSVC(C=2.0)
fine: group-specific specialized_x2 TF-IDF + LogisticRegression(C=2.0)
transition prior: P(action | last2_action), alpha=0.3
pair resolver: 주요 top2 pair 전용 binary LogisticRegression, threshold=0.08
```

zip 크기:

```text
submit_advanced_router.zip: 약 29.2 MB
advanced_router.pkl: 약 29.3 MB
```

`script.py`는 이제 다음 우선순위로 모델을 로드한다.

```text
advanced_router.pkl
-> routing_margin_router.pkl
-> compact_flags_router.pkl
-> older research/baseline models
```

## 14. NEXT_EXPERIMENT_v2 결론

v2 실험에서는 각 coarse group만 따로 떼어 group 내부 분류 상한을 측정했다.

| Group | Best Isolated Macro-F1 | Best Variant |
|---|---:|---|
| `inspect` | `0.553876` | `inspect_specialized_x2_word` |
| `modify` | `0.916111` | `modify_specialized_word_char` |
| `execute` | `0.702804` | `execute_specialized_word` |
| `communicate` | `0.703610` | `communicate_specialized_x2_word_char_num` |

isolated macro upper estimate는 `0.706192`였다. 이는 group-isolated 평가라 full pipeline 점수와 완전히 동일하게 비교할 수는 없지만, 현재 lightweight TF-IDF/linear 계열이 low `0.71` 근처에서 수렴하고 있다는 신호로 볼 수 있다.

해석:

1. `modify`는 거의 해결된 그룹이다.
2. `inspect`가 여전히 가장 큰 병목이다.
3. `execute`와 `communicate`는 개선 여지가 있으나, 현재 feature family로는 `0.70` 부근에서 막힌다.
4. leaderboard `0.78`에 도달하려면 OOF stacking, 강한 representation learning, LLM distillation, 또는 inspect/communicate에 대한 더 강한 specialist가 필요할 가능성이 높다.

현재 제출 우선순위는 다음과 같다.

```text
1. submit_advanced_router.zip
2. submit_routing_margin_router.zip
```
