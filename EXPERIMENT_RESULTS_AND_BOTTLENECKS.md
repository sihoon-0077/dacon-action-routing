# DACON Action Routing 실험 전체 결과 및 현재 문제점

작성일: 2026-07-02  
작업 디렉터리: `C:\Users\kiros\Desktop\데이콘라워`

## 1. 현재 최종 결론

현재 가장 성능이 높은 제출 후보는 다음 파일이다.

```text
C:\Users\kiros\Desktop\데이콘라워\submit_advanced_router.zip
```

검증 기준 성능:

```text
GroupShuffle Macro-F1: 0.711324
Accuracy:              0.710974
Experiment:            phase6_pair_resolver_t0.08
```

현재 제출 우선순위:

| 순위 | 제출 파일 | Local Validation |
|---:|---|---:|
| 1 | `submit_advanced_router.zip` | GroupShuffle `0.711324` |
| 2 | `submit_routing_margin_router.zip` | GroupShuffle `0.687763` |
| 3 | `submit_compact_flags_router.zip` | Stratified `0.666351` |

## 2. 전체 실험 타임라인

| 단계 | 핵심 방법 | Macro-F1 | 판단 |
|---|---|---:|---|
| Baseline | `current_prompt` only TF-IDF + LogReg | `0.436882` | 시작 기준선 |
| Feature LogReg | compact history/meta/action sequence | `0.633169` | history가 핵심임을 확인 |
| Embedding 추가 | TF-IDF + MiniLM + numeric | `0.639156` | embedding 단독은 약함, 보조 신호 정도 |
| Model zoo | Voting ensemble | `0.658509` | RandomForest/Tree 계열은 약함, linear가 강함 |
| Compact score router | compact flags + transition/group/rule prior | `0.666351` | 제출 가능한 첫 강한 모델 |
| Coarse/Fine v1 | 4-group coarse SVC + fine LogReg | `0.694036` | hierarchical routing이 크게 개선 |
| Coarse/Fine C=2.0 | coarse SVC C 튜닝 | `0.695150` | 기존 routing 제출 모델 |
| Group-specific fine | group별 vectorizer/serializer | `0.706314` | fine action 병목 개선 |
| Transition prior | `P(action | last2_action)` 보정 | `0.710302` | workflow state prior 유효 |
| Pair resolver | 주요 top2 pair binary resolver | `0.711324` | 현재 best |

주의:

```text
0.695150은 Stratified split 기준
0.711324는 GroupShuffleSplit 기준
```

현재 모델 선택은 GroupShuffleSplit을 우선 기준으로 했다. 같은 session의 step이 train/valid에 섞이면 validation이 과대평가될 수 있기 때문이다.

## 3. 최종 Advanced Router 구조

최종 `advanced_router.pkl` 구조는 다음과 같다.

```text
input sample
  -> compact_flags text
  -> coarse LinearSVC(C=2.0)로 4개 group 예측
  -> group-specific specialized_x2 text 생성
  -> group별 fine LogisticRegression(C=2.0)
  -> last2_action transition prior 보정
  -> 주요 top2 pair에 대해 binary resolver 적용
  -> final action
```

4개 coarse group:

| Group | Actions |
|---|---|
| `inspect` | `read_file`, `grep_search`, `list_directory`, `glob_pattern` |
| `modify` | `edit_file`, `write_file`, `apply_patch` |
| `execute` | `run_bash`, `run_tests`, `lint_or_typecheck` |
| `communicate` | `ask_user`, `plan_task`, `web_search`, `respond_only` |

advanced 설정:

```text
group_text_variant: specialized_x2
transition prior:   P(action | last2_action)
prior_alpha:        0.3
prior_smooth:       1.0
pair_threshold:     0.08
pair_resolvers:     10개
```

## 4. 최종 모델의 클래스별 성능

현재 best `phase6_pair_resolver_t0.08`의 GroupShuffle class별 F1은 다음과 같다.

| Action | Precision | Recall | F1 |
|---|---:|---:|---:|
| `list_directory` | `0.3724` | `0.5360` | `0.4394` |
| `read_file` | `0.5276` | `0.5251` | `0.5263` |
| `grep_search` | `0.6326` | `0.5684` | `0.5988` |
| `glob_pattern` | `0.6727` | `0.5571` | `0.6095` |
| `ask_user` | `0.6496` | `0.5903` | `0.6186` |
| `lint_or_typecheck` | `0.6056` | `0.6527` | `0.6282` |
| `plan_task` | `0.6499` | `0.6272` | `0.6383` |
| `web_search` | `0.5913` | `0.6996` | `0.6409` |
| `run_tests` | `0.7408` | `0.7538` | `0.7473` |
| `run_bash` | `0.8036` | `0.7666` | `0.7847` |
| `apply_patch` | `0.8230` | `0.8954` | `0.8577` |
| `edit_file` | `0.9439` | `0.9105` | `0.9269` |
| `write_file` | `0.9399` | `0.9568` | `0.9483` |
| `respond_only` | `0.9940` | `0.9930` | `0.9935` |

## 5. 현재 가장 큰 문제점

### 문제 1. `inspect` 4형제가 여전히 가장 큰 병목

가장 낮은 클래스가 모두 inspect group에 몰려 있다.

```text
list_directory 0.439
read_file      0.526
grep_search    0.599
glob_pattern   0.610
```

모델은 “코드베이스를 봐야 한다”는 coarse group은 잘 맞힌다. 하지만 그 안에서 다음 네 action을 계속 헷갈린다.

```text
read_file
grep_search
list_directory
glob_pattern
```

대표 confusion:

```text
grep_search -> read_file       478
read_file   -> list_directory  370
read_file   -> grep_search     367
grep_search -> list_directory  272
list_directory -> read_file    213
glob_pattern -> read_file      168
glob_pattern -> grep_search    150
```

해석:

```text
"파일을 열어야 하는지"
"코드 검색을 해야 하는지"
"디렉터리 목록을 봐야 하는지"
"glob 패턴으로 파일 후보를 찾아야 하는지"
```

이 네 가지가 데이터 텍스트상 매우 비슷하게 나타난다.

### 문제 2. `communicate` 내부에서 `ask_user / plan_task / web_search`가 섞임

`respond_only`는 거의 해결됐다.

```text
respond_only F1 = 0.9935
```

하지만 나머지 communicate 클래스는 아직 낮다.

```text
ask_user   0.619
plan_task  0.638
web_search 0.641
```

대표 confusion:

```text
ask_user  -> plan_task   142
plan_task -> ask_user    130
ask_user  -> web_search   75
plan_task -> web_search   52
web_search -> ask_user    45
web_search -> plan_task   30
```

해석:

```text
"어떻게 할까?"        -> ask_user인지 plan_task인지 애매
"찾아봐"              -> web_search인지 grep_search인지 애매
"권장 방식 알려줘"    -> web_search인지 respond_only/plan_task인지 애매
```

### 문제 3. `execute`에서 lint/typecheck가 아직 약함

execute group 성능:

```text
run_bash          0.785
run_tests         0.747
lint_or_typecheck 0.628
```

대표 confusion:

```text
run_bash          -> run_tests          133
run_tests         -> lint_or_typecheck  116
run_tests         -> run_bash           101
lint_or_typecheck -> run_tests          100
run_bash          -> lint_or_typecheck   89
lint_or_typecheck -> run_bash            71
```

해석:

```text
"돌려봐"
"확인해봐"
"검증해봐"
```

같은 표현이 bash 실행, test 실행, lint/typecheck 모두에 쓰인다.

### 문제 4. 현재 lightweight linear 계열의 상한이 보임

`NEXT_EXPERIMENT_v2`에서 group을 따로 떼어 isolate 실험을 했다.

| Group | Best Isolated Macro-F1 | 해석 |
|---|---:|---|
| `inspect` | `0.553876` | 가장 어려움 |
| `modify` | `0.916111` | 거의 해결 |
| `execute` | `0.702804` | 중간 난이도 |
| `communicate` | `0.703610` | 중간 난이도 |

isolated macro upper estimate:

```text
0.706192
```

이 값은 full pipeline 점수와 완전히 같은 의미는 아니지만, 현재 TF-IDF + feature engineering + linear model 계열이 `0.71` 근처에서 한 번 막히고 있다는 신호다.

### 문제 5. 1등 `0.78`과의 차이는 단순 튜닝 문제가 아님

현재 best:

```text
0.711324
```

리더보드 1등 근처:

```text
0.78
```

gap:

```text
약 0.0687
```

이 정도 차이는 C값, threshold, ngram_range 같은 단순 튜닝만으로 메우기 어렵다.

가능한 차이 요인:

```text
OOF stacking
강한 앙상블
세션/템플릿 패턴 memory
pseudo-labeling
hidden test 분포에 맞는 rule
더 강한 embedding/representation
LLM distillation
데이터 생성 규칙 역추정
```

## 6. 실험별 의미 있는 발견

### 발견 1. `history`는 필수다

`current_prompt`만 쓰면 성능이 낮다.

```text
Baseline current_prompt only: 0.436882
compact history/action/meta:  0.633169
```

이 문제는 단순 사용자 의도 분류가 아니라 agent workflow state transition 문제다.

### 발견 2. coarse group routing은 거의 해결됨

coarse group accuracy는 약 `0.99` 수준이다. 따라서 더 이상 coarse routing을 파는 것보다, group 내부 fine action을 개선하는 편이 낫다.

### 발견 3. margin gating은 손해였다

초기에는 coarse margin이 높을 때만 fine model을 쓰는 전략을 검토했다. 하지만 실제로는 항상 coarse/fine routing을 적용하는 `threshold=0.0`이 가장 좋았다.

### 발견 4. group-specific feature가 가장 큰 개선을 만들었다

공유 fine vectorizer에서 group별 specialized feature로 바꾸면서 크게 올랐다.

```text
shared compact fine:          0.687763
group-specific specialized:   0.699546
group-specific specialized_x2:0.706314
```

### 발견 5. transition prior는 여전히 유효하다

`P(action | last2_action)` prior를 fine score에 더했을 때 성능이 올랐다.

```text
specialized_x2:                    0.706314
specialized_x2 + last2 prior:      0.710302
```

### 발견 6. pair resolver는 작지만 의미 있는 개선

주요 top2 혼동 pair에 대해 binary resolver를 붙였을 때 현재 최고점이 나왔다.

```text
last2 prior best:           0.710302
pair resolver threshold .08:0.711324
```

## 7. 최종 산출물

현재 가장 좋은 제출 zip:

```text
C:\Users\kiros\Desktop\데이콘라워\submit_advanced_router.zip
```

구성:

```text
model/advanced_router.pkl
requirements.txt
script.py
```

크기:

```text
submit_advanced_router.zip: 약 29.2 MB
advanced_router.pkl:        약 29.3 MB
```

검증:

```text
python script.py
-> output/submission.csv 생성 성공
```

## 8. 다음 실험 우선순위

### 1순위. OOF stacking

현재 가장 현실적인 점프 후보.

base model 후보:

```text
M1 advanced_router
M2 routing_margin_router
M3 compact_flags_router
M4 flat14 compact LogReg
M5 char-only model
M6 prompt-only model
M7 history-action-only model
```

meta feature:

```text
각 모델 14-class score/proba
coarse group
fine margin
last action
rule hint
turn bucket
CI status
```

기대:

```text
0.711 -> 0.72~0.73 가능성
```

### 2순위. inspect 전용 강한 specialist

현재 가장 낮은 group이므로 집중 개선 필요.

개선 후보:

```text
path/pattern parser 강화
open_files와 prompt path 매칭
last action result의 file/match count 구조화
read vs grep binary resolver 재설계
list vs glob binary resolver 재설계
```

### 3순위. communicate 전용 specialist

목표:

```text
ask_user vs plan_task
plan_task vs web_search
web_search vs respond_only
```

고정밀 rule 후보:

```text
latest / official docs / recommended / paper -> web_search
plan / step / roadmap / approach -> plan_task
which / should I / choose / confirm -> ask_user
summary / recap / wrap up -> respond_only
```

### 4순위. execute 전용 specialist

목표:

```text
run_tests vs lint_or_typecheck vs run_bash
```

강화 feature:

```text
pytest / npm test / cargo test / unit / e2e -> run_tests
lint / typecheck / mypy / ruff / tsc -> lint_or_typecheck
build / install / dev server / command -> run_bash
```

## 9. 현재 판단

현재 모델은 대회 제출 조건에는 매우 안정적이다.

```text
offline 실행 가능
GPU 불필요
zip 1GB 제한 여유
추론 10분 제한 여유 예상
requirements 최소
```

하지만 리더보드 `0.78`까지 가려면 현재 방식만으로는 부족해 보인다. 지금 병목은 모델 하나의 hyperparameter가 아니라, 다음 action을 결정하는 미세한 문맥 구분이다.

가장 큰 병목 순서:

```text
1. inspect 4형제 구분
2. communicate 3형제 구분
3. lint_or_typecheck 구분
4. 현재 linear feature family의 표현력 한계
```

현재 추천:

```text
제출은 submit_advanced_router.zip 사용
다음 실험은 OOF stacking부터 진행
```
