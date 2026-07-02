# DACON AI Agent Action Routing

AI 코딩 에이전트 세션 상태에서 다음 action을 14개 클래스 중 하나로 예측하는 DACON 대회 실험 코드입니다.

## Current Best

현재 로컬 검증 기준 최고 모델은 `advanced_router`입니다.

```text
GroupShuffle Macro-F1: 0.711324
Experiment: phase6_pair_resolver_t0.08
Submission zip: submit_advanced_router.zip
```

제출용 모델 파일과 zip은 git에 포함하지 않습니다. 팀 공유 시 별도 드라이브나 GitHub Release를 사용하세요.

## Repository Contents

```text
script.py                              # 제출용 추론 스크립트
train_advanced_router.py               # 현재 best artifact 학습 스크립트
advanced_action_routing_experiments.py # group-specific/router 실험
group_ceiling_v2_experiments.py        # group별 병목/상한 분석
train_routing_margin_router.py         # 이전 coarse/fine router 학습
compact_state_experiments.py           # compact flags 기반 실험
*_experiments.py                       # 기타 실험 스크립트
research.md                            # 실험 로그
DACON_ACTION_ROUTING_REPORT.md         # 결과 보고서
EXPERIMENT_RESULTS_AND_BOTTLENECKS.md  # 전체 결과 및 병목 정리
requirements.txt
```

## Not Included

다음 파일/폴더는 git에 올리지 않습니다.

```text
data/
model/
output/
reports/
submit_*.zip
*.pkl
```

이유:

- 대회 데이터는 공개 repo에 올리면 규정 문제가 생길 수 있습니다.
- `.pkl`, `.zip`은 크고 binary라 git history를 더럽힙니다.
- 실험 로그/예측 CSV는 재생성 가능합니다.

## Local Setup

대회 배포 데이터를 직접 받아서 아래 구조로 둡니다.

```text
data/
  train.jsonl
  train_labels.csv
  test.jsonl
  sample_submission.csv
```

패키지 설치:

```bash
pip install -r requirements.txt
```

현재 best 모델 학습:

```bash
python train_advanced_router.py
```

추론 실행:

```bash
python script.py
```

출력:

```text
output/submission.csv
```

## Main Findings

- `current_prompt`만으로는 부족하고, `history`의 최근 action sequence가 핵심입니다.
- coarse group routing은 거의 해결됐고, 병목은 group 내부 fine action 구분입니다.
- 가장 큰 병목은 `inspect` 그룹입니다.
- 현재 lightweight TF-IDF/linear 계열은 GroupShuffle 기준 low `0.71` 부근에서 한 번 막힙니다.
- 다음 점프 후보는 OOF stacking, 강한 representation/distillation, inspect/communicate specialist입니다.
