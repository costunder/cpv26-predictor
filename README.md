# CPV26 Predictor

공개 KBO 데이터로 타석 결과, 경기 승무패, 선수별 PA·안타 분포를 학습하는
RelGNN 프로젝트입니다. 설치 명령은 Linux/Bash 기준이며, Python은 Conda로 관리합니다.

V26 계정별 당일 추천 기능은 아직 구현되지 않았습니다.

v5는 더블헤더 구분, 원천 스냅샷 선택, 겹치는 기록 처리와 사용 건수 보고를 수정합니다.
기존 v3/v4 사용자는 [캐시를 이용한 v5 갱신](#4-1-기존-v3v4에서-v5로-갱신)을 실행합니다.
환경 재설치·원천 재다운로드·파일 삭제 없이, 캐시 재수입 후 새 그래프를 만듭니다.

## 1. 프로젝트 받기

공개 저장소이므로 로그인이나 토큰 없이 받을 수 있습니다.

~~~bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/costunder/cpv26-predictor.git
cd cpv26-predictor
~~~

이미 받았다면 [코드 업데이트](#9-환경-활성화와-코드-업데이트)로 이동합니다.
다른 폴더에 받았다면 아래 `cd` 경로를 바꾸면 됩니다.

## 2. Conda 설치

### 2-1. 환경 생성

기존 환경을 확인합니다.

~~~bash
conda env list
~~~

`cpv26`이 없으면 프로젝트 폴더에서 생성합니다. Python 3.12와 pip가 설치됩니다.

~~~bash
conda env create -f environment.yml
~~~

이미 `cpv26`이 있으면 생성 명령은 건너뜁니다. 지원 Python 버전은 3.10~3.12입니다.

### 2-2. 환경 활성화

~~~bash
conda activate cpv26
echo "$CONDA_DEFAULT_ENV"
echo "$CONDA_PREFIX"
command -v python
python --version
python -m pip --version
~~~

환경 이름은 `cpv26`, Python 경로는 `$CONDA_PREFIX/bin/python`이어야 합니다.
`base` 상태이거나 경로가 다르면 아래 Conda 오류 항목을 확인합니다.

### 2-3. CUDA PyTorch와 프로젝트 설치

아래는 **A100 MIG 10GB / driver 535.104.05 / CUDA 12.2 표시**에 맞춘 CUDA 12.1 설치 예시입니다.

~~~bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
TORCH_INDEX_URL="$TORCH_INDEX_URL" bash scripts/setup.sh ml-cuda
~~~

다른 driver에서는 [PyTorch 설치 안내](https://pytorch.org/get-started/locally/) 또는
[이전 버전 안내](https://pytorch.org/get-started/previous-versions/)에서 호환되는 index URL을
골라 `TORCH_INDEX_URL`을 바꿉니다.
[CUDA 12.1 driver 조건](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/)도 참고할 수 있습니다.

`setup.sh`는 활성화된 Conda 환경의 pip로 프로젝트와 검사 도구를 설치합니다.
환경 생성·활성화는 앞에서 해야 하며, `base`에는 설치할 수 없습니다.
기존 PyTorch가 지원 범위(`torch>=2.4,<3`)와 AMP·CUDA 연산 검사를 통과하면 그대로 사용합니다.
검사에 실패하면 지정한 index로 재설치하며, CPU로 자동 전환하지 않습니다.

### 2-4. 프로젝트 설정

설치 끝에 `Conda environment ready: ...`가 출력되면 실행합니다. 설치 오류가 있으면 먼저 해결합니다.

~~~bash
if [ ! -f .env ]; then cp .env.example .env; fi
chmod 600 .env
source scripts/activate.sh
~~~

`activate.sh`는 Conda 환경을 검사하고 `.env`를 읽습니다. Conda 활성화 명령은 아닙니다.
기존 `.env`, `var/`의 데이터와 checkpoint는 유지됩니다.

`.env` 기본값은 아래와 같습니다. 상대 경로는 프로젝트 루트 기준입니다.

~~~dotenv
CPV26_HOME=./var
CPV26_DB_PATH=./var/cpv26.duckdb
CPV26_TIMEZONE=Asia/Seoul
CPV26_DEVICE=auto
CPV26_RANDOM_SEED=2026
CPV26_LOG_LEVEL=INFO
~~~

설정 키는 `CPV26_` 뒤에 영문 대문자·숫자·밑줄만 허용합니다.
`PATH`, `PYTHONPATH`, `CONDA_PREFIX`, `TORCH_INDEX_URL`은 넣지 않습니다.
값에 따옴표나 `export`, shell 변수 치환을 쓰지 않으며, 주석은 별도 `#` 줄에 씁니다.
학습 장치는 아래 명령의 `--device cuda:0`으로 지정합니다.

## 3. GPU 확인과 DB 초기화

~~~bash
cpv26 show-config
cpv26 gpu-check --device cuda:0
cpv26 db-init
cpv26 db-check
~~~

`gpu-check`는 CUDA forward/backward 연산까지 검사합니다. 실패하면 학습을 시작하지 않습니다.
`db-check`는 schema와 참조 관계를 검사합니다. 아직 데이터를 넣지 않아 DB가 비어 있어도 정상입니다.

## 4. KBO 데이터와 그래프 생성

처음 적재한다면 아래를 실행합니다. 기존 v3/v4 사용자는 [4-1절](#4-1-기존-v3v4에서-v5로-갱신)로 이동합니다.

~~~bash
cpv26 kbo-history-fetch &&
cpv26 kbo-history-import &&
cpv26 kbo-fetch --year 2023 --year 2024 --year 2025 --year 2026 &&
cpv26 kbo-import --year 2023 --year 2024 --year 2025 --year 2026 &&
cpv26 db-check &&
cpv26 kbo-graph-build \
  --output var/datasets/kbo_graph_2001_2026_v5 \
  --start-date 2001-01-01 \
  --end-date 2026-07-26
~~~

**2001년부터** 적재합니다. 기간별로 확보된 기록은 다릅니다.

| 기간 | 학습에 쓰는 기록 |
|---|---|
| 2001~2022 | 경기·득점, 선수별 타격·투구 합계, 확인된 이닝별 타격 결과 → 경기·안타·타격 집계·투구 헤드 |
| 2023~2025 | 경기 결과와 관측 타석, 그 타석의 선수·경기별 집계 → 같은 통계 입력과 전체 헤드 |
| 2026 | 7월 26일까지의 같은 항목, 별도 테스트 |

타자·투수 원문을 전부 보존하고, 읽을 수 있는 항목은 각각 학습에 연결합니다.
안타 수는 있지만 정확한 PA가 없는 행도 사용합니다. 확인되지 않은 타석 순서·상대 투수·
주자 상태는 만들지 않습니다. `kbo-history-import`와 그래프 `manifest.json`에 연도별
타자·투수·안타·타격 결과 사용 건수와 사용할 수 없는 항목의 사유를 남깁니다.

옛 원천에는 고유 선수 ID가 없어 이름이 같다고 동일 선수로 합치지 않습니다.
각 원천 행을 별도로 유지하며, 과거의 같은 이름·팀·역할로 묶인 통계를 입력으로 사용합니다.
동명이인이 섞일 수 있는 그룹이지 확인된 개인 이력이 아닙니다. 이름이 없거나 이전
그룹 기록이 없으면 팀 통계를 사용합니다. 개인의 2001~2026 경력을 연결했다는 뜻은 아닙니다.

최근 타석도 선수·경기별로 집계하므로 2023년 이후 통계 블록이 통째로 0이 되던 문제를
해결합니다. 필드별 관측 횟수는 타석 수가 아닌 선수·경기 수입니다. 타석만으로 확정할
수 없는 타자 득점/RBI, 투수 투구 수/아웃/실점/자책점은 결측으로 유지합니다.

생성 위치는 다음과 같습니다.

| 위치 | 내용 |
|---|---|
| var/datasets/kbo_history/ | 2001~2022년 고정 원천 JSON과 SOURCE.json |
| var/datasets/kbo_playbyplay/v0/ | 내려받은 원천 Parquet와 SOURCE.json |
| var/cpv26.duckdb | 정규화한 선수·팀·경기·타석 DB |
| var/reports/kbo_history_import.json | 역사 경기 적재·중복·연도별 제공 범위 |
| var/reports/kbo_import.json | 타석 원천 적재 결과와 품질 검사 |
| var/datasets/kbo_graph_2001_2026_v5/ | 정정된 원천 선택·중복 처리 기준을 적용한 v5 날짜별 그래프 |

두 fetch 명령은 고정된 공개 파일과 SHA-256을 확인합니다. 같은 파일과 적재 행은
재실행 시 재사용합니다. 기존 `var/datasets/kbo_graph/`와 checkpoint는 건드리지 않습니다.
역사 원천의 출처와 보완 기록은 [데이터 설명](docs/GPU_TRAINING.md#데이터-출처와-고정-스냅샷)에 있습니다.

기존 경기 점수만 적재했던 DB는 `cpv26 db-init` 후 `cpv26 kbo-history-import`를
다시 실행합니다. schema v5가 기존 경기·타석을 보존하면서 선수 기록을 추가합니다.
이미 내려받은 원천은 다시 받을 필요가 없습니다. 그래프와 학습 run은 새 경로를 사용합니다.

kbo-graph-build는 대상 날짜 이전의 최대 90일 관계를 사용해 날짜별 그래프를 만듭니다.
그날의 경기 결과는 정답이며, 같은 날의 결과를 과거 관계 입력으로 사용하지 않습니다.
90일은 입력 이력 범위이고, 학습 기간을 최근 90일로 제한한다는 뜻은 아닙니다.
원천의 당시 발표 시각은 재구성한 값이므로 실시간 운영 재현과는 구분합니다.

생성 후 전 기간 입력을 검사합니다. 파일 무결성·정보 시점·연도별 입력 사용량과 선수
입력의 다양성을 검사하며, 최근 자료의 통계 입력이 비어 있으면 오류로 종료합니다.

~~~bash
python scripts/audit_cross_era_graph.py var/datasets/kbo_graph_2001_2026_v5 \
  --output var/reports/cross_era_v5.json
~~~

검사는 입력 연결을 확인하는 것이며 예측 성능 향상을 입증하는 검사는 아닙니다.
[v5 오류 수정 검증](docs/ERROR_FIX_VALIDATION.md)에서 전 기간 대조와 검사 범위를 확인합니다.
[v3→v4 입력 연결 기록](docs/CROSS_ERA_VALIDATION.md)은 이전 검사로 보존합니다.

### 4-1. 기존 v3/v4에서 v5로 갱신

기존 학습 프로세스가 끝난 뒤 프로젝트 폴더에서 실행합니다. 같은 Conda 환경과
이미 받은 2023~2026 Parquet를 사용합니다. `kbo-fetch`, `kbo-history-fetch`,
`kbo-history-import`, 환경 재설치는 다시 하지 않습니다.

~~~bash
git pull --ff-only
conda activate cpv26
source scripts/activate.sh
cpv26 kbo-import --year 2023 --year 2024 --year 2025 --year 2026 &&
cpv26 db-check &&
cpv26 kbo-graph-build \
  --output var/datasets/kbo_graph_2001_2026_v5 \
  --start-date 2001-01-01 \
  --end-date 2026-07-26 &&
python scripts/audit_cross_era_graph.py var/datasets/kbo_graph_2001_2026_v5 \
  --output var/reports/cross_era_v5.json
~~~

`kbo-import`는 더블헤더 번호를 반영한 adapter v2 revision을 추가합니다. DB schema는
기존과 같은 **5**이며 이전 원천 행은 삭제하지 않습니다. 같은 배포자·연도의 파일은
전체 스냅샷으로 취급하고, `knowledge_at`까지 적재된 최신 스냅샷만 현재 조회에 사용합니다.
새 스냅샷에서 빠진 경기·타석을 옛 행에서 되살려 합산하지 않습니다.

겹치는 자료는 실제 관측된 필드와 학습 항목별로 우선순위를 적용합니다. 불완전한
PA 집계를 완전한 박스스코어로 간주하지 않습니다. 집계 사용 건수는 두 출처 전체를
보고하고, 아카이브 원문 기준 건수는 `raw_archive_boxscore`로 따로 남깁니다.

그래프 reader는 v2/v3/v4/v5를 읽습니다. 기존 그래프·run·checkpoint는 그대로 보관하고,
5절에서 **새 v5 run**을 시작합니다. 옛 checkpoint를 새 그래프에 `--resume`하지 않습니다.

## 5. 2001년부터 RelGNN 학습

4절이 끝난 뒤 실행합니다. A100 MIG 10GB 예시는 날짜 8개를 한 minibatch로 학습합니다.

~~~bash
cpv26 relgnn-train \
  --dataset var/datasets/kbo_graph_2001_2026_v5 \
  --run-dir var/runs/relgnn/kbo_2001_2024_v5 \
  --train-start-year 2001 \
  --train-end-year 2024 \
  --validation-year 2025 \
  --test-year 2026 \
  --chronological \
  --device cuda:0 \
  --epochs 30 \
  --batch-days 8 \
  --max-pa-per-day 0 \
  --max-edges-per-route 0 \
  --amp auto
~~~

| 시즌 | 역할 |
|---|---|
| 2001~2024 | 가중치를 학습하는 train |
| 2025 | validation, early stopping, best.pt 선택 |
| 2026-07-26까지 | 다음 절에서 별도로 실행하는 test |

`--chronological`은 매 epoch 2001년부터 날짜순으로 학습합니다. 연도가 바뀌어도
가중치·optimizer를 유지하고, 다음 epoch에는 2001년부터 다시 학습합니다.
요청한 학습연도의 자료가 빠졌으면 오류로 중단합니다.

2026 test는 학습 중 실행하지 않습니다. validation이 6 epoch 동안 개선되지 않으면
early stopping하며, `--patience 0`은 이를 끕니다. `--epochs`는 최대 전체 epoch 수입니다.
이미 결과를 확인한 2025년은 여기서 검증용으로 사용하고 독립적인 최종 테스트로 부르지 않습니다.

2001~2022년 타격 합계는 안타 분포와 타격 결과 집계를, 투구 합계는 관측된 수치별
투구 손실을 학습합니다. 정확한 PA가 없으면 가능한 PA에 걸쳐 안타 likelihood를 합산합니다.
빠진 항목만 손실에서 마스킹하며, 해당 선수 행 전체를 버리지 않습니다.
LiveHit는 관측 PA가 한 개 이상임을 확인할 수 있는 선수-경기에 조건부인 분포입니다.
미출장 확률이나 V26 계정별 최종 추천과 같지 않습니다.

기본 모델은 hidden dimension 64, 관계 layer 2개, attention head 4개이며,
learning rate 0.0003, weight decay 0.0001입니다. 상세 구조는 [GPU 학습 설명](docs/GPU_TRAINING.md)에 있습니다.

위 run 이름은 이번 실험용입니다. 같은 이름으로 이미 실행했다면 새 이름을 사용합니다.
기존 v2/v3/v4 checkpoint를 이번 v5 데이터에 `--resume`하지 않습니다.
기존 checkpoint는 기존 그래프에서 계속 평가할 수 있습니다.

~~~text
var/runs/relgnn/kbo_2001_2024_v5/
├── config.json
├── history.jsonl
├── last.pt
├── best.pt
└── training_report.json
~~~

history.jsonl에서 진행 기록을, training_report.json에서 학습 요약을 확인합니다.
best.pt는 validation으로 선택한 평가용 모델이고, last.pt는 학습 재개용입니다.
GPU를 사용할 수 없으면 오류로 끝나며 CPU 학습으로 자동 전환하지 않습니다.

기본 선택 기준은 기존과 같은 가중 손실입니다. v4부터 최근 연도에도 집계 정답이 있으므로
해당 손실도 포함됩니다. `training_report.json`과 평가 출력에 각 손실·관측 수·가중 기여도를
남깁니다. 승무패 log loss로 best.pt를 고르려면 새 학습에 `--selection-target match`를
명시합니다. 안타/타격 성능까지 동시에 좋아진다는 의미는 아닙니다.

기본은 모든 손실을 공유 모델에 역전파합니다. 집계 손실의 간섭 여부를 별도로 비교하려면
`--box-gradient-mode head_only`로 집계 출력층만 학습할 수 있습니다. 이 옵션은 집계 손실이
공유 표현을 학습하지 못하게 하므로 기본값이 아닙니다. 비교 시 다른 조건은 고정합니다.

### 5-1. 다시 설계하기 전 그래프 의존도 확인

`best.pt` 하나를 고정한 채 정상 입력과 관계 제거·재배선 입력을 같은 2025 validation에서
비교합니다. 2026 test를 구조 선택에 사용하지 않도록 이 명령의 기본 split은 validation입니다.

~~~bash
cpv26 relgnn-graph-diagnose \
  --checkpoint var/runs/relgnn/kbo_2001_2024_v5/best.pt \
  --dataset var/datasets/kbo_graph_2001_2026_v5 \
  --split validation \
  --device cuda:0 \
  --batch-days 1 \
  --workers 0 \
  --seed 2026
~~~

한 번의 실행에서 아래 조건을 비교합니다.

- `intact`: 원래 그래프
- `no_routes`: 모든 관계 메시지를 제거한 같은 checkpoint
- `permuted_endpoints`: 날짜와 관계별로 endpoint만 재배선한 그래프
- `permuted_edge_attributes`: endpoint는 유지하고 edge attribute 대응을 섞은 그래프
- `without_<route>`: 관계 한 종류씩 제거한 그래프

표의 `Delta`는 정상 그래프 대비 selection loss 변화입니다. 양수면 해당 조작에서 손실이
커졌다는 뜻입니다. `Mean TV`는 같은 질의의 예측 확률이 얼마나 달라졌는지 나타내며,
0에 가까우면 조작 전후 예측이 거의 같습니다. Live Hit TV는 전체 PA×H 결합분포가 아니라
`안타 없음/한 개 이상` 이진 주변분포의 변화입니다. 재배선이 실제로 바꾼 endpoint 수와
정상(`intact`) 입력의 관계별 메시지 크기·gate·attention 진단을 포함한 전체 결과는 터미널에
표시된 `report.json`에서 확인합니다. intervention 내부 통계는 동기화 비용을 피하기 위해
수집하지 않습니다. `--max-days N`은 날짜를 줄인 실행 확인용이며 전체 validation 결과가
아닙니다. 아주 작은 Delta/TV는 반복한 정상 평가의 변동폭과 여러 intervention seed에서 다시
확인합니다.

이 검사는 **이미 학습된 checkpoint가 현재 그래프에 의존하는지** 확인합니다. 관계를 없앴을 때
성능이 떨어져도 GNN이 더 좋은 모델이라는 증명은 아닙니다. 구조를 확정하기 전에는 같은 분할,
특징, loss, seed와 학습 예산으로 원래 그래프·node-only·재배선 모델을 각각 새로 학습하고
여러 seed로 비교해야 합니다. 그 결정까지 validation에서 하고 2026 test는 마지막 평가에만
사용합니다.

### 5-2. 이번 범위의 1~5단계: 크기와 그래프 해상도 확인

이번 작업은 이미 끝난 **단일 seed × 6조건 × 64 hidden × 2 layer** 결과를 다시 돌리지
않습니다. 아래 1~5단계까지만 실행하며 multi-seed 확장은 포함하지 않습니다.

1. 기존 v5 그래프의 노드·관계 수, route별 복원 관계 이벤트의 edge 압축률, 질의 노드 고립률·차수,
   1-hop/2-hop 도달 범위를 감사합니다.

~~~bash
cpv26 kbo-graph-audit \
  --dataset var/datasets/kbo_graph_2001_2026_v5 \
  --end-date 2025-12-31 \
  --output var/reports/kbo_graph_audit_v5.json
~~~

이 감사는 그래프 자체의 크기와 연결 범위를 재는 정적 검사입니다. 5-1절의
`relgnn-graph-diagnose`처럼 학습된 checkpoint의 예측 의존도를 재평가하는 명령이 아닙니다.
`--end-date`는 validation 종료일로 지정해야 하며 held-out test 날짜를 포함하면 안 됩니다.

2. 완료된 단일-seed 64×2 suite의 `full`과 `node_only` validation 결과를 그대로 읽습니다.
3. 같은 seed·분할·loss·optimizer·epoch 예산으로 **128×3의 `full`과 `node_only`만** 새로
   학습해 용량 증가 전후의 `node_only - full` validation 차이를 비교합니다.

~~~bash
cpv26 relgnn-capacity-compare \
  --baseline-suite var/runs/relgnn_ablations/kbo_2001_2024_v5 \
  --baseline-seed 2026 \
  --dataset var/datasets/kbo_graph_2001_2026_v5 \
  --output var/runs/relgnn_capacity/kbo_2001_2024_v5_64x2_vs_128x3
~~~

`--baseline-seed`는 기존 matched suite에 선언되고 완료된 seed 하나를 선택합니다. suite 전체가
실패 상태여도 선택한 seed의 `full`과 `node_only`가 완전히 검증되면 재사용하지만, 실행 중인
suite는 거부합니다. 이 명령은 그 안의 64×2 모델을 재학습하지 않고, 선택한 seed로 128×3
두 run만 추가합니다. 데이터 fingerprint,
각 용량 안의 `full`/`node_only` 초기화, 학습 예산과 split lineage가 맞지 않거나 test 봉인을
증명하지 못하면 중단합니다.

실패한 suite가 최상위 `runs` 기록을 쓰기 전에 중단됐더라도 로컬
`seed-<seed>/{full,node_only}`의 완료된 training report와 `best.pt`를 검증해 복구합니다.
`last.pt`가 남아 있으면 그 checkpoint와 hash도 엄격히 검증하지만, 정리되어 없어도 복구할 수 있습니다.
기존 validation artifact가 없을 때만 비교 출력 디렉터리에서 `best.pt`를 validation으로 다시
평가하며, 원본 suite는 수정하지 않고 test split은 로드하지 않습니다.

4. 관계를 한 쌍의 90일 집계 edge에만 압축하는 한계를 확인하기 위해 graph-vNext를 별도
   디렉터리에 만듭니다. DB schema는 5를 유지하고 graph cache version만 6이 됩니다.

~~~bash
cpv26 kbo-graph-build \
  --output var/datasets/kbo_graph_2001_2026_vnext \
  --start-date 2001-01-01 \
  --end-date 2026-07-26 \
  --graph-schema vnext
~~~

vNext는 과거 경기와 당일 질의 경기의 `game` 노드를 추가합니다. 확인된 **과거**
타자/투수 출전만 player-game 관계로 연결하고, 과거·현재 경기의 홈/원정 팀은
team-game 관계로 연결합니다. 현재 경기 노드와 team-game 관계에는 대진·예정 시작시각만 있으며
점수·결과·실제 출전은 없습니다. 시점이 명시된 사전 라인업·선발 원천이 없으므로 현재
player-game, 라인업, 선발 관계를 추정하거나 만들지 않습니다.

5. vNext에서 같은 초기화와 예산의 `full`과 `node_only`를 seed 하나로 한 번씩만 학습합니다.

~~~bash
cpv26 relgnn-pair-train \
  --dataset var/datasets/kbo_graph_2001_2026_vnext \
  --output var/runs/relgnn_pairs/kbo_2001_2024_vnext_128x3_seed2026 \
  --train-start-year 2001 \
  --train-end-year 2024 \
  --validation-year 2025 \
  --test-year 2026 \
  --chronological \
  --device cuda:0 \
  --epochs 30 \
  --batch-days 8 \
  --hidden-dim 128 \
  --layers 3 \
  --heads 4 \
  --workers 0 \
  --max-pa-per-day 0 \
  --max-edges-per-route 0 \
  --amp auto \
  --seed 2026
~~~

두 비교 명령은 checkpoint 선택과 조건 비교에 validation만 사용하며, 지정한 test 시즌의
graph와 label을 로드하거나 평가하지 않습니다. 단일 seed 결과는 용량·schema 변경의 다음
실험 방향을 고르는 screening일 뿐 seed 간 분산이나 안정성의 근거가 아닙니다. 이번 범위에는
seed 추가와 multi-seed 결론이 없으며 test는 계속 봉인합니다.

### 5-3. 기존 여섯 조건 전체 비교 (이번 범위에서는 사용하지 않음)

고정 checkpoint 진단 다음에는 여섯 조건을 처음부터 다시 학습합니다. 아래 명령은 seed 3개마다
조건 6개, 즉 전체 학습 18회를 순서대로 실행하므로 단일 학습보다 훨씬 오래 걸립니다. 먼저
`--seed 2026` 하나와 `--max-days-per-split`으로 실행 경로만 확인할 수 있지만 그 결과는
전체 기간 비교가 아닙니다.

이 절은 기존 전체 ablation 재현용으로 보존합니다. 위 1~5단계에서는 이 명령을 실행하지
않으며, 완료된 단일-seed suite를 2~3단계의 읽기 전용 baseline으로만 사용합니다.

~~~bash
cpv26 relgnn-ablation-train \
  --dataset var/datasets/kbo_graph_2001_2026_v5 \
  --suite-dir var/runs/relgnn_ablations/kbo_2001_2024_v5 \
  --train-start-year 2001 \
  --train-end-year 2024 \
  --validation-year 2025 \
  --test-year 2026 \
  --chronological \
  --device cuda:0 \
  --epochs 30 \
  --batch-days 8 \
  --workers 0 \
  --max-pa-per-day 0 \
  --max-edges-per-route 0 \
  --amp auto \
  --seed 2026 \
  --seed 2027 \
  --seed 2028 \
  --graph-control-seed 2026
~~~

짧은 실행 확인은 full suite 디렉터리에 쓰지 않습니다. 아래처럼 이름에 `_smoke`를 붙인
별도 디렉터리를 사용해야 `max_days_per_split` 설정으로 본 실험 manifest를 오염시키지 않습니다.

~~~bash
cpv26 relgnn-ablation-train \
  --dataset var/datasets/kbo_graph_2001_2026_v5 \
  --suite-dir var/runs/relgnn_ablations/kbo_2001_2024_v5_smoke \
  --train-start-year 2001 \
  --train-end-year 2024 \
  --validation-year 2025 \
  --test-year 2026 \
  --chronological \
  --device cuda:0 \
  --epochs 1 \
  --batch-days 8 \
  --workers 0 \
  --max-pa-per-day 0 \
  --max-edges-per-route 0 \
  --max-days-per-split 3 \
  --amp auto \
  --seed 2026 \
  --graph-control-seed 2026
~~~

| 조건 | 관계 메시지 설정 | 그래프 입력 |
|---|---|---|
| `full` | 모든 layer에서 모든 관계·방향, normalization 없음 | 원본 |
| `normalized` | `full` + parameter 없는 route message layer normalization | 원본 |
| `staged` | 첫 layer는 PA 양방향, 타자→팀, 팀→투수, 홈↔원정 팀만; 이후 core | 원본 |
| `core` | 모든 layer에서 PA 양방향과 타자↔팀만 | 원본 |
| `node_only` | 관계 메시지 전달만 제거 | 원본 |
| `rewired` | `full`, normalization 없음 | 날짜·관계별 endpoint 고정 재배선 |

같은 seed의 여섯 모델은 초기 `state_dict` SHA-256과 parameter 수가 완전히 같은지 먼저
검사합니다. `node_only`도 사용하지 않는 route parameter를 그대로 보유해 이 비교에서 모델
parameter 예산을 바꾸지 않습니다. 따라서 `node_only`는 순수 tabular 모델이 아닙니다.
그래프에서 만든 node/role feature는 그대로 남고 **관계 message passing만** 제거됩니다.

모든 조건은 early stopping 없이 같은 epoch와 optimizer-step 시도 예산을 받습니다. FP16
overflow로 실제 갱신을 건너뛴 수는 별도 기록합니다. `rewired`의 변환 seed는 학습 seed,
epoch, batch 순서와 무관하며 같은 날짜의 train·validation에서 같은 방식으로 적용됩니다.
변환 mode·seed·algorithm version과 fingerprint는 config/checkpoint/report에 저장되고 재개 시
일치하지 않으면 중단합니다.
해석된 precision, PyTorch/CUDA runtime, GPU 이름·compute capability·메모리도 suite manifest에
고정합니다. 같은 `cuda:0` 문자열이어도 다른 GPU나 runtime이면 기존 suite 재개를 거부합니다.

각 `best.pt`는 **2025 validation으로만** 다시 평가하며 seed별 full 대비 paired delta와
selection loss, Match/LiveHit/PA의 log loss·accuracy·ECE·Brier mean/std를
`matched_retraining_report.json`에 저장합니다. 이 명령은 2026 test graph나 label을 로드하거나
평가하지 않습니다. 구조를 선택한 뒤 선택한 run의 `best.pt`만 6절에서 test 평가합니다.

이미 끝난 suite에서 어떤 task가 차이를 만들었는지 확인할 때는 재학습하지 않고 아래 명령을
실행합니다.

~~~bash
cpv26 relgnn-ablation-report \
  --suite-dir var/runs/relgnn_ablations/kbo_2001_2024_v5
~~~

이 명령은 저장된 suite/학습/validation JSON만 읽으며 checkpoint, graph, test를 로드하지
않습니다. 여섯 task의 weighted contribution과 Match/LiveHit/PA 지표, best epoch, 마지막 epoch,
마지막 5 epoch 평균을 함께 표시합니다. route 효과는 `core-full`처럼 normalization까지 섞인
차이가 아니라 `core-normalized`, normalization 효과는 `normalized-full`, message 제거와 endpoint
재배선은 각각 `node_only-full`, `rewired-full` contrast로 확인합니다. task별 validation sample 수,
history의 best epoch와 저장된 best epoch, selection-loss 구성이나 test 봉인 정보가 맞지 않으면
수치를 출력하지 않고 중단합니다.

중단되면 같은 `--suite-dir`과 동일 옵션으로 다시 실행합니다. 완료 조건은 건너뛰고
`last.pt`가 있는 조건만 재개합니다. `--epochs`는 늘릴 수 있지만 device, batch, AMP,
worker, accumulation, 분할, 모델, sampling, loss와 graph-control 설정은 바꿀 수 없습니다.
기존 seed는 제거하거나 재정렬할 수 없고, 기존 순서를 유지한 채 뒤에 새 seed만 추가할 수
있습니다.
MIG 10GB에서 `--batch-days 8`이 OOM이면 실행을 시작하기 전에 4, 그다음 2로 낮춥니다.
이미 child checkpoint가 생긴 suite의 batch days는 바꿀 수 없으므로, OOM 뒤 값을 바꿀 때는
새 `--suite-dir`에서 여섯 조건을 같은 값으로 다시 시작합니다.

처음 같은 suite를 `--seed 2026` 하나로 끝낸 뒤 seed를 추가할 때는 기존 seed를 빼지 않고
아래처럼 같은 순서의 앞부분으로 유지합니다. 그러면 2026의 완료 run 6개는 재사용하고
2027·2028 run만 추가합니다. 기존 seed 제거·재정렬은 거부됩니다.

~~~bash
cpv26 relgnn-ablation-train \
  --dataset var/datasets/kbo_graph_2001_2026_v5 \
  --suite-dir var/runs/relgnn_ablations/kbo_2001_2024_v5 \
  --train-start-year 2001 \
  --train-end-year 2024 \
  --validation-year 2025 \
  --test-year 2026 \
  --chronological \
  --device cuda:0 \
  --epochs 30 \
  --batch-days 8 \
  --workers 0 \
  --max-pa-per-day 0 \
  --max-edges-per-route 0 \
  --amp auto \
  --seed 2026 \
  --seed 2027 \
  --seed 2028 \
  --graph-control-seed 2026
~~~

## 6. 2026 부분 시즌 test 평가

5-3절의 기존 전체 비교를 실행했다면 `matched_retraining_report.json`의 **seed aggregate validation**으로
variant를 선택합니다. 그 뒤 가장 validation이 좋았던 seed까지 고르면 seed cherry-picking이
되므로 checkpoint seed는 결과를 보기 전에 고정합니다. 아래는 seed 2026을 사전 고정했고
aggregate 결과에서 `normalized`를 선택했다는 실행 예입니다. 사전 정의했다면 모든 seed를
각각 test하거나 ensemble하는 별도 protocol도 가능하지만 이 CLI가 자동 ensemble하지는 않습니다.
child checkpoint 경로 형식은
`var/runs/relgnn_ablations/kbo_2001_2024_v5/seed-<seed>/<selected_variant>/best.pt`입니다.
보고서에서 다른 variant가 선택됐다면 `SELECTED_VARIANT`를 그 값으로 바꿉니다. standalone
5절 경로를 그대로 평가하면 matched 비교에서 선택한 모델의 test가 아닙니다.

~~~bash
SELECTED_SEED=2026
SELECTED_VARIANT=normalized
cpv26 relgnn-evaluate \
  --checkpoint "var/runs/relgnn_ablations/kbo_2001_2024_v5/seed-${SELECTED_SEED}/${SELECTED_VARIANT}/best.pt" \
  --split test \
  --device cuda:0
~~~

평가 결과는 checkpoint 폴더 아래의 새 evaluations/test-<run-id>/에 저장됩니다.
터미널에 출력되는 실제 경로를 확인합니다.

~~~text
evaluations/test-<run-id>/
├── metrics.json
├── match_predictions.parquet
├── live_hit_predictions.parquet
└── pa_predictions.parquet
~~~

평가 연도는 checkpoint에 저장한 분할에서 읽습니다. 2026 자료는 부분 시즌이므로
전체 시즌 성능이라고 표시하지 않습니다. test 결과를 보고 설정을 골랐다면 그 사실도 기록합니다.

저장된 학습 요약과 가장 최근 test 평가 결과만 다시 보려면 아래 명령을 실행합니다.
학습이나 평가를 새로 실행하지 않습니다.

~~~bash
SELECTED_SEED=2026
SELECTED_VARIANT=normalized
python scripts/show_relgnn_results.py \
  --run-dir "var/runs/relgnn_ablations/kbo_2001_2024_v5/seed-${SELECTED_SEED}/${SELECTED_VARIANT}"
~~~

GPU 사용률이 낮을 때는 [병목 진단](docs/GPU_BOTTLENECK_DIAGNOSIS.md)을 사용합니다.
기존 학습·평가가 끝난 뒤 같은 MIG가 비어 있을 때만 실행하며, 원래 checkpoint는 변경하지 않습니다.

최적화 전후의 속도만 비교하려면 기존 `last.pt`로 아래를 실행합니다. 전체 학습을 다시
할 필요는 없습니다. 기존과 같은 worker 2개를 유지합니다.

~~~bash
python scripts/profile_relgnn.py \
  --run-dir var/runs/relgnn/kbo_2001_2024_v5 \
  --device cuda:0 \
  --device-idle \
  --workers 2 \
  --compare-optimizations \
  --repeats 3 \
  --trace-steps 0
~~~

양방향 관계의 event/time 인코딩 재사용과 dtype별 묶음 CUDA 전송을 끈 실행/켠 실행을
같은 조건으로 비교합니다. 두 최적화는 기존 학습·평가 명령에도 기본 적용되며 모델 설정,
데이터와 checkpoint를 바꾸지 않습니다. CPU 등가성 검사와 A100 속도 검증은 별개이며,
실제 가속량은 아직 측정하지 않았습니다. 구간별 `Optimization comparison`과 새 `report.json`을
확인합니다. [측정 조건과 결과 읽기](docs/GPU_BOTTLENECK_DIAGNOSIS.md#최적화-전후-비교)를 참고합니다.

## 7. 학습 재개

학습 프로세스가 종료됐다면 마지막으로 저장된 last.pt에서 재개합니다.
best.pt는 재개용으로 사용하지 않습니다.
새 터미널에서는 먼저 [환경을 활성화](#9-환경-활성화와-코드-업데이트)합니다.

~~~bash
cpv26 relgnn-train \
  --dataset var/datasets/kbo_graph_2001_2026_v5 \
  --resume var/runs/relgnn/kbo_2001_2024_v5/last.pt \
  --train-start-year 2001 \
  --train-end-year 2024 \
  --validation-year 2025 \
  --test-year 2026 \
  --chronological \
  --device cuda:0 \
  --epochs 50 \
  --batch-days 8 \
  --max-pa-per-day 0 \
  --max-edges-per-route 0 \
  --amp auto
~~~

--run-dir를 생략하면 checkpoint의 부모 폴더를 사용합니다. --epochs 50은 추가
50 epoch가 아니라 전체 목표 50 epoch입니다. 30 epoch까지 저장됐다면 최대 20 epoch를
더 실행합니다. 이 예시는 5절의 v5 run을 재개하며 같은 데이터셋과 모델 설정을 유지합니다.
이전 버전의 run은 그 run을 만들 때 사용한 그래프 경로와 설정으로만 재개합니다.
이 명령은 같은 실행의 중단 지점을 잇습니다. 새 경기 자동 수집·추가 학습은 하지 않습니다.
재개는 마지막 checkpoint에 저장된 지점부터이며, 이후 저장되지 않은 batch의 진행은
복구하지 않습니다. 기존 프로세스가 종료됐는지 확인한 뒤 재개합니다.

## 8. VRAM 부족

5절은 `--batch-days 8`입니다. 메모리가 부족하면 먼저 4, 이어서 2로 낮추고,
그래도 부족하면 `--hidden-dim 32`처럼 모델 크기를 줄여 새 run 이름으로 실행합니다.

CLI의 `--max-pa-per-day`와 `--max-edges-per-route` 기본값은 `0`으로, 질의·관계를
전부 사용합니다. 양수를 직접 지정한 경우에만 표본화하며, 그 제한을 학습 보고서에 기록합니다.
기존 제한 학습을 재개할 때는 원래 값(예: 128, 20000)을 명시합니다.
--amp auto는 GPU가 지원하는 혼합정밀도를 선택합니다. 데이터 로딩 프로세스가
문제라면 --workers 0으로 확인할 수 있습니다.

다중 GPU·분산 학습은 지원하지 않습니다. 2001년부터의 전체 학습·평가 메모리와 성능은
실제 실행 결과로 확인해야 합니다.

## 9. 환경 활성화와 코드 업데이트

설치 후 새 터미널에서 작업할 때:

~~~bash
cd ~/projects/cpv26-predictor
conda activate cpv26
source scripts/activate.sh
~~~

코드 업데이트는 학습이 끝난 뒤 진행합니다.

~~~bash
cd ~/projects/cpv26-predictor
git pull --ff-only
conda activate cpv26
source scripts/activate.sh
cpv26 gpu-check --device cuda:0
~~~

이번 v5 갱신에는 패키지 재설치가 필요 없습니다. 기존 v3/v4 데이터는
[4-1절](#4-1-기존-v3v4에서-v5로-갱신)의 캐시 재수입부터 이어갑니다.
`.env`, 기존 그래프와 checkpoint는 그대로 둡니다. Conda로 처음 전환한다면
[Conda 설치](#2-conda-설치)를 진행합니다.

전체 코드 검사는 다음과 같습니다.

~~~bash
conda activate cpv26
bash scripts/check.sh
~~~

Python compile, Ruff, strict mypy, pytest, 패키지 충돌 검사를 실행합니다.
검사에서 만드는 임시 DB·모델·입력 파일은 실제 학습 데이터가 아니며, pytest가 종료되면
보관하지 않도록 설정되어 있습니다. `--basetemp var/...`로 테스트 출력을 실행 데이터 폴더에 두지 않습니다.
GitHub CI도 전용 Conda 환경의 Python 3.12에서 `base` profile 설치·CLI 도움말과
CPU PyTorch가 있는 neural 테스트를 검사합니다. CI에서 전체 원천 데이터를 다운로드하거나 NVIDIA GPU 학습을
수행하지는 않습니다.
여기서 `base`는 runtime만 설치하는 스크립트 profile 이름이며 **Conda의 `(base)` 환경이 아닙니다.**

### 설치·실행 후 저장소를 private으로 되돌리기

실행 환경에서 필요한 설치·학습·평가를 마친 뒤 저장소 소유자가 GitHub 저장소의
Settings → General → Danger Zone → Change repository visibility에서 private으로
직접 바꿉니다. 이 프로젝트가 자동으로 비공개 전환하지는 않습니다.

private으로 바꿔도 실행 환경에 이미 받은 소스·설치 환경·데이터·checkpoint는 그대로이며,
이 파일들을 사용하는 학습·재개·평가를 계속할 수 있습니다. 이후 새 clone이나
git pull에는 저장소 접근 권한과 GitHub 인증이 필요합니다. HTTPS에서는 저장소 읽기
권한이 있는 personal access token 또는 설정된 credential helper를 사용하고,
토큰을 clone URL이나 문서에 넣지 않습니다.

주의: public인 동안 다른 사람이 받은 복제본은 private으로 되돌려도 회수되지 않습니다.

## 선택 사항: CPU에서 작은 동작 검사만 하기

GPU 주 학습과 별개의 코드 검사입니다. 이 검사도 4절에서 만든 실제 graph dataset을
사용하므로 데이터 다운로드·적재·graph build가 먼저 끝나 있어야 합니다.
GPU용 `cpv26`의 CUDA PyTorch를 바꾸지 않도록 별도 Conda 환경 `cpv26-cpu`를 만듭니다.
처음 한 번만 생성하며, 이미 있으면 생성 명령을 건너뛰고 활성화합니다.

~~~bash
conda env create -f environment.yml -n cpv26-cpu
conda activate cpv26-cpu
command -v python
python --version
bash scripts/setup.sh ml-cpu
~~~

`Conda environment ready: ...`를 확인한 뒤 실행합니다. `.env`는 2-4절에서 만든 파일을 사용합니다.

~~~bash
source scripts/activate.sh
cpv26 relgnn-train \
  --dataset var/datasets/kbo_graph_2001_2026_v5 \
  --run-dir var/runs/relgnn/kbo_2001_2024_v5_cpu_smoke \
  --train-start-year 2001 \
  --train-end-year 2024 \
  --validation-year 2025 \
  --test-year 2026 \
  --chronological \
  --device cpu \
  --amp off \
  --workers 0 \
  --epochs 1 \
  --batch-days 1 \
  --max-days-per-split 3
~~~

날짜를 줄인 1 epoch 검사는 CUDA 검증이나 전체 시즌 학습 성능을 대신하지 않습니다.
GPU 학습 명령에서 --device cpu로 자동 전환되는 일도 없습니다.
GPU 학습으로 돌아갈 때는 `conda activate cpv26` 후 `source scripts/activate.sh`를 실행합니다.

## 선택 사항: CatBoost 기준선과 비교하기

RelGNN과 별개의 tabular 기준선이 필요할 때만 설치합니다.

~~~bash
conda activate cpv26
bash scripts/setup.sh tabular
~~~

`Conda environment ready: ...`를 확인한 뒤 실행합니다.

~~~bash
source scripts/activate.sh
cpv26 kbo-match-evaluate
cpv26 kbo-live-hit-evaluate
~~~

두 명령은 CatBoost를 학습하며 RelGNN checkpoint를 만들지 않습니다.
승부예측은 경기 패·무·승, 라이브 히트 기준선은 실제 PA가 기록된 선수의 1안타 이상
확률을 평가합니다. 상세한 시즌 경계와 결과 경로는
[CatBoost 비교 실험 설명](docs/KBO_BASELINE.md)에 있습니다.

## 자주 발생하는 오류

### cpv26: command not found / ModuleNotFoundError

~~~bash
cd ~/projects/cpv26-predictor
conda activate cpv26
source scripts/activate.sh
~~~

설치 자체가 실패했다면 먼저 setup 오류를 해결합니다.

### CUDA-enabled PyTorch is missing / CUDA kernel check failed

`nvidia-smi` 출력과 설치한 PyTorch 버전을 확인하고 [CUDA 설치](#2-3-cuda-pytorch와-프로젝트-설치)를
다시 진행합니다. 설치 후 다음 검사까지 통과해야 합니다.

~~~bash
cpv26 gpu-check --device cuda:0
~~~

### KBO source file not found / graph dataset not found

[4절의 데이터·그래프 생성 명령](#4-kbo-데이터와-그래프-생성)을 실행합니다.
다른 경로를 사용했다면 각 명령의 --help에서 source/dataset 옵션을 확인하고,
학습 명령도 같은 그래프 경로를 가리키도록 맞춥니다.

### SHA-256 mismatch

오류 파일명과 SOURCE.json의 revision을 확인하고 해당 원천의
`kbo-history-fetch` 또는 `kbo-fetch`를 다시 실행합니다.
다른 revision의 파일을 검증 없이 같은 파일명으로 바꾸지 않습니다.

### Conda가 없거나 conda activate가 동작하지 않음

Conda가 설치돼 있지만 활성화되지 않을 때:

~~~bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cpv26
~~~

`cpv26` 환경이 없으면 [환경 생성](#2-1-환경-생성)을 먼저 진행합니다.
Conda 자체가 없다면 [공식 설치 안내](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)를 참고합니다.

### Python 경로가 다름 / 중첩된 환경 / 지원하지 않는 Python

다른 Python 가상환경을 비활성화한 뒤 Conda `cpv26`을 활성화합니다.
`command -v python`, `python --version`, `echo "$CONDA_PREFIX"`로 경로와 버전을 확인합니다.
설치 스크립트는 중첩된 가상환경이나 Conda 환경과 다른 Python을 거부합니다.

## 생성 파일과 라이선스

`environment.yml`은 Git에 포함하며, `.env`, `var/` 아래의 실행 데이터와 모델은
Git에서 제외됩니다. 예전 `.venv` 폴더도 추적하지 않으며 실행 경로에서 사용하지 않습니다.
원본 아카이브와 학습 모델을 GitHub에 재배포하지 않습니다. 아카이브에서 빠진 10경기는
공식 기록으로 확인한 최종 점수와 출처만 별도 보완 자료로 포함합니다.

- 원천 출처·라이선스·품질 검사: [docs/KBO_BASELINE.md](docs/KBO_BASELINE.md)
- RelGNN 그래프·GPU 학습의 기술 설명: [docs/GPU_TRAINING.md](docs/GPU_TRAINING.md)
- 상세 설계와 데이터·학습 계약: [docs/HANDOFF.md](docs/HANDOFF.md)
- 소프트웨어 라이선스와 비제휴 고지: [LICENSE.md](LICENSE.md)

외부 코드 검토용 전체 소스 문서가 필요할 때만 다음을 실행합니다.

~~~bash
python scripts/build_code_summary.py
~~~

생성된 code_summary.md는 Git에 올라가지 않습니다. 이 프로젝트는 KBO나 게임
개발·배급·운영사와 제휴하거나 이들의 보증을 받은 프로젝트가 아닙니다.
