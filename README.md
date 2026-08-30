# CPV26 Predictor

공개 KBO 데이터로 타석 결과, 경기 승무패, 선수별 PA·안타 분포를 학습하는
RelGNN 프로젝트입니다. 설치 명령은 Linux/Bash 기준이며, Python은 Conda로 관리합니다.

V26 계정별 당일 추천 기능은 아직 구현되지 않았습니다.

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

~~~bash
cpv26 kbo-fetch
cpv26 kbo-import
cpv26 db-check
cpv26 kbo-graph-build
~~~

기본 대상은 2023·2024·2025 정규시즌입니다. 생성 위치는 다음과 같습니다.

| 위치 | 내용 |
|---|---|
| var/datasets/kbo_playbyplay/v0/ | 내려받은 원천 Parquet와 SOURCE.json |
| var/cpv26.duckdb | 정규화한 선수·팀·경기·타석 DB |
| var/reports/kbo_import.json | 적재 결과와 원천 품질 검사 |
| var/datasets/kbo_graph/ | 실제 날짜별 RelGNN graph dataset/cache |

kbo-fetch는 고정된 공개 파일과 SHA-256을 확인합니다. KBO·NAVER·Statiz 사이트를
새로 크롤링하거나 실시간으로 갱신하는 명령이 아닙니다. 같은 revision의 파일과
적재 행은 재실행 시 재사용합니다.

kbo-graph-build는 대상 날짜 이전의 최대 90일 관계를 사용해 날짜별 그래프를 만듭니다.
그날의 경기 결과는 정답이며, 같은 날의 결과를 과거 관계 입력으로 사용하지 않습니다.
원본의 누락·불완전한 라벨은 임의의 안타나 아웃으로 채우지 않습니다. 다운로드·적재
보고서와 [원천 데이터 설명](docs/KBO_BASELINE.md)을 함께 확인하세요.
원천의 당시 발표 시각은 재구성된 값이므로 이 실험은 retrospective benchmark이며,
실시간으로 수집한 당시 정보만으로 재현한 V26 운영 검증과는 구분합니다.

~~~bash
ls -lah var/datasets/kbo_graph
~~~

## 5. RelGNN 학습

아래 run 이름은 이번 실행을 구분하는 폴더 이름입니다. 새 실험에는 다른 이름을
사용하고, 기존 실행을 이어갈 때만 --resume를 사용합니다.

~~~bash
cpv26 relgnn-train \
  --dataset var/datasets/kbo_graph \
  --run-dir var/runs/relgnn/kbo_2023_2024_v1 \
  --device cuda:0 \
  --epochs 30 \
  --batch-days 2 \
  --amp auto
~~~

학습·평가 시즌은 고정해서 구분합니다.

| 시즌 | 역할 |
|---|---|
| 2023 | 가중치를 학습하는 train |
| 2024 | validation, early stopping, best.pt 선택 |
| 2025 | 다음 절에서 명시적으로 실행하는 test |

2025 test는 학습 중 자동 실행하지 않습니다. 기본 early stopping은 validation이
6 epoch 동안 개선되지 않으면 멈추며, --patience 0은 이를 끕니다. --epochs는 최대
전체 epoch 수이므로 early stopping이 먼저 끝낼 수 있습니다.

기본 모델은 hidden dimension 64, 관계 layer 2개, attention head 4개입니다.
기본 learning rate는 0.0003, weight decay는 0.0001입니다. 서로 다른 정답을 쓰는
세 Head가 공유 RelGNN backbone을 학습합니다.

~~~text
실제 KBO 기록 → 날짜 이전 관계 graph → 공유 RelGNN
                                      ├─ PA: 타석 결과 10개 분류
                                      ├─ Match: 홈팀 패·무·승
                                      └─ Live Hit: 선수별 PA·안타 수 결합 분포
                         2024 validation → best.pt
                         별도 2025 test → 평가 보고서와 예측 Parquet
~~~

현재 Live Hit 학습은 완료된 관측 PA가 1개 이상인 선수-경기 기록에 조건부인 분포입니다.
기록에 없는 미출장 후보까지 포함한 무조건부 출전확률이나 V26 계정 점수와 같지 않습니다.
같은 경기 선수들의 joint scenario와 게임 보너스를 결합하는 최종 추천도 별도입니다.
PA 보조 task는 해당 타석 직전 상태를 사용하고, Match·Live Hit는 날짜 이전 입력을
사용합니다.

학습 결과는 지정한 run 폴더에 저장됩니다.

~~~text
var/runs/relgnn/kbo_2023_2024_v1/
├── config.json
├── history.jsonl
├── last.pt
├── best.pt
└── training_report.json
~~~

history.jsonl에서 진행 기록을, training_report.json에서 학습 요약을 확인합니다.
best.pt는 validation으로 선택한 평가용 모델이고, last.pt는 학습 재개용입니다.
GPU를 사용할 수 없으면 오류로 끝나며 CPU 학습으로 자동 전환하지 않습니다.

## 6. 2025 test 평가

학습이 끝난 뒤 best.pt를 명시해 실행합니다.

~~~bash
cpv26 relgnn-evaluate \
  --checkpoint var/runs/relgnn/kbo_2023_2024_v1/best.pt \
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

2025 결과를 본 뒤 모델 설정을 바꾸고 같은 2025 성능을 다시 고르면 독립적인 최종
테스트가 아닙니다. 설정 선택은 2024 validation으로 하고 test 결과는 따로 기록합니다.

## 7. 학습 재개

학습 프로세스가 종료됐다면 마지막으로 저장된 last.pt에서 재개합니다.
best.pt는 재개용으로 사용하지 않습니다.
새 터미널에서는 먼저 [환경을 활성화](#9-환경-활성화와-코드-업데이트)합니다.

~~~bash
cpv26 relgnn-train \
  --dataset var/datasets/kbo_graph \
  --resume var/runs/relgnn/kbo_2023_2024_v1/last.pt \
  --device cuda:0 \
  --epochs 50 \
  --batch-days 2 \
  --amp auto
~~~

--run-dir를 생략하면 checkpoint의 부모 폴더를 사용합니다. --epochs 50은 추가
50 epoch가 아니라 전체 목표 50 epoch입니다. 30 epoch까지 저장됐다면 최대 20 epoch를
더 실행합니다. 재개할 때는 같은 데이터셋과 모델 설정을 유지합니다.
재개는 마지막 checkpoint에 저장된 지점부터이며, 이후 저장되지 않은 batch의 진행은
복구하지 않습니다. 기존 프로세스가 종료됐는지 확인한 뒤 재개합니다.

## 8. VRAM 부족

먼저 한 batch의 날짜 수를 줄여 실행합니다.

~~~bash
cpv26 relgnn-train \
  --dataset var/datasets/kbo_graph \
  --run-dir var/runs/relgnn/kbo_2023_2024_small_batch \
  --device cuda:0 \
  --epochs 30 \
  --batch-days 1 \
  --max-pa-per-day 128 \
  --amp auto
~~~

--max-pa-per-day의 기본값은 128이며 훈련 PA query만 표본화합니다. validation과
test PA는 전부 평가하므로 이 옵션이 평가 메모리까지 제한하지는 않습니다.
--amp auto는 GPU가 지원하는 혼합정밀도를 선택합니다. 데이터 로딩 프로세스가
문제라면 --workers 0으로 확인할 수 있습니다.

다중 GPU·분산 학습은 지원하지 않습니다. A100 MIG 10GB에서 전체 학습의 최대 메모리
사용량은 아직 측정하지 않았습니다. `--batch-days 1`부터 시험하고 학습·평가의
peak memory를 각각 확인합니다.

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
~~~

Conda로 처음 전환한다면 [Conda 설치](#2-conda-설치)를 진행합니다.
이미 설치했다면 다음 명령으로 패키지를 갱신합니다. `.env`와 `var/`는 그대로 둡니다.

~~~bash
conda activate cpv26
bash scripts/setup.sh ml-cuda
~~~

`Conda environment ready: ...`를 확인한 뒤 실행합니다.

~~~bash
source scripts/activate.sh
cpv26 gpu-check --device cuda:0
cpv26 db-init
cpv26 db-check
~~~

기존 CUDA PyTorch가 정상이라면 이 명령은 해당 설치를 보존합니다. 새 graph 생성에
영향을 주는 코드가 바뀌었다면 새 데이터셋과 새 run으로 실험하고 기존 checkpoint와
섞지 않습니다.

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
  --dataset var/datasets/kbo_graph \
  --run-dir var/runs/relgnn/cpu_validation \
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

~~~bash
cpv26 kbo-fetch
cpv26 kbo-import
cpv26 kbo-graph-build
~~~

다른 경로를 사용했다면 각 명령의 --help에서 source/dataset 옵션을 확인하고 같은
데이터셋을 가리키도록 맞춥니다.

### SHA-256 mismatch

오류 파일명과 SOURCE.json의 revision을 확인하고 kbo-fetch를 다시 실행합니다.
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
원본 KBO 파일과 학습 모델을 GitHub에 재배포하지 않습니다.

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
