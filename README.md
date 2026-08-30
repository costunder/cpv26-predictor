# CPV26 Predictor

컴프야 V26의 승부예측과 라이브 히트를 연구하는 KBO 예측 프로젝트입니다.
공개 KBO 데이터를 내려받아 DB에 적재하고, 두 목적에 맞는 모델을 각각 학습·평가할 수
있습니다.

이 문서의 명령은 모두 **MobaXterm으로 접속한 Linux 서버의 Bash 터미널**에서
실행합니다. Windows의 .venv나 DB를 서버에 복사할 필요는 없습니다.

현재 실행 명령은 CatBoost 기준선 실험입니다. RelGNN 모델·학습 부품도 저장소에 있지만,
아래 명령이 RelGNN을 학습하는 것은 아닙니다. 당일 V26 후보·포지션·도감·선택률을 받아
계정 추천을 만드는 명령과도 구분합니다.

## 1. 서버 접속과 프로젝트 받기

MobaXterm에서 Session → SSH로 서버에 접속합니다. 아래 조건이 필요합니다.

- Linux와 Bash
- Python 3.10~3.12
- Git과 인터넷 연결
- 이 비공개 GitHub 저장소에 대한 접근 권한

먼저 확인합니다.

~~~bash
git --version
python3 --version
python3 -m venv --help >/dev/null
~~~

Git 또는 venv가 없는 Ubuntu 서버에서는 관리 권한이 있는 사용자가 설치합니다.

~~~bash
sudo apt-get update
sudo apt-get install -y git python3-venv
~~~

서버에 GitHub SSH key가 등록돼 있다면 다음처럼 받습니다.

~~~bash
mkdir -p ~/projects
cd ~/projects
git clone git@github.com:costunder/cpv26-predictor.git
cd cpv26-predictor
~~~

HTTPS 인증을 사용하는 경우 clone 명령만 바꿉니다.

~~~bash
git clone https://github.com/costunder/cpv26-predictor.git
~~~

인증이 어려우면 GitHub에서 ZIP을 받아 압축을 풀고 MobaXterm 왼쪽 SFTP 패널로
~/projects/cpv26-predictor에 소스만 올려도 됩니다.

## 2. 학습 환경 설치

프로젝트 루트에서 실행합니다.

~~~bash
bash scripts/setup.sh tabular
if [ ! -f .env ]; then cp .env.example .env; fi
source scripts/activate.sh
~~~

tabular 프로필은 가상환경, 프로젝트, 개발 검사 도구와 CatBoost를 설치합니다.
이번 실험에는 GPU나 PyTorch가 필요하지 않습니다.

서버의 Python 명령이 다른 이름이면 직접 지정합니다.

~~~bash
PYTHON_BIN=python3.11 bash scripts/setup.sh tabular
~~~

기본 .env는 다음과 같습니다. 처음에는 바꾸지 않아도 됩니다.

~~~dotenv
CPV26_HOME=./var
CPV26_DB_PATH=./var/cpv26.duckdb
CPV26_TIMEZONE=Asia/Seoul
CPV26_DEVICE=auto
CPV26_RANDOM_SEED=2026
CPV26_LOG_LEVEL=INFO
~~~

설치가 됐는지 확인합니다.

~~~bash
cpv26 show-config
cpv26 --help
python -c "import catboost; print(catboost.__version__)"
~~~

## 3. 실제 KBO 데이터 다운로드와 적재

다음 세 명령을 순서대로 실행합니다.

~~~bash
cpv26 kbo-fetch
cpv26 kbo-import
cpv26 db-check
~~~

기본값은 완결된 2023·2024·2025 정규시즌입니다.

- 다운로드: var/datasets/kbo_playbyplay/v0/
- 출처·revision·SHA-256 기록: 같은 폴더의 SOURCE.json
- DB: var/cpv26.duckdb
- 적재·품질 검사 결과: var/reports/kbo_import.json

kbo-fetch는 고정된 공개 Parquet를 받습니다. KBO·NAVER·Statiz를 새로 크롤링하지
않습니다. 같은 파일을 다시 실행하면 체크섬을 확인하고 재사용합니다.

kbo-import는 DB가 없으면 생성하고, 투구별 행을 경기·선수·팀·완료 타석으로 변환합니다.
같은 revision을 다시 적재해도 행이 중복 추가되지 않습니다.

2023~2025 기본 데이터에서 확인할 주요 숫자는 다음과 같습니다.

~~~text
경기: 2,160
완료 타석: 169,481
결과 라벨 없는 타석: 460
~~~

원본에는 주루 중단·누락 라벨·일부 점수 경계 오류가 있습니다. 이를 임의로 안타나
아웃으로 채우지 않고 보고서에 남깁니다. 적재 성공이 모든 타석 전이를 완벽하게
복원했다는 뜻은 아닙니다. 자세한 출처와 검사 결과는
[실데이터 실험 기록](docs/KBO_BASELINE.md)을 참고합니다.

## 4. 승부예측 모델 학습·평가

~~~bash
cpv26 kbo-match-evaluate
~~~

이 명령은 경기당 한 행을 사용해 홈팀 기준 패·무·승 확률을 학습합니다.
입력은 이전 경기로 계산한 팀 득실점·최근 성적·Elo 등 26개 값입니다.

학습은 두 번 실행됩니다.

1. 2023 학습 → 2024 검증
2. 2023~2024로 새로 학습 → 2025 테스트

현재 경기 결과를 입력에 넣지 않습니다. 정확한 경기 시작 시각이 없는 자료이므로 같은
날의 다른 경기 결과도 그날 입력에는 사용하지 않습니다.

결과는 다음 위치에 저장됩니다.

~~~text
var/reports/kbo_match_baseline.json
var/models/kbo_match_baseline/<run-id>/validation_2024.cbm
var/models/kbo_match_baseline/<run-id>/test_2025.cbm
var/models/kbo_match_baseline/<run-id>/evaluation.json
~~~

JSON에는 log loss, Brier score, ECE, 정확도와 학습 시즌의 단순 결과 비율 기준선이
함께 기록됩니다. log loss·Brier·ECE는 낮을수록 좋습니다. 기준선보다 나쁜 결과도
그대로 보고하며, 테스트 결과를 보고 설정을 자동으로 고치지 않습니다.

## 5. 라이브 히트 모델 학습·평가

~~~bash
cpv26 kbo-live-hit-evaluate
~~~

승부예측과 다른 모델입니다. 선수-경기당 한 행으로 **적어도 한 번 안타를 칠 확률**을
학습합니다. 선수 개인의 과거 성적, 최근 기록, 소속팀 공격, 상대팀 수비,
선수-상대팀의 과거 관계 등 25개 값을 사용합니다.

학습·검증 시즌 경계는 승부예측과 같습니다. 해당 경기의 실제 타석 수와 안타 수는
정답에만 쓰며 입력으로 넣지 않습니다.

~~~text
var/reports/kbo_live_hit_baseline.json
var/models/kbo_live_hit_baseline/<run-id>/validation_2024.cbm
var/models/kbo_live_hit_baseline/<run-id>/test_2025.cbm
var/models/kbo_live_hit_baseline/<run-id>/evaluation.json
~~~

중요한 조건이 있습니다. 이 실험의 대상은 실제로 기록된 타석이 1개 이상인 선수입니다.
따라서 아직 출전 여부가 모르는 모든 V26 후보의 무조건부 안타 확률이나 최종 추천
점수는 아닙니다. 출전확률, 당일 후보·포지션, V26 보너스 규칙은 별도 입력과 검증이
필요합니다.

빠른 동작 확인만 할 때는 트리 수를 줄일 수 있습니다.

~~~bash
cpv26 kbo-match-evaluate --iterations 20
cpv26 kbo-live-hit-evaluate --iterations 20
~~~

기본값은 400입니다. 실행마다 시각과 고유 ID로 된 모델 폴더가 생깁니다. 터미널과
JSON의 model_directory에서 실제 경로를 확인할 수 있습니다. 이전 모델과 그 폴더의
evaluation.json은 보존되고, var/reports 아래의 기본 JSON만 최신 결과로 바뀝니다.
보고서에는 모델 파일의 SHA-256도 기록됩니다.

최신 보고서도 별도 이름으로 남기려면 다음처럼 실행합니다.

~~~bash
cpv26 kbo-match-evaluate --iterations 20 --report var/reports/match_20_trees.json
~~~

## 6. 선택 사항: 2026 부분시즌 추가

고정 snapshot의 2026 자료는 2026-07-26까지 470경기입니다. 최신 실시간 자료가 아닙니다.

~~~bash
cpv26 kbo-fetch --year 2026
cpv26 kbo-import --year 2026
cpv26 db-check
~~~

기본 학습·평가 명령은 2023~2025만 사용하므로 2026을 추가해도 테스트 경계는
바뀌지 않습니다. 여러 시즌을 지정하려면 --year를 반복합니다.

~~~bash
cpv26 kbo-import --year 2023 --year 2024 --year 2025 --year 2026
~~~

다른 폴더에 받은 파일은 다음처럼 지정합니다.

~~~bash
cpv26 kbo-import --source-dir /path/to/kbo_playbyplay/v0
~~~

## 7. 다시 접속하거나 코드를 업데이트할 때

새 SSH 접속마다 환경을 활성화합니다.

~~~bash
cd ~/projects/cpv26-predictor
source scripts/activate.sh
~~~

Git으로 받은 소스를 업데이트할 때는 다음 순서입니다.

~~~bash
git pull --ff-only
bash scripts/setup.sh tabular
source scripts/activate.sh
cpv26 db-init
cpv26 db-check
~~~

전체 검사는 다음 한 줄입니다.

~~~bash
bash scripts/check.sh
~~~

Python compile, Ruff, strict mypy, pytest, 패키지 충돌 검사를 실행합니다.
PyTorch가 없으면 neural 관련 선택 테스트는 skip될 수 있습니다.

## 8. RelGNN 개발용 PyTorch 설치

CatBoost 실험만 할 때는 이 단계를 건너뜁니다.

CPU에서 neural 테스트까지 실행하려면:

~~~bash
bash scripts/setup.sh ml-cpu
source scripts/activate.sh
bash scripts/check.sh
~~~

NVIDIA GPU 서버에서는 먼저 확인합니다.

~~~bash
nvidia-smi
~~~

[PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 서버 driver에
맞는 CUDA 설치 명령을 골라 활성화된 .venv 안에서 실행한 뒤 다음을 실행합니다.

~~~bash
bash scripts/setup.sh ml-cuda
source scripts/activate.sh
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
~~~

RelGNN의 graph·head·loss·학습 부품과 남은 연결 작업은
[상세 인수인계서](docs/HANDOFF.md)에 정리돼 있습니다.

## CLI 요약

| 명령 | 하는 일 |
|---|---|
| cpv26 show-config | 실행 경로·device·timezone 확인 |
| cpv26 db-init | DB 생성 또는 schema migration |
| cpv26 db-check | schema와 단일·복합 참조 검사 |
| cpv26 kbo-fetch | 고정 공개 KBO 파일 다운로드·SHA-256 검사 |
| cpv26 kbo-import | 원본→canonical DB 적재와 품질 보고 |
| cpv26 kbo-match-evaluate | 경기 패·무·승 기준선 학습·평가·모델 저장 |
| cpv26 kbo-live-hit-evaluate | 출전 선수의 1안타 이상 기준선 학습·평가·모델 저장 |
| cpv26 snapshot-build RUN_ID | 기존 prediction run의 시점 보존 snapshot 생성 |

snapshot-build는 prediction run이 먼저 적재돼 있어야 하는 개발용 기능입니다.
kbo-import가 prediction run이나 V26 계정 입력을 자동 생성하지는 않습니다.

## 자주 발생하는 오류

### cpv26: command not found / ModuleNotFoundError

~~~bash
cd ~/projects/cpv26-predictor
source scripts/activate.sh
~~~

### CatBoost is required

~~~bash
bash scripts/setup.sh tabular
source scripts/activate.sh
~~~

### KBO source file not found

~~~bash
cpv26 kbo-fetch
cpv26 kbo-import
~~~

다른 폴더로 다운로드했다면 kbo-import의 --source-dir도 그 폴더로 맞춥니다.

### Database not found / has no training games

~~~bash
cpv26 kbo-import --year 2023 --year 2024 --year 2025
cpv26 db-check
~~~

### SHA-256 mismatch

부분 다운로드나 다른 버전의 파일을 섞었을 수 있습니다. 오류가 난 파일명과
SOURCE.json의 revision을 확인한 뒤 kbo-fetch를 다시 실행합니다.
다른 revision의 데이터를 검증 없이 같은 파일명으로 바꾸지 않습니다.

### Python 버전 또는 .venv 오류

Python 3.10~3.12가 필요합니다. Windows의 .venv를 복사했다면 사용하지 말고 Linux에서
새로 설치합니다. 기존 환경을 보존해야 하면 먼저 다른 이름으로 옮깁니다.

~~~bash
mv .venv .venv.previous
PYTHON_BIN=python3.11 bash scripts/setup.sh tabular
~~~

## 생성 파일과 라이선스

소스 외 실행 파일은 .venv, .env, var 아래에 생기며 Git에서 제외됩니다.
원본 KBO Parquet와 학습 모델은 GitHub에 재배포하지 않습니다.

- 출처·라이선스·실제 품질 검사와 결과: [docs/KBO_BASELINE.md](docs/KBO_BASELINE.md)
- 전체 설계·데이터 계약·구현 범위: [docs/HANDOFF.md](docs/HANDOFF.md)
- 소프트웨어 라이선스·비제휴 고지: [LICENSE.md](LICENSE.md)

외부 코드 검토용 전체 소스 문서를 만들려면:

~~~bash
python scripts/build_code_summary.py
~~~

생성된 code_summary.md는 Git에 올라가지 않습니다. 이 프로젝트는 KBO나 게임
개발·배급·운영사와 제휴하거나 이들의 보증을 받은 프로젝트가 아닙니다.
