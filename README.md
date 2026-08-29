# CPV26 Predictor

컴프야 V26의 **승부예측**과 **라이브 히트**를 연구하기 위한 KBO 확률 예측
프로젝트입니다.

이 README는 처음 사용하는 사람이 MobaXterm으로 Linux 서버에 접속한 뒤 저장소를
받고, 설치하고, 현재 구현된 기능을 검증하는 데 필요한 순서만 설명합니다. 모델 구조,
데이터 계약, 테이블 정의의 상세 내용은 [docs/HANDOFF.md](docs/HANDOFF.md)에 있습니다.

> 현재 버전: 0.4.0 · DuckDB schema: v4
> 상태: 연구용 기반 코드 · 비공식 프로젝트 · 독점 소프트웨어

## 먼저 알아둘 점

현재 저장소에서 바로 실행할 수 있는 범위는 다음과 같습니다.

- Python 환경 설치
- 설정 확인
- DuckDB schema 생성과 무결성 검사
- 전체 단위 테스트
- CatBoost·PyTorch·RelGNN 코드 로드와 테스트
- 이미 적재된 prediction run의 시점 보존 snapshot 생성

아직 바로 실행할 수 없는 범위는 다음과 같습니다.

- KBO 또는 V26 실제 데이터 자동 수집
- 공급자 CSV/API를 DuckDB에 적재하는 adapter
- 실제 시즌 데이터로 학습하는 단일 명령
- 당일 승부예측·라이브 히트 추천을 만드는 단일 명령

따라서 아래 빠른 시작을 완료하면 **프로젝트 설치와 기반 코드가 정상이라는 것**까지
확인됩니다. 실제 예측값이 자동으로 만들어지는 단계는 아닙니다.

## 1. MobaXterm으로 서버 접속

MobaXterm에서 `Session` → `SSH`를 선택하고 Linux 서버 주소와 사용자 이름을 입력해
접속합니다. 이후 이 문서의 명령은 모두 MobaXterm 오른쪽 터미널에서 실행합니다.
왼쪽 SFTP 패널은 파일을 올리거나 내려받을 때만 사용합니다.

지원 환경은 다음과 같습니다.

- Linux, Bash
- Python 3.10~3.12
- Git
- 인터넷 연결
- GPU 학습 시 NVIDIA driver와 CUDA 지원 GPU

먼저 서버 상태를 확인합니다.

```bash
git --version
python3 --version
python3 -m venv --help >/dev/null
```

Python 버전이 3.10보다 낮거나 3.13 이상이면 설치를 진행하지 말고 서버 관리자에게
Python 3.10, 3.11 또는 3.12 설치를 요청합니다. Ubuntu에서 Git 또는 venv 모듈만
없는 경우에는 권한이 있는 사용자가 다음처럼 설치할 수 있습니다.

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv
```

## 2. GitHub에서 프로젝트 받기

저장소는 비공개이므로 GitHub 계정에 접근 권한이 있어야 합니다.

### SSH key가 서버에 등록된 경우

```bash
ssh -T git@github.com
mkdir -p ~/projects
cd ~/projects
git clone git@github.com:costunder/cpv26-predictor.git
cd cpv26-predictor
```

`Permission denied (publickey)`가 나오면 서버의 SSH 공개키를 GitHub 계정에 먼저
등록해야 합니다. 절차는
[GitHub SSH 연결 문서](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)를
따릅니다.

### HTTPS를 사용하는 경우

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/costunder/cpv26-predictor.git
cd cpv26-predictor
```

비공개 저장소 인증이 불가능한 서버에서는 Windows 브라우저로 저장소 ZIP을 받은 뒤
압축을 풀고, MobaXterm 왼쪽 SFTP 패널로 `~/projects/cpv26-predictor`에 올려도 됩니다.
`.venv`, `var`, cache 폴더는 Windows에서 복사하지 않습니다.

현재 위치가 맞는지 확인합니다.

```bash
pwd
ls
```

`README.md`, `pyproject.toml`, `scripts`, `src`, `tests`가 보여야 합니다.

## 3. 기본 환경 설치

프로젝트 루트에서 다음 명령을 실행합니다.

```bash
bash scripts/setup.sh base
cp .env.example .env
source scripts/activate.sh
```

`setup.sh`는 `.venv` 가상환경을 만들고 프로젝트를 editable mode로 설치합니다.
서버의 기본 명령이 `python3`이 아니라면 설치된 Python을 직접 지정합니다.

```bash
PYTHON_BIN=python3.11 bash scripts/setup.sh base
```

`setup.sh`는 정상적인 기존 `.venv`를 재사용합니다. Python 버전을 바꾸려면 기존
가상환경을 다른 이름으로 옮긴 뒤 원하는 `PYTHON_BIN`으로 다시 실행합니다.

`.env`의 기본값은 바로 사용할 수 있습니다.

```dotenv
CPV26_HOME=./var
CPV26_DB_PATH=./var/cpv26.duckdb
CPV26_TIMEZONE=Asia/Seoul
CPV26_DEVICE=auto
CPV26_RANDOM_SEED=2026
CPV26_LOG_LEVEL=INFO
```

`.env`에는 나중에 추가될 비밀값을 넣을 수 있으므로 Git에 올리지 않습니다.

## 4. 첫 실행

설정과 데이터베이스를 순서대로 확인합니다.

```bash
cpv26 show-config
cpv26 db-init
cpv26 db-check
```

정상이면 마지막 두 명령에 다음 내용이 표시됩니다.

```text
Database ready: .../var/cpv26.duckdb (schema=4, tables=36)
Database schema and references are current: version 4, 36 tables
```

생성된 파일도 확인합니다.

```bash
ls -lh var/cpv26.duckdb
cpv26 --help
```

여기까지 오류 없이 끝나면 기본 설치가 완료된 것입니다.

## 5. 서버에 다시 접속했을 때

새 SSH 세션에서는 가상환경과 `.env`가 자동으로 로드되지 않습니다. 매번 다음 두
명령을 실행합니다.

```bash
cd ~/projects/cpv26-predictor
source scripts/activate.sh
```

프롬프트 앞에 `(.venv)`가 표시되고 `CPV26 environment activated`가 출력되면
준비된 상태입니다.

## 6. 전체 코드 검사

코드를 수정하거나 서버에서 전체 동작을 검증하려면 개발 의존성을 설치합니다.

```bash
cd ~/projects/cpv26-predictor
bash scripts/setup.sh dev
source scripts/activate.sh
bash scripts/check.sh
```

검사는 다음 순서로 실행됩니다.

1. Python compile
2. Ruff 정적 검사
3. strict mypy 타입 검사
4. pytest
5. 설치된 패키지 충돌 검사

모두 통과하면 명령이 exit code 0으로 끝납니다. `dev` 환경에 PyTorch나 CatBoost가
없으면 해당 선택 기능의 테스트는 skip될 수 있습니다. GitHub CI는 README 빠른 시작과
모든 shell script의 문법도 Linux에서 함께 검사합니다.

## 7. ML 의존성 설치

데이터베이스와 설정만 확인할 때는 `base`면 충분합니다. CatBoost 또는 RelGNN 코드를
실행할 때만 아래 프로필을 사용합니다.

### CPU

```bash
cd ~/projects/cpv26-predictor
bash scripts/setup.sh ml-cpu
source scripts/activate.sh
python -c "import catboost, torch; print(catboost.__version__, torch.__version__, torch.cuda.is_available())"
bash scripts/check.sh
```

마지막 값이 `False`인 것이 CPU 환경에서는 정상입니다.

### NVIDIA GPU

먼저 GPU와 driver를 확인합니다.

```bash
nvidia-smi
cd ~/projects/cpv26-predictor
bash scripts/setup.sh dev
source scripts/activate.sh
```

[PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 다음 조건을
선택하고 표시되는 명령을 **활성화된 `.venv` 안에서** 실행합니다.

- OS: Linux
- Package: Pip
- Language: Python
- Compute Platform: 서버 driver에 맞는 CUDA

그다음 CatBoost를 설치하고 CUDA 인식을 검사합니다.

```bash
bash scripts/setup.sh ml-cuda
source scripts/activate.sh
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
bash scripts/check.sh
```

`torch.cuda.is_available()`이 `True`여야 합니다.

## 목표 파이프라인과 현재 연결 상태

아래는 프로젝트가 완성됐을 때의 **목표 흐름**입니다. 현재 한 명령으로 이어지는
end-to-end 파이프라인이 아니며, 각 상자의 구현 상태가 다릅니다.

```text
[미구현] 공급자 adapter + 선수·팀·경기 ID 통합
    ↓
[구현] DuckDB schema + 시점 보존 query + 물리 참조 무결성 검사
    ↓
[일부 구현] prediction run 기반 Parquet snapshot
    ↓
[미구현] feature build job + Parquet→tensor/subgraph loader
    ↓
[부품 구현] 개인 능력 feature + 관계 graph + 공유 RelGNN backbone
    ├─ PA task: 타석 결과 분포
    ├─ Match task: 승/무/패 + 홈·원정 득점 marginal
    └─ Live Hit task: 선수별 출장·PA·안타 수 분포
    ↓
[미구현] 학습·calibration·추론·walk-forward orchestration
```

세 task는 관계형 backbone을 공유할 수 있지만 row 단위와 label이 다릅니다. 따라서
PA·경기·선수경기 loader와 Head/loss를 분리하고, 학습 시에는 별도 batch를 번갈아
처리하도록 구현했습니다.

### 승부예측 목표 경로

PA Head의 10개 결과 확률은 adapter를 거쳐 simulator의 14개 terminal event 확률이
됩니다. Simulator는 정규 9이닝부터 설정된 최대 12회까지 타석을 순서대로 표본화해
승패·득점·선수 안타를 만듭니다. Match Head의 승/무/패와 홈·원정 득점 marginal은
별도 직접 예측이며 simulator 입력이 아닙니다. 둘을 비교하거나 보정해 최종 market을
만드는 job은 아직 없습니다.

`MatchPredictionOptimizer`는 호출자가 만든 경기별 option 확률과 보상 점수를
입력받습니다. 모든 경기에서 정확히 하나씩 선택하고, 올킬 보상·최소 올킬 확률·위험
회피를 포함한 expected utility로 전체 조합을 정렬합니다.

### 라이브 히트 목표 경로

`DirectPlayerGameHead`는 선수별 `P(PA 수, 안타 수 | 경기 진행)`를 예측합니다. 이
Head 자체가 같은 경기 선수들의 결합 상관을 학습하는 것은 아닙니다. 같은 simulation
path에서 만든 joint `HitScenario`를 사용한 경우에만 `LiveHitOptimizer`가 입력에 들어
있는 선수 간 상관을 보존해 계산합니다. 두 경로를 결합하는 production job은 아직
없습니다.

Optimizer에는 포지션, 중복 금지, 선택률, 도감, 선택 구단 보너스를 계산하는 부품이
있습니다. 다만 내장 `fixed_300`과 기본 안타 점수표는 provisional 비교 규칙이며 공식
V26 ruleset이 아닙니다. 실제 account 모드는 당시의 포지션별 선택률 구간표·도감·점수표를
호출자가 제공해야 합니다.

## 실제 데이터를 연결하려면

저장소에는 라이선스 문제가 없는 실제 KBO/V26 데이터가 포함되어 있지 않습니다.
실전 학습 전에는 다음 구현이 추가되어야 합니다.

1. 데이터 공급자별 adapter와 안정적인 내부 ID mapping을 구현합니다.
2. 변동 행에 `event_at`, `available_at`, `ingested_at`, `valid_from`, `valid_to`를
   기록합니다.
3. Python `DuckDBStore.append()` API로 원천 table을 적재하고 `cpv26 db-check`를
   실행합니다. ingest CLI는 아직 없습니다.
4. Python append API로 prediction run과 후보를 만듭니다. run-create CLI는 아직
   없습니다.
5. 일반 야구 입력은 `cpv26 snapshot-build RUN_ID`로 고정합니다. 이 기본 CLI는
   V26 ruleset·slate·선택률·도감 table을 포함하지 않습니다.
6. Live Hit 계정 snapshot은 명시적 scope와 `live_hit_snapshot_specs()`를 사용하는
   Python 코드가 필요합니다. 이를 위한 CLI 옵션은 아직 없습니다.
7. Feature/graph build job과 Parquet→tensor/subgraph loader를 구현합니다.
8. Train·evaluate·predict·optimize job과 CLI를 각각 연결합니다.

현재 ingest, run-create, feature-build, graph-build, train, evaluate, predict,
optimize 명령은 없습니다. 테스트 fixture를 실제 예측값으로 오해하지 마세요. 필요한
column, 무결성 규칙과 학습 경계는 [docs/HANDOFF.md](docs/HANDOFF.md)를 참고합니다.

## CLI 명령

| 명령 | 용도 | 선행 조건 |
|---|---|---|
| `cpv26 show-config` | 적용된 경로·device·timezone 확인 | 환경 활성화 |
| `cpv26 db-init` | DuckDB 생성 또는 schema migration | 환경 활성화 |
| `cpv26 db-check` | schema와 단일·복합 물리 참조 무결성 검사 | `db-init` 완료 |
| `cpv26 snapshot-build RUN_ID` | 기본 야구 table의 시점 보존 snapshot | 실제 prediction run 적재 |

`snapshot-build` 결과는 `var/snapshots/RUN_ID/`에 생성됩니다. 같은 run ID의 내용을
바꾸어 기존 snapshot을 덮어쓰지 않습니다.

## 생성되는 로컬 파일과 폴더

```text
.venv/                 서버용 Python 가상환경
.env                   서버별 설정
var/cpv26.duckdb       실행 데이터베이스
var/snapshots/         prediction snapshot
```

이 파일들은 Git에 올라가지 않습니다. Windows의 `.venv`를 Linux 서버에 복사하지
말고 서버에서 다시 설치합니다.

## 자주 발생하는 오류

### `Python executable not found`

```bash
python3 --version
PYTHON_BIN=python3.12 bash scripts/setup.sh base
```

실제로 설치된 3.10~3.12 명령으로 바꿉니다.

### `The existing .venv is incomplete`

중간에 venv 설치가 실패해 불완전한 폴더가 남은 경우입니다. 기존 폴더를 보존한 채
이름을 바꾸고 다시 설치합니다.

```bash
mv .venv .venv.broken
bash scripts/setup.sh base
```

### `cpv26: command not found` 또는 `ModuleNotFoundError`

```bash
cd ~/projects/cpv26-predictor
source scripts/activate.sh
```

### `.env file not found`

```bash
cp .env.example .env
source scripts/activate.sh
```

### `Database not found`

```bash
cpv26 db-init
cpv26 db-check
```

### `CUDA-enabled PyTorch is missing`

활성화된 `.venv` 안에 서버 driver와 맞는 CUDA PyTorch를 먼저 설치한 뒤
`bash scripts/setup.sh ml-cuda`를 다시 실행합니다.

## 코드 업데이트

Git으로 받은 저장소라면 다음처럼 최신 코드를 적용합니다.

```bash
cd ~/projects/cpv26-predictor
git pull --ff-only
bash scripts/setup.sh base
source scripts/activate.sh
cpv26 db-init
cpv26 db-check
```

개발 또는 ML 프로필을 사용했다면 `base` 대신 기존 프로필을 다시 실행합니다.

## 문서와 라이선스

- 상세 설계·데이터 계약·구현 상태: [docs/HANDOFF.md](docs/HANDOFF.md)
- 독점 라이선스와 비제휴 고지: [LICENSE.md](LICENSE.md)

외부 코드 검토용 전체 소스 문서가 필요할 때만 다음 명령으로 로컬
`code_summary.md`를 생성합니다.

```bash
python scripts/build_code_summary.py
```

`code_summary.md`는 Git에서 제외됩니다. 이 프로젝트는 KBO 또는 게임
개발·배급·운영사와 제휴하거나 이들의 보증을 받은 프로젝트가 아닙니다.
