# CPV26 Predictor

공개 KBO 데이터를 내려받아 **Linux NVIDIA GPU에서 RelGNN을 학습하고 평가하는**
프로젝트입니다. 공유 관계 그래프 모델에서 타석 결과, 경기 승무패, 선수별 PA·안타
분포를 학습합니다.

이 문서의 명령은 MobaXterm으로 접속한 Linux 서버에서 순서대로 실행합니다.
Python 환경은 서버에 별도 Conda 환경 `cpv26`으로 만듭니다. Conda `base`에 설치하지
않으며, Windows에서 만든 Python 환경, DB, 모델 파일을 서버로 복사해 시작하지 않습니다.
CatBoost는 주 학습 경로가 아니며, 맨 아래의 선택적 비교 실험으로 분리했습니다.

이 과정을 마쳐도 V26 계정의 당일 추천이 자동 생성되는 것은 아닙니다. 당일 후보,
포지션 자격, 도감, 선택률, 공식 점수 규칙을 연결한 계정 추천은 별도 작업입니다.
아래는 단일 GPU 학습 경로이며, 사용자 서버의 학습 시간·최대 VRAM 사용량은 아직
확인되지 않았습니다.

## 1. MobaXterm으로 서버에 접속하기

MobaXterm에서 Session → SSH를 선택하고 서버 주소와 사용자 이름으로 접속합니다.
이후 명령은 오른쪽 터미널에 입력합니다. 왼쪽 SFTP 패널은 파일 전송용입니다.

먼저 Bash로 들어가 서버를 확인합니다.

~~~bash
bash
git --version
conda --version
python3 --version
nvidia-smi
~~~

필요한 환경은 Linux, Bash, Conda, Git, 인터넷 연결, NVIDIA GPU와 동작하는 NVIDIA
driver입니다. 프로젝트 Python은 3절에서 Conda로 3.12를 설치합니다(지원 범위 3.10~3.12).
지금 보이는 `(base)`와 Python 3.10.9는 기존 서버 환경이며 그대로 프로젝트에 쓰지 않습니다.
Conda가 없거나 nvidia-smi가 실패하면 서버 관리자에게 환경 구성을 요청합니다.

Ubuntu에서 Git이나 tmux만 없다면 관리 권한이 있는 사용자가 설치합니다.

~~~bash
sudo apt-get update
sudo apt-get install -y git tmux
~~~

CUDA wheel은 서버 driver에 맞게 직접 선택해야 하며, 설치 스크립트가 추측하지 않습니다.

## 2. 프로젝트 받기

저장소를 public으로 공개한 동안에는 GitHub 로그인, 토큰, SSH 키 없이 HTTPS로
받을 수 있습니다. 다음 명령을 그대로 실행합니다.

~~~bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/costunder/cpv26-predictor.git
cd cpv26-predictor
~~~

이미 이 폴더에 clone했다면 다시 clone하지 않고 11절의 업데이트 절차를 따릅니다.

프로젝트 폴더에 다음 파일이 있는지 확인합니다.

~~~bash
pwd
ls README.md environment.yml pyproject.toml .env.example scripts src tests
~~~

## 3. Conda 환경 생성 → 활성화 → CUDA 패키지 설치하기

각 블록이 성공한 것을 확인한 뒤 다음 블록을 실행합니다. 환경 생성, 환경 활성화,
패키지 설치는 서로 다른 단계입니다. `setup.sh`가 Conda 환경을 대신 만들거나 활성화하지 않습니다.

### 3-1. 프로젝트 전용 Conda 환경 만들기 — 처음 한 번

먼저 기존 환경 목록을 확인합니다.

~~~bash
conda env list
~~~

`cpv26` 환경이 없다면 프로젝트 폴더에서 다음을 실행합니다. `environment.yml`은
Conda 환경 이름 `cpv26`, Python 3.12, pip를 지정합니다.

~~~bash
conda env create -f environment.yml
~~~

이미 `conda create -n cpv26 python=3.12 pip` 등으로 만들었다면 다시 생성하지 않고
다음 단계로 갑니다. 기존 환경을 삭제하거나 덮어쓰지 않습니다.
[Conda 환경 생성·활성화 안내](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)

### 3-2. Conda 환경 활성화하고 Python 확인하기

~~~bash
conda activate cpv26
echo "$CONDA_DEFAULT_ENV"
which python
python --version
python -m pip --version
~~~

환경 이름이 `cpv26`, Python 경로가 `…/envs/cpv26/bin/python`, 버전이 `Python 3.12.x`
인지 확인합니다. `/tools/anaconda3/bin/python`이 그대로 나오거나 `(base)` 상태라면
패키지를 설치하지 말고 아래의 Conda 활성화 오류 항목부터 확인합니다.

### 3-3. 활성화된 Conda 환경 안에 CUDA PyTorch와 프로젝트 설치하기

확인된 서버 **A100 MIG 10GB / driver 535.104.05 / nvidia-smi CUDA 12.2 표시**에서는
아래의 공식 CUDA 12.1 wheel index를 사용합니다. 이는 모든 서버에 공통인 기본값이
아닙니다. [PyTorch 공식 이전 버전 안내](https://pytorch.org/get-started/previous-versions/)와
[NVIDIA CUDA 12.1 드라이버 조건](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/)을
기준으로 선택한 예시이며, 실제 CUDA 연산 확인은 설치 중과 4절에서 별도로 합니다.

~~~bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
TORCH_INDEX_URL="$TORCH_INDEX_URL" bash scripts/setup.sh ml-cuda
~~~

다른 서버에서는 [공식 설치 선택기](https://pytorch.org/get-started/locally/) 또는
위 이전 버전 안내에서 driver에 맞는 CUDA wheel을 선택하고 `--index-url` 뒤의 URL을
`TORCH_INDEX_URL`에 넣습니다. 최신 선택기에 `cu121`이 없다고 임의로 `cu128` 등으로
바꾸지 않습니다. 어떤 버전이 맞는지 모르면 nvidia-smi 결과로 서버 관리자에게 확인합니다.

`ml-cuda`는 현재 활성화한 Conda 환경에 프로젝트와 개발 검사 도구를 설치하며
CatBoost는 설치하지 않습니다. Conda로 Python을 관리하고, 패키지는 **그 환경의 pip**로
설치합니다. Conda `base`, 미활성 환경, Python 경로가 다른 환경은 거부합니다.

이미 현재 Conda 환경의 CUDA PyTorch가 지원 범위(torch>=2.4,<3)와 필요한 AMP API,
실제 forward/backward 검사를 통과하면 보존합니다. 그렇지 않을 때만 명시한 index로
torch를 upgrade/reinstall하고 다시 검사합니다. URL도 정상 CUDA PyTorch도 없으면
오류로 멈추며 CPU로 자동 전환하지 않습니다.

### 3-4. 설치 성공 후 프로젝트 설정 불러오기

설치 마지막에 **`Conda environment ready: ...`**가 나와야 다음을 실행합니다. 오류가 났다면
그 오류부터 해결하고 넘어가지 않습니다.

~~~bash
if [ ! -f .env ]; then cp .env.example .env; fi
chmod 600 .env
source scripts/activate.sh
~~~

`conda activate cpv26`은 Python 환경을 활성화하고, `source scripts/activate.sh`는
이미 활성화된 Conda 환경을 검사한 뒤 `.env`의 프로젝트 설정만 읽습니다. 서로 대체하지 않습니다.
기존 `.venv`가 남아 있어도 생성·활성화·재사용·삭제하지 않습니다. `var/`의 데이터와
checkpoint도 그대로 보존되므로 환경을 전환한다고 다시 만들 필요는 없습니다.

이미 동작하는 Conda CUDA 환경의 패키지를 다시 설치할 때는 11절의 업데이트 절차를 따릅니다.

.env 기본값은 바꾸지 않아도 됩니다. 상대 경로는 프로젝트 루트 기준입니다.
학습 명령의 --device cuda:0은 일반 설정인 CPV26_DEVICE와 별도로 명시합니다.

~~~dotenv
CPV26_HOME=./var
CPV26_DB_PATH=./var/cpv26.duckdb
CPV26_TIMEZONE=Asia/Seoul
CPV26_DEVICE=auto
CPV26_RANDOM_SEED=2026
CPV26_LOG_LEVEL=INFO
~~~

.env는 한 줄에 `CPV26_이름=값` 형식으로 쓰며 키에는 `CPV26_` 뒤 영문 대문자·숫자·밑줄만
허용합니다. `PATH`, `PYTHONPATH`, `CONDA_PREFIX`처럼 Python 환경을 바꾸는 변수는
넣지 않습니다. `TORCH_INDEX_URL`도 `.env`가 아니라 위 설치 명령에 지정합니다.
값 둘레의 따옴표, export 접두사, $HOME 같은 shell 변수 치환, 값 뒤의 주석은
사용하지 않습니다. 주석은 별도 # 줄에
씁니다. .env는 Git에 올리지 않습니다.

## 4. GPU와 DB가 실제로 동작하는지 확인하기

~~~bash
cpv26 show-config
cpv26 gpu-check --device cuda:0
cpv26 db-init
cpv26 db-check
~~~

gpu-check는 GPU 이름·메모리·PyTorch/CUDA 버전을 확인하고 실제 CUDA
forward/backward kernel을 실행합니다. 단순히 GPU가 목록에 보이는지만 확인하는
명령이 아닙니다. 이 명령이 성공해야 다음 GPU 학습 단계로 넘어갑니다.

db-check는 schema와 단일·복합 물리 참조를 검사합니다. 이때 DB가 비어 있어도
정상이며 실제 데이터는 다음 단계에서 받습니다.

## 5. 실제 KBO 데이터 다운로드 → 적재 → 그래프 만들기

다음 명령을 순서대로 실행합니다.

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

## 6. SSH가 끊겨도 학습이 계속되게 하기

장시간 학습은 tmux 안에서 실행합니다. 다음 명령으로 Bash 세션을 만듭니다.

~~~bash
tmux new -s cpv26-train bash
~~~

tmux 화면 안에서 프로젝트 환경을 다시 활성화합니다.

~~~bash
cd ~/projects/cpv26-predictor
conda activate cpv26
source scripts/activate.sh
cpv26 gpu-check --device cuda:0
~~~

`conda activate`를 찾지 못하면 아래의 Conda 활성화 오류 항목처럼 이 Bash에 Conda를
먼저 로드합니다. tmux 안에서도 `base`나 다른 Python으로 학습을 시작하지 않습니다.

이제 다음 절의 학습 명령을 실행합니다. 학습을 남겨두고 나가려면 Ctrl+B를 누른 뒤
손을 떼고 D를 누릅니다. 다시 SSH로 접속한 뒤에는 다음으로 돌아옵니다.

~~~bash
tmux attach -t cpv26-train
~~~

기존 학습이 살아 있는지 먼저 확인하고 같은 run을 두 번 실행하지 마세요.
tmux는 SSH 연결 종료를 견디게 해 주지만 서버 재부팅이나 프로세스 오류까지 막지는
않습니다. 프로세스가 종료됐다면 9절의 checkpoint 재개를 사용합니다.

## 7. RelGNN GPU 학습하기

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

## 8. 2025 test 평가하기

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

## 9. 중단된 학습 재개하기

학습 프로세스가 종료됐다면 마지막으로 저장된 last.pt에서 재개합니다.
best.pt는 재개용으로 사용하지 않습니다.
새 SSH/tmux 셸이라면 먼저 11절처럼 `conda activate cpv26` 후 프로젝트 설정을 로드합니다.

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
더 실행합니다. 재개할 때는 같은 데이터셋과 모델 설정을 유지합니다. 저장되지 않은
진행 중 batch는 복구되지 않을 수 있으므로 SSH 연결만 끊을 때는 Ctrl+C 대신 tmux
detach를 사용하세요.

## 10. VRAM이 부족할 때

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

이 설정도 모든 GPU에서 동작한다고 보장할 수는 없습니다. gpu-check 결과와 오류,
사용한 옵션을 함께 확인해야 하며 다중 GPU·분산 학습 경로는 제공하지 않습니다.
A100 MIG 10GB도 전체 A100이 아니라 할당된 MIG 장치를 사용하며, 이 프로젝트의
10GB 내 학습 성공 여부는 아직 측정하지 않았습니다. 해당 환경은 `--batch-days 1`부터
시험하고 학습·평가 각각의 peak memory를 확인합니다.

## 11. 다시 접속하거나 코드 업데이트하기

3절의 Conda 설치를 마쳤다면 새 SSH 세션에서 매번 환경을 활성화합니다.
아직 Conda로 전환하지 않은 기존 checkout은 아래의 코드 업데이트부터 합니다.

~~~bash
bash
cd ~/projects/cpv26-predictor
conda activate cpv26
source scripts/activate.sh
~~~

실행 중인 학습을 먼저 종료하거나 별도 checkout에서 작업한 뒤 업데이트합니다.
학습 중인 코드와 데이터셋을 바꾸지 않습니다.

~~~bash
cd ~/projects/cpv26-predictor
git pull --ff-only
~~~

이전 설치 방식에서 처음 업데이트하는 경우에는 여기서 **3-1~3-4절을 먼저** 수행합니다.
`cpv26` Conda 환경을 만든 적이 없다면 바로 `conda activate cpv26`부터 실행할 수 없습니다.
기존 `.env`, `var/`의 DB·데이터셋·checkpoint는 삭제하지 않습니다.

이미 Conda 환경을 만들고 설치한 경우에는 다음을 실행합니다.

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
GitHub CI도 전용 Conda 환경의 Python 3.12에서 `base` profile 설치·CLI 도움말과
CPU PyTorch가 있는 neural 테스트를 검사합니다. CI에서 전체 원천 데이터를 다운로드하거나 NVIDIA GPU 학습을
수행하지는 않습니다.
여기서 `base`는 runtime만 설치하는 스크립트 profile 이름이며 **Conda의 `(base)` 환경이 아닙니다.**

### 서버 실행 후 저장소를 private으로 되돌리기

서버에서 필요한 설치·학습·평가를 마친 뒤 저장소 소유자가 GitHub 저장소의
Settings → General → Danger Zone → Change repository visibility에서 private으로
직접 바꿉니다. 이 프로젝트가 자동으로 비공개 전환하지는 않습니다.

private으로 바꿔도 서버에 이미 받은 소스·설치 환경·데이터·checkpoint는 그대로이며,
이 파일들을 사용하는 학습·재개·평가를 계속할 수 있습니다. 이후 새 clone이나
git pull에는 저장소 접근 권한과 GitHub 인증이 필요합니다. HTTPS에서는 저장소 읽기
권한이 있는 personal access token 또는 설정된 credential helper를 사용하고,
토큰을 clone URL이나 문서에 넣지 않습니다.

주의: public인 동안 다른 사람이 받은 복제본은 private으로 되돌려도 회수되지 않습니다.

## 선택 사항: CPU에서 작은 동작 검사만 하기

GPU 주 학습과 별개의 코드 검사입니다. 이 검사도 5절에서 만든 실제 graph dataset을
사용하므로 데이터 다운로드·적재·graph build가 먼저 끝나 있어야 합니다.
GPU용 `cpv26`의 CUDA PyTorch를 바꾸지 않도록 별도 Conda 환경 `cpv26-cpu`를 만듭니다.
처음 한 번만 생성하며, 이미 있으면 생성 명령을 건너뛰고 활성화합니다.

~~~bash
conda env create -f environment.yml -n cpv26-cpu
conda activate cpv26-cpu
which python
python --version
bash scripts/setup.sh ml-cpu
~~~

`Conda environment ready: ...`를 확인한 뒤 실행합니다. `.env`는 3-4절에서 만든 파일을 사용합니다.

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

nvidia-smi를 확인하고 3절에서 서버에 맞는 공식 CUDA index를 다시 선택합니다.
CPU wheel을 그대로 둔 채 반복 실행하지 않습니다. driver 권한이나 GPU 할당 문제는
서버 관리자에게 확인합니다. 설치 후에는 반드시 다음을 통과해야 합니다.

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

`conda --version`부터 확인합니다. 명령이 없다면 서버 관리자가 제공한 Conda 초기화
방법을 사용합니다. Conda 명령은 있지만 Bash 활성화 오류가 나면 현재 셸에서만 다음을 실행합니다.

~~~bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cpv26
~~~

환경을 찾지 못하면 `conda env list`로 확인하고 3-1절에서 먼저 생성합니다.
Conda 설치 경로 자체인 `base`에는 프로젝트 패키지를 설치하지 않습니다.

### Python 경로가 다름 / 중첩된 환경 / 지원하지 않는 Python

기존 Python 가상환경을 활성화한 셸이라면 새 SSH 세션에서 시작하고 Conda `cpv26`만
활성화합니다. `which python`, `python --version`, `echo "$CONDA_PREFIX"`로 경로와 버전을
확인합니다. 스크립트는 중첩된 가상환경이나 Conda 환경과 다른 Python을 거부합니다.
`PYTHON_BIN`으로 다른 Python을 강제로 지정하지 말고, 3절처럼 올바른 Conda 환경을 선택합니다.
기존 환경을 삭제하거나 base의 Python을 교체하는 방식으로 해결하지 않습니다.

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
