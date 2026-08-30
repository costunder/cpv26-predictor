# CPV26 Predictor 인수인계서

- 작성 기준일: 2026-08-30 KST
- 프로젝트 버전: `0.4.0`
- DuckDB schema version: `4`
- 실행 대상: MobaXterm으로 접속하는 Linux 서버
- 현재 검증 host: Windows CPU / Python 3.12.13

## 1. 최종 판정

이 저장소는 다음 핵심 구성요소를 실제 코드와 테스트로 제공합니다.

- append-only point-in-time DuckDB와 schema v1→v2→v3→v4 migration
- 경기 상태·선발·타석 전이·교체·주루·수비·날씨·V26 slate를 포함한 36-table 계약
- PA/박스스코어/전이/V26 capture/날씨/시즌 분할 데이터 감사
- 재현 가능한 Parquet snapshot과 SHA-256 manifest
- 타격 feature primitive와 확률 평가 도구
- 선수 역할별 state encoder
- destination-query multi-head atomic-route GNN
- 10종 neural PA와 14종 simulator event 사이의 양방향 label 계약
- 순차 Monte Carlo 야구 시뮬레이터
- 승부예측 exhaustive optimizer
- 출처가 명시된 라이브 히트 ruleset, exact/beam optimizer와 탐색 진단
- PA·선수경기·경기 단위 별도 loss와 alternating multi-task trainer
- 시간순 OOF stacking과 calibration primitive
- revision·SHA-256을 고정한 공개 KBO Parquet 다운로드와 canonical DB importer
- 날짜별 과거 90일 관계 그래프를 만드는 safe NPZ cache와 hash manifest
- 공유 role-aware RelGNN의 Match WDL/득점, 조건부 Live Hit, PA10 학습·평가 CLI
- CUDA 검사, AMP, 날짜별 mini-batch, atomic checkpoint와 중단 후 재개
- 실제 2023~2025 시즌의 경기 승/무/패 CatBoost 학습·평가·모델 저장
- 별도 선수-경기 1안타 이상 CatBoost 학습·평가·모델 저장

다만 이것은 아직 당일 데이터를 받아 자동으로 추천을 내는 운영 서비스가 아니다.
공개 데이터 adapter→날짜별 graph cache→RelGNN 학습·평가 job을 연결했고 실제 데이터의
축소 CPU 학습을 확인했다. 주 실행 경로는 Linux 단일 GPU RelGNN이다. 현재 host에는
NVIDIA GPU가 없어 CUDA 학습 성능·메모리·전체 시즌 학습 완료를 확인하지 못했다.
당일 후보·출전 확률 입력, 실제 V26 ruleset replay와 scheduler도 남아 있다.
정확한 표현은 다음과 같다.

> 실제 KBO 데이터로 공유 RelGNN과 작업별 head를 학습·평가·재개할 수 있는 연구
> 프레임워크다. GPU 전체 학습이나 성능 우위를 입증한 모델, 완성된 당일 추천기는 아니다.

설치부터 GPU 학습·재개·평가까지는 [README](../README.md)와
[GPU_TRAINING.md](GPU_TRAINING.md)를 따른다. 선택적 CatBoost baseline의 과거 실행
결과는 [KBO_BASELINE.md](KBO_BASELINE.md)에 기록했다. 2025 test에서 경기 모델은
정확도 46.53%로 학습 빈도 기준선 49.72%보다 낮다. 선수 안타 모델은 출전 선수 조건부
정확도 59.80%이며 기준선 53.80%보다 높지만, 2024 validation에서는 기준선보다 낮았다.
성능 개선이나 실전 일반화를 입증했다고 해석하지 않는다.

## 2. 전달 파일

- `README.md`
  - 초보자가 MobaXterm에서 clone, CUDA 설치·검사, 데이터 적재·graph 생성,
    RelGNN 학습·재개·평가를 순서대로 실행하는 안내서다. CatBoost는 선택적 baseline이다.
- `docs/GPU_TRAINING.md`
  - Linux 단일 GPU 학습, 메모리 조절, checkpoint 재개와 검증 범위의 상세 안내다.
- `docs/KBO_BASELINE.md`
  - 원본 출처·revision·hash, adapter 정책, 데이터 결함, 실제 시즌 평가 결과다.
- `docs/HANDOFF.md`
  - 이 문서다. 설계 결정, 공개 계약, 검증 결과와 다음 작업을 설명한다.
- `code_summary.md`
  - Git에 포함하지 않는 외부 검토용 생성물이다. `code_summary.md`와 별도 문서인
    `docs/HANDOFF.md`, 실행 데이터·모델·캐시를 제외한 소스·설정·테스트·README와
    실행 안내 문서를 다음 형식으로 이어 붙인다.
- `scripts/build_code_summary.py`
  - README와 handoff 수정 후 `code_summary.md`를 같은 형식으로 재생성한다.

```text
# `파일경로`

````
코드 내용
````
```

- 최신 `code_summary.md` SHA-256: `b35c137e1540b35e6301871dc9d3e2ad35550e2c118468e84121c29fdc9b6ed7`
- 포함 section 수: `86`
- 고유 경로 수: `86`

이 hash는 GPU RelGNN 구현과 `docs/GPU_TRAINING.md`를 포함해 다시 생성한 전달본의 값이다.
`docs/HANDOFF.md` 자체는 자기참조를 피하기 위해 summary에서 제외한다.

## 3. 저장소 구조

```text
.
├── .env.example
├── .gitattributes
├── .gitignore
├── .github/workflows/ci.yml
├── LICENSE.md
├── pyproject.toml
├── README.md
├── docs/
│   ├── GPU_TRAINING.md
│   ├── HANDOFF.md
│   └── KBO_BASELINE.md
├── requirements/
│   └── constraints.txt
├── scripts/
│   ├── activate.sh
│   ├── build_code_summary.py
│   ├── setup.sh
│   └── check.sh
├── src/cpv26/
│   ├── data/
│   │   ├── dataset_contracts.py
│   │   ├── kbo_playbyplay.py
│   │   ├── kbo_ingest.py
│   │   ├── kbo_graph_dataset.py
│   │   ├── schema.py
│   │   ├── schema_v4.py
│   │   ├── store.py
│   │   ├── integrity.py
│   │   └── snapshots.py
│   ├── features/
│   │   ├── batting.py
│   │   ├── statistics.py
│   │   └── state.py
│   ├── graph/
│   │   ├── routes.py
│   │   └── snapshot.py
│   ├── models/
│   │   ├── _torch.py
│   │   ├── baseline.py
│   │   ├── kbo_relgnn.py
│   │   ├── player_encoder.py
│   │   ├── relgnn.py
│   │   ├── interaction.py
│   │   ├── heads.py
│   │   └── stacking.py
│   ├── simulation/
│   │   ├── events.py
│   │   ├── adapter.py
│   │   ├── probability.py
│   │   ├── state.py
│   │   ├── transition.py
│   │   └── simulator.py
│   ├── optimization/
│   │   ├── match_prediction.py
│   │   └── live_hit.py
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── kbo_match_baseline.py
│   │   └── kbo_live_hit_baseline.py
│   ├── training/
│   │   ├── contracts.py
│   │   ├── kbo_runner.py
│   │   ├── losses.py
│   │   ├── model.py
│   │   └── trainer.py
│   ├── cli.py
│   ├── config.py
│   ├── domain.py
│   └── evaluation.py
└── tests/
    ├── test_cli.py
    ├── test_config.py
    ├── test_dataset_contracts.py
    ├── test_dataset_integrity_v4.py
    ├── test_dataset_schema_v4.py
    ├── test_domain.py
    ├── test_evaluation.py
    ├── test_graph_models.py
    ├── test_kbo_graph_dataset.py
    ├── test_kbo_ingest.py
    ├── test_kbo_live_hit_baseline.py
    ├── test_kbo_match_baseline.py
    ├── test_kbo_playbyplay_source.py
    ├── test_kbo_relgnn.py
    ├── test_kbo_runner.py
    ├── test_live_hit_point_in_time.py
    ├── test_live_hit_rules.py
    ├── test_model_output_contracts.py
    ├── test_pa_adapter.py
    ├── test_point_in_time.py
    ├── test_simulation_optimization.py
    └── test_task_training.py
```

가짜 데이터 파일이나 fixture CSV용 빈 폴더는 없다. 원본·DB·모델·보고서는
`CPV26_HOME` 아래에 생성하며 기본값은 Git에서 제외되는 `var/`다.

## 4. Linux 설치와 profile

Windows `.venv`는 복사하지 않는다. 저장소가 public인 동안 Linux에서 익명 HTTPS로
clone한 뒤 새로 만든다. GitHub 로그인·토큰·SSH 키는 필요하지 않다. 전체 초보자
안내는 [README](../README.md), GPU 운용은 [GPU_TRAINING.md](GPU_TRAINING.md)를 따른다.
아래 명령은 Bash에서 실행한다. 이미 clone한 폴더가 있다면 README의 업데이트
절차를 사용한다.

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/costunder/cpv26-predictor.git
cd ~/projects/cpv26-predictor
nvidia-smi
# 공식 PyTorch 설치 화면에서 이 서버에 맞게 선택한 --index-url 값을 붙여 넣는다.
read -r -p '공식 PyTorch CUDA index URL: ' TORCH_INDEX_URL
TORCH_INDEX_URL="$TORCH_INDEX_URL" bash scripts/setup.sh ml-cuda
test -f .env || cp .env.example .env
chmod 600 .env
source scripts/activate.sh
cpv26 gpu-check --device cuda:0
cpv26 db-init
cpv26 db-check
cpv26 kbo-fetch
cpv26 kbo-import
cpv26 db-check
cpv26 kbo-graph-build
cpv26 relgnn-train --device cuda:0 --epochs 30 --batch-days 2 --amp auto \
  --run-dir var/runs/relgnn/kbo_2023_2024_v1
cpv26 relgnn-evaluate --checkpoint var/runs/relgnn/kbo_2023_2024_v1/best.pt \
  --split test --device cuda:0
```

`setup.sh`는 기본적으로 `python3`를 사용하고 Python 3.10~3.12인지 검사한다. 서버의
Python 명령이 다르면 `PYTHON_BIN`을 지정한다. CUDA wheel 선택은
[공식 PyTorch 설치 화면](https://pytorch.org/get-started/locally/)에서 확인한다.

```bash
PYTHON_BIN=python3.12 TORCH_INDEX_URL="$TORCH_INDEX_URL" bash scripts/setup.sh ml-cuda
```

Profile은 다섯 개다.

| profile | 설치 내용 | 용도 |
|---|---|---|
| `base` | runtime만 | DB, feature, snapshot, CLI |
| `dev` | runtime + Ruff/mypy/pytest/coverage | 개발·검증 |
| `tabular` | runtime + dev + CatBoost | 실제 KBO baseline; PyTorch 불필요 |
| `ml-cpu` | runtime + dev + CatBoost + 공식 CPU PyTorch wheel | 별도 CPU 검증 환경 |
| `ml-cuda` | runtime + dev, 작동하는 CUDA torch 유지 또는 명시적 공식 CUDA wheel 설치 | 주 실행 경로: GPU RelGNN |

`setup.sh`는 기존 `.venv`를 재사용하되 Python·pip가 빠진 불완전 환경은 거부한다.
`ml-cuda`는 CatBoost를 설치하지 않는다. CPU 검증용 `ml-cpu`를 GPU 환경에 다시 실행하면
CPU torch로 바뀔 수 있으므로 두 용도를 구분한다.

기존 CUDA torch가 실제 forward/backward 검사까지 통과하면 변경하지 않고 재사용한다.
그렇지 않으면 사용자가 선택한 `TORCH_INDEX_URL`이 필요하다. 공식 `cu숫자` index만
허용하며 CPU torch나 불완전 CUDA 설치는 upgrade/재설치 후 constraints를 다시 적용한다.
CUDA wheel을 임의 추측하지 않는다. 설치 후 실제 CUDA 연산·gradient 검사가 실패하면
exit code 3으로 실패한다. 학습 시에도 CUDA 실패를 CPU로 조용히 대체하지 않는다.

오래 걸리는 실행은 README의 `tmux` 절차를 따른다. `last.pt`는 재개용,
`best.pt`는 validation 최적 checkpoint 평가용이다.

서버 실행을 마치면 소유자가 GitHub 설정에서 저장소를 private으로 직접 전환한다.
자동 전환 작업은 없다. 이미 받은 소스·설치 환경·데이터·checkpoint를 사용하는
학습·재개·평가는 계속 가능하지만, 이후 새 clone과 pull에는 저장소 접근 권한과
인증이 필요하다. public인 동안 다른 사람이 받은 복제본은 비공개 전환으로 회수할
수 없다. 원본 데이터·모델·인증 정보는 계속 Git에서 제외한다.

## 5. 의존성 결정

Runtime direct dependency:

- DuckDB
- NumPy
- Typer
- Rich
- tzdata
- pytz

Optional dependency:

- `tabular`: CatBoost
- `neural`: PyTorch
- `dev`: Ruff, mypy, pytest, pytest-cov

`scikit-learn`은 현재 코드가 직접 사용하지 않아 제거했다. `pytz`는 소스에서 직접
import하지 않지만 DuckDB `TIMESTAMPTZ`→Python timezone datetime 변환 경로에서 실제로
필요해 runtime에 남겼다.

`requirements/constraints.txt`는 Python 3.12에서 검증한 core/dev와 CatBoost 버전을
고정한다. CUDA PyTorch wheel만 GPU host의 driver/runtime에 맞춰 선택한다.

이전 tabular 실제 데이터 CPU 검증 버전:

```text
Python    3.12.13
DuckDB    1.5.5
NumPy     2.2.6
CatBoost  1.2.10
pytest    8.4.2
Ruff      0.16.5
mypy      1.20.2
```

위 표는 CatBoost 검증 당시 환경 기록이다. 이후 별도 CPU PyTorch 환경에서 실제
KBO graph cache를 읽는 RelGNN 축소 학습을 추가로 실행했다. 예전 neural 단위 테스트와
실제 데이터 CPU 실행, 아직 하지 못한 NVIDIA GPU 학습을 구분한다. GPU host에서는
`gpu-check` 출력의 PyTorch/CUDA 버전·장치 capability를 실행 기록과 함께 보존한다.

## 6. 환경 설정

```dotenv
CPV26_HOME=./var
CPV26_DB_PATH=./var/cpv26.duckdb
CPV26_TIMEZONE=Asia/Seoul
CPV26_DEVICE=auto
CPV26_RANDOM_SEED=2026
CPV26_LOG_LEVEL=INFO
```

`Settings.from_environment()`는 timezone/device/seed를 검증하고 log level을 대문자로
정규화한다. 상대 경로는 repository root 기준으로 해석한다. `relgnn-train`의 기본
장치는 설정의 `auto`와 별개로 `cuda:0`이며 CPU 검증은 `--device cpu`를 명시해야 한다.

## 7. 의도한 전체 흐름과 현재 연결 상태

목표 흐름:

```text
허가된 KBO feed
  → source adapter / internal ID mapping
  → append-only PIT DuckDB
  → cutoff feature + graph snapshot
  → CatBoost / role-aware RelGNN / direct heads
  → neural PA adapter
  → sequential Monte Carlo simulator
  → temporal OOF stacker + calibration
  → V26 match/live-hit optimizer
  → recommendation artifact
```

현재 실제 데이터로 연결된 주 학습 경로(CPU 축소 학습 확인, GPU 실행은 서버 검증 필요):

```text
공개 KBO Parquet(revision + SHA-256 고정)
  → pitch를 PA·경기 단위로 축약하고 품질 보고
  → canonical DuckDB
  → 과거 90일 관계·역할 feature → 날짜별 NPZ graph cache + manifest
  → 여러 날짜의 disjoint graph batch → 공유 role-aware RelGNN
  ├→ Match: 경기 승/무/패 + 양 팀 NB2 득점 분포
  ├→ Conditional Live Hit: 관측 PA≥1 선수의 PA/안타 joint 분포
  └→ PA: 별도 pre-PA context를 쓰는 10종 타석 결과
  → 2023 train / 2024 validation → best.pt + last.pt + 학습 기록
  → 명시적 relgnn-evaluate → 2025 test metrics + 작업별 prediction Parquet
```

공유 graph 표현을 사용하되 작업별 query·label·loss·출력을 분리한다. 같은 날짜 결과는
과거 관계 feature에 넣지 않고 label로만 사용한다. 현재 타석 직전 상태는 PA 보조 작업에만
주며 경기 전 Match/Live Hit head로 넘기지 않는다. Live Hit는 실제 PA가 관측된 선수
조건부이므로 후보 중 미출전까지 포함한 추천 확률과 같지 않다.

선택적 CatBoost 경로는 과거 팀 성적·Elo→WDL, 과거 선수·상대팀 성적→any-hit의 두
별도 모델이다. 이 경로의 전체 시즌 실행 결과와 `.cbm` 기록은 유지하지만 GPU 주경로가
아니며 RelGNN 성적과 혼동하지 않는다.

기존 연구 프레임워크에서 단위 테스트로 연결한 구간:

```text
append-only DB
  → snapshot
  → feature/graph/model primitive
  → task-separated model/loss/alternating trainer
  → simulator primitive
  → optimizer primitive
```

당일 RelGNN 추천기로 연결하려면 남은 구간:

- NVIDIA 서버에서 전체 시즌 GPU 학습·메모리·재개 검증
- 학습 산출물→당일 inference orchestration
- 동일 조건의 시간순 OOF 학습·calibration과 모델 비교
- 출전 확률·라인업·포지션 자격·선택률을 포함한 실제 후보 입력
- 공식 V26 점수표·구간표 replay→verified scoring configuration

`kbo-graph-build`, `relgnn-train`, `relgnn-evaluate`가 실제 KBO neural 경로다.
`predict-today`는 아직 없다. 선택적 `kbo-match-evaluate`와 `kbo-live-hit-evaluate`도
실제 학습을 수행한다. 공개 파일에는 당시 게시 시각이 없어 날짜 기준 보수적 이력
재구성을 사용한다. 엄밀한 과거 공개시각 replay는 아니다.

## 8. 시간 계약

각 변동 행은 다섯 temporal column을 갖는다.

| column | 의미 |
|---|---|
| `event_at` | 사건 발생 또는 효력 시각 |
| `available_at` | 예측자가 정보를 사용할 수 있게 된 시각 |
| `ingested_at` | 시스템이 행을 관측한 시각 |
| `valid_from` | 업무 버전 효력 시작 |
| `valid_to` | 업무 버전 효력 종료, null이면 open |

규칙:

- naive datetime은 public domain API에서 거부한다.
- 저장과 manifest serialization은 UTC다.
- 일반 snapshot source row는 `available_at <= cutoff_at`이다.
- bitemporal replay는 `ingested_at <= knowledge_at`이다.
- 관측 결과 table은 필요할 때 `event_at < cutoff_at`도 적용한다.
- 생성된 run-scoped state/candidate는 cutoff 이후 만들어질 수 있어
  `ingested_at <= knowledge_at`을 사용한다.

Graph temporal contract:

```text
eligibility            available_at <= cutoff
event recency          cutoff - event_at
publication delay      available_at - event_at  # opt-in feature
```

`available_at`을 recency로 쓰지 않는다. 오래된 경기가 최근 정정됐다고 최근 경기로
취급하는 오류를 막는다.

## 9. Schema v4 table 목록

Metadata 포함 36개다.

| table | 역할 |
|---|---|
| `schema_migration` | 설치된 schema version 기록 |
| `source_revision` | 원천 content hash와 lineage |
| `prediction_run` | immutable 예측 실행 계약 |
| `prediction_run_status_event` | created/snapshotted/scored/failed 상태 이벤트 |
| `v26_live_hit_rule_set` | 규칙 JSON과 official/observed/provisional 출처 |
| `v26_player_position_eligibility` | slate·Live 카드별 선수 포지션 자격 |
| `v26_selection_snapshot` | phase·포지션별 선택률 snapshot |
| `user_collection_snapshot` | 사용자·Live 카드별 도감 보유 상태 |
| `player` | 선수 identity와 정적 정보 revision |
| `team` | 팀 identity revision |
| `game` | 일정·홈/원정·결과·더블헤더·재개 경기 revision |
| `team_season` | 시즌 단위 팀 context |
| `team_game` | 경기별 팀 결과 |
| `roster_spell` | 선수 소속·1군 상태 유효기간 |
| `lineup_version` | projected/announced/official/final 라인업과 발표 시각 |
| `lineup_entry` | 라인업별 선수·타순·포지션·선발 여부 |
| `observed_plate_appearance` | 실제 PA와 전후 점수·아웃·주자 전이 |
| `pitching_appearance` | 등판 역할·BF·투구 수·아웃·실점 |
| `player_game_candidate` | 예측 run별 미래 경기 선수 시나리오 |
| `player_state_snapshot` | run별 선수 feature snapshot |
| `team_state_snapshot` | run별 팀 feature snapshot |
| `model_prediction` | run·entity별 예측 artifact |
| `stadium` | 구장 좌표·돔 여부·시간대 revision |
| `game_status_snapshot` | scheduled/delayed/cancelled/postponed/suspended/no-result 상태 |
| `starter_announcement` | projected/announced/confirmed/changed/scratched 선발 |
| `player_game_batting` | 공식 선수 타격 박스스코어 정합성 기준 |
| `substitution_event` | 대타·대주자·투수·수비 교체 sequence |
| `runner_event` | 도루·견제·폭투 등 비PA 주자 상태 변화 |
| `fielding_assignment` | sequence 구간별 수비 포지션 |
| `catcher_assignment` | sequence 구간별 포수 |
| `weather_station_version` | ASOS/AWS 등 관측 지점 revision |
| `stadium_weather_station_map` | 구장과 예보/관측 지점의 versioned mapping |
| `weather_forecast_snapshot` | 발표·대상·캡처 시각과 원문 hash가 있는 예보 revision |
| `weather_observation` | 실제 관측; 사후 검증 또는 oracle-weather 전용 |
| `v26_slate` | 날짜·lock·규칙·Live 카드·자격 snapshot의 결합 단위 |
| `v26_submission` | 선수와 시너지 팀을 포함한 실제 선택 행동 |

v4에서 `observed_plate_appearance`에 추가된 핵심 컬럼:

```text
home_score_before / away_score_before
home_score_after / away_score_after
outs_added
runners_after
event_subsequence
transition_complete
```

v3 legacy 행에는 알 수 없는 전후 상태를 만들어 넣지 않는다. migration은 nullable 상태와
`transition_complete=false`를 사용하고, simulator-ready 감사만 해당 행을 거부한다.

현재 schema에도 없는 범위:

- pitch-by-pitch 구종·구속·location 원장
- 자유형 부상 텍스트와 의학적 상태 추론
- 공식 V26 점수 화면에서 검증한 과거 규칙·선택률 replay fixture
- 공식 기록원 수준의 RBI/earned-run 판정 원장

선택한 공개 데이터 adapter는 `source_revision`, `team`, `player`, `game`, `team_game`,
`observed_plate_appearance` 여섯 테이블을 채운다. 나머지 테이블의 데이터까지 확보한
것은 아니다. 선수 공식 박스스코어·RBI·실책·라인업·주루 이벤트를 임의로 만들지 않는다.

## 10. As-of query

`DuckDBStore.as_of_sql(..., current_only=True)`는 2단계 version ranking을 한다.

```text
eligible:
  available_at <= cutoff
  ingested_at <= knowledge

correction ranking:
  natural_identity + valid_from별
  available_at DESC, ingested_at DESC, row_id DESC

validity:
  valid_from <= cutoff < valid_to 또는 valid_to IS NULL

business version ranking:
  natural_identity별 valid_from DESC
```

이 순서가 해결하는 두 문제:

1. 미리 발표된 미래 valid_from 행이 현재 행을 rank 2로 밀어 빈 결과를 만드는 문제
2. 최신 correction이 version을 닫았는데 이전 open revision이 다시 살아나는 문제

`observed_before_cutoff`는 correction 선택 후 적용해 event timestamp correction 때문에
superseded row가 부활하지 않게 한다.

`player_game_candidate` natural identity는
`(prediction_run_id, candidate_id)`다. 다른 run에서 같은 candidate ID를 재사용해도 서로
가리지 않는다.

## 11. Prediction run 불변성과 migration

Schema v4에서 `prediction_run_id`는 `UNIQUE`다. 같은 ID의 두 번째 append는
`AppendValidationError`다. 다음 값은 한 run ID에 고정된다.

- target game
- cutoff / knowledge
- horizon
- feature version / fingerprint
- model version
- simulator version
- V26 rule version
- config JSON

상태 변경은 `prediction_run_status_event`에 추가한다.

v1→v2→v3→v4 migration:

1. transaction 시작
2. v1 column contract 확인
3. duplicate prediction run ID 탐지
4. v1 `prediction_run`에 의존하는 index 제거
5. legacy table rename
6. v2 immutable run table 생성·복사
7. legacy status를 status event로 이관
8. backup table 삭제
9. canonical index 재생성
10. schema version 2 기록과 signature 검사
11. Live Hit 4개 table·index를 설치하고 schema version 3 기록
12. 경기 상태·선발·박스스코어·전이·날씨·V26 slate 등 v4 table·index 설치
13. legacy 선택률·자격·도감 행의 `captured_at`을 기존 `event_at`으로 보존 backfill
14. legacy PA는 전이값을 추측하지 않고 `transition_complete=false`로 표시
15. v4 필수 컬럼·TIMESTAMPTZ·index·참조 계약 검사 후 commit

중복 run ID나 column mismatch가 있으면 전부 rollback한다. 실제 v1
`idx_prediction_run_id`가 존재하는 fixture, 실제 v3 행 보존, v4 fresh install을 모두
검증한다. DuckDB는 의존 index가 있는 컬럼에 `SET NOT NULL`을 적용하기 어려우므로
migrated DB의 backfill 컬럼은 값 계약으로 검사하고, fresh v4 DDL은 `NOT NULL`을 강제한다.

중요한 보장 경계:

- `DuckDBStore`는 update/delete API를 제공하지 않는다.
- append API와 DB unique constraint는 run 재사용을 막는다.
- 그러나 public `store.connection`으로 직접 raw `UPDATE`를 실행하면 DuckDB 자체가 이를
  막지는 않는다.

운영에서는 DB 파일 쓰기 권한을 ingestion process로 제한하고, 애플리케이션 코드가
`connection.execute("UPDATE ...")`를 사용하지 않도록 해야 한다. 완전한 DB-level
immutability가 필요하면 write service 분리나 별도 audit hash chain이 다음 단계다.

## 12. Append transaction 원자성

다중 행 `append()`는 호출 자체 transaction을 갖는다. 두 번째 행에서 CHECK constraint가
실패해도 첫 번째 행이 남지 않는다.

명시적 `with store.transaction():` 안에서는 append가 outer transaction을 사용한다.
내부 append 예외를 호출자가 잡더라도 store가 failed flag를 기억하고 context 종료 시
전체 rollback 후 오류를 낸다. nested explicit transaction은 지원하지 않고 명확히
거부한다.

## 13. 참조 무결성 감사

Feed는 부모와 자식이 다른 순서로 도착할 수 있으므로 physical FK를 DDL에 강제하지
않는다. 대신 `data/integrity.py`가 다음 reference를 감사한다.

- source-backed row→`source_revision`
- game→home/away team
- roster/lineup/PA/pitching→player/team/game
- candidate/state/prediction/status→`prediction_run`
- generated entity→player/team/game

API:

```python
violations = store.reference_violations(sample_limit=10)
store.assert_referential_integrity()
store.assert_composite_referential_integrity()
pit_violations = store.as_of_reference_violations(
    cutoff_at=cutoff_at,
    knowledge_at=knowledge_at,
)
```

Violation은 rule, missing distinct value count, sample value를 포함한다. `db-check`는
schema뿐 아니라 이 감사를 실행한다.

단일 컬럼 규칙 외에 lineup, player-game batting, V26 slate/rule/eligibility/selection/
submission의 composite business key도 검사한다. `as_of_reference_violations()`는 부모가
물리적으로 존재하더라도 해당 cutoff/knowledge에는 아직 발표되지 않았으면 위반으로
보고한다. 적재 완료 후 물리 감사와 snapshot cutoff별 시간 감사를 둘 다 실행해야 한다.

## 14. Snapshot과 fingerprint

`SnapshotBuilder`는 저장된 prediction run에서 cutoff/knowledge를 다시 읽는다. 호출자가
다른 값을 덮어쓰면 거부한다.

Source table:

```text
available_at <= cutoff
ingested_at <= knowledge
```

Run-generated table:

```text
prediction_run_id = current run
ingested_at <= knowledge
```

Manifest artifact별 기록:

- table 이름
- Parquet 상대 경로
- row 수
- column 이름
- logical content SHA-256
- file SHA-256
- current/observed/run-scoped flags
- 명시적 table filter

전체 fingerprint는 schema version, run ID, cutoff, knowledge와 table digest에서 계산한다.
같은 destination이 이미 있으면 manifest와 파일 hash를 검증한다. 동일 fingerprint면
idempotent return, 다르면 `FileExistsError`다.

Status event는 기본 snapshot에서 제외한다. snapshot 완료 status가 fingerprint를 다시
바꾸는 순환을 막기 위한 의도적 선택이다.

Live Hit 입력은 기본 baseball snapshot에 모든 사용자 계정을 넣지 않는다.
`live_hit_snapshot_specs()`에 다음 scope를 반드시 제공한다.

```text
user_id / slate_id / live_card_version / rule_version
position_eligibility_snapshot_id / selection_snapshot_id
```

이 filter도 manifest와 fingerprint에 포함되므로 다른 계정·다른 선택률 snapshot을 같은
run artifact로 오인하지 않는다.

`v26_slate`는 scoped 입력에 포함하지만 `v26_submission`은 제외한다. submission의
`selected_player_id`와 `selected_synergy_team_id`는 예측 전에 알려진 feature가 아니라
optimizer가 결정한 행동이기 때문이다.

### 14.1 데이터셋 계약과 시즌 역할

`data/dataset_contracts.py`는 provider별 parsing이 끝난 뒤 다음 네 계층을 감사한다.

1. 시즌·game-group 시간 분할
2. PA↔선수 박스스코어↔팀 득점 정합성
3. PA 전후 상태와 비PA runner event 연속성
4. 날씨 provenance와 V26 capture phase 완전성

고정 시즌 역할:

| season | role | 허용되는 의사결정 |
|---|---|---|
| 2018~2022 | `base_train` | 기본 모델 적합 |
| 2023 | `model_selection` | feature·route·hyperparameter 선택 |
| 2024 | `calibration` | OOF stacking·확률 보정 |
| 2025 | `holdout` | 구조·feature 변경 없이 최종 평가 |
| 2026 | `live` | 2018~2025 재학습 후 shadow/live |

동일 `game_id`의 선수·PA·horizon 행은 같은 partition에만 존재해야 한다. 2025 결과를 본
뒤 구조를 바꾸면 holdout이 아니므로 새 미래기간 평가가 필요하다.

PA 전이 감사 항목:

```text
before score + runs_scored == after score
outs_before + outs_added <= 3
third out 이후 runners_after == 000
동일 game의 canonical (sequence_in_game, event_subsequence) 중복 금지
인접 PA의 inning/half/score/outs/runners 연속성
```

두 PA 사이에 `runner_event`가 있으면 주자 상태 직접 비교는 생략하되 이벤트 누락으로
간주하지 않는다. `transition_complete=false` legacy 행은 일반 PA 학습에는 남길 수 있지만
`assert_simulator_ready_transitions()`를 통과하지 못한다.

날씨 실험:

- `forecast`: `forecast_issued_at`과 `available_at`이 cutoff 이하인 revision만 허용
- `oracle_weather`: 경기 시점 observation만 허용하며 일반 backtest와 별도 보고

2018~2025 당시 예보 archive가 없으면 날씨 feature를 제외하거나 oracle이라고 표시한다.
실제 관측을 당시 예보로 위장하면 데이터 누수다.

V26 capture phase는 정확히 다음 순서다.

```text
early → starter_known → lineup_known → near_lock
```

`audit_v26_capture_consistency()`는 네 phase 누락·순서, snapshot ID 재사용, slate/rule/Live
카드/eligibility 불일치, ineligible 선수, lock 이후 capture, 포지션별 중복을 거부한다.

평가 이름은 데이터 가용성에 따라 분리한다.

```text
2018~2025: fixed_300 counterfactual
2026 capture 시작 이후: 네 phase가 모두 있는 slate만 actual replay 후보
```

과거 선택률·포지션 자격·규칙·사용자 도감 snapshot이 없으면 실제 V26 replay라고 부르지
않는다.

## 15. Feature 계층

현재 구현 범위는 강한 전체 KBO feature set이 아니라 primitive다.

`features/statistics.py`:

- `CountRate`
- beta prior/posterior
- empirical Bayes shrinkage
- EWMA와 time-decay EWMA
- rolling count rate/sum/mean

`features/batting.py`:

- cutoff 이전 PA parsing
- 타격 count rate
- recent window와 EWMA
- shrinkage된 batter feature vector

`features/state.py`:

- player/team feature JSON row 생성
- numerator/denominator 분리
- source fingerprint 검증

아직 필요한 feature:

- starter/bullpen workload와 pitch count
- platoon/handedness split
- catcher framing/game-calling proxy
- defense/position value
- park factor와 weather
- travel/rest/doubleheader
- injury/roster availability
- lineup uncertainty scenario

## 16. Graph snapshot과 route registry

Graph는 foreign-key path를 자동 탐색하지 않는다. `RouteRegistry` whitelist에 등록한
atomic route만 허용한다.

기본 route:

| route | source role | event/bridge | destination role |
|---|---|---|---|
| `batter_pa_pitcher` | player/batting | observed PA | player/pitching |
| `pitcher_pa_catcher` | player/pitching | observed PA | player/catcher |
| `batter_pa_game` | player/batting | observed PA | game |
| `player_lineup_game` | player/shared | lineup entry | game |
| `pitcher_appearance_team_game` | player/pitching | pitching appearance | team game |
| `player_roster_team_season` | player/shared | roster spell | team season |
| `home_team_game_away_team` | team/home | game | team/away |
| `player_candidate_game` | player/batting | player game candidate | game |

`AtomicRouteBatch`:

- source/destination indices
- event_at/available_at
- numeric event feature matrix
- edge weights

Validation:

- route name whitelist
- endpoint type 일치
- available_at cutoff 차단
- index 범위
- non-negative finite weight
- rectangular event feature

`GraphSnapshot.node_feature_dims`는 빈 node type의 width를 보존한다. 빈 rows를 torch로
바꿀 때 `(0,)`가 아니라 `[0, feature_dim]`이다.

## 17. 선수 역할 state

`RoleAwarePlayerEncoder` 구조:

```text
shared feature
  → shared MLP core
  → role-specific gated residual adapter
      batting
      pitching
      defense
      baserunning
      catcher
```

Route alias `batter`, `pitcher`, `fielder`, `runner`는 canonical role로 normalize한다.

Backbone 내부 channel:

```text
player
player__batting
player__pitching
player__defense
player__baserunning
player__catcher
game
team
...
```

Role state가 주어지면 player endpoint는 선언된 role channel을 반드시 사용한다. 필요한
role이 없으면 오류다. Role state를 주지 않은 기존 호출은 shared player fallback을
유지한다.

## 18. RelGNN backbone

`CompositeRelGNNBackbone` layer의 두 aggregation 수준:

### Route 내부

```text
source projection
+ event encoder
+ signed log event age / decay
+ optional publication delay
  → key/value

destination state
  → query

destination별 multi-head attention softmax
  → one route summary per destination
```

Edge weight 0은 mask된다. 각 destination/head별 stable max와 denominator를 계산한다.

### Route 사이

```text
[previous destination state, route summary]
  → learned route gate score
  → available route mask
  → route softmax
  → combined message
  → GRUCell + LayerNorm
```

따라서 한 route의 edge 100개와 다른 route의 edge 1개를 한 degree denominator로 나누지
않는다. 동일 unrelated edge를 복제해도 다른 route의 summary가 희석되지 않는 regression
test가 있다.

이름은 RelGNN 계열 구조를 나타낸다. destination-query MHA, intermediate event encoder,
route-level combination이라는 핵심은 구현했다. 다만 특정 논문 공식 code와 완전 동일한
재현이라고 주장하지 않는다. 논문 비교를 하려면 dataset, route construction, loss와
training budget까지 맞춰야 한다.

## 19. Neural PA와 direct head

`PlateAppearanceInteractionDecoder` input:

- batter embedding
- pitcher embedding
- elementwise product
- absolute difference
- optional catcher embedding
- optional defense embedding
- game context

Output은 기본 10개다.

```text
strikeout
walk_or_hbp
single
double
triple
home_run
ball_in_play_out
reached_on_error
sacrifice_hit
sacrifice_fly
```

`DirectPlayerGameHead`:

- `appearance_probability`
- PA 0 + positive count + overflow distribution
- PA bucket마다 별도 parameter를 갖는 conditional `P(H | PA, x)`
- masked joint `P(PA, hits)` with `hits <= PA`
- hit marginal
- marginal에서 계산한 expected hits
- 동일 joint distribution을 학습하는 `negative_log_likelihood()`

Overflow bucket 예:

```text
max_plate_appearances=4 → 0,1,2,3,4,5+
max_hits=3              → 0,1,2,3,4+
```

`DirectRunDistributionHead`는 home/away mean과 dispersion 네 개만 출력한다. marginal NB
loss에서 gradient가 없던 `score_correlation`은 제거했다.

`WDLHead`는 away win/draw/home win logits를 출력한다.

### 19.1 작업별 학습 경계

범용 `TaskSeparatedModel`/`AlternatingMultiTaskTrainer`는 서로 다른 row granularity를
하나의 task batch로 섞지 않는다. 아래 계약과 별개로 실제 KBO runner는 날짜별 graph를
공유하면서 작업별 query·label·loss를 분리한다(§19.2).

| task | 한 행 | label | loss |
|---|---|---|---|
| PA | 과거 타석 | 10-way neural target | cross entropy |
| Live Hit | 선수-경기-cutoff | 출장, PA, hits | joint PA/hit NLL |
| Match | 경기-cutoff | away/draw/home, 양 팀 득점 | WDL CE + 두 NB NLL |

감독 mask도 작업별로 분리한다.

- Live Hit: `label_observed & game_played`만 PA/hit NLL에 포함한다.
- 취소·노게임: `label_observed=true`, `game_played=false`; 정상 경기 벤치/0 PA와 다르다.
- Match: `result_observed & completed`만 CE/NB에 포함한다.
- 미완료·노게임: 점수와 WDL에 `-1` sentinel을 강제해 가짜 0:0 무승부를 막는다.
- 유효 감독 행이 하나도 없는 배치는 오류이며, `sample_count`는 실제 사용 행 수다.

`TaskSeparatedModel`은 하나의 registered backbone 뒤에 PA, Live Hit, Match adapter와
head를 각각 등록한다. `AlternatingMultiTaskTrainer`는 세 finite loader를 고정 순서로
번갈아 소비하며 shared backbone에는 세 작업의 gradient가 들어가고 다른 작업 head에는
해당 step의 gradient만 들어간다.

Checkpoint에 포함:

- model / optimizer state
- epoch, global step, task별 step
- CPU/CUDA RNG state
- loss weights와 task order
- feature, route, label schema, model version lineage

Lineage가 현재 trainer와 다르면 resume를 거부한다. 이 범용 trainer는 in-memory tensor
batch 이후의 계약이며, 실제 KBO data loader·파일 checkpoint·실행 CLI는 다음 계층이다.

### 19.2 실제 KBO RelGNN 학습 경로

- `data/kbo_graph_dataset.py`: DB에서 날짜별 과거 90일 관계를 구성해
  `days/YYYY-MM-DD.npz`와 sidecar, `manifest.json`을 만든다. `allow_pickle=False`로
  읽고 파일 hash를 검사한다. 해당 날짜 전의 event만 사용하며 availability/validity
  cutoff를 적용한다. 같은 날 경기는 순서에 상관없이 그날 graph의 과거 이력에 넣지 않는다.
- `models/kbo_relgnn.py`: `RoleAwarePlayerEncoder`를 쓰는 공유
  `CompositeRelGNNBackbone` 뒤에 Match WDL, 조건부 Live Hit joint PA/hit, PA10 head와
  보조 NB2 양 팀 득점 head를 연결한다. 과거 타자↔투수, 타자/투수↔팀, 홈↔원정팀
  관계를 사용하며 실제 당일 라인업·선발을 사전에 알려진 feature로 위장하지 않는다.
- `training/kbo_runner.py`: graph mini-batch, GPU 전송, 학습·validation, checkpoint,
  재개, 별도 holdout 평가를 연결한다. 실제 KBO data loader는 generic snapshot Parquet를
  직접 받는 대신 위 전용 NPZ cache를 읽는다.

세 작업은 다음과 같이 분리된다.

| 작업 | 예측 시점·대상 | 학습 출력 |
|---|---|---|
| Match | 해당 날짜 전 이력, home/away team query | away/draw/home CE + 보조 양 팀 NB2 득점 NLL |
| Conditional Live Hit | 해당 날짜 전 이력, 실제 완료 관측 PA≥1 선수-경기 | PA/안타 joint NLL와 `P(H≥1 | 관측 PA≥1)` |
| PA | 같은 과거 graph와 별도의 현재 pre-PA context | 10-way CE |

조건부 Live Hit는 범용 `DirectPlayerGameHead`의 미출전·0 PA 예측과 다르다.
최대 count를 넘는 bucket의 기대값은 lower bound로 보고한다. 포수 타격방해는 PA10에서
제외한다. 점수 전이가 불완전한 PA는 라벨을 보존하되 직전 양 팀 점수를 unknown으로
마스킹해 현재 타석의 결과 점수가 context로 새지 않게 한다. cache version 변경 후에는
`kbo-graph-build`를 다시 실행하며 이전 cache/checkpoint를 서로 섞지 않는다.

기본 모델은 hidden dimension 64, 2 layers, 4 attention heads다. 기본 runner는
단일 `cuda:0`, 2일/batch, DataLoader workers 2, pinned-memory/non-blocking 전송,
날짜·route별 edge 상한 20,000, 훈련 PA query 128개/일을 사용한다. validation/test의
PA query는 전부 평가한다. graph edge 상한은 평가에도 적용되므로 무제한 전체 관계
graph라고 주장하지 않는다. AMP `auto`는 장치에 따라 BF16/FP16을 선택하며 `off`도
가능하다. AdamW, gradient clipping, gradient accumulation을 지원한다.

2023만 train, 2024만 validation으로 사용한다. `--patience 6`은 validation loss가
개선되지 않는 epoch가 연속 6개면 조기 종료하며 0이면 비활성화한다. 2025 test는
학습에서 평가하지 않고 `relgnn-evaluate --split test`로 명시적으로 실행한다.
`best.pt`는 validation 최적 모델, `last.pt`는 마지막 완료 epoch의 재개용 상태다.
atomic checkpoint에는 모델·optimizer·AMP scaler·CPU/CUDA RNG·설정·dataset fingerprint가
포함된다. 재개는 같은 run directory의 `last.pt`만 허용하며 호환되지 않는 데이터나
모델 설정은 거부한다. `--epochs`는 추가 횟수가 아니라 전체 목표 epoch다.

`--device cpu --amp off --workers 0 --epochs 1 --max-days-per-split 3`은 별도 축소 검증이다.
각 시즌 전체 날짜 범위에서 균등하게 3일을 고르며 첫 3일만 쓰는 방식이 아니다. 이를
전체 시즌 모델 성적이나 GPU 메모리 검증으로 해석하면 안 된다. 실행 절차·산출물은
[GPU_TRAINING.md](GPU_TRAINING.md)를 따른다.

## 20. 14 terminal event와 10 neural target 계약

Simulator terminal event:

```text
single, double, triple, home_run,
walk, hit_by_pitch, strikeout,
ball_in_play_out, double_play,
sacrifice_fly, sacrifice_bunt,
reached_on_error, fielders_choice,
catcher_interference
```

Historical target coarsening:

| terminal event | neural target |
|---|---|
| K | `strikeout` |
| BB, HBP | `walk_or_hbp` |
| 1B/2B/3B/HR | 같은 hit type |
| BIP out, DP, FC | `ball_in_play_out` |
| ROE | `reached_on_error` |
| SH | `sacrifice_hit` |
| SF | `sacrifice_fly` |
| CI | `None`, adapter rare-event rate 학습 |

`neural_training_target()`가 이 mapping을 단일 source of truth로 제공한다.

Inference decomposition:

- `walk_or_hbp`→BB/HBP
- `ball_in_play_out`→ordinary out/DP/FC
- CI absolute rare event mass
- base/outs state에서 불가능한 event mass를 legal event로 이동
- 마지막에 14-way normalization

Split은 `NeuralTerminalAdapterConfig.estimate()`가 training fold records에서 추정한다.
기록하는 lineage:

- training cutoff
- source fold ID
- records used
- global split
- state bucket split

Known revision을 PA ID별 latest available_at으로 먼저 고른 뒤 event_at cutoff를 적용한다.
최신 correction이 event를 미래로 옮겨도 과거 revision이 부활하지 않는다. 동일
available_at의 서로 다른 correction은 ambiguous 오류다.

## 21. 순차 시뮬레이터

State:

- inning / top-bottom
- outs
- 1·2·3루 runner ID
- away/home score
- 양 팀 batting index

Pitching plan:

- starter
- ordered relievers
- starter max batters faced / through inning
- reliever max batters faced

한 simulation path에서 매 PA마다:

1. 현재 batter/pitcher/catcher/context 생성
2. 14-way probability 요청
3. legality normalization
4. event sampling
5. runner advancement와 outs/score transition
6. pitcher workload와 batting order 갱신
7. 종료 조건 검사

Illegal event handling:

- 2아웃 또는 1루 주자 없는 DP→generic BIP out
- 2아웃 또는 3루 주자 없는 SF→generic BIP out
- 2아웃 또는 무주자 SH→generic BIP out
- 무주자 FC→generic BIP out

Walk-off:

- bottom regulation inning 이후에 새로 home lead를 얻어야 walk-off다.
- 이미 home lead인 initial bottom state는 거부한다.
- non-HR는 `away_score - home_score + 1`에 필요한 run만 인정한다.
- non-HR 타자의 공식 인정 루타는 결승 주자의 시작 베이스에 따라 1B·2B·3B로 조정한다.
- 원 확률 표본은 `sampled_event`, 공식 적용 결과는 `applied_event`, 인정 루타는
  `credited_total_bases`로 분리한다.
- HR은 모든 runner와 batter run을 인정한다.
- walk-off terminal state의 bases는 비운다. full advancement 뒤 남은 가짜 base state를
  노출하지 않기 위함이다.

한계:

- runner advancement rate는 현재 configuration 값이며 실제 fold estimator가 없다.
- error/automatic award와 official scorer 세부 판정은 단순화돼 있다.
- RBI, scoring runner identity, earned run 판정은 출력하지 않는다.
- live resume pitcher workload를 초기화하는 상세 API는 없다.

Live Hit가 요구하는 hit 여부와 joint player hit distribution에는 현재 계약이 충분하지만,
공식 기록 예측까지 확장하려면 scorekeeping layer가 필요하다.

## 22. 승부예측 optimizer

`MatchPredictionOptimizer`는 game별 option product를 전부 열거한다.

각 조합에서 계산:

- expected points
- point variance
- all-correct probability
- expected correct picks
- expected utility

```text
utility = expected_points - risk_aversion × variance
```

All-correct bonus는 joint outcome enumeration에 포함한다. 사용되지 않던
`MatchPickOption.selection_rate`는 제거했다. 선택률 기반 reward가 실제 ruleset에 있으면
호출자가 명시적 reward rule을 추가해야 한다.

## 23. Live Hit optimizer

Input:

- position slots
- candidate player / team / eligible positions
- 포지션별 selection rate
- account의 collection owned boolean
- joint hit scenarios
- versioned `LiveHitRuleSet`
- risk와 별도 all-hit reward utility

Ruleset mode:

| mode | 의미 |
|---|---|
| `pure_hits` | 배율 없이 안타 분포만 비교 |
| `fixed_300` | provisional 전 선수 3배, 선택 구단 선수 4배 |
| `account` | 포지션별 명시 구간표 + 실제 도감 + 선택 구단 |

제거한 확인되지 않은 가정:

- `selection_bonus_rate × (1-selection_rate)^exponent` 연속함수
- collection과 selection을 더한 뒤 다시 암묵적 `1 +` 적용
- team synergy 최소 인원과 activation points
- 올 히트 item reward를 공식 hit points에 합산
- 선수별 임의 base multiplier

`LiveHitRuleSet`은 rule version, 안타 점수표와 출처, 선택률 구간표, additive percentage
point 결합 방식, collection/선택구단 가산, all-hit reward ID를 가진다. `fixed_300`의
기본 버전과 기본 선형 안타표는 `provisional`로 표시한다. `account`는 caller가 안타표
출처와 구간표를 명시해야 한다.

제약:

- slot별 eligible position
- 동일 선수 중복 금지
- fixed/account mode에서 선택 synergy team 정확히 하나
- remaining slots의 bipartite matching 가능성 검사

출력 의미는 분리한다.

```text
expected_hit_points    ruleset으로 계산한 공식 점수 기대값
hit_point_variance     공식 점수 분산
all_hit_probability    9명 모두 1안타 이상일 공동 확률
all_hit_reward_id      별도 item reward 식별자
custom_utility         기대점수 - risk×variance + 사용자 all-hit 가치×확률
```

`all_hit_reward_utility`가 달라져도 `expected_hit_points`는 절대 변하지 않는다.

Search mode:

- `exact`: legal roster 전부 열거
- `beam`: partial rank 상위 `beam_width`만 유지

Diagnostics:

- mode / exact 여부
- beam width
- slot count
- candidate count
- eligible slot-candidate assignment count
- expanded state
- constraint-pruned state
- beam-pruned state
- completed roster
- known optimality gap

Exact 또는 beam pruning이 없으면 gap 0이다. Beam pruning이 있으면 upper bound를 계산하지
않으므로 gap은 `None`, `is_exact=False`다. 작은 독립 brute-force oracle과 exact 결과를
비균등 scenario weight, 포지션별 selection band, collection, 선택 구단 multiplier,
risk variance와 별도 all-hit utility까지 포함해 비교하는 테스트가 있다.

## 24. CatBoost와 stacking

`CatBoostClassifierBaseline`:

- lazy import
- binary/multiclass loss 자동 선택
- sample weight와 categorical feature 전달
- probability shape/finite/normalization 검증
- fitted classes와 feature importance 노출

실제 CatBoost 1.2.10 `fit()`/`predict_proba()` smoke test를 수행했다.

Stacking/calibration primitive:

- `OOFPredictionSet`
- `OOFProbabilityStacker`
- temperature calibration
- stagewise temperature calibration
- binary isotonic calibration
- OOF stacking pipeline

Meta prediction은 이전 OOF stage만 사용하도록 설계한다. 개별 CatBoost/RelGNN의 실제
학습·평가·artifact 저장은 연결했지만, 이 산출물로 시간순 OOF stage와 calibration을
자동 fitting하는 통합 orchestrator는 아직 없다.

## 25. 평가 계약

`evaluation.py`:

- expanding walk-forward split
- multiclass log loss
- multiclass Brier score
- expected calibration error
- aggregate probability metrics

Random row split은 금지한다. Split unit은 최소 game/cutoff로 묶어 동일 경기 정보가
train/validation 양쪽에 섞이지 않게 해야 한다.

권장 model promotion 조건:

```text
동일 cutoff
동일 feature set
동일 train period
동일 tuning budget
동일 calibration
동일 future holdout
```

GPU 학습 실행 경로는 RelGNN이며 CatBoost는 선택적 baseline이다. 성능 주장에 필요한
비교 순서는 별개다.

1. empirical baseline
2. CatBoost
3. MLP
4. HeteroGraphSAGE
5. destination-query RelGNN
6. sequential simulator
7. temporal OOF ensemble

RelGNN을 미리 우승 모델로 정하지 않는다. KBO의 제한된 시즌 표본에서는 CatBoost가
매우 강할 수 있다.

## 26. CLI

```bash
cpv26 show-config
cpv26 db-init
cpv26 db-check
# 기존 prediction run이 있을 때만 실제 ID로 실행: cpv26 snapshot-build RUN_ID
cpv26 kbo-fetch
cpv26 kbo-import
cpv26 gpu-check --device cuda:0
cpv26 kbo-graph-build
cpv26 relgnn-train --device cuda:0 --epochs 30 --batch-days 2 --amp auto \
  --run-dir var/runs/relgnn/kbo_2023_2024_v1
cpv26 relgnn-evaluate --checkpoint var/runs/relgnn/kbo_2023_2024_v1/best.pt \
  --split test --device cuda:0
# 아래 두 명령만 선택적 CatBoost baseline이다.
cpv26 kbo-match-evaluate
cpv26 kbo-live-hit-evaluate
```

`db-init`:

- runtime directory 생성
- 새 schema v4 설치 또는 compatible v1/v2/v3 migration

`db-check`:

- read-only open
- schema version/table/temporal/필수 column/unique constraint 검사
- reference audit

`snapshot-build`:

- 기존 run 조회
- deterministic Parquet materialization
- manifest/hash 검증

`kbo-fetch` / `kbo-import`:

- 기본 시즌은 완결된 2023~2025다. `--year` 반복으로 선택하며 2026은 7월 26일까지다.
- 다운로드는 고정 revision과 파일별 SHA-256을 검사하고 `SOURCE.json`을 남긴다.
- importer는 schema 초기화, deterministic ID 적재, 참조 감사와 품질 보고를 수행한다.
- 기본 경로는 `var/datasets/kbo_playbyplay/v0/`와 `var/reports/kbo_import.json`이다.

`gpu-check`:

- 기본 `cuda:0`에서 실제 matrix 연산 forward/backward와 finite gradient를 검사한다.
- 장치 이름, capability, PyTorch/CUDA 버전, AMP 선택과 메모리를 보고한다.
- 성공은 exit code 0, CUDA 사용 불가·연산 실패는 exit code 1이다. CPU로 fallback하지 않는다.

`kbo-graph-build`:

- 기본 2023~2025 날짜를 대상으로 과거 90일의 safe NPZ graph cache를 생성한다.
- 기본 위치는 `var/datasets/kbo_graph`, `--output`으로 변경한다.
- 기존 날짜별 cache는 입력 fingerprint와 파일 hash가 맞을 때만 재사용한다.
- 494일·2,160경기·49,091 Live Hit query·169,477 PA10 label을 생성 확인했다.
  포수 타격방해 4건은 PA10에서 제외된다. 전이 불완전 PA 11건의 직전 점수 unknown
  mask가 포함된 cache version으로 재생성하여 사용한다.

`relgnn-train`:

- 기본 dataset은 위 graph cache다. `--dataset`/`--run-dir`로 경로를 지정한다.
- 기본 `--device cuda:0 --epochs 30 --batch-days 2 --amp auto --workers 2`다.
- 2023 train/2024 validation만 사용하며 2025 test를 학습 중 들여다보지 않는다.
- `config.json`, `history.jsonl`, `best.pt`, `last.pt`, `training_report.json`을 저장한다.
- `--resume .../last.pt`와 전체 목표 `--epochs`로 재개한다. `--run-dir` 생략 시
  해당 checkpoint의 부모 경로를 사용하며 dataset fingerprint/config 호환성을 검사한다.
- `--batch-days 1`, `--accumulate-steps`로 메모리·유효 batch를 조절한다.
  `--max-pa-per-day`는 훈련 PA query에만 적용된다.

`relgnn-evaluate`:

- 저장된 checkpoint와 동일 fingerprint의 dataset을 읽고 명시한 split을 평가한다.
- 기본 출력은 checkpoint 폴더의 `evaluations/test-<run-id>/metrics.json` 및
  `match_predictions.parquet`, `live_hit_predictions.parquet`, `pa_predictions.parquet`이다.
- 학습 날짜를 제한한 CPU smoke checkpoint는 평가 결과에도 `smoke_test_only`로 표시한다.

`kbo-match-evaluate` / `kbo-live-hit-evaluate`:

- `tabular` 설치가 필요하다. 각각 2023→2024, 2023~2024→2025 두 fold를 학습한다.
- 과거 날짜만 feature에 반영하고 같은 날짜 경기는 일괄 예측한 뒤 이력을 갱신한다.
- `--iterations` 기본값은 400, `--report`로 JSON 경로를 지정할 수 있다.
- 모델별 `var/reports/kbo_*_baseline.json`과
  `var/models/kbo_*_baseline/<run-id>/{validation_2024,test_2025}.cbm`을 저장한다.
- 실행별 모델 폴더에는 `evaluation.json`도 보존한다. 기본 보고서만 최신 결과로
  교체하고 이전 모델은 덮어쓰지 않는다. 모델 SHA-256을 평가 JSON에 기록한다.
- 평가 JSON에는 학습 기간, parameter, feature 목록, log loss/Brier/ECE/accuracy,
  학습 구간 빈도 기준선과 조건부 모집단 한계를 함께 기록한다.

아직 없는 CLI는 범용 provider ingest, 당일 inference, `optimize-today`, 통합 OOF
calibration이다. 실제 KBO graph 생성·RelGNN 학습·재개·holdout 평가 CLI는 구현되어 있다.

## 27. 테스트와 검증 결과

검증 명령:

```bash
python -m compileall -q src tests scripts
ruff check src tests scripts/build_code_summary.py
mypy --no-incremental src/cpv26
pytest
python -m pip check
python -m pip wheel . --no-deps -w /tmp/cpv26-wheel
```

최신 RelGNN 구현 검증 결과(Windows CPU, Python 3.12.13 / PyTorch 2.13.0+cpu):

```text
compileall: passed
Ruff:       all checks passed
mypy:       50 source files, no issues
pytest:     193 passed, 1 skipped (CUDA unavailable)
pip check:  no broken requirements
shell:      setup/activate/check syntax passed
wheel:      cpv26_predictor-0.4.0-py3-none-any.whl built
```

새 테스트는 graph 10개, RelGNN model 15개, runner 7개다. CUDA 전용 테스트 1개만
skip됐으며 CPU mixed-precision, 실제 graph 연결, DataLoader worker, checkpoint 재개와
held-out 평가를 포함한다. 연속 2 epoch와 1 epoch 후 재개한 2 epoch의 CPU 가중치가
정확히 동일함을 검증했다. CUDA가 없으면 기본 명령이 명시적으로 실패하고 CPU로
전환하지 않는 것도 확인했다. Wheel에 신규 dataset/model/runner 세 모듈이 포함된다.

이전 tabular 단계 검증 결과(아래 숫자는 GPU RelGNN 추가 전 기록):

```text
compileall: passed
Ruff:       all checks passed
mypy:       47 source files, no issues
pytest:     145 passed, 17 skipped (tabular; PyTorch not installed)
pip check:  no broken requirements
wheel:      cpv26_predictor-0.4.0-py3-none-any.whl built
```

이전 검증 환경은 Python 3.12.13, DuckDB 1.5.5, CatBoost 1.2.10의 `tabular` 환경이었다.
이때 17개 neural 테스트는 PyTorch 미설치로 skip됐다. 더 이전의 별도 full-ML 환경에서는
당시 117개 테스트가 skip 없이 통과했다. 어느 숫자도 새 KBO graph/model/runner 테스트의
최종 집계가 아니며, 전체 GPU 검증 결과도 아니다.

Windows 실행 중 native 의존성 검사·import 경로에서 `Windows fatal exception: access violation`
진단이 출력되었지만 pytest는 계속 실행되어 exit code 0으로 종료했다. 원인을 해결했다고
주장하지 않는다. Linux가 실제 실행 대상이며 동일 suite와 필요한 CUDA 검증을 수행해야 한다.

이전 tabular 단계 test file별 수(새 `test_kbo_graph_dataset.py`, `test_kbo_relgnn.py`,
`test_kbo_runner.py` 추가 전):

| file | tests |
|---|---:|
| `test_cli.py` | 6 |
| `test_config.py` | 2 |
| `test_dataset_contracts.py` | 19 |
| `test_dataset_integrity_v4.py` | 2 |
| `test_dataset_schema_v4.py` | 6 |
| `test_domain.py` | 3 |
| `test_evaluation.py` | 6 |
| `test_graph_models.py` | 13 |
| `test_kbo_ingest.py` | 4 |
| `test_kbo_live_hit_baseline.py` | 19 |
| `test_kbo_match_baseline.py` | 11 |
| `test_kbo_playbyplay_source.py` | 6 |
| `test_live_hit_point_in_time.py` | 7 |
| `test_live_hit_rules.py` | 8 |
| `test_model_output_contracts.py` | 3 |
| `test_pa_adapter.py` | 7 |
| `test_point_in_time.py` | 14 |
| `test_simulation_optimization.py` | 15 |
| `test_task_training.py` | 11 |
| 합계 | 162 |

이전 tabular 실제 데이터 실행:

- 2023~2026 원본 805,960 pitches 다운로드·SHA-256 검증
- 2,630경기·206,583 완료 라벨 PA의 DB 적재와 재적재 중복 방지
- 누락 label 550개, 점수 전이 불일치 13개, 원본 sequence gap 53개의 품질 보고
- 2023~2025 경기 2,160개로 경기 모델 두 fold 학습·평가
- 같은 기간 선수-경기 49,091개로 안타 모델 두 fold 학습·평가
- fold별 학습 빈도 기준선 비교와 네 개의 CatBoost `.cbm` 파일 저장
- 모델 재로드 후 보고서 log loss 일치, 모델 SHA-256·실행별 평가 JSON 보존

추가된 실제 데이터 RelGNN 검증:

- 기본 2023~2025의 날짜별 graph 494개, Match 2,160개, 조건부 Live Hit 49,091개,
  PA10 169,477개를 생성했다. 최초 cache 크기는 약 45.6 MiB였다.
- dataset version 2 fingerprint:
  `9eb9ea4b538e83a3fb83d3566cde17ed1cb0bc6cdddbc41678641b586fd473ed`.
  전이 불완전 PA 11건만 context 점수를 unknown으로 마스크했고 다른 모든 배열은
  v1과 동일함을 494일 전체에서 비교했다.
- 2023 train 3일/2024 validation 3일을 시즌 전체에서 균등 선정해 CPU 1 epoch를
  수행한 뒤 `last.pt`에서 총 2 epoch까지 재개했다. 모델 구조는 README 기본값인
  hidden 64 / 2 layers / 4 heads, 1,154,640 parameters이고 batch-days 1을 사용했다.
- 최종 optimizer step 6, AMP skipped step 0, best epoch 2. validation selection loss는
  6.005682 → 5.645183이었다. 이후 별도로 2025의 균등 선정 3일만 평가했다.
- 테스트 예측 Parquet는 Match 12행, 조건부 Live Hit 286행, PA10 970행으로 저장됐다.
  이 소표본의 정확도를 전체 시즌 모델의 성능 수치로 제시하지 않는다.
- 실행 위치는 `var/runs/relgnn/cpu_validation/`이고 최신 `best.pt` SHA-256은
  `d7f9b8d36fbbc3bc97e85f592f1cc8697e0581a0fe66d9c4c6e6e7468e1b3275`이다.
  `training_report.json`과 `evaluations/test-20260830T013137942015Z-5141e8b6/metrics.json`에
  실행 설정·데이터 fingerprint·정확한 평가 지표가 있다. 실행물은 Git에 포함하지 않는다.
- 이 축소 실행은 데이터/graph/model/runner 연결 검증이며 전체 시즌 학습 성적이 아니다.
  `smoke_test_only=true`, `test_used_during_training=false`를 보고서에 기록한다.

이전 full-ML 환경에서 실행한 optional neural runtime 검증:

- role encoder→RelGNN route→PA decoder/direct player head forward/backward
- finite gradient 확인
- distinct batter/pitcher role state routing
- unrelated route edge duplication 불변성
- empty node `[0, D]`
- DirectPlayer joint NLL backward
- PA bucket별 conditional hit distribution
- 세 task loader alternating update와 shared-backbone/task-head gradient 분리
- checkpoint restore와 lineage mismatch 차단
- DirectRun output contract

실행하지 못한 검증:

- CUDA forward/backward
- A6000 48GB peak memory
- A100 10GB MIG peak memory/OOM boundary
- Linux driver별 CUDA wheel compatibility
- NVIDIA GPU의 실제 KBO RelGNN 전체 시즌 학습·재개·holdout 성능
- 당일 V26 추천 replay

Windows CPU 통과를 GPU production 검증으로 해석하면 안 된다.

## 28. 외부 코드 검토 지적 처리표

| 지적 | 처리 |
|---|---|
| global weighted mean/GRU라 엄밀한 RelGNN 아님 | route-local destination-query MHA + route attention으로 교체 |
| role encoder가 backbone에 연결되지 않음 | role channel을 route endpoint가 실제 선택하도록 연결 |
| recency가 available_at 기반 | event_at age로 수정, availability는 eligibility만 사용 |
| validity filter가 rank 뒤라 current row 탈락 | correction/business version 2단계 ranking 구현 |
| neural 10종과 simulator 14종 미연결 | fold-safe inference adapter + historical target mapping 구현 |
| non-HR walk-off 과다 득점·루타 불일치 | required run cap, 공식 applied event/루타, HR 예외, terminal bases clear |
| prediction run ID 재사용 가능 | UNIQUE immutable run + status event + append validation |
| score correlation gradient 없음 | output 제거 |
| empty node shape `(0,)` | explicit feature dim으로 `[0,D]` 보존 |
| DirectPlayer outputs 모순 가능 | joint masked distribution, overflow, derived expectation, joint NLL |
| Match selection_rate 미사용 | 필드 제거 |
| beam global optimum 아님 | exact mode와 detailed diagnostics, brute-force oracle 추가 |

추가 교차감사에서 발견해 수정한 항목:

| 추가 결함 | 처리 |
|---|---|
| 실제 v1 index가 table rename을 막음 | dependent index drop 후 canonical recreate |
| multi-row append partial commit | append transaction과 failed outer transaction flag |
| run 간 candidate ID가 서로 가림 | natural identity에 prediction_run_id 포함 |
| status table 필수 column 누락을 signature가 놓침 | explicit v2 status contract 검사 |
| reference lineage 누락 검사 없음 | 46개 business reference rule audit 추가 |
| historical 14→10 target 불명확 | 전 event mapping과 CI 별도 정책 |
| DirectPlayer inference만 consistent | target bucket encoder와 joint NLL |
| 이미 home lead인 bottom initial state가 walk-off로 오판 | spec validation + 새 lead 조건 |
| Live Hit 연속 선택률 함수·임의 bonus 합산 | 출처 있는 versioned ruleset과 명시 구간표로 교체 |
| 최소 인원 synergy·activation point 가정 | 선택 팀 한 개의 선수별 가산만 유지 |
| 올 히트를 공식 hit points에 합산 | 공식 기대점수와 all-hit 확률/item/custom utility 분리 |
| 선수당 selection rate 하나 | 배치 포지션별 selection rate로 교체 |
| ruleset·position·selection·collection PIT state 없음 | schema v4의 versioned state와 scoped snapshot 유지 |
| PA 이후 점수·아웃·주자 상태가 없음 | v4 PA transition 컬럼과 완전성 flag·감사 추가 |
| 도루·폭투·교체 등 비PA 사건이 없음 | runner/substitution/fielding/catcher event table 추가 |
| 취소·순연·선발 발표가 없음 | game status와 starter announcement PIT table 추가 |
| 공식 선수 박스스코어 대조가 없음 | player_game_batting과 PA/team reconciliation 추가 |
| 과거 실제 관측 날씨를 예보처럼 쓸 위험 | forecast snapshot과 observation/oracle 계약 분리 |
| selected_team이 candidate feature에 섞임 | optimizer 결과인 v26_submission에만 저장 |
| 선택률을 한 번만 capture | early/starter_known/lineup_known/near_lock 4-phase 계약 |
| 과거 선택률 없이 실제 V26 replay 주장 | 2018~2025 fixed_300과 2026 actual capture 구간 분리 |
| 취소를 정상 0 PA로 학습 | game_played/label_observed mask로 분리 |
| 미완료 경기를 0:0으로 학습 | completed/result_observed와 -1 sentinel 계약 추가 |
| 공유 backbone 뒤 작업별 학습 연결 없음 | 3개 batch/loss/adapter와 alternating trainer 추가 |
| hit logits가 PA mask만 공유 | PA bucket별 별도 conditional logits로 확장 |

현재 선언은 단일 참조 107개, composite 참조 7개다. schema relation을 추가하면 rule과
테스트를 함께 늘려야 하며 이 숫자는 고정 API가 아니다.

## 29. 알려진 한계와 위험

### 데이터

- 공개 Hugging Face KBO source의 downloader와 importer는 있다. 다중 공급자 crosswalk,
  공식 licensed production feed와 V26 실제 화면 데이터는 없다.
- 실제 KBO 원본·DB는 로컬 `var/`에 보관하고 Git에는 넣지 않는다.
- 원본 선수/팀/경기 ID를 namespace로 보존하지만 다른 공급자와의 ID 통합은 하지 않았다.
- dataset 작성자의 CC BY 4.0 표기와 출처·revision·hash를 기록했다. 원출처의 권리·약관과
  서비스 운영 허가는 별도 확인이 필요하다.
- 원본의 정확한 발표·정정 시각이 없어 날짜 기반 retrospective 이력 재구성만 가능하다.
- 누락 PA와 점수·sequence 불일치가 있어 `simulator_ready=false`다.
- 2018~2025 당시 발표된 날씨 예보 revision archive를 확보하지 않았다.
- 2026 V26 네 phase 선택률·자격·규칙 snapshot 수집을 아직 시작하지 않았다.

### 모델

- 전용 safe NPZ graph loader와 실제 KBO RelGNN 학습·평가·재개 job은 있다. 범용
  snapshot Parquet를 직접 받는 loader, 당일 inference와 production orchestrator는 없다.
- 실제 데이터 축소 CPU 학습은 확인했지만 NVIDIA GPU 전체 시즌 학습·성능·메모리는
  미검증이다. graph를 공유하는 것만으로 CatBoost보다 좋다고 주장하지 않는다.
- 실제 CatBoost 모델은 학습했으나 승부예측이 빈도 기준선보다 낮고, Live Hit는 연도별
  성능이 일관되지 않는다. 두 모델의 calibration은 아직 fitting하지 않았다.
- CatBoost 안타 baseline과 KBO RelGNN Live Hit head는 관측 PA가 있는 선수 조건부다.
  미출전 확률이나 실제 V26 후보·보너스 최적화 전체를 학습한 모델이 아니다.
- GraphSAGE benchmark가 없다.
- hyperparameter tuning budget contract가 없다.
- AMP, gradient accumulation, atomic checkpoint 재개는 있다. distributed training,
  activation checkpointing과 model registry는 없다.

### 시뮬레이션

- runner advancement가 empirical fold artifact가 아니다.
- bullpen substitution rule이 단순하다.
- official scorekeeping은 범위 밖이다.
- rain-shortened/called game, suspended game 등은 없다.

### V26

- 실제 2026 점수표와 알려진 결과를 대조한 verified ruleset replay가 없다.
- slate, lock, selection, collection, position, submission schema는 있으나 실제 화면 ingestion
  adapter가 없다.
- account/UI automation은 의도적으로 없다.

### 운영

- scheduler/API/dashboard가 없다.
- concurrent writer process coordination이 없다.
- raw connection을 사용하면 application immutability를 우회할 수 있다.
- CUDA 장치 검사와 학습 peak-memory 보고 경로는 있다. 실제 NVIDIA 서버별 GPU memory
  profile과 OOM 경계는 아직 측정하지 못했다.

## 30. 다음 구현 우선순위

### P0 — GPU 실행 검증과 실제 데이터에서 남은 과제

공개 source 선택, checksum 다운로드, namespaced ID adapter, 실제 DB 참조 감사,
WDL·조건부 player-hit CatBoost 학습/평가/저장과 KBO RelGNN graph cache·학습·평가 CLI는
구현했다. RelGNN의 실제 데이터 축소 CPU 실행까지 확인했으며 다음은 별도 과제다.

1. Linux GPU에서 `gpu-check`, 실제 graph 학습, 재개, 2025 test를 실행하고 시간·메모리·
   성능을 기록한다. 전체 test 결과로 hyperparameter를 되돌려 조정하지 않는다.
2. 원본 누락·점수·교체 상태를 공식 기록과 대조하여 simulator용 데이터 품질을 확보한다.
3. 선발·라인업·출전 후보와 발표 시각을 확보하고 당일 출전 확률을 별도로 모델링한다.
4. 시간순 validation에서 feature·regularization·calibration을 검증한다. PA10 head도
   실제 학습 경로에 있으나 모델 성능과 simulator 활용 적합성은 별도 평가해야 한다.
5. 2026 V26 네 phase capture와 weather forecast revision 수집 정책을 확정한다.
6. 과거 weather는 확인 가능한 발표 revision만 forecast 실험에 넣는다.

### P1 — 학습 infrastructure

날짜별 temporal graph cache, disjoint mini-batch, AMP/gradient accumulation, atomic
best/last checkpoint, optimizer/scaler/RNG 재개, 작업별 prediction writer는 구현되어 있다.

1. 현재 전용 KBO NPZ cache와 별개로 범용 snapshot/provider 입력 확장
2. 여러 temporal fold의 비교 가능한 학습/OOF orchestrator
3. calibration/stacking fitting과 artifact writer 연결
4. 실제 GPU 측정에 근거한 route/PA sampling과 batch 기본값 검증
5. 필요성이 확인된 경우 activation checkpointing·분산 학습
6. model registry와 production inference artifact 계약

### P2 — feature 확장

1. starter/bullpen fatigue
2. handedness/platoon
3. catcher/defense
4. lineup uncertainty scenarios
5. stadium/weather
6. travel/rest/injury
7. empirical runner advancement rate

### P3 — model benchmark

1. 현재 CatBoost와 빈도 기준선을 동일 조건으로 재검증
2. MLP
3. GraphSAGE
4. 현재 RelGNN
5. direct heads + simulator
6. OOF ensemble

동일 cutoff/feature/fold/tuning budget으로 log loss, Brier, ECE, V26 replay utility를
비교한다.

### P4 — 운영

1. daily run orchestrator
2. recommendation artifact JSON
3. model/data drift report
4. read-only API/dashboard
5. failure notification

게임 계정 자동 선택은 별도 보안·약관 검토 없이는 추가하지 않는다.

## 31. 실제 데이터 adapter 최소 출력 계약

Source adapter는 최소 다음을 보장해야 한다.

- stable physical row ID
- natural business identity
- internal player/team/game ID
- source revision ID
- event/available/ingested time
- valid interval
- original locator
- content SHA-256
- parsing version

원천별 최소 산출물:

| 원천 범주 | 필수 출력 |
|---|---|
| 경기 일정·결과 | stable game ID, 더블헤더 번호, 재개 원경기, status revision |
| 라인업·선발 | projected/announced/official/final 상태와 실제 `published_at`/`announced_at` |
| PA | canonical sequence/subsequence, 전후 점수·아웃·주자, 결과·안타·루타·득점 |
| 비PA 사건 | 교체와 runner event의 같은 canonical sequence |
| 박스스코어 | player_game_batting, team_game, pitching_appearance reconciliation 필드 |
| 날씨 예보 | issued/target/captured/available time, 구장 grid, raw response hash |
| 날씨 관측 | station version/map, observed/available time, raw response hash |
| V26 | slate/rule/card/eligibility와 네 phase selection snapshot, 사용자 도감 |

날씨의 실제 관측은 `weather_observation`에만 적재하고 forecast feature로 승격하지 않는다.
V26의 `selected_synergy_team_id`는 수집된 후보 속성이 아니라 submission 결과다.

잘못된 fallback:

- scrape time을 event time으로 사용
- lineup 발표 시각을 경기 시각으로 사용
- 수정된 box score로 과거 cutoff를 덮어쓰기
- future official lineup을 projected lineup으로 위장
- 실제 출전 선수를 player candidate 정답으로 사용

Source별 availability policy를 문서화하고 replay test를 만들어야 한다.

## 32. 현재 artifact와 향후 권장 layout

현재 KBO RelGNN 경로가 실제 생성하는 기본 구조:

```text
var/
├── cpv26.duckdb
├── datasets/kbo_graph/
│   ├── manifest.json
│   └── days/
│       ├── YYYY-MM-DD.npz
│       └── YYYY-MM-DD.json
└── runs/relgnn/<run-id>/
    ├── config.json
    ├── history.jsonl
    ├── best.pt
    ├── last.pt
    ├── training_report.json
    └── evaluations/test-<evaluation-id>/
        ├── metrics.json
        ├── match_predictions.parquet
        ├── live_hit_predictions.parquet
        └── pa_predictions.parquet
```

Evaluation directory는 `relgnn-evaluate`를 별도로 실행할 때 생성한다. 학습이 test까지
자동 평가한 것처럼 해석하지 않는다. `--dataset`, `--run-dir`, 평가 `--output`으로
위치를 변경할 수 있다. cache fingerprint가 checkpoint와 다르면 재개·평가를 거부한다.
`best.pt`/`last.pt`가 존재한다는 사실만으로 GPU 실행을 입증하지 않으며 `config.json`과
학습 보고서의 device/smoke 표시를 함께 확인한다.

아래는 기존 snapshot을 포함한 향후 prediction-run 통합 layout 제안이다. fold와
recommendation 부분을 모두 구현한 것은 아니다.

```text
var/
├── cpv26.duckdb
├── snapshots/<prediction_run_id>/
│   ├── manifest.json
│   └── *.parquet
├── folds/<experiment_id>/<fold_id>/
│   ├── train_manifest.json
│   ├── feature_schema.json
│   ├── route_schema.json
│   ├── label_schema.json
│   ├── model.ckpt
│   ├── calibration.json
│   └── metrics.json
└── predictions/<prediction_run_id>/
    ├── probabilities.parquet
    ├── simulation_summary.json
    └── recommendations.json
```

모든 artifact는 prediction run, snapshot fingerprint, source commit 또는 source archive
hash를 참조해야 한다.

## 33. GPU 인계 체크리스트

A6000 등 실제 Linux GPU host에서([GPU_TRAINING.md](GPU_TRAINING.md) 실행 절차 참고):

1. `nvidia-smi`
2. 공식 selector로 CUDA index를 선택하고 `setup.sh ml-cuda` 실행
3. `cpv26 gpu-check --device cuda:0`의 실제 forward/backward 통과
4. 동일 환경의 전체 test 실행
5. `kbo-graph-build`의 날짜 수·label 품질·cache fingerprint 확인
6. 실제 데이터 RelGNN 짧은 GPU 실행 후 전체 목표 epoch 학습
7. 학습 보고서의 peak allocated/reserved memory와 epoch 시간 기록
8. `--batch-days 1`부터 필요한 크기까지 GPU별 OOM 경계 확인
9. `last.pt` 재개와 별도 `best.pt --split test` 평가 산출물 확인

A100 10GB MIG에서는 위와 별도로:

- BF16 지원 확인
- route 상한·훈련 PA query 상한과 `--batch-days 1`에서 시작
- 필요한 경우 `--accumulate-steps`로 유효 batch 조절
- activation checkpointing은 아직 미구현이므로 지원한다고 가정하지 않기
- max allocated/reserved memory
- fragmentation/OOM recovery

현재 날짜별 graph loader, route/훈련 PA 상한, AMP, gradient accumulation 경로는 있다.
이것이 10GB 장치에서의 성공을 보장하지는 않는다. 실제 장치에서 데이터·설정·AMP mode와
peak memory를 함께 측정해야 하며 A6000/A100 MIG 수치를 추정해 기재하지 않는다.

## 34. ChatGPT 교차검증 요청 방법

`code_summary.md`와 이 문서를 함께 전달하고 다음 순서로 검토를 요청한다.

1. 코드를 실행 가능한 tree로 복원한다.
2. README/hand-off의 주장을 코드와 대조한다.
3. PIT SQL을 adversarial timestamp case로 검증한다.
4. v1 index가 있는 migration과 rollback을 실행한다.
5. append batch partial failure를 실행한다.
6. route role/time/attention gradient를 검사한다.
7. 14↔10 PA label 정책과 cutoff dedup을 검사한다.
8. walk-off edge case를 property test한다.
9. exact optimizer를 독립 brute-force와 비교한다.
10. CatBoost/RelGNN을 동일 future holdout에서 비교할 수 있는지 판정한다.

검토 우선순위:

```text
correctness / leakage / migration / baseball rule
  > reproducibility
  > model performance claim
  > style
```

반드시 물어볼 질문:

- 최신 correction이 과거 business version을 부활시키는가?
- run-scoped identity가 다른 run에서 충돌하는가?
- route가 declared player role을 실제로 읽는가?
- unrelated route degree가 matchup message를 희석하는가?
- training target과 inference adapter가 같은 label policy를 쓰는가?
- simulator output이 주장하지 않는 official scoring field는 무엇인가?
- exact/beam 결과 화면에서 exact 여부가 보이는가?
- generic trainer와 실제 데이터 학습 service를 구분하는가?

## 35. 완료 기준

연구 프레임워크 단계 완료 기준은 현재 충족했다.

- schema v4와 v1/v2/v3 migration regression
- PIT correction/business version regression
- immutable run/status event contract
- append atomicity
- 단일·composite·cutoff 시점 reference audit
- PA/박스스코어/전이/날씨/V26 capture 데이터 계약
- 취소·노게임·미완료 경기를 가짜 0 PA/0:0으로 만들지 않는 training mask
- role/time/route attention regression
- 10↔14 label contract
- walk-off score regression
- direct head joint training contract
- task-separated loss/gradient/checkpoint lineage contract
- exact optimizer oracle
- CPU optional runtime smoke
- Ruff/mypy/pytest/wheel 검증
- README/code summary/hand-off 일치

이전 실제 데이터 CatBoost baseline 단계에서 추가로 완료한 항목:

- 고정 revision 다운로드·hash·출처 manifest
- pitch→경기/PA 정규화와 source 품질 보고
- 경기 WDL과 선수 any-hit의 별도 데이터셋·모델·loss
- 2024 validation / 2025 test 및 학습 빈도 기준선 비교
- 평가 JSON과 fold별 학습 모델 파일 저장

실제 데이터 RelGNN 경로에서 추가로 구현·확인한 항목:

- 과거 날짜·availability·validity를 적용한 날짜별 safe NPZ graph와 hash manifest
- 공유 role-aware backbone + Match WDL/NB2, 조건부 Live Hit joint, PA10의 작업별 loss
- 실제 데이터 축소 CPU 학습·validation·checkpoint 저장
- CUDA 검사, AMP, DataLoader mini-batch, gradient clipping/accumulation 실행 코드
- atomic best/last checkpoint와 optimizer/scaler/RNG·dataset lineage 재개 계약
- 2025 holdout을 분리한 명시적 평가 CLI와 작업별 prediction Parquet writer

마지막 세 항목은 실행 경로의 구현을 뜻하며 NVIDIA GPU에서 검증 완료했다는 뜻은 아니다.

실전 추천기 완료 기준은 아직 충족하지 않았다.

- licensed production data
- provider replay
- 기준선 우위와 calibration이 확인된 모델(CatBoost 전체 실행·RelGNN 축소 CPU 실행과 구분)
- comparable graph benchmark
- 실제 NVIDIA 전체 시즌 GPU training artifacts와 재개·성능·메모리 검증
- actual V26 ruleset replay
- daily orchestrator
- monitoring and operational security

이 구분을 유지해야 프로젝트가 과장 없이 확장 가능하다.
