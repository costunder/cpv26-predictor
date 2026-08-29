# CPV26 Predictor 인수인계서

- 작성 기준일: 2026-08-29 KST
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

다만 이것은 아직 당일 데이터를 받아 자동으로 추천을 내는 운영 서비스가 아니다.
공급자 adapter, 실제 KBO 데이터, Parquet graph loader, production 학습 job, 실제 V26
ruleset replay와 scheduler가 없다. 정확한 표현은 다음과 같다.

> 시점·관계·야구 규칙의 correctness를 검증한 연구용 프레임워크이며, 완성된 실전
> 추천기는 아니다.

## 2. 전달 파일

- `README.md`
  - Linux/MobaXterm 설치, 실행 profile, 구조, 현재 한계를 설명한다.
- `docs/HANDOFF.md`
  - 이 문서다. 설계 결정, 공개 계약, 검증 결과와 다음 작업을 설명한다.
- `code_summary.md`
  - Git에 포함하지 않는 외부 검토용 생성물이다. `code_summary.md`와 별도 문서인
    `docs/HANDOFF.md`를 제외한 프로젝트 파일을 다음 형식으로 이어 붙인다.
- `scripts/build_code_summary.py`
  - README와 handoff 수정 후 `code_summary.md`를 같은 형식으로 재생성한다.

```text
# `파일경로`

````
코드 내용
````
```

- 최종 `code_summary.md` SHA-256: `32c0c8f1771ebd887b4bc2940d316d6fa4926ad023df38b2baabc103f5ddbbe6`
- 포함 section 수: `68`
- 고유 경로 수: `68`

이 hash로 외부 전달 과정에서 코드 요약이 변형되지 않았는지 확인할 수 있다.

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
│   └── HANDOFF.md
├── requirements/
│   └── constraints.txt
├── scripts/
│   ├── build_code_summary.py
│   ├── setup.sh
│   └── check.sh
├── src/cpv26/
│   ├── data/
│   │   ├── dataset_contracts.py
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
│   ├── training/
│   │   ├── contracts.py
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
    ├── test_live_hit_point_in_time.py
    ├── test_live_hit_rules.py
    ├── test_model_output_contracts.py
    ├── test_pa_adapter.py
    ├── test_point_in_time.py
    ├── test_simulation_optimization.py
    └── test_task_training.py
```

원본 데이터나 가짜 fixture CSV용 빈 폴더는 없다. 실행 산출물은 `CPV26_HOME` 아래에
생성한다.

## 4. Linux 설치와 profile

Windows `.venv`는 복사하지 않는다. MobaXterm SFTP로 소스만 올린 뒤 Linux에서 새로
만든다.

```bash
cd ~/projects/cpv26-predictor
bash scripts/setup.sh base
source .venv/bin/activate
cp .env.example .env
set -a
source .env
set +a
cpv26 db-init
cpv26 db-check
```

Python 명령이 `python3.10`이 아니면 다음처럼 지정한다.

```bash
PYTHON_BIN=python3.11 bash scripts/setup.sh base
```

Profile은 네 개다.

| profile | 설치 내용 | 용도 |
|---|---|---|
| `base` | runtime만 | DB, feature, snapshot, CLI |
| `dev` | runtime + Ruff/mypy/pytest/coverage | 개발·검증 |
| `ml-cpu` | runtime + CatBoost + CPU PyTorch | CPU smoke/train |
| `ml-cuda` | runtime + CatBoost, 기존 CUDA torch 검사 | GPU 서버 |

`setup.sh`는 기존 `.venv`를 재사용한다. 개발 host에서는 `dev` 다음 `ml-cpu`를 실행해도
된다.

GPU에서는 다음 순서를 사용한다.

```bash
bash scripts/setup.sh base
source .venv/bin/activate
nvidia-smi
# 공식 PyTorch selector에서 driver/runtime에 맞는 CUDA wheel 설치
bash scripts/setup.sh ml-cuda
```

`ml-cuda`는 `torch.cuda.is_available()`이 false면 exit code 3으로 실패한다.

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

현재 CPU 검증 버전:

```text
Python    3.12.13
DuckDB    1.5.5
NumPy     2.2.6
PyTorch   2.13.0+cpu
CatBoost  1.2.10
pytest    8.4.2
Ruff      0.16.5
mypy      1.20.2
```

## 6. 환경 설정

```dotenv
CPV26_HOME=./var
CPV26_DB_PATH=./var/cpv26.duckdb
CPV26_TIMEZONE=Asia/Seoul
CPV26_DEVICE=auto
CPV26_RANDOM_SEED=2026
CPV26_LOG_LEVEL=INFO
```

`Settings.from_environment()`는 naive path나 잘못된 timezone/device/log level을
검증한다. 상대 경로는 repository root 기준으로 해석한다.

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

현재 repository 내부에서 구현된 구간:

```text
append-only DB
  → snapshot
  → feature/graph/model primitive
  → task-separated model/loss/alternating trainer
  → simulator primitive
  → optimizer primitive
```

현재 끊긴 구간:

- 공급자 feed→정규화 row
- snapshot Parquet→mini-batch graph tensor
- 실제 시즌 fold dataset/job→trainer 입력
- 학습 산출물→당일 inference orchestration
- 공식 V26 점수표·구간표 replay→verified scoring configuration

따라서 각 부품을 import해 사용할 수는 있지만 단일 `train` 또는 `predict-today` 명령은
아직 없다.

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

테이블이 있다는 것은 수집기가 있다는 뜻이 아니다. 이 저장소에는 provider adapter나
실데이터가 아직 없다.

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

`src/cpv26/training/`은 서로 다른 row granularity를 하나의 batch로 섞지 않는다.

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

Lineage가 현재 trainer와 다르면 resume를 거부한다. 이 패키지는 in-memory tensor batch
이후의 학습 계약이며 provider나 Parquet loader를 구현했다고 주장하지 않는다.

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

Meta prediction은 이전 OOF stage만 사용하도록 설계한다. 실제 model training
orchestrator가 없으므로 base prediction 생성과 artifact 저장은 호출자가 연결해야 한다.

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

비교 순서:

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
cpv26 snapshot-build <prediction-run-id>
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

아직 없는 CLI:

- ingest
- feature-build
- graph-build
- train
- evaluate
- predict
- optimize-today

## 27. 테스트와 검증 결과

최종 CPU 검증 명령:

```bash
python -m compileall -q src tests scripts
ruff check src tests scripts/build_code_summary.py
mypy --no-incremental src/cpv26
pytest
python -m pip check
python -m pip wheel . --no-deps -w /tmp/cpv26-wheel
```

결과:

```text
compileall: passed
Ruff:       all checks passed
mypy:       42 source files, no issues
pytest:     117 passed in 11.63s
pip check:  no broken requirements
wheel:      cpv26_predictor-0.4.0-py3-none-any.whl built
```

검증 환경은 Python 3.12.13, DuckDB 1.5.5, PyTorch 2.13.0+cpu, CatBoost 1.2.10이다.
선택 의존성이 없는 번들 환경에서는 99 passed/18 skipped였고, 별도 임시 full-ML
환경에서는 117개가 한 프로세스에서 skip 없이 통과했다. wheel 크기는 134,218 bytes,
SHA-256은 `88aa809689dafb1369007da7930064aeb1f952da021cb6799908c016ae6dcc41`이며
`dataset_contracts.py`와 `schema_v4.py` 포함을 확인했다. 최종 운영 대상 Linux에서는
동일 suite와 CUDA 검증을 다시 실행해야 한다.

Test file별 수:

| file | tests |
|---|---:|
| `test_cli.py` | 1 |
| `test_config.py` | 2 |
| `test_dataset_contracts.py` | 19 |
| `test_dataset_integrity_v4.py` | 2 |
| `test_dataset_schema_v4.py` | 6 |
| `test_domain.py` | 3 |
| `test_evaluation.py` | 6 |
| `test_graph_models.py` | 13 |
| `test_live_hit_point_in_time.py` | 7 |
| `test_live_hit_rules.py` | 8 |
| `test_model_output_contracts.py` | 3 |
| `test_pa_adapter.py` | 7 |
| `test_point_in_time.py` | 14 |
| `test_simulation_optimization.py` | 15 |
| `test_task_training.py` | 11 |
| 합계 | 117 |

실제로 실행한 optional runtime 검증:

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
- CatBoost fit/predict

실행하지 못한 검증:

- CUDA forward/backward
- A6000 48GB peak memory
- A100 10GB MIG peak memory/OOM boundary
- Linux driver별 CUDA wheel compatibility
- 실제 KBO multi-season walk-forward

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

- 실제 provider adapter가 없다.
- 저장소에 실제 KBO/V26 row가 없다.
- KBO 내부 ID crosswalk가 없다.
- source licensing/retention policy가 코드화되지 않았다.
- 2018~2025 당시 발표된 날씨 예보 revision archive를 확보하지 않았다.
- 2026 V26 네 phase 선택률·자격·규칙 snapshot 수집을 아직 시작하지 않았다.

### 모델

- in-memory alternating trainer는 있으나 snapshot/Parquet data loader와 production job이 없다.
- GraphSAGE benchmark가 없다.
- hyperparameter tuning budget contract가 없다.
- AMP, distributed training, gradient accumulation과 model registry가 없다.

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
- GPU memory profile이 없다.

## 30. 다음 구현 우선순위

### P0 — 실제 데이터 baseline

1. 허가된 source 한 개를 선택한다.
2. immutable raw landing + checksum을 만든다.
3. player/team/game ID crosswalk를 만든다.
4. PIT normalization adapter를 만든다.
5. `db-check`와 snapshot을 실제 시즌 slice에서 통과시킨다.
6. CatBoost PA/player-hit/WDL baseline을 end-to-end로 만든다.
7. 2026 V26 네 phase capture와 날씨 forecast revision 수집을 즉시 시작한다.
8. 2018~2025 historical weather는 archive가 확인된 revision만 forecast 실험에 넣는다.

### P1 — 학습 infrastructure

1. snapshot Parquet loader
2. target-game temporal subgraph builder
3. fold artifact directory contract
4. 현재 alternating trainer에 AMP / gradient accumulation 연결
5. checkpoint 파일 저장·atomic resume orchestration
6. fold prediction writer
7. calibration/stacking artifact writer

### P2 — feature 확장

1. starter/bullpen fatigue
2. handedness/platoon
3. catcher/defense
4. lineup uncertainty scenarios
5. stadium/weather
6. travel/rest/injury
7. empirical runner advancement rate

### P3 — model benchmark

1. CatBoost
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

## 32. 권장 artifact layout

아직 code로 고정되지 않았지만 다음 구조를 권장한다.

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

A6000 host에서:

1. `nvidia-smi`
2. CUDA wheel 설치
3. `torch.cuda.is_available()`
4. CPU test 전체
5. neural test CUDA device variant
6. small graph forward/backward
7. actual subgraph peak allocation 기록
8. batch size 증가 OOM boundary 기록
9. checkpoint save/load 동일 output 검사

A100 10GB MIG에서는 위와 별도로:

- BF16 지원 확인
- neighbor/route sampling 활성화
- gradient accumulation
- activation checkpointing
- max allocated/reserved memory
- fragmentation/OOM recovery

현재 subgraph loader와 memory-control 경로가 없어 10GB 검증을 시작할 단계는 아니다.

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

실전 추천기 완료 기준은 아직 충족하지 않았다.

- licensed production data
- provider replay
- trained and calibrated baseline
- comparable graph benchmark
- GPU training artifacts
- actual V26 ruleset replay
- daily orchestrator
- monitoring and operational security

이 구분을 유지해야 프로젝트가 과장 없이 확장 가능하다.
