# CPV26 Predictor

컴프야 V26의 승부예측과 라이브 히트 의사결정을 연구하기 위한 KBO 확률 예측
프레임워크입니다. 개인 능력, 타자-투수 관계, 라인업과 팀 상태를 시점 보존 데이터로
구성하고, 관계 모델·순차 경기 시뮬레이션·게임 규칙 최적화를 분리합니다.

현재 버전은 `0.4.0`, DuckDB schema는 `v4`입니다. 이 저장소는 검증 가능한 핵심
부품을 제공하지만, 데이터 수집부터 당일 추천까지 자동 실행되는 완성 서비스는
아닙니다. 허가된 실제 데이터와 학습 산출물이 없으면 임의 예측을 만들지 않습니다.

> 상태: 연구 프로토타입 · 독점 소프트웨어 · 비공식 프로젝트
> KBO 또는 게임 개발·배급·운영사와 제휴하거나 이들의 보증을 받은 프로젝트가 아닙니다.

## 설계 원칙

- 학습·예측 입력은 `available_at <= cutoff_at`을 만족해야 합니다.
- 사건의 최근성은 `event_at`, 정보 사용 가능성은 `available_at`으로 계산합니다.
- 관측된 과거 타석과 미래 경기의 선수 후보를 별도 테이블로 관리합니다.
- 선수 공통 identity state와 타격·투구·포수·수비·주루 state를 분리합니다.
- 야구 확률과 V26 점수·보너스 규칙을 분리합니다.
- 예측 run ID는 한 번 생성하면 재사용하지 않고, 상태 변화는 별도 이벤트로 남깁니다.
- 게임 계정 로그인, 선택 자동화, 비인가 대량 수집은 제공하지 않습니다.

## 대상 환경

실제 실행 대상은 MobaXterm으로 접속하는 Linux 서버입니다.

- Ubuntu 22.04 이상 권장
- Bash
- Python 3.10~3.12
- RAM 32GB 이상 권장
- 전체 graph 실험은 A6000 48GB 권장

A100 10GB MIG는 현재 주 학습 환경으로 가정하지 않습니다. 10GB에서 안정적으로
학습하려면 target-game subgraph, temporal neighbor sampling, route mini-batch loader,
BF16/AMP, gradient accumulation, activation checkpointing과 CPU→GPU streaming이 먼저
필요합니다. 이 경량화 경로는 아직 구현되지 않았습니다.

## MobaXterm에서 설치

MobaXterm SFTP 패널로 프로젝트를 `~/projects/cpv26-predictor`에 전송합니다. 다음은
전송하지 마세요.

```text
.venv/
var/
__pycache__/
.pytest_cache/
.ruff_cache/
.mypy_cache/
```

Windows 가상환경은 Linux에서 재사용할 수 없습니다. 서버에서 새로 만듭니다.

### 운영 최소 환경

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

서버의 Python 명령이 다르면 지정할 수 있습니다.

```bash
PYTHON_BIN=python3.11 bash scripts/setup.sh base
```

### 개발 검사 환경

```bash
bash scripts/setup.sh dev
source .venv/bin/activate
bash scripts/check.sh
```

### CPU ML 환경

CatBoost와 CPU PyTorch를 설치합니다.

```bash
bash scripts/setup.sh ml-cpu
source .venv/bin/activate
bash scripts/check.sh
```

### NVIDIA CUDA 환경

CUDA PyTorch wheel은 서버 드라이버와 CUDA runtime에 맞춰 공식 PyTorch 설치 선택기로
고릅니다. 일반 PyPI torch로 덮어쓰지 않도록 `ml-cuda` 프로필은 CatBoost만 설치한
뒤 이미 설치된 torch의 CUDA 가용성을 검사합니다.

```bash
cd ~/projects/cpv26-predictor
bash scripts/setup.sh base
source .venv/bin/activate
nvidia-smi

# 이 위치에서 서버에 맞는 공식 CUDA PyTorch wheel을 설치합니다.
bash scripts/setup.sh ml-cuda
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`requirements/constraints.txt`는 Python 3.12에서 검증한 core·dev·CatBoost 버전을
고정합니다. CUDA PyTorch는 target Linux host에서 결정하고, 버전을 바꿀 때 전체
검사를 다시 실행합니다. `scikit-learn`은 직접 사용하지 않아 의존성에서 제외했습니다.
`pytz`는 DuckDB `TIMESTAMPTZ`를 Python datetime으로 변환하는 실제 runtime 경로에서
필요하므로 유지합니다.

## 환경 설정

`.env`는 저장소에 포함하지 않습니다. 기본 항목은 다음과 같습니다.

```dotenv
CPV26_HOME=./var
CPV26_DB_PATH=./var/cpv26.duckdb
CPV26_TIMEZONE=Asia/Seoul
CPV26_DEVICE=auto
CPV26_RANDOM_SEED=2026
CPV26_LOG_LEVEL=INFO
```

상대 경로는 저장소 루트를 기준으로 해석됩니다. 실행 데이터베이스, snapshot,
checkpoint와 로그는 `CPV26_HOME` 아래에 생성합니다.

## 코드 구조

```text
src/cpv26/
├── data/          schema v4, PIT query, 데이터 계약·무결성 감사, Parquet snapshot
├── features/      타격 상태, 베이지안 shrinkage, 시점별 상태 primitive
├── graph/         atomic route registry와 누수 방지 graph snapshot
├── models/        CatBoost, 역할별 encoder, RelGNN, direct heads, OOF stacker
├── simulation/    10→14 PA adapter와 순차 야구 경기 시뮬레이션
├── optimization/  승부예측 exhaustive search와 라이브 히트 exact/beam search
├── training/      작업별 batch/loss, 공유 backbone 경계, alternating trainer
├── cli.py         Linux 배치 명령
├── config.py      환경변수 기반 설정
├── domain.py      공통 도메인 타입
└── evaluation.py  날짜순 walk-forward와 확률 평가 지표
```

빈 데이터 폴더나 가짜 CSV는 두지 않습니다.

## 저장소 문서

- [프로젝트 인수인계와 설계 상세](docs/HANDOFF.md)
- [독점 라이선스와 비제휴 고지](LICENSE.md)

외부 코드 검토가 필요하면 `python scripts/build_code_summary.py`를 실행해
`code_summary.md`를 생성합니다. 이 파일은 전체 소스를 복제하는 전달용 산출물이므로
Git에는 포함하지 않습니다.

## 데이터와 재현성 계약

변동 가능한 원천 행은 다음 다섯 시점을 갖습니다.

- `event_at`: 실제 사건 또는 효력 발생 시각
- `available_at`: 예측자가 그 정보를 사용할 수 있게 된 시각
- `ingested_at`: 이 시스템이 행을 관측한 시각
- `valid_from`, `valid_to`: 업무 버전의 반개구간 유효기간

`current_only=True` as-of query는 다음 순서로 동작합니다.

```text
available_at / ingested_at eligibility
  → natural identity + valid_from별 최신 correction
  → cutoff에서 유효한 business version
  → natural identity별 최신 valid_from
```

따라서 미리 발표된 미래 버전이 현재 행을 숨기지 않고, 최신 correction이 닫은 과거
행도 다시 살아나지 않습니다. run-scoped 후보의 identity에는 `prediction_run_id`가
포함됩니다.

`prediction_run`은 `prediction_run_id UNIQUE` 단일 행이며 cutoff, knowledge,
feature/model/simulator/ruleset 버전을 고정합니다. 상태는
`prediction_run_status_event`에 append합니다. v1→v2 migration은 실제 v1 index를
제거·재생성하며 중복 run ID가 있으면 전체 transaction을 rollback합니다.

Schema v4는 라이브 히트 의사결정 상태와 순차 경기 재생에 필요한 데이터 원장을 야구
feature와 분리해 저장합니다. Metadata를 포함해 36개 table입니다.

```text
game_status_snapshot               취소·순연·지연·중단·노게임의 발표 시점
starter_announcement               예상·예고·확정·변경 선발의 발표 시점
player_game_batting                공식 박스스코어 정합성 기준
substitution_event                 대타·대주자·투수·수비 교체
runner_event                       도루·도루실패·견제사·폭투·포일·보크
fielding_assignment                경기 sequence별 수비 역할
catcher_assignment                 경기 sequence별 포수 역할
weather_forecast_snapshot          예측 시점에 사용 가능한 예보 revision
weather_observation                사후 검증·oracle 실험용 실제 관측
v26_live_hit_rule_set              규칙 JSON, 출처와 공식/관측/가정 구분
v26_slate                          lock·규칙·Live 카드·자격 snapshot의 결합 단위
v26_player_position_eligibility    slate·Live 카드 버전별 포지션 자격
v26_selection_snapshot             phase·포지션별 선택률 snapshot
user_collection_snapshot           사용자·Live 카드 버전별 도감 보유 상태
v26_submission                     선택 선수·선택 시너지 팀이라는 실제 행동 결과
```

각 source-backed table은 시점 컬럼을 가지며 v1→v2→v3→v4, v2→v3→v4,
v3→v4 migration을 지원합니다. `observed_plate_appearance`에는 타석 전후 점수,
`outs_added`, `runners_after`, 동일 sequence 안의 `event_subsequence`와
`transition_complete`를 추가했습니다. 과거 행의 전이 상태를 추측해 채우지 않고
`transition_complete=false`로 남기며 simulator-ready 데이터 감사에서 제외합니다.

`live_hit_snapshot_specs()`는 특정 사용자·슬레이트·규칙·선택률 snapshot만 Parquet
fingerprint에 포함합니다. 사용자 도감 전체가 기본 snapshot에 섞이지 않도록 명시적인
scope가 없으면 Live Hit 계정 snapshot을 만들지 않습니다. `v26_submission`은 모델 입력이
아닌 optimizer 행동이므로 입력 snapshot에서 제외합니다.

다중 행 `append()`도 원자적입니다. 중간 CHECK 실패가 나면 해당 호출의 앞선 행까지
rollback하며, 명시적 transaction 안에서 실패를 잡아도 transaction 전체가 실패로
표시됩니다.

물리 foreign key는 공급자별 적재 순서를 막을 수 있어 강제하지 않습니다. 대신 다음
감사를 snapshot·학습 전에 실행합니다.

```python
from cpv26.data import DuckDBStore

with DuckDBStore("var/cpv26.duckdb", read_only=True) as store:
    store.assert_referential_integrity()
    store.assert_composite_referential_integrity()
    violations = store.as_of_reference_violations(
        cutoff_at=cutoff_at,
        knowledge_at=knowledge_at,
    )
```

단일 ID 존재 검사뿐 아니라 `(game_id, team_id, player_id)` 같은 composite 관계와, 물리
row가 있더라도 예측 cutoff에는 아직 발표되지 않았던 부모를 구분합니다. `cpv26 db-check`도
schema signature와 선수·팀·경기·source revision·prediction run 참조를 함께 검사합니다.

Snapshot builder는 cutoff의 논리 입력을 Parquet으로 고정하고 table별 논리 SHA-256,
파일 SHA-256과 전체 manifest fingerprint를 기록합니다. 같은 run ID에 다른 내용이
생기면 기존 디렉터리를 덮어쓰지 않습니다.

## 데이터셋 구성과 평가 경계

이 저장소에는 실데이터 자체나 비어 있는 더미 파일을 넣지 않았습니다. 대신 공급자
adapter가 채워야 할 schema와 적재 후 실행할 데이터 계약을 제공합니다.

권장 야구 원천 범위는 다음과 같습니다.

```text
2018~2025 KBO 경기·타석·박스스코어
  → PA 결과, 선수 안타 분포, 득점·승패, 관계 모델 학습·검증

2026 shadow/live 수집
  → 발표 당시 선발·라인업·경기 상태·예보 revision
  → V26 early/starter_known/lineup_known/near_lock snapshot
```

엄격한 시즌 역할은 코드의 `DEFAULT_EXPANDING_TEMPORAL_POLICY`에 고정했습니다.

| 시즌 | 역할 |
|---|---|
| 2018~2022 | base model 학습 |
| 2023 | feature·route·hyperparameter 선택 |
| 2024 | OOF stacking·확률 보정 |
| 2025 | 구조를 바꾸지 않는 최종 holdout |
| 2026 | 2018~2025 재학습 후 shadow/live |

같은 `game_id`의 타석·선수·horizon 행은 반드시 같은 fold에 둡니다. PA와
`player_game_batting` 및 `team_game` 집계, 타석 전후 점수·아웃·주자 연속성도 계약으로
검사합니다. 명시적 `runner_event`가 두 PA 사이에 있으면 직접 주자 연속성 비교만
건너뛰고, 이벤트 자체를 원인으로 보존합니다.

날씨는 두 실험을 섞지 않습니다.

```text
forecast       forecast_issued_at·available_at <= cutoff인 예보만 입력
oracle_weather 경기 시점 실제 관측을 사후 상한선 분석에만 사용
```

2018~2025 당시 예보 revision을 확보하지 못하면 날씨를 제외하거나
`oracle_weather`라고 명시해야 합니다. 실제 관측을 당시 예보처럼 사용하면 누수입니다.

V26도 가용성에 따라 평가 이름을 분리합니다.

```text
2018~2025  과거 선택률·자격·규칙 snapshot이 없으면 fixed_300 counterfactual
2026~      수집 시작 이후 네 phase가 모두 있는 slate만 실제 replay 후보
```

`audit_v26_capture_consistency()`는 네 phase의 완전성·순서, slate/rule/Live 카드/자격의
일치, 포지션별 중복, lock 이후 capture를 검사합니다. `selected_synergy_team_id`는 후보
feature가 아니라 `v26_submission`에 기록되는 optimizer 출력입니다.

## 관계 모델

`AtomicRouteBatch`는 `event_at`과 `available_at`을 모두 보존합니다.

- eligibility: `available_at <= cutoff_at`
- recency encoding: `cutoff_at - event_at`
- 선택 feature: `available_at - event_at` publication delay

빈 node type은 `node_feature_dims`를 명시해 항상 `[0, feature_dim]` tensor로 변환합니다.

선수 state channel은 다음과 같습니다.

```text
shared player core
  ├── batting
  ├── pitching
  ├── catcher
  ├── defense
  └── baserunning
```

Route의 `source_role`과 `destination_role`이 실제 channel을 선택합니다. Backbone의 한
layer는 다음을 수행합니다.

```text
source state + encoded event + event-time context
  → destination state를 query로 쓰는 route-local multi-head attention
  → route별 독립 summary
  → destination별 learned route attention/gate
  → GRU + LayerNorm update
```

따라서 unrelated route의 edge 수가 많아도 하나의 공통 degree denominator로 핵심
matchup message를 희석하지 않습니다. 기본 whitelist에는 타자–타석–투수,
투수–타석–포수, 타자–타석–경기, 홈팀–경기–원정팀,
선수후보–player_game_candidate–경기 등이 포함됩니다.

이 구현은 외부 검토에서 지적된 destination-query attention과 route-level 결합을
반영한 RelGNN 계열 backbone입니다. 특정 논문의 공식 repository와 parameter-by-parameter
동일하다고 주장하지 않으며, CatBoost·MLP·GraphSAGE와 동일 cutoff/feature/tuning budget의
미래기간 비교가 끝나기 전에는 최고 모델로 간주하지 않습니다.

## 타석 확률과 시뮬레이션

Neural decoder의 10개 label은 다음 adapter를 통해 시뮬레이터의 14개 terminal event로
변환됩니다.

```text
10-way neural probability
  → fold/cutoff에서 추정한 BB:HBP, CI rate, BIP:DP:FC split
  → outs/base-state legality normalization
  → 14-way terminal probability
```

반대 방향의 학습 label 계약도 `neural_training_target()`으로 고정합니다. HBP는
`walk_or_hbp`, DP와 FC는 `ball_in_play_out`, catcher interference는 별도 희귀-event
rate 학습 대상으로 처리합니다. 최신 known revision을 먼저 고른 뒤 `event_at < cutoff`
행만 학습해 정정본 때문에 오래된 행이 부활하지 않습니다.

Simulator는 미래 타석만 순차 표본화하며 한 simulation path 안의 선수 안타와 경기
득점 상관을 보존합니다. 비홈런 끝내기는 결승에 필요한 득점까지만 인정하고 결승
주자의 시작 베이스에 따라 공식 `applied_event`와 `credited_total_bases`를 1B·2B·3B로
조정하며, 원 표본은 `sampled_event`에 보존합니다. terminal base state는 비웁니다.
홈런은 모든 주자 득점을 인정합니다. 무주자 또는 2아웃의 희생 번트, 불가능한
병살·희생플라이는 합법 event로 변환합니다.

이 simulator는 승패·득점·안타·끝내기 타자의 인정 루타용이며 공식 기록원의 타점,
득점 주자 identity, 자책점까지 판정하지 않습니다. 해당 통계가 필요하면 별도
scorekeeping 계층을 추가해야 합니다.

## 직접 Head와 V26 최적화

- `DirectPlayerGameHead`: 미출장=PA 0, PA/hit overflow bucket, `hits <= PA` mask와
  PA bucket별 `P(H | PA, x)`를 가진 joint distribution입니다. 기대 안타와
  `negative_log_likelihood()`는 같은 분포에서 계산됩니다.
- `DirectRunDistributionHead`: 실제 loss가 있는 홈/원정 negative-binomial marginal만
  출력합니다. 학습되지 않던 scalar correlation 출력은 제거했습니다.
- `MatchPredictionOptimizer`: 가능한 일일 선택 조합을 전부 열거하고 기대점수,
  variance와 올킬 보상을 계산합니다. 사용되지 않던 `selection_rate` 필드는 제거했습니다.
- `LiveHitOptimizer`: joint Monte Carlo scenario로 포지션, 선수 중복, 포지션별 선택률,
  도감과 선택 구단 배율을 계산합니다. 임의의 연속 선택률 함수, 최소 인원 시너지,
  활성화 점수와 공식 포인트에 섞인 올 히트 점수는 제거했습니다.

Live Hit ruleset은 세 모드입니다.

```text
pure_hits   모든 배율을 제거하고 야구 안타 예측만 비교
fixed_300   전 선수 3배, 선택 구단 선수 4배인 provisional 비교 모드
account     명시적 포지션별 선택률 구간표 + 실제 도감 + 선택 구단
```

`fixed_300`의 기본 규칙명과 기본 안타 점수표 출처는 명시적으로 `provisional`입니다.
실제 게임 점수 상세 화면과 replay 사례를 확보하기 전에는 공식 ruleset으로 표시하지
않습니다. `account` 모드는 선택률 구간표와 안타 점수표 출처를 호출자가 반드시
제공해야 합니다.

추천 결과의 `expected_hit_points`에는 공식 히트 포인트만 들어갑니다. 올 히트는
`all_hit_probability`와 `all_hit_reward_id`로 분리되며, 사용자가 보상 가치를 평가하고
싶을 때만 `custom_utility`에 반영됩니다.

`search_mode="exact"`와 `"beam"`을 제공하고 slot/candidate 수, expanded/pruned state,
beam width, exact 여부와 known optimality gap을 진단에 남깁니다.

Beam은 전역 최적을 보장하지 않습니다. 작은 후보군이나 최종 검증에는 exact를 사용하고,
큰 후보군에서는 beam 결과의 `diagnostics.is_exact`를 반드시 확인합니다.

## 작업별 학습 계약

승부예측과 라이브 히트는 관계형 backbone을 공유할 수 있지만 같은 라벨이나 같은
batch로 학습하지 않습니다.

```text
PA loader          → shared backbone → PA adapter        → 10-way PA CE
PlayerGame loader  → shared backbone → Live Hit adapter  → joint PA/hit NLL
Game loader        → shared backbone → Match adapter     → WDL CE + run NB NLL
```

`TaskSeparatedModel`은 하나의 등록된 backbone과 세 작업별 adapter/head를 연결합니다.
`AlternatingMultiTaskTrainer`는 서로 다른 row granularity의 세 finite loader를 결정적인
순서로 번갈아 소비하며, 하나의 batch로 억지로 합치지 않습니다. Checkpoint에는 모델,
optimizer, RNG, epoch/step과 feature·route·label·model lineage가 들어가며 lineage가 다른
실행으로는 복구되지 않습니다.

Live Hit target은 `game_played`, `started`, `label_observed`를 별도로 갖습니다. 취소·노게임은
관측된 `game_played=false`이며 정상 경기의 벤치·0 PA와 같은 0으로 학습하지 않습니다.
승부예측 target도 `completed`, `result_observed` mask를 요구하고, 미완료 경기의 점수·WDL은
`-1` sentinel로 강제해 가짜 0:0 무승부를 만들지 않습니다. 각 loss의 `sample_count`는
원시 배치 크기가 아니라 실제 감독 신호에 사용된 행 수입니다.

이 계층은 실제 학습 루프이지 데이터 공급자 구현은 아닙니다. Parquet→tensor/subgraph
loader와 실제 KBO 학습 job은 별도로 연결해야 합니다.

## 명령

```bash
cpv26 show-config
cpv26 db-init
cpv26 db-check
cpv26 snapshot-build <prediction-run-id>
```

`snapshot-build`는 기존 `prediction_run`을 읽어
`CPV26_HOME/snapshots/<prediction-run-id>/`에 Parquet과 `manifest.json`을 만듭니다.

## 검증

```bash
source .venv/bin/activate
bash scripts/check.sh
```

개별 명령은 다음과 같습니다.

```bash
ruff check src tests scripts/build_code_summary.py
mypy --no-incremental src/cpv26
pytest
```

Windows CPU 검증 환경에서는 Python 3.12, DuckDB 1.5.5, PyTorch 2.13.0+cpu,
CatBoost 1.2.10으로 compileall, Ruff, strict mypy 42개 source, 117개 pytest, neural
forward/backward, CatBoost fit/predict와 `cpv26_predictor-0.4.0-py3-none-any.whl` 빌드를
통과했습니다. CUDA/A6000/A100 peak-memory 검증은
Linux GPU host에서 별도로 수행해야 합니다.

평가는 random row split이 아니라 game/cutoff 단위 날짜순 walk-forward를 전제로 하며,
정확도보다 log loss, Brier score, calibration과 실제 snapshot이 존재하는 기간의 V26
replay 효용을 우선합니다.

## 아직 구현하지 않은 범위

- 허가된 KBO 공급자별 CSV/API adapter와 안정적인 내부 ID mapping
- 투수·불펜·포수·수비·구장·날씨·부상 전체 feature pipeline
- Parquet→temporal subgraph mini-batch loader
- AMP, gradient accumulation, distributed training과 model registry
- 실제 시즌 dataset을 사용하는 학습·calibration·fold artifact job
- CatBoost·MLP·GraphSAGE·RelGNN 동일 조건 walk-forward benchmark
- empirical runner advancement fitting과 공식 기록원 수준 scorekeeping
- V26 실제 시즌 ruleset replay 자료와 당일 추천 orchestrator
- 운영 API, dashboard, scheduler와 게임 계정 연동

따라서 지금 단계의 올바른 용도는 실제 데이터를 연결하기 전 correctness가 검증된 연구
기반이며, 실전 추천기라고 부르기에는 아직 이릅니다.
