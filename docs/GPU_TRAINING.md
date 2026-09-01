# KBO RelGNN GPU 학습 기술 설명

이 문서는 구현된 학습 경로의 데이터 계약·모델·재현성·한계를 설명합니다.
설치와 처음부터 실행하는 순서는 [README](../README.md)를 따릅니다.
핵심 구현은 `src/cpv26/data/kbo_graph_dataset.py`,
`src/cpv26/models/kbo_relgnn.py`, `src/cpv26/training/kbo_runner.py`입니다.

## 실행 환경

환경은 `environment.yml`로 만든 Python 3.12 **Conda `cpv26` 환경**을 사용합니다.
README의 Conda 설치 절차에 따라 환경을 생성·활성화한 뒤 CUDA 패키지를 설치합니다.

설치 후 새 Bash 터미널에서 학습을 시작하거나 재개할 때는 다음을 실행합니다.
프로젝트 경로는 설치한 위치에 맞춥니다.

```bash
cd ~/projects/cpv26-predictor
conda activate cpv26
source scripts/activate.sh
cpv26 gpu-check --device cuda:0
```

`source scripts/activate.sh`는 Conda 활성화가 아니라 `.env`의 프로젝트 설정 로드입니다.
CPU smoke는 CUDA 패키지를 바꾸지 않도록 README의 별도 Conda `cpv26-cpu` 환경에서
실행합니다. 데이터·DB·checkpoint 경로와 학습 계약은 환경 전환으로 바뀌지 않습니다.

## 데이터 출처와 고정 스냅샷

### 2001~2022 경기·타격·투구 기록

`kbo-history-fetch`와 `kbo-history-import`는 다음 공개 아카이브에서 실제 경기 결과를 읽습니다.
기본 범위는 2001~2022년이며 `--start-year`와 `--end-year`로 양끝을 포함한 범위를 지정합니다.

| 기간 | 원천 | 고정 revision |
| --- | --- | --- |
| 2001~2020 | [dialektike/KBO-league](https://github.com/dialektike/KBO-league) 연도별 JSON | `00c63c74c3c0590f3ca2fae5c03d4d2eeaa18296` |
| 2021~2022 | [LOPES-HUFS/KBO_data](https://github.com/LOPES-HUFS/KBO_data) 월별 JSON | `94e72c797e07b3b72167c92258728bef599ed5fc` |

아카이브의 경기 ID·홈/원정·최종 득점·승패를 검사합니다. 2021년 중복 8건은 한 번만
사용하고 순위결정전은 `tiebreaker` 유형으로 보존·학습합니다. 원천에서 빠진 2015년 1경기,
2018년 1경기, 2021년 8경기는 **KBO 공식 기록에서 확인한 최종 점수**로 보완합니다.
확인한 요청 조건·경기 ID·공식 응답 해시는
[보완 기록 출처](../src/cpv26/data/history_supplement_sources.json)에 보존합니다.

schema v5는 `historical_boxscore`에 타자·투수의 모든 원문 행, 파싱한 수치, 품질 사유를
보존하고 `historical_game_detail`에 스코어보드·ETC 등 나머지 경기 원문을 보존합니다.
기존 점수 전용 source revision과 경기 행을 수정하지 않고 별도 box-score revision을 추가합니다.

| 원천 항목 | 학습에서의 사용 |
| --- | --- |
| 최종 승패·득점 | Match와 득점 정답, 다음 날부터의 팀 이력 |
| 타수·안타·득점·타점 | 관측 항목별 과거 타격 이력, 확인된 안타 정답 |
| 이닝별 타격 결과 | AB/H와 상대 투수 BF 대조를 통과한 10종 결과 집계 정답 |
| 타자수·이닝·투구수·피안타·홈런·4사구·삼진·실점·자책 | 관측 항목별 투구 정답과 과거 이력 |
| 선수명 | 미확정 이름·팀·역할 그룹의 과거 통계 조회에 사용; 개인 ID 병합 없음 |
| 포지션·누적 승패/ERA/타율·ETC | 원문·감사 정보 보존, 확인되지 않은 사전 입력으로 사용하지 않음 |

빈 값이나 모순이 있으면 해당 수치만 마스킹하고 다른 관측값은 유지합니다.
정확한 PA를 확인하지 못해도 안타와 출전 증거가 있으면 LiveHit를 학습합니다.
포수방해는 별도로 보존하며 기존 10분류로 강제 변환하지 않습니다.
과거 이닝 결과는 **집계 정답**이지 타석 순서·상대 투수·주자 상태를 복원한 PA 로그가 아닙니다.

고유 선수 ID가 없으므로 이름으로 경력을 합치지 않습니다. 행별 source-observation ID를
사용하고 과거의 같은 이름·팀·역할 그룹 통계를 prior로 줍니다. 동명이인을 포함할 수
있는 그룹이며 개인의 경력이 아닙니다. 그룹 이력이 없으면 팀 통계를 사용합니다.
2023년 이후 실제 ID와 타석 경로는 그대로 유지합니다. manifest의
`label_quality.historical_boxscore_identity`에 미확정 행·그룹·이름 없는 행·동명이인
충돌 그룹 수를 남기며, 공식 개인 ID를 확보한 것처럼 보고하지 않습니다.

역사 파일의 SHA-256은 `kbo_history_source.py`에 고정되어 있습니다. 출처는 다운로드
`SOURCE.json`, DB의 `source_revision`, 그래프의 `source_provenance`에 남깁니다.
`kbo_history_import.json`에는 연도별 경기·타자·투수·라벨 수, 원문 필드와 품질 사유를 기록합니다.
동일 파일의 재적재는 기존 행을 재사용합니다. 점수 또는 중복 경기의 박스스코어 원문이
충돌하면 전체 적재를 rollback하며 한쪽을 임의 선택하지 않습니다.

두 아카이브는 KBO 공식 배포본이 아닙니다. 첫 저장소는 GPL-3.0을 표시하고 두 번째는
저장소 라이선스를 명시하지 않습니다. 이 표기를 원자료에 대한 포괄적 이용 허락으로
해석하지 않으며, 원본 파일이나 외부 scraper 코드를 이 프로젝트에 복사하지 않습니다.

### 2023~2026 관측 타석

타석 데이터는 **KBO 공식 배포 데이터가 아니라**, NAVER Sports 중계를 가공해
`slothman3878`이 공개한 Parquet입니다. 작성자와 추출 방식은
[배포자 데이터 카드](https://huggingface.co/datasets/slothman3878/kbo_playbyplay)와
[원본 변환 프로젝트](https://github.com/slothman3878/kbo_pbp_naver_sports)를 참고합니다.
배포자는 CC BY 4.0을 표시하고 있지만, 원천 중계 자료의 별도 권리·이용조건까지
해결됐다는 뜻은 아닙니다. 재배포 시 작성자·출처·가공 사실을 보존해야 합니다.

다운로더는 변하는 `main`이 아니라 revision
`6afc8af044e3bba5f326b688e8cb41d7ff7065ec`의
`v0/kbo_pbp_2023.parquet`, `kbo_pbp_2024.parquet`, `kbo_pbp_2025.parquet`를 기본으로 사용합니다.
`kbo-fetch --year 2026`으로 같은 revision의 `kbo_pbp_2026.parquet`도 받을 수 있으며,
포함 범위는 2026년 7월 26일까지입니다. 2001~2022년 경기 기록과 구분해 적재합니다.
파일별 예상 SHA-256은 `kbo_playbyplay.py`에 고정돼 있습니다.
`.part` 다운로드 → 체크섬 검증 → 파일 교체 순서이며, 유효한 기존 파일은 재사용합니다.
다운로드 출처는 `SOURCE.json`, canonical 변환 버전과 원본 체크섬은
DuckDB `source_revision` 및 그래프 manifest의 `source_provenance`에 남습니다.
README 실행은 2023~2026년을 모두 명시적으로 내려받고 2026년을 테스트에만 사용합니다.

CPV26은 공개 원본을 canonical PA/경기 기록과 학습 그래프로 다시 가공합니다.
이 저장소의 변환·모델 결과를 데이터 작성자, KBO 또는 NAVER의 공식 결과로 표시하지 않습니다.

adapter v2는 더블헤더 번호를 반영한 경기·타석 revision을 추가합니다. v3/v4 사용자는
[README의 갱신 명령](../README.md#4-1-기존-v3v4에서-v5로-갱신)에 따라 `git pull` 후
기존 Conda 환경에서 `source scripts/activate.sh`, 캐시의 2023~2026 `kbo-import`,
`db-check`, 새 v5 그래프 생성·검사만 실행합니다. 원천 다운로드나 환경 재설치,
DB 삭제는 필요 없으며 DB schema는 **5**를 유지합니다.

연도별 파일은 같은 배포자·시즌의 **전체 스냅샷**으로 취급합니다. `knowledge_at`까지
수집된 revision 중 최신 하나를 먼저 고른 뒤 그 안에서 경기·타석을 조회합니다.
새 파일에서 삭제된 경기·타석이 이전 파일의 행으로 다시 나타나지 않도록 합니다.
이전 DB 행과 출처는 보존하여 이전 knowledge 시점 조회에 사용할 수 있습니다.
이 선택 기준은 예측일 cutoff가 아니라 수집 지식 시각이며, 선택한 행에는 별도의
공개 시각·유효기간·과거 날짜 조건을 적용합니다. 기본 graph build는 현재 DB에 수집된
내용을 아는 회고적 snapshot을 사용합니다.

## 날짜별 그래프: 버전 5 (기존 버전 2·3·4 읽기 지원)

v3의 박스스코어 입력은 2001~2022에만 채워지고 최근 타석 입력과 연결되지 않았습니다.
v4부터 현재 cutoff에서 유효한 PA만 먼저 선택한 뒤 선수·경기·역할별 합계로 묶어 같은
19/21차원 입력에 넣습니다. 합계와 관측 횟수를 짝지으며, 관측 횟수는 타석 수가 아니라
선수·경기 수입니다. 지연 공개·수정·만료 시 영향을 받은 그룹만 다시 계산합니다.

타석만으로 확정할 수 없는 타자 득점/RBI, 투수 아웃·투구 수·실점·자책점은 결측입니다.
과거 타격 자료의 PA/AB/H/TB/BBHBP/K/HR가 모두 확인되면 기존 8차원 타격 입력에도
연결합니다. 과거 투수 자료에서 TB가 없다고 0으로 꾸며 기존 비율을 만들지 않습니다.

v5는 겹치는 출처를 행 전체가 아니라 **관측 필드·학습 항목별로** 처리합니다.
확인된 동일 선수는 박스스코어의 관측 필드를 우선하며, PA 집계와 모순되지 않는
결측 항목만 채웁니다. 불완전한 PA 결과별 집계를 박스스코어의 완전한 타격 합계로
격상하지 않습니다. 예를 들어 박스스코어 H=2와 PA 집계 H=1이 다르면 후자의
결과별 횟수나 총루타를 전자의 완전한 기록인 것처럼 붙이지 않습니다.

신원 미확정 행은 개인별로 병합하지 않습니다. 팀 이력도 관측 필드별로 중복을 막고,
LiveHit·타격 집계·투구 정답은 해당 항목에 실제로 쓸 수 있는 기록을 기준으로
우선순위를 정합니다. 아카이브 행이 있다는 이유만으로 다른 출처의 모든 정답을
버리지 않습니다. 실제 PA 개인 이력과 순서 있는 PA 정답, 두 출처의 원문은 유지합니다.

하루 `D`의 공통 기준 시각은 **Asia/Seoul 00:00**입니다. 역사 입력은 다음 조건을
동시에 만족해야 합니다.

- 경기/사건 날짜가 `D`보다 이르고, 최근 90일 범위 안에 있습니다.
- `available_at <= cutoff`이며 해당 cutoff에서 `valid_from`/`valid_to`가 유효합니다.
- 데이터베이스 knowledge snapshot에 들어온 revision만 사용합니다.

원본에는 실제 발표 시각이 없으므로 importer는 경기 종료 자료를 다음 날 00:00에
공개된 것으로 재구성합니다. `ingested_at`은 실제 로컬 수집 시각입니다.
따라서 이것은 **과거 공개 시각을 재구성한 회고적 벤치마크**이지,
해당 시점에 실제로 수집해 둔 자료만으로 한 실시간 운영 재현은 아닙니다.
날짜 필터로 일부 기간만 출력해도 그 이전 원본 이력은 버리지 않습니다.

| 입력 | 차원 | 내용 |
| --- | ---: | --- |
| 선수 공통 특성 | 4 | 과거 타격/투구 PA 규모와 최근성 |
| 선수 타격 역할 | 8 | PA, AB·안타·총루타·볼넷/HBP·삼진·홈런 비율, 최근성 |
| 선수 투구 역할 | 8 | 상대 PA와 같은 사건의 허용/삼진 비율, 최근성 |
| 팀 특성 | 8 | 경기 수, 승/무 비율, 득실점, 홈 비율, 최근성 |
| 박스스코어 타격 확장 | 19 | 9개 항목의 관측 합계·관측 개수와 최근성 |
| 박스스코어 투구 확장 | 21 | 10개 항목의 관측 합계·관측 개수와 최근성 |
| `batter_pa_pitcher` | 관계당 6 | 과거 완료 PA의 타격 선수 → 투구 선수 집계 |
| `batter_participation_team` | 관계당 6 | 과거 타격 출전 선수 → 당시 타격 팀 |
| `pitcher_participation_team` | 관계당 6 | 과거 투구 출전 선수 → 당시 수비 팀 |
| `home_team_game_away_team` | 관계당 6 | 과거 홈 팀 → 원정 팀의 완료 경기 집계 |

선수 ID는 공통 노드 집합을 쓰지만 타격·투구 역할 상태는 분리합니다.
관계에는 사건의 경과 시간, 공개 지연, 가중치도 전달합니다.
수치 특성은 고정 로그/비율 스케일을 사용하며, 전체 시즌 평균으로 정규화하지 않습니다.
정확한 특성명·차원·역할은 manifest의 `feature_names`, `*_feature_dims`,
`route_metadata`가 기준입니다. 선수–팀 관계는 **과거 출전 이력**이며 실제 등록 명단이 아닙니다.

당일 결과·실제 선발·타순·출전 명단을 WDL 그래프 관계로 연결하지 않습니다.
당일 처음 등장한 선수는 정답 질의를 만들기 위해 노드 ID가 필요하지만,
직접 이력이 없으면 확인된 과거 그룹/팀 prior만 사용할 수 있습니다.
박스스코어 관측 노드에는 이전 날짜의 이름·팀·역할 그룹 통계를 주며, 당일 타격·투구
수치는 정답에만 둡니다. 그룹 노드를 별도로 유지하므로 관측 ID를 개인 ID로 합치지 않습니다.
더블헤더 첫 경기 결과도 같은 날 다른 경기의 역사 입력으로 들어가지 않습니다.

README 실행의 캐시는 `var/datasets/kbo_graph_2001_2026_v5/manifest.json`과
`days/YYYY-MM-DD.npz`에 저장합니다. 기존 v2/v3/v4 그래프를 덮어쓰지 않습니다.
NPZ는 object 배열 없이 저장하고 `allow_pickle=False`로 읽습니다.
설정·정책 버전·canonical 행/출처의 fingerprint와 파일 SHA-256을 확인하며,
날짜별 sidecar로 동일한 입력의 파일을 재사용합니다. 미래 날짜가 추가되어도
입력이 변하지 않은 과거 날짜는 재사용할 수 있습니다.

manifest의 `season_coverage`와 날짜별 항목은 `games`, `games_with_pa`,
`game_only_games`, `observed_completed_pa`, `live_hit_queries`, `pa_queries`를 기록합니다.
`games_with_pa`는 관측된 완료 PA가 있다는 뜻이며 모든 공식 PA가 빠짐없다는 뜻은 아닙니다.
경기 기록만 있는 날짜도 팀 노드·과거 경기 관계와 최종 점수 정답을 유지합니다.
완료 PA가 없어도 박스스코어가 있으면 별도 선수 질의와 집계 정답을 유지합니다.
집계 정답의 질의 수·결과 횟수·관측 필드 수는 아카이브와 PA 파생 자료를 모두 포함해
`label_quality.training_targets`에 기록하며 날짜·연도별 건수도 같은 기준입니다.
`raw_archive_boxscore`는 아카이브 원문만의 건수를 별도로 보존하는 항목이며
전체 학습 사용량과 구분합니다. 원문 행 수, 실제 질의 수, 손실의 분모는 서로 다릅니다.

## 라벨과 시간순 분할

README의 실행은 **2001~2024년 학습, 2025년 검증, 2026년 부분 시즌 테스트**입니다.
2001~2022년에는 경기·안타·집계 타격 결과·투구 라벨을 사용합니다.

아래는 adapter v1로 적재했던 2023~2025년의 이전 분할 수치를 비교용으로 보존한 표입니다.
v5 재수입 후 건수는 새 manifest를 확인합니다. 현재 2001년부터의 학습 경로를
이 세 시즌으로 제한한다는 뜻이 아닙니다.

| 역할 | 시즌 | 날짜 수 | 경기/WDL | 선수–경기 LiveHit | PA 10분류 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 학습 | 2023 | 167 | 720 | 16,363 | 56,230 |
| 검증·모델 선택 | 2024 | 163 | 720 | 16,271 | 57,253 |
| 최종 보류 테스트 | 2025 | 164 | 720 | 16,457 | 55,994 |
| 합계 | — | 494 | 2,160 | 49,091 | 169,477 |

이전 분할의 완료된 canonical PA는 169,481개입니다. 포수방해 4개는 PA 10분류에서만 제외하고
관측 PA 수와 LiveHit 집계에는 남깁니다. 점수 전이가 불완전한 11개도 유효한 PA
결과 라벨은 유지합니다. 버전 2에서는 해당 PA의 타석 전 홈/원정 점수만
`0, 0`으로 바꾸고 두 결측 플래그를 `1, 1`로 설정합니다.
WDL·LiveHit 라벨과 과거 집계는 이 마스킹으로 바뀌지 않습니다.
두 건수는 manifest `label_quality`에 따로 기록됩니다.

연도 옵션을 생략한 CLI 기본값은 기존 checkpoint와의 호환을 위해 2023/2024/2025로
유지합니다. README는 모든 연도를 명시하여 이 기본값을 사용하지 않습니다.
검증/테스트 시즌에서도 각 평가일 이전에 끝난 같은 시즌 경기는 역사 입력에 포함됩니다.
이는 매일 관측 이력을 갱신하는 평가이며, 그 이력으로 모델 파라미터를 재학습한다는 뜻은 아닙니다.
`max_days_per_split`은 각 split 전체에 걸쳐 날짜를 균등 간격으로 줄이는 smoke 옵션입니다.
그 결과는 전체 시즌 성능으로 해석하면 안 됩니다.

### 다년 학습과 날짜 순서

`relgnn-train`의 `--train-start-year`, `--train-end-year`는 양끝 연도를 포함한 학습
범위입니다. `--validation-year`, `--test-year`로 이후 검증·테스트 시즌을 지정합니다.
학습 종료연도 < 검증연도 < 테스트연도여야 하며, 요청한 학습연도마다 데이터가 있어야 합니다.
기본값은 기존대로 2023년 학습·2024년 검증·2025년 테스트입니다.

[README의 실행 예시](../README.md#5-2001년부터-relgnn-학습)는
2001~2024년으로 학습하고, 2025년으로 모델을 선택한 뒤 2026년 부분
시즌을 별도로 평가합니다. 이미 평가 결과를 본 2025년은 이 실험에서 개발용 검증으로
전환되며, 최종 보류 테스트라고 표시하지 않습니다. 기존 그래프 대신
`var/datasets/kbo_graph_2001_2026_v5`를 만들고 새 run에서 시작합니다.

v4부터 최근 관측 타석에서도 집계 정답을 만들므로 2025/2026의 `box_pa`와
`box_pitch` 평가가 더 이상 단순히 null이 아닙니다. 기본 가중 손실 공식/가중치는 유지하지만
관측되는 정답이 늘어 전체 선택 손실을 v3 숫자와 그대로 비교하면 안 됩니다. 승무패·안타·
PA의 같은 테스트 경기 지표를 각각 비교하고, 한 번 본 테스트를 반복 튜닝 기준으로 삼지 않습니다.

`--selection-target match`는 validation의 경기 log loss로 best.pt를 고르는 명시적 옵션입니다.
기본 `auto`/`weighted`는 기존 가중 손실을 사용합니다. `--box-gradient-mode head_only`는
집계 출력층에만 해당 손실을 역전파하고 주 작업/집계층 gradient clipping도 분리합니다.
기본 `auto`/`shared`는 모든 손실을 공유층으로 역전파합니다. 간섭이 실제 성능 저하의
원인이었다고 확정한 것이 아니므로 이러한 비교 옵션을 자동으로 적용하지 않습니다.

`--chronological`이면 매 epoch 전체 학습 날짜를 오름차순으로 순회하며 시즌 경계에서
모델·optimizer를 초기화하지 않습니다. 다음 epoch는 학습 시작 날짜로 돌아갑니다.
기본값은 날짜 shuffle이며, 시간순 모드는 온라인 predict-before-learn 평가나 새 데이터
추가 학습을 구현한 것이 아닙니다. 각 날짜의 입력은 여전히 그 날짜 이전 최대 90일입니다.
검증·테스트 결과로 가중치를 갱신하지 않습니다.

## 실제 모델과 학습 신호

`RoleAwarePlayerEncoder`와 `CompositeRelGNNBackbone`을 공유하고,
관계별 메시지 전달로 선수 역할 상태와 팀 상태를 만듭니다.
여러 날짜를 묶을 때 노드 인덱스를 offset한 서로 연결되지 않은 그래프로 구성합니다.

| 출력 헤드 | 입력 | 학습 목표 |
| --- | --- | --- |
| Match WDL | 과거 그래프의 홈/원정 팀 상태 | 원정승·무승부·홈승 3분류 CE |
| 득점 NB2 | 같은 홈/원정 팀 상태 | 양 팀 최종 점수의 음이항 NLL 합 |
| LiveHit | 타자 역할 상태, 타격 팀·상대 팀 상태와 상호작용 | 선수–경기 `(PA, H)` 결합분포 CE |
| PA auxiliary | 타자·투수 역할 상태와 타석 전 context | 10종 타석 결과 CE |
| Box batting | 타자 역할·팀·상대 팀 상태 | 실제 결과별 횟수로 가중한 10종 집계 CE |
| Box pitching | 투수 역할·팀·상대 팀 상태 | 관측된 10개 수치만 사용하는 Poisson NLL |

득점 헤드는 홈/원정 각각의 양수 평균과 dispersion을 출력하는 독립 NB2 주변분포입니다.
분산은 `mean + mean² / dispersion` 형태입니다. WDL은 별도 직접 분류 헤드이므로
NB2 분포를 적분해 얻은 승률과 일치하도록 강제하지 않습니다.

PA context는 이닝, 초/말, 아웃, 세 베이스 점유, 홈/원정 점수와 두 결측 플래그의
10개 값입니다. **현재 타석 직전 상태는 PA decoder에만** 전달하며,
WDL이나 LiveHit 헤드의 입력에 추가하지 않습니다.
PA 클래스 순서는 삼진, 볼넷/HBP, 단타, 2루타, 3루타, 홈런, 인플레이 아웃,
실책 출루, 희생번트, 희생플라이입니다. 병살·야수선택은 인플레이 아웃에 합칩니다.

LiveHit의 모집단은 **완료된 관측 PA가 1개 이상인 선수–경기**입니다.
출전하지 않은 선수를 음성 라벨로 만들어 넣지 않습니다.
따라서 `P(H >= 1)`은 이 조건부 모집단의 확률이며, 미출전 가능성을 포함한 전체 후보 확률이 아닙니다.
기본 PA bin은 `1..8, 9+`, 안타 bin은 `0..5, 6+`이고 불가능한 `H > PA` 영역을 마스킹합니다.
overflow bin의 평균을 알 수 없으므로 출력의 기대 PA/안타는 마지막 bin을 각각 9/6으로
계산한 **하한 기대값**입니다. 정확한 꼬리 평균이나 출전 확률로 해석하지 않습니다.

PA가 불명확한 안타 정답은 `-log Σ P(PA, H관측)`으로 학습합니다. 합산 범위는 확인된
타수 이상의 PA이며, 임의의 PA 값을 정답으로 넣지 않습니다. PA가 모르는 행은
PA 기대값 MAE 분모에서 제외하고 안타 지표에는 포함합니다. PA 상한 bin 이상의
하한은 해당 overflow 범위로 제한해 근사하며 그 건수를 보고합니다.

기본 총손실은 `WDL + LiveHit + 0.2×PA + 0.1×득점 + 0.2×Box batting + 0.1×Box pitching`입니다.
각 항은 해당 task의 평균이며, 검증에서도 task별 표본 수로 평균을 집계한 뒤
같은 가중합으로 checkpoint를 선택합니다.
각 날짜에서 실제 정답이 있는 항만 계산합니다. 기존 v2 그래프/모델은 추가 헤드를 사용하지 않습니다.
정답이 없는 task의 평가 지표는 `null`이며 가짜 0점 정확도를 만들지 않습니다.

## 학습 기본값과 CUDA 동작

| 항목 | 기본값 |
| --- | --- |
| 장치 / AMP | `cuda:0` / `auto` |
| 최대 epoch / early stopping | 30 / 검증 개선 없는 6 epoch |
| 은닉 차원 / 레이어 / attention head | 64 / 2 / 4 |
| dropout | 0.1 |
| optimizer / learning rate / weight decay | AdamW / `3e-4` / `1e-4` |
| minibatch / gradient accumulation / clip | 날짜 2개 / 1 step / norm 1.0 |
| 데이터 로더 worker / seed | 2 / 2026 |
| 학습 날짜 순서 | shuffle (`--chronological`로 오름차순 선택) |
| CLI 학습 PA 질의 상한 | `0`: 전부 사용 |
| CLI 관계 상한 | `0`: 전부 사용 |

양수 상한을 명시했을 때만 PA를 표본화하거나 최근 관계부터 제한합니다.
`sampling_limits`에 적용 값을 남깁니다. 검증·테스트에는 PA 상한을 적용하지 않습니다.
Python API의 기존 기본값(128/20000)은 이전 설정 재현을 위해 남아 있고 CLI는 0을 전달합니다.
이 경로는 지정한 한 CUDA 장치에서 실행하며 DDP 다중 GPU 학습은 구현하지 않았습니다.

`gpu-check`는 실제 CUDA 행렬곱·autocast·역전파와 유한한 gradient를 검사합니다.
CUDA를 요청했는데 사용할 수 없으면 즉시 실패하고 CPU로 자동 전환하지 않습니다.
`auto`는 BF16 지원 GPU에서 BF16, 아니면 FP16을 고릅니다.
FP16은 GradScaler를 사용하고, overflow로 건너뛴 optimizer step 수를 기록합니다.
CPU 검증은 명시적으로 CPU를 선택하며 runner에서는 AMP를 끕니다.
GPU 이름, VRAM, PyTorch/CUDA 버전, precision과 peak memory를 보고서에 기록합니다.
seed를 고정해도 CUDA atomic 연산까지 bitwise 재현된다고 보장하지 않습니다.

데이터 로더 워커는 `spawn`으로 시작합니다. 이미 열린 PyTorch 스레드/가속기 상태를
`fork`로 복제하지 않으며, 프로그램 전역의 multiprocessing 설정은 바꾸지 않습니다.
`--workers 0`이면 별도 워커를 만들지 않습니다. CLI 실행 방법은 같고, Python API를
직접 스크립트에서 호출할 때는 진입점을 `if __name__ == "__main__":`로 보호합니다.
워커 시작에 드는 비용이 있으므로 이 변경 자체가 속도 향상을 뜻하지는 않습니다.

## 체크포인트·재개·평가 산출물

학습 run 디렉터리에는 `config.json`, `history.jsonl`, `best.pt`, `last.pt`,
`training_report.json`이 생성됩니다. `best.pt`는 선택한 검증 기준의 최저 모델
(기본 가중손실, `--selection-target match`이면 경기 log loss),
`last.pt`는 마지막으로 완료된 epoch 경계입니다.
모델·optimizer·scaler·난수 상태·epoch·step·학습 이력·데이터 fingerprint를 함께 저장합니다.

재개는 **같은 run 디렉터리의 `last.pt`**로만 합니다. 데이터 fingerprint와 모델/특성/관계
설정, 학습·검증·테스트 연도 또는 날짜 순서가 다르면 거부합니다. 새 시즌을 추가한
데이터에 기존 checkpoint를 `--resume`하는 기능은 아닙니다.
v5 갱신도 새 그래프·새 run에서 시작합니다. v2/v3/v4 checkpoint와 그래프는 보존하며,
기존 checkpoint의 평가·재개에는 당시의 그래프를 그대로 사용합니다.
재개 시 바꿀 수 있는 설정은 총 목표 epoch, 장치, worker,
batch days, AMP, accumulation, patience이며, epoch는 추가 횟수가 아니라 총 목표입니다.
batch/precision 변경을 허용한다는 것이 이전과 동일한 수치 경로를 보장한다는 뜻은 아닙니다.
저장되지 않은 진행 중 batch는 복구되지 않으며, epoch 도중 중단된 계산은 마지막으로
저장한 완료 epoch 지점부터 다음 재개에서 다시 수행합니다. 새 학습이나 평가가
이미 파일이 있는 출력 디렉터리를 조용히 덮어쓰지는 않습니다.

`relgnn-evaluate`는 checkpoint를 다시 읽어 평가하며, 기본 출력은 run 아래
`evaluations/<split>-<timestamp>-<id>/`입니다.

- `metrics.json`: split, checkpoint/데이터 fingerprint, 실행 환경, task별 손실,
  log loss·Brier score·ECE·accuracy, LiveHit 결합 NLL 및 하한 기대값 MAE.
- `match_predictions.parquet`: 경기 질의 ID, 라벨, 원정승/무승부/홈승 확률.
- `live_hit_predictions.parquet`: 선수–경기 ID, 안타 유무 확률, 실제 PA/H,
  하한 기대 PA/H.
- `pa_predictions.parquet`: PA ID, 라벨, 10종 결과 확률.
- `box_pa_predictions.parquet`, `box_pitch_predictions.parquet`: 집계 정답·마스크와 예측값.

예측 파일별 행 수와 SHA-256도 남깁니다. NB2 손실은 평가하지만 현재 별도의 득점
분포 Parquet는 내보내지 않습니다. ECE를 계산하는 것과 calibration 모델을 학습·적용하는
것은 다릅니다. 이 runner에는 OOF stacking이나 후처리 calibration이 자동 연결돼 있지 않습니다.

## 고정 checkpoint 그래프 의존도 진단

모델 크기나 그래프 schema를 바꾸기 전에는 같은 `best.pt`에서 관계 메시지를 실제로
사용하는지 먼저 확인합니다. 진단 기본값은 test가 아니라 checkpoint의 validation split입니다.

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

진단은 node feature, 질의, 정답과 checkpoint를 고정하고 route 입력만 바꿉니다.
`intact`, 전체 route 제거, 날짜·관계 안에서의 endpoint 재배선, endpoint를 유지한
edge attribute 순열, 관계별 단독 제거를 각각 평가합니다. 서로 다른 날짜의 노드를
연결하지 않으며 조작 전후 edge 수와 실제로 변경된 항목을 `transform`에 기록합니다.
같은 질의 ID와 라벨이 유지되지 않으면 비교 결과를 만들지 않습니다.

조건별 보고 내용은 다음과 같습니다.

- 원래 task별 metric과 selection loss
- `intact` 대비 metric 변화. selection loss delta가 양수면 조작 후 손실이 증가한 것입니다.
- 같은 질의별 확률분포의 total variation과 예측 class 변경률. Live Hit TV는 전체 PA×H
  결합분포가 아니라 `안타 없음/한 개 이상` 이진 주변분포입니다.
- 정상(`intact`) 조건의 layer·route·방향별 메시지 전달 내부 진단. intervention 조건은
  동기화 비용을 피하기 위해 이 내부 통계를 수집하지 않습니다.
- checkpoint SHA-256, 데이터 fingerprint, split 날짜, seed와 실행 환경

`no_routes`와 endpoint 재배선에서 loss와 예측 확률이 거의 변하지 않으면 현재 checkpoint가
관계 topology에 거의 의존하지 않는다는 근거입니다. 특정 `without_<route>`만 변화가 작으면
그 관계의 현재 기여가 작을 가능성을 먼저 확인합니다. 내부 메시지가 작거나 gate가 한 관계로
몰리는 현상도 함께 볼 수 있지만, 하나의 수치만으로 원인을 확정하지 않습니다.

Endpoint·edge attribute 순열은 지정한 seed의 한 번의 조작입니다. 아주 작은 loss/TV 차이는
반복한 정상 평가의 변동폭과 여러 intervention seed에서 같은 방향으로 재현되는지 확인합니다.

반대로 조작 후 손실이 커지는 것은 고정 checkpoint가 그 입력에 의존한다는 뜻이지,
GNN 구조가 node-only 모델보다 일반화 성능이 좋다는 뜻은 아닙니다. 구조 선택에는 동일한
특징·분할·loss·학습 예산으로 원래 그래프, node-only, 재배선 control을 각각 처음부터
여러 seed로 학습하는 비교가 추가로 필요합니다. 이 선택은 validation에서 마치고 test는
마지막 한 번의 평가에 사용합니다. `--max-days` 결과는 실행 경로를 확인하는 smoke 진단으로만
취급합니다.

## 아직 연결되지 않은 범위와 해석 한계

출처의 `events`가 없는 구간은 완료 PA 라벨에 포함되지 않습니다. canonical PA만 보고
이 누락 건수를 0으로 판단해서는 안 되며 원본 import 보고서를 함께 확인해야 합니다.
누락 PA와 중계/공식 기록 차이 때문에 관측 PA·안타 집계가 공식 기록과 완전히 같다고
보장할 수 없습니다. 공식 실책 수, 신뢰할 수 있는 사전 라인업 발표, 선수 좌우 정보,
정확한 실제 경기 시작 시각 등도 이 학습 입력에 임의로 채우지 않습니다.

타자는 importer의 타석 귀속 정책, 투수는 마지막 관측 투수 ID를 사용합니다.
투수 ID는 공식 실점 책임 투수라는 뜻이 아니며, 2스트라이크 대타 등에서는 귀속 타자와
마지막 투수가 실제 마지막 공에서 대면한 조합이 아닐 수 있습니다.

이 학습 경로는 V26 실제 후보 목록·선택 시각·미출전 확률·최적 조합 선택에 아직 연결되지
않았습니다. 별도로 존재하는 순차 경기 simulator에 완전한 교체·주루 사건 열을 공급하는
경로도 아닙니다. 원본의 runner-only 사건과 전이 누락을 추정으로 메우지 않으므로
manifest의 `simulator_ready`는 `false`입니다.

CatBoost 경로는 별도 표형 비교 baseline으로 선택할 수 있습니다.
그 실행 결과는 여기 설명한 RelGNN 학습이나 GPU 검증을 대신하지 않으며,
실제 검증 전에는 어느 모델이 더 정확하다고 단정하지 않습니다.
기존 2023년 학습 경로는 A100 MIG 10GB 실행 로그가 있으나, 그것이 확장된 2001년부터의
전체 학습 완료나 성능 향상을 검증한 것은 아닙니다. 테스트·smoke와 실제 전체 학습은 구분합니다.

검증 결과는 HANDOFF.md를 참고.
