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

### 2001~2022 경기 기록

`kbo-history-fetch`와 `kbo-history-import`는 다음 공개 아카이브에서 실제 경기 결과를 읽습니다.
기본 범위는 2001~2022년이며 `--start-year`와 `--end-year`로 양끝을 포함한 범위를 지정합니다.

| 기간 | 원천 | 고정 revision |
| --- | --- | --- |
| 2001~2020 | [dialektike/KBO-league](https://github.com/dialektike/KBO-league) 연도별 JSON | `00c63c74c3c0590f3ca2fae5c03d4d2eeaa18296` |
| 2021~2022 | [LOPES-HUFS/KBO_data](https://github.com/LOPES-HUFS/KBO_data) 월별 JSON | `94e72c797e07b3b72167c92258728bef599ed5fc` |

아카이브의 경기 ID·홈/원정·최종 득점·승패를 검사합니다. 2021년 중복 8건은 한 번만
사용하고 정규시즌과 별도인 순위결정전은 제외합니다. 원천에서 빠진 2015년 1경기,
2018년 1경기, 2021년 8경기는 **KBO 공식 기록에서 확인한 최종 점수**로 보완합니다.
확인한 요청 조건·경기 ID·공식 응답 해시는
[보완 기록 출처](../src/cpv26/data/history_supplement_sources.json)에 보존합니다.

원본 박스스코어에 선수별 합계가 있어도 이 어댑터는 실제 순서가 없는 타석을 만들어내지
않습니다. 2001~2022년은 경기·팀·팀-경기·출처만 적재하며, **Match와 득점 헤드만**
학습합니다. 선수 노드나 PA·LiveHit 정답은 생성하지 않습니다.

역사 파일의 SHA-256은 `kbo_history_source.py`에 고정되어 있습니다. 출처는 다운로드
`SOURCE.json`, DB의 `source_revision`, 그래프의 `source_provenance`에 남깁니다.
`kbo_history_import.json`에는 연도별 경기 수, 중복 제거, 날짜 범위를 기록합니다.
동일 파일의 재적재는 기존 행을 재사용하고 점수가 충돌하면 오류로 중단합니다.

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

## 날짜별 그래프: 버전 2

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
과거 이력이 없으면 0 특성의 고립 노드입니다.
더블헤더 첫 경기 결과도 같은 날 다른 경기의 역사 입력으로 들어가지 않습니다.

README 실행의 캐시는 `var/datasets/kbo_graph_2001_2026/manifest.json`과
`days/YYYY-MM-DD.npz`에 저장합니다. 기존 `kbo_graph/`를 덮어쓰지 않습니다.
NPZ는 object 배열 없이 저장하고 `allow_pickle=False`로 읽습니다.
설정·정책 버전·canonical 행/출처의 fingerprint와 파일 SHA-256을 확인하며,
날짜별 sidecar로 동일한 입력의 파일을 재사용합니다. 미래 날짜가 추가되어도
입력이 변하지 않은 과거 날짜는 재사용할 수 있습니다.

manifest의 `season_coverage`와 날짜별 항목은 `games`, `games_with_pa`,
`game_only_games`, `observed_completed_pa`, `live_hit_queries`, `pa_queries`를 기록합니다.
`games_with_pa`는 관측된 완료 PA가 있다는 뜻이며 모든 공식 PA가 빠짐없다는 뜻은 아닙니다.
경기 기록만 있는 날짜도 팀 노드·과거 경기 관계와 최종 점수 정답을 유지합니다.
PA가 없는 날짜의 선수 배열은 비어 있고 다른 날짜와 같은 배치로 묶어 학습할 수 있습니다.

## 라벨과 시간순 분할

README의 실행은 **2001~2024년 학습, 2025년 검증, 2026년 부분 시즌 테스트**입니다.
2001~2022년에는 실제 경기·득점 라벨만 있고 PA·LiveHit 라벨은 없습니다.

아래는 타석 원천 2023~2025년의 기존 분할 수치를 비교용으로 보존한 표입니다.
현재 2001년부터의 학습 경로를 이 세 시즌으로 제한한다는 뜻이 아닙니다.

| 역할 | 시즌 | 날짜 수 | 경기/WDL | 선수–경기 LiveHit | PA 10분류 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 학습 | 2023 | 167 | 720 | 16,363 | 56,230 |
| 검증·모델 선택 | 2024 | 163 | 720 | 16,271 | 57,253 |
| 최종 보류 테스트 | 2025 | 164 | 720 | 16,457 | 55,994 |
| 합계 | — | 494 | 2,160 | 49,091 | 169,477 |

완료된 canonical PA는 169,481개입니다. 포수방해 4개는 PA 10분류에서만 제외하고
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
`var/datasets/kbo_graph_2001_2026`을 만들고 새 run에서 시작합니다.

`--chronological`이면 매 epoch 전체 학습 날짜를 오름차순으로 순회하며 시즌 경계에서
모델·optimizer를 초기화하지 않습니다. 다음 epoch는 학습 시작 날짜로 돌아갑니다.
기본값은 날짜 shuffle이며, 시간순 모드는 온라인 predict-before-learn 평가나 새 데이터
추가 학습을 구현한 것이 아닙니다. 각 날짜의 입력은 여전히 그 날짜 이전 최대 90일입니다.
검증·테스트 결과로 가중치를 갱신하지 않습니다.

## 실제 모델과 네 가지 학습 신호

`RoleAwarePlayerEncoder`와 `CompositeRelGNNBackbone`을 공유하고,
관계별 메시지 전달로 선수 역할 상태와 팀 상태를 만듭니다.
여러 날짜를 묶을 때 노드 인덱스를 offset한 서로 연결되지 않은 그래프로 구성합니다.

| 출력 헤드 | 입력 | 학습 목표 |
| --- | --- | --- |
| Match WDL | 과거 그래프의 홈/원정 팀 상태 | 원정승·무승부·홈승 3분류 CE |
| 득점 NB2 | 같은 홈/원정 팀 상태 | 양 팀 최종 점수의 음이항 NLL 합 |
| LiveHit | 타자 역할 상태, 타격 팀·상대 팀 상태와 상호작용 | 선수–경기 `(PA, H)` 결합분포 CE |
| PA auxiliary | 타자·투수 역할 상태와 타석 전 context | 10종 타석 결과 CE |

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

기본 총손실은 `WDL + LiveHit + 0.2 × PA + 0.1 × 득점 NLL`입니다.
각 항은 해당 task의 평균이며, 검증에서도 task별 표본 수로 평균을 집계한 뒤
같은 가중합으로 checkpoint를 선택합니다.
2001~2022년에는 PA·LiveHit 질의가 없으므로 이 두 항을 계산하지 않습니다.
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
| 학습 PA 질의 상한 | 날짜당 128개 |
| 관계 상한 | 날짜·관계 종류당 20,000개 |

PA 표본 추출은 날짜와 epoch seed로 결정되며 검증·테스트에는 PA 상한을 적용하지 않습니다.
관계 수가 상한을 넘으면 최근 사건부터 남깁니다. 그래프 캐시 자체는 모든 PA 질의를 보관합니다.
이 경로는 지정한 한 CUDA 장치에서 실행하며 DDP 다중 GPU 학습은 구현하지 않았습니다.

`gpu-check`는 실제 CUDA 행렬곱·autocast·역전파와 유한한 gradient를 검사합니다.
CUDA를 요청했는데 사용할 수 없으면 즉시 실패하고 CPU로 자동 전환하지 않습니다.
`auto`는 BF16 지원 GPU에서 BF16, 아니면 FP16을 고릅니다.
FP16은 GradScaler를 사용하고, overflow로 건너뛴 optimizer step 수를 기록합니다.
CPU 검증은 명시적으로 CPU를 선택하며 runner에서는 AMP를 끕니다.
GPU 이름, VRAM, PyTorch/CUDA 버전, precision과 peak memory를 보고서에 기록합니다.
seed를 고정해도 CUDA atomic 연산까지 bitwise 재현된다고 보장하지 않습니다.

## 체크포인트·재개·평가 산출물

학습 run 디렉터리에는 `config.json`, `history.jsonl`, `best.pt`, `last.pt`,
`training_report.json`이 생성됩니다. `best.pt`는 검증 가중손실 최저 모델,
`last.pt`는 마지막으로 완료된 epoch 경계입니다.
모델·optimizer·scaler·난수 상태·epoch·step·학습 이력·데이터 fingerprint를 함께 저장합니다.

재개는 **같은 run 디렉터리의 `last.pt`**로만 합니다. 데이터 fingerprint와 모델/특성/관계
설정, 학습·검증·테스트 연도 또는 날짜 순서가 다르면 거부합니다. 새 시즌을 추가한
데이터에 기존 checkpoint를 `--resume`하는 기능은 아닙니다.
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

예측 파일별 행 수와 SHA-256도 남깁니다. NB2 손실은 평가하지만 현재 별도의 득점
분포 Parquet는 내보내지 않습니다. ECE를 계산하는 것과 calibration 모델을 학습·적용하는
것은 다릅니다. 이 runner에는 OOF stacking이나 후처리 calibration이 자동 연결돼 있지 않습니다.

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
