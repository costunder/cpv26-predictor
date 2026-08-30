# 공개 KBO 실데이터 기준선 실험

실행일: 2026-08-30 KST. 이 문서는 실제 다운로드·적재·학습·평가 결과를 기록한다.
실행 순서는 [README](../README.md)에 있으며, 이 실험을 RelGNN이나 완성된 V26 추천기로
해석하지 않는다.

## 1. 사용한 데이터와 출처

- 데이터셋: [slothman3878/kbo_playbyplay](https://huggingface.co/datasets/slothman3878/kbo_playbyplay)
- 생성 코드: [kbo_pbp_naver_sports](https://github.com/slothman3878/kbo_pbp_naver_sports)
- 성격: NAVER 스포츠 중계를 구조화한 비공식 파생 데이터, 투구당 한 행
- 고정 revision: `6afc8af044e3bba5f326b688e8cb41d7ff7065ec`
- 배포자가 표기한 라이선스: CC BY 4.0
- 원자료의 별도 권리·이용 조건까지 이 표기로 해결됐다고 가정하지 않는다.
- 이 저장소는 공개 산출물을 다운로드하며 원 사이트 크롤러를 실행하지 않는다.
- 원본 Parquet와 학습 모델은 GitHub에 포함하지 않는다. 출처·해시는 `SOURCE.json`과
  import report에 함께 보존한다.

| 시즌 | 투구 행 | 경기 | 완료 라벨 PA | 무라벨 PA | 범위 |
|---|---:|---:|---:|---:|---|
| 2023 | 219,934 | 720 | 56,231 | 170 | 정규시즌 |
| 2024 | 223,277 | 720 | 57,255 | 146 | 정규시즌 |
| 2025 | 217,970 | 720 | 55,995 | 144 | 정규시즌 |
| 2026 | 144,779 | 470 | 37,102 | 90 | 2026-07-26까지 |
| 합계 | 805,960 | 2,630 | 206,583 | 550 | 포스트시즌 제외 |

기본 학습·평가는 완결된 2023~2025의 2,160경기·169,481 PA만 사용한다.
2026은 다운로드·적재 smoke에만 포함했고 모델 평가에는 사용하지 않았다.

### 원본 SHA-256

| 파일 | SHA-256 |
|---|---|
| kbo_pbp_2023.parquet | `818f6016655b02fe48b8118281d1b04bfe3548d376fdc70131a41ea539341edb` |
| kbo_pbp_2024.parquet | `8332cd716cf0126a4ab0bf390383f43deff22ab320a57fb70d02b31025bdf553` |
| kbo_pbp_2025.parquet | `2c824919495809722a5ff0290a823ff9a44d88f61640ad9b288ff3dca2652f2c` |
| kbo_pbp_2026.parquet | `9d330311d28371806028b878191fcc85b9170839c8951b00ff9c64ec8aa28630` |

## 2. 실제 연결한 경로

```text
kbo-fetch
  → 고정 Parquet + SHA-256 + SOURCE.json
kbo-import
  → source_revision / team / player / game / team_game / observed_plate_appearance
  → 무결성 검사 + 품질 보고서
  ├─ kbo-match-evaluate: 경기 단위 L/D/W CatBoost
  └─ kbo-live-hit-evaluate: 선수-경기 단위 any-hit CatBoost
      → fold별 평가 JSON + .cbm 모델
```

4시즌을 적재한 최종 DB에는 source revision 4개, team version 40개, player version
2,280개, game 2,630개, team_game 5,260개, 완료 PA 206,583개가 있다. 선수 2,280명이라는
뜻이 아니라 같은 선수의 시즌별 version도 포함한 물리 행 수다.

동일 파일·adapter version의 재실행은 primary-key 기준으로 중복을 만들지 않는다.
공급자 ID에는 `kbo-game:`, `kbo-team:`, `kbo-player:` namespace를 붙인다. 다른 공급자와의
crosswalk까지 해결한 것은 아니다.

## 3. 정규화 정책과 원본 품질 검사

### 시각과 정보 가용성

원본에는 정확한 과거 발표 시각과 경기 시작 시각이 없다.

- scheduled_start: 경기일 00:00 KST로 명시적 대체
- event_at: 경기일 23:59:59 KST
- available_at: 다음 달력 날짜 00:00 KST로 재구성
- ingested_at: 이 환경이 파일을 실제로 적재한 시각
- 같은 날 경기는 입력 생성 시 동시 사건으로 취급

이는 이전 날짜 결과만 사용하는 retrospective benchmark다. 실제 당시 발표 revision이
보존된 strict point-in-time replay라고 주장하지 않는다. 원본의 사후 기록 정정 가능성도
별도 관리 대상이다.

### 타석과 선수 귀속

`game_pk + at_bat_number`로 묶고 첫 투구의 전 상태와 마지막 투구의 후 상태를 사용한다.
NULL 주자 상태를 건너뛰는 arg_min/arg_max 대신 NULL을 보존하는 first/last를 쓴다.

- 일반 결과·안타: 마지막 타자에게 귀속
- 2스트라이크 이후 교체 삼진: 두 번째 스트라이크를 부담한 타자에게 귀속
- pitcher_id: 마지막으로 관측된 투수이며 공식 실점 책임 투수를 뜻하지 않음
- 3아웃 후 잔루: canonical 후 상태를 `000`으로 정규화
- triple_play: `ball_in_play_out`으로 묶되 실제 `outs_added=3`은 보존

교체 삼진 정책은 [KBO 공식 야구규칙 9.15(b)](https://6ptotvmi5753.edge.naverncp.com/KBO_FILE/ebook/pdf/2025_%EC%95%BC%EA%B5%AC%EA%B7%9C%EC%B9%99.pdf)의
기록 원칙을 따른다. 원본의 타자 교체 51타석 중 8타석이 안타였으므로, 첫 타자로
묶는 단순 집계는 실제 선수 안타 라벨을 바꾼다. 회귀 확인 예시는
`20230502KTSK02023`, 김민혁(`64004`)의 4안타다.

### 확인된 결함과 처리

| 검사 | 원본 건수 | 처리 |
|---|---:|---|
| 종료 결과가 NULL인 PA | 550 | 학습 PA 라벨에서 제외, 원본 보존 |
| 위 무라벨 행의 득점 | 13점 | 보고서의 unlabelled_runs로 별도 표시 |
| 원본 PA 번호 내부 공백 | 53개 | source_sequence_gaps로 표시 |
| 3아웃인데 잔루가 남은 후 상태 | 27,426 | canonical 주자 상태를 000으로 변경 |
| PA 득점과 first-pitch→post 점수차 불일치 | 13행 | transition_complete=false |
| 모든 PA 득점 합과 최종 점수 불일치 | 5경기, 순부족 5점 | 별도 보고, 임의 주루 사건 생성 금지 |

모든 무라벨 행이 중단 타석은 아니다. `20240504OBLG02024`에는 완료된 아웃 결과의
누락도 있다. 해당 경기의 김현수·홍창기 안타는 정상 라벨에 남아 있지만 PA/AB 분모는
완전하지 않다. [KBO 경기 기사](https://www.koreabaseball.com/MediaNews/News/BreakingNews/View.aspx?bdSe=56593)

원본에서 경기별 max(post_score)와 마지막 post_score는 2,630경기 모두 같았다.
그러나 이를 모든 공식 박스스코어와 외부 대조했다는 의미로 사용하지 않는다.

팀 실책을 reached-on-error 횟수로, 선수 득점·타점을 0으로, 타격 방향을 시즌 전체
관측으로 추정해 채우지 않는다. 확인되지 않은 값은 NULL 또는 미적재로 남긴다.
따라서 player_game_batting, pitching_appearance, runner_event, lineup, weather, V26
계정 테이블은 이 importer가 허구의 값으로 채우지 않는다.

개별 PA에 후 상태가 있어도 누락된 주루 사건과 번호 공백 때문에 전체 순차 재생은
완전하지 않다. import report의 `simulator_ready`는 명시적으로 false다.

## 4. 학습과 분할

| 평가 | 학습 시즌 | 평가 시즌 | 경기 학습/평가 | 선수-경기 학습/평가 |
|---|---|---|---:|---:|
| validation_2024 | 2023 | 2024 | 720 / 720 | 16,363 / 16,271 |
| test_2025 | 2023~2024 | 2025 | 1,440 / 720 | 32,634 / 16,457 |

두 fold의 모델은 독립적으로 새로 학습한다. 평가 시즌에서도 전날까지 완료된 경기
기록은 다음 날 feature에 반영되지만 모델 파라미터는 재학습하지 않는다. 시즌 시작 때
한 번만 예측하는 실험과는 다르다.

공통 CatBoost 설정:

```text
CatBoost 1.2.10
iterations=400, depth=7, learning_rate=0.05
random_seed=2026, thread_count=1
allow_writing_files=false, verbose=false
match loss=MultiClass, live-hit loss=Logloss
```

- Match: 26개 팀 누적/최근 성적·득실점·Elo feature, 홈 기준 0=L/1=D/2=W
- Live Hit: 25개 개인/최근/팀 공격/상대 수비/선수-상대팀 관계 feature, 0=무안타/1=안타
- 현재 경기 최종점수·타석 수·안타 수는 feature에 넣지 않음
- prior 기준선: 학습 데이터의 class count에 Laplace +1 smoothing
- calibration metric은 계산하지만 확률 calibration 모델은 아직 fit하지 않음
- 테스트 성능을 보고 hyperparameter를 바꾸거나 좋은 결과만 선택하지 않음

## 5. 실제 결과

| Task / 평가 | 모델 log loss | prior log loss | 모델 정확도 | prior 정확도 |
|---|---:|---:|---:|---:|
| Match / 2024 검증 | 0.888346 | 0.757743 | 50.00% | 50.42% |
| Match / 2025 테스트 | 0.869101 | 0.813822 | 46.53% | 49.72% |
| Live Hit / 2024 검증 | 0.734888 | 0.684762 | 52.60% | 57.04% |
| Live Hit / 2025 테스트 | 0.679467 | 0.690944 | 59.80% | 53.80% |

2025 테스트의 추가 확률 지표:

| Task | Brier | ECE |
|---|---:|---:|
| Match | 0.567129 | 0.127499 |
| Live Hit | 0.482902 | 0.050742 |

프로젝트의 Brier는 모든 class의 제곱오차 합을 평균한다. 이진 task에서도 2개 class를
더하므로 단일 양성확률만 쓰는 외부 binary Brier 수치와 바로 비교하지 않는다.

판정:

- Match 기준선은 두 시즌 모두 단순 prior보다 나쁘다. 실전 추천 성능을 확보하지 못했다.
- Live Hit는 2025에서 개선됐지만 2024에서는 prior보다 나빴다. 안정적인 일반화나
  실전 V26 효용이 입증된 것은 아니다.
- 이번 결과는 다운로드→정규화→서로 다른 target 학습→시간 분할 평가가 실제로
  실행됐다는 증거이며 높은 정확도나 RelGNN 우월성의 증거가 아니다.

## 6. 산출물과 재현

```bash
bash scripts/setup.sh tabular
source scripts/activate.sh
cpv26 kbo-fetch
cpv26 kbo-import
cpv26 db-check
cpv26 kbo-match-evaluate --iterations 400
cpv26 kbo-live-hit-evaluate --iterations 400
```

생성 위치:

```text
var/datasets/kbo_playbyplay/v0/SOURCE.json
var/reports/kbo_import.json
var/reports/kbo_match_baseline.json
var/reports/kbo_live_hit_baseline.json
var/models/kbo_match_baseline/<run-id>/{validation_2024,test_2025}.cbm
var/models/kbo_live_hit_baseline/<run-id>/{validation_2024,test_2025}.cbm
```

실행마다 모델 폴더를 분리하고 그 안에 `evaluation.json`을 함께 보존한다. 기본
`var/reports/` JSON은 최신 실행을 가리키며, JSON의 `model_directory`, `model_path`,
`model_sha256`으로 실행·모델을 연결한다. `--report` 경로를 바꿔도 이전 모델을 덮어쓰지
않는다. 저장한 네 모델을 재로드한 평가가 원 보고서와 일치함도 확인했다.

모델은 해당 fold의 학습 종료 시점까지만 배웠다. test_2025 모델을 2026 최신 학습 모델로
해석하지 않는다. 실제 V26 후보의 출전확률, 포지션·선택률·도감·점수 규칙은 이 데이터에
없고 최종 optimizer 추천도 이번 실험의 평가 대상이 아니다.
