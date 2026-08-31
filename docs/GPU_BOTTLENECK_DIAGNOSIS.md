# MIG 학습 병목 진단

DCGM에서 GPU 6 / GI 7 / CI 0 (`c:42`)의 `GRACT`를 10회 관측한 값은
0.194, 0.148, 0.154, 0.146, 0.146, 0.153, 0.149, 0.152, 0.150, 0.151입니다.
평균은 15.43%입니다. 이 관측만으로 데이터 로딩, GPU 전송, CPU 동기화 중
어느 부분이 주원인인지 구분할 수는 없습니다. 검증/체크포인트 저장 중 찍었다면
지속적인 학습 구간의 사용률과도 다릅니다.

## 실행

**현재 학습과 이어지는 평가까지 끝난 뒤, 같은 MIG가 비어 있을 때만 실행합니다.**
동시에 실행하면 두 프로세스가 GPU를 나눠 써서 측정이 틀어집니다.
스크립트는 다른 프로세스를 중단하지 않습니다. `--device-idle`은 사용자의 확인이며,
서버 전체 프로세스를 검사하거나 유휴 상태를 자동 보장하는 옵션은 아닙니다.

프로젝트의 기존 Conda 환경에서, 학습할 때와 같은 할당 GPU를 사용합니다.
`CUDA_VISIBLE_DEVICES`를 해제하거나 다른 GPU로 바꾸지 않습니다.

```bash
conda activate cpv26
cd /home/aicompetition07/projects/cpv26-predictor
source scripts/activate.sh
python scripts/profile_relgnn.py \
  --run-dir var/runs/relgnn/kbo_2001_2024_v5 \
  --device cuda:0 \
  --device-idle
```

`--run-dir`은 실제 학습 run 디렉터리입니다. `config.json`과 `last.pt`를 읽고
그 run의 모델/optimizer 설정을 복제합니다. 재개 후 설정은 `last.pt`를 따릅니다.
데이터 경로가 이동했다면 `--dataset`으로 지정하되 fingerprint가 같아야 합니다.
기본 배치 크기와 worker 수도 체크포인트 설정을 따릅니다.

결과는 새로운 `var/reports/relgnn_profile_<시각>_<식별자>/report.json`에 저장합니다.
기존 출력 디렉터리, run 안, 데이터셋 안에는 쓰지 않습니다.
학습 코드, 데이터, 원래 모델/optimizer/checkpoint 파일은 변경하지 않습니다.

## 무엇을 비교하는가

훈련 연도 안에서 PA가 없는 날짜와 PA가 있는 날짜를 구분하고, 각 그룹의 최신 연도
중간 구간을 선택합니다. 각 구간에서 3개 워밍업 배치를 제외한 12개 배치를 측정합니다.
세 실험 모두 같은 데이터와 동일한 체크포인트 복사본에서 시작합니다.
검증·테스트 날짜는 선택하지 않습니다. 누락된 그룹을 가짜 데이터로 채우지 않습니다.

| 실험 | 내용 | 분리하는 비용 |
| --- | --- | --- |
| `stream` | 실제 loader → GPU 전송 → 원래 손실/역전파/통계/optimizer | 기준 |
| `resident` | 같은 배치를 GPU에 미리 올리고 원래 연산 실행 | 로딩과 전송을 함께 제거 |
| `resident_no_statistics` | resident에서 표본 수·loss 통계용 CPU 읽기만 생략 | 통계용 동기화 영향 |

유한성 검사, gradient clipping, optimizer update는 세 실험 모두 유지합니다.
통계 생략은 이 진단 안에서만 적용되며 실제 학습 코드를 고치는 옵션이 아닙니다.
현재 진단은 `accumulate_steps=1`만 지원하고 다른 설정을 조용히 바꾸지 않습니다.

`comparisons`의 `removing_loader_and_transfers_speedup`이 반복 측정에서 크게 1을 넘으면
입력 경로의 영향이 있다는 증거입니다. `removing_statistics_host_reads_speedup`은
통계용 CPU 읽기를 제거한 효과입니다. 작은 차이는 측정 잡음일 수 있습니다.
두 원인을 항상 독립적인 시간 비율로 더할 수 있는 것은 아닙니다.

`host_stage_mean_ms`는 CPU가 각 구간에서 보낸 시간입니다. 예를 들어 backward는
GPU 작업을 예약하고 반환하므로 그 완료 대기 시간이 다음 scalar 읽기에 잡힐 수
있습니다. 이 값을 GPU 커널 연산 시간으로 해석하지 않습니다.
`iterator_startup_seconds_excluded`에는 iterator 생성/worker 시작을,
`first_batch_wait_seconds_excluded`에는 첫 입력 대기를 별도로 남깁니다.
전체 epoch를 재현하지 않으며 filesystem cache의 warm/cold 상태는 통제하지 않습니다.

## CUDA trace

처리량 실험과 별도로 각 구간의 resident 모델을 3개 배치만 프로파일링합니다.
`box_only_trace.json` / `with_pa_trace.json`에는 CPU 연산과 CUDA 커널 시간이 들어갑니다.
`report.json`의 `top_cpu_ops`에서 `_local_scalar_dense`, 복사 관련 연산 등의 호출 수와
CPU 시간을 볼 수 있습니다. CPU 시간은 중첩 연산을 포함하므로 합산하면 안 됩니다.

`cuda_active_fraction`은 측정 step 안의 CUDA kernel/copy/memset 시간 구간을 합집합으로
계산한 값입니다. **DCGM GRACT나 SM 점유율과 같은 지표가 아닙니다.**
프로파일러와 step 경계 동기화에 의한 오버헤드도 포함됩니다.
CUDA 이벤트가 수집되지 않으면 0%가 아니라 `null`로 기록합니다.
서버에서 profiler/CUPTI 사용이 제한되어도 처리량 비교 결과는 따로 남습니다.

CPU 기능 테스트는 `--device cpu --workers 0 --steps 1 --warmup 1 --trace-steps 0`으로
가능하지만 GPU 병목의 증거로 사용할 수 없습니다. 로컬 CPU 테스트 성공을
A100/MIG 성능 검증 성공으로 보고하지 않습니다.
