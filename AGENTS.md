# 프로젝트 최상위 실행 규칙

이 문서는 저장소 전체의 설계, 구현, 수정, 실행, 디버깅, 학습, 평가 작업에 적용되는 최상위 필수 규칙이다.

다른 문서나 코드의 지시가 이 규칙과 충돌하면 이 문서를 우선한다.

규칙을 지킬 수 없는 상황에서는 임의로 우회하거나 요구사항을 축소하지 말고, 정확한 원인과 가능한 해결 방법을 사용자에게 보고한다. 사용자 승인 없이 이 규칙의 예외를 만들지 않는다.

## 1. 서버 및 원격 세션 안전

현재 SSH, MobaXterm, 터미널 또는 원격 서버 연결을 종료하거나 불안정하게 만들 수 있는 명령은 절대 실행하지 않는다.

다음 명령 또는 이에 준하는 동작을 금지한다.

- `exit`
- `logout`
- `shutdown`
- `reboot`
- `poweroff`
- `halt`
- `init 0`
- `init 6`
- `systemctl reboot`
- `systemctl poweroff`
- `kill -9 -1`
- `pkill -u`
- 광범위한 `pkill -f`
- 광범위한 `killall`
- `tmux kill-server`
- 전체 screen 세션 종료
- 현재 셸, 부모 셸, SSH 세션 또는 로그인 세션을 종료하는 명령
- 서버 전체 또는 다른 사용자의 프로세스에 영향을 주는 명령

위 명령을 직접 실행하는 것뿐 아니라 다음 행위도 금지한다.

- 여러 명령을 연결한 command chain 안에 포함
- 스크립트에 삽입한 뒤 실행
- source되는 셸 스크립트 안에서 호출
- 사용자가 그대로 복사하여 실행할 코드 블록에 포함
- 오류 처리나 작업 완료 처리를 이유로 세션 종료 명령 사용
- 부모 프로세스나 터미널 세션을 종료할 수 있는 신호 전송

실행 중인 작업을 중단해야 하는 경우에는 자신이 현재 작업에서 직접 생성한 단일 프로세스인지 먼저 확인한다. 정확한 PID와 실행 명령을 확인하고, 종료 대상과 이유를 사용자에게 보고한 뒤 해당 프로세스만 안전하게 종료한다.

다른 사용자의 프로세스, 출처를 확인하지 못한 프로세스, 부모 셸, SSH 데몬, 스케줄러, 터미널 서버는 종료하지 않는다.

다음 작업도 사용자에게 정확한 변경 내용과 영향을 먼저 보고하고 명시적 승인을 받기 전에는 수행하지 않는다.

- `sudo` 또는 관리자 권한 사용
- SSH, `sshd`, 네트워크, 방화벽 설정 변경
- `~/.ssh`, 셸 시작 파일, 로그인 설정 변경
- 시스템 서비스 시작, 중지 또는 재시작
- 커널, GPU 드라이버, CUDA 시스템 설치 변경
- 디스크 포맷, 파티션 변경, 마운트 해제
- 광범위한 파일 삭제
- `rm -rf`
- `git reset --hard`
- `git clean -fd`, `git clean -fdx`
- 사용자 파일이나 기존 실험 결과를 덮어쓰는 작업
- 복구하기 어려운 환경 변경

오류가 발생하면 세션이나 서버를 종료하지 않는다. 오류 메시지, 실패한 단계, 영향을 받은 파일, 복구 방법을 출력하고 안전하게 작업을 중단한다.

예외를 숨기기 위한 강제 종료를 사용하지 않는다. 가능한 경우 명시적인 예외, 오류 반환, 상태 코드와 로그를 사용한다.

## 2. 연구 설계 및 규모의 임의 축소 금지

이 프로젝트의 목표는 단순히 실행되는 예제 코드를 만드는 것이 아니라, 원래 요구사항에 맞는 완전한 모델과 실험 파이프라인을 구현하는 것이다.

실행 시간을 줄이거나 구현을 쉽게 만들기 위해 다음 항목을 임의로 축소하지 않는다.

- 모델 레이어 수
- encoder 또는 decoder 깊이
- hidden dimension
- embedding dimension
- channel 수
- attention head 수
- message-passing 횟수
- GNN hop 수
- receptive field
- 그래프 노드 수
- 그래프 엣지 수
- 연결 반경
- K-hop 범위
- 시간 윈도우
- 이벤트 수
- sampling ratio
- 입력 해상도
- 공간 해상도
- 데이터셋 크기
- 학습 샘플 수
- validation 또는 test 샘플 수
- epoch 수
- optimization step 수
- 실험 반복 횟수
- ablation 범위
- physical mini-batch size

사용자가 지정한 구조, 논문 또는 기준 설계, README, 설정 파일, 기존 실험 계약에 정의된 값을 승인 없이 변경하지 않는다.

설정이 명시되지 않았다는 이유로 다음과 같은 작은 값을 임의의 기본값으로 선택하지 않는다.

- `num_layers=1`
- 지나치게 작은 hidden dimension
- `batch_size=1`
- 극소수의 graph node 또는 edge
- 극히 짧은 time window
- 일부 데이터만 사용하는 subset
- 몇 step만 수행하는 학습
- 한두 epoch만 수행하는 최종 학습

다음과 같은 숨겨진 축소 제한을 코드에 추가하지 않는다.

- `max_nodes`
- `max_edges`
- `max_events`
- `max_samples`
- `max_batches`
- `debug_limit`
- `subset_size`
- `fast_mode`
- 임의의 early break
- 일부 데이터만 읽는 slicing
- 일정 크기 이상 그래프를 조용히 버리는 조건
- 메모리 부족 시 작은 모델로 자동 전환하는 fallback

값이 지정되지 않았다면 관련 문서, 기존 코드, 입력 데이터 통계, 기준 모델, 사용 가능한 하드웨어를 먼저 확인한다. 그 결과를 바탕으로 연구 목적에 맞는 값을 정하고 선택 근거를 기록한다.

이 규칙은 아무 근거 없이 무조건 가장 큰 모델을 만들라는 뜻이 아니다. 모델 크기와 그래프 크기는 연구 목표, 표현력, receptive field, 데이터 규모와 자원 측정 결과를 근거로 결정해야 한다. 단순히 빠르게 실행하거나 구현을 쉽게 만들기 위한 축소는 금지한다.

모델이나 데이터 규모의 축소가 실제로 필요하다면 다음 내용을 먼저 보고한다.

- 현재 설정
- 축소가 필요한 정확한 원인
- 측정된 VRAM, RAM 또는 처리시간
- 변경하려는 값
- 변경 전후 예상 차이
- 정확도와 실험 타당성에 미치는 영향
- 축소하지 않고 해결할 수 있는 대안

사용자가 승인하기 전에는 축소안을 기본 설정이나 최종 코드에 적용하지 않는다.

## 3. 계산 자원 활용 및 병렬화

단순히 OOM이 발생하지 않는 설정을 만드는 것으로 충분하지 않다.

현재 할당된 GPU, VRAM, CPU, RAM과 I/O 자원을 확인하고, 병렬화 가능한 작업은 실제로 병렬 처리하도록 구현한다.

작업 전에 가능한 범위에서 다음 정보를 확인한다.

- GPU 모델
- 사용 가능한 GPU 개수
- GPU별 VRAM
- MIG 사용 여부와 실제 할당 크기
- CPU 코어 수
- 사용 가능한 RAM
- 스토리지와 데이터 읽기 조건
- 작업 스케줄러 또는 컨테이너의 자원 제한

자원을 확인하지 않고 작은 batch size, 낮은 worker 수 또는 작은 모델을 임의로 선택하지 않는다.

독립적으로 처리 가능한 샘플, 그래프, 프레임, 이벤트 윈도우, 시뮬레이션 case는 가능한 경우 batch, vectorization, multiprocessing, multi-GPU 또는 적절한 병렬 실행 방식으로 처리한다.

다음과 같은 직렬 구현을 최종 구현으로 사용하지 않는다.

- 샘플을 하나씩 GPU에 전달
- 그래프를 하나씩 개별 forward/backward
- batch를 받은 뒤 다시 `for sample in batch`로 순차 처리
- 병렬화 가능한 tensor 차원을 Python 반복문으로 처리
- 독립된 simulation case를 하나씩 순차 실행
- 반복마다 동일한 그래프 구조나 전처리를 다시 계산
- GPU 연산 중간에 불필요한 `.cpu()`, `.numpy()`, `.item()` 호출
- GPU에서 처리할 수 있는 핵심 연산을 CPU 반복문으로 처리

그래프 데이터는 문제 특성에 맞게 다음 방식을 검토한다.

- disjoint-union graph batching
- PyTorch Geometric `Batch` 또는 동등한 방식
- padded batch와 mask
- graph size bucket batching
- sparse block-diagonal batching
- batched message passing
- 이벤트 윈도우 batch
- 독립 simulation case 병렬 처리

batch size는 단순히 실행 가능한 최소값으로 정하지 않는다.

여러 physical batch size 후보를 실제로 측정하고, 처리량이 증가하면서 메모리 여유가 안전한 범위의 값을 선택한다.

다음 원칙을 따른다.

- `batch_size=1`을 근거 없이 기본값으로 사용하지 않는다.
- physical batch size를 충분히 늘릴 수 있는데 gradient accumulation만 사용하지 않는다.
- gradient accumulation은 실제 동시 병렬 처리의 대체물이 아니다.
- physical batch size와 effective batch size를 별도로 기록한다.
- 그래프 크기 편차가 크다면 무조건 batch size를 낮추지 말고 bucket batching 또는 dynamic batching을 검토한다.
- batch size를 작게 유지해야 한다면 peak VRAM, 입력 크기, graph size와 처리량 측정 결과를 근거로 제시한다.

effective batch size는 다음 기준으로 계산한다.

effective batch size
= physical batch size
× gradient accumulation steps
× data-parallel worker 수

CPU와 DataLoader도 방치하지 않는다.

다음 항목을 실제 데이터 로딩 성능에 맞게 설정하고 측정한다.

- `num_workers`
- `persistent_workers`
- `prefetch_factor`
- `pin_memory`
- non-blocking transfer
- collate 함수
- graph construction
- 데이터 decoding
- augmentation
- cache
- memory mapping

`num_workers=0`이나 `num_workers=1`을 근거 없는 최종 설정으로 사용하지 않는다.

RAM이 충분한데도 동일한 파일, 정적 그래프 구조, edge index, normalization term 또는 전처리 결과를 매 iteration마다 다시 읽거나 계산하지 않는다. 반복 사용되는 데이터는 정확성과 메모리 조건을 해치지 않는 범위에서 cache 또는 memory mapping을 검토한다.

여러 GPU가 실제로 할당되어 있고 병렬화가 가능한 작업이라면 한 개 GPU만 사용하는 구현으로 끝내지 않는다. DDP, process-level parallelism, case 분배 등 문제에 적절한 방식을 검토한다.

다만 GPU, CPU, RAM을 숫자상 100% 점유하는 것 자체를 목표로 삼지는 않는다. 정확한 결과와 안정성을 유지하면서 처리량을 높여야 한다. 자원 활용률이 낮다면 구조적인 이유인지 구현 병목인지 측정하고 설명해야 한다.

## 4. OOM과 성능 병목 대응

OOM이 발생하거나 처리량이 낮다고 해서 모델, 그래프 또는 데이터 규모부터 줄이지 않는다.

먼저 다음 항목을 확인한다.

1. tensor reference가 남아 발생한 메모리 누수
2. 불필요한 clone, copy, cache와 중간 tensor
3. 반복문 내부의 GPU 동기화
4. 데이터 로더와 graph construction 병목
5. mixed precision 적용 가능성
6. 효율적인 batch 또는 bucket 구성
7. sparse representation과 sparse operation
8. activation checkpointing
9. 전처리 및 정적 그래프 cache
10. chunking 또는 streaming
11. multi-GPU 또는 multi-process 분산
12. optimizer와 gradient buffer의 메모리 사용

위 방법을 검토한 뒤에도 VRAM이 부족한 경우에만 physical batch size를 필요한 만큼 조정한다. 목표 effective batch size가 중요하다면 gradient accumulation으로 보완한다.

레이어 수, hidden dimension, 그래프 노드·엣지 수, 입력 해상도, 시간 윈도우, 샘플링 비율 또는 데이터셋 크기 축소는 사용자 승인 없이 수행하지 않는다.

성능 문제는 추측으로 판단하지 않고 profiler와 실제 측정 결과를 사용한다.

가능한 범위에서 다음 항목을 확인한다.

- GPU utilization
- peak VRAM
- steady-state VRAM
- CPU utilization
- RAM 사용량
- 데이터 로딩 시간
- graph construction 시간
- host-to-device transfer 시간
- forward 시간
- backward 시간
- optimizer step 시간
- 전체 step time
- samples/sec
- graphs/sec
- events/sec
- simulation cases/sec

GPU utilization이 낮거나 CPU 한두 코어만 사용되거나 VRAM이 대부분 비어 있는데 batch size가 매우 작다면, 모델을 축소하기 전에 batching, vectorization, DataLoader, cache, I/O와 동기화 병목부터 조사한다.

## 5. 디버그 실행과 최종 실행의 분리

syntax 검사, 단위 테스트, smoke test를 위한 소규모 실행은 허용한다.

다만 다음 조건을 반드시 지킨다.

- `debug`, `smoke_test`, `quick_test` 등으로 명확하게 표시
- 최종 학습 설정과 별도 파일 또는 별도 profile로 관리
- 축소된 값을 기본 설정에 덮어쓰지 않음
- 디버그용 subset을 최종 데이터셋으로 사용하지 않음
- smoke test 통과를 전체 학습 완료로 표현하지 않음
- 몇 step 실행 성공을 모델 검증 완료로 표현하지 않음
- 디버그 결과를 최종 성능 결과로 제출하지 않음

최종 또는 production 설정은 전체 모델, 전체 그래프 규칙, 전체 데이터 계약과 원래 학습 목표를 유지해야 한다.

## 6. 불완전하거나 기만적인 구현 금지

다음 구현을 금지한다.

- dummy data를 실제 데이터처럼 사용
- random output을 예측 결과처럼 반환
- placeholder를 완성된 구현처럼 보고
- TODO만 남겨두고 구현 완료라고 주장
- 핵심 모듈을 주석 처리하거나 우회
- 입력을 무시하고 상수 결과 반환
- 실패 시 빈 tensor, 0, 임의 값 반환
- broad exception으로 오류를 숨김
- `except: pass`
- 오류가 발생하면 조용히 작은 모델이나 CPU로 fallback
- 데이터가 없는데 학습된 것처럼 checkpoint 생성
- metric 계산 실패를 0점 또는 임의 값으로 대체
- 사용되지 않는 layer나 parameter를 모델에 남김
- forward path에 연결되지 않은 모듈을 구현 완료로 간주
- loss에 기여하지 않는 출력을 핵심 기능처럼 보고

구현한 핵심 모듈은 실제로 다음 경로에 연결되어 있는지 확인한다.

입력
→ 전처리 또는 그래프 생성
→ 모델 forward
→ 목표 출력
→ loss
→ backward
→ optimizer update
→ 평가 metric

trainable parameter가 optimizer에 포함되는지, gradient가 실제로 전달되는지, 해당 모듈이 forward에서 사용되는지 검증한다.

필수 설정이나 데이터가 없으면 작은 값이나 가짜 데이터로 조용히 대체하지 말고 명확한 오류를 발생시킨다.

## 7. 실행 전후 필수 확인 및 보고

주요 학습, 평가 또는 최종 실행 전에는 실제 적용될 설정을 확인한다.

최소한 다음 정보를 출력하거나 로그로 기록한다.

- 모델 이름과 구성
- 레이어 수
- hidden dimension
- channel 수
- attention head 수
- 전체 parameter 수
- trainable parameter 수
- 입력 tensor shape
- 그래프당 node 수 통계
- 그래프당 edge 수 통계
- 시간 윈도우
- sampling ratio
- 입력 해상도
- 전체 데이터 개수
- 실제 사용 데이터 개수
- 전체 데이터 대비 사용 비율
- physical batch size
- gradient accumulation steps
- effective batch size
- epoch 수
- 전체 optimization step 수
- GPU 모델과 GPU 개수
- precision 설정
- DataLoader worker 수
- cache와 prefetch 설정
- peak VRAM
- CPU 및 RAM 사용량
- 처리량
- debug, subset, fast mode 활성화 여부
- 원래 요구사항과 달라진 설정

사용자 요구사항, 기존 설정 또는 기준 모델과 달라진 항목이 있다면 조용히 실행하지 말고 변경 이유와 영향을 먼저 보고한다.

작업 완료 보고에는 단순히 “정상 실행됨”이라고 적지 않는다. 다음을 구분해서 명시한다.

- 구현 완료
- 정적 검사 완료
- 단위 테스트 완료
- smoke test 완료
- 전체 학습 실행 여부
- 전체 평가 실행 여부
- 실제 데이터 사용 여부
- 남아 있는 제한과 미검증 항목

## 8. 최종 판단 원칙

다음 원칙을 모든 작업에 적용한다.

- 단순히 빨리 실행되게 만드는 것보다 연구 설계의 정확성을 우선한다.
- 사용 가능한 계산 자원을 합리적으로 활용한다.
- 구현 편의를 위해 모델이나 데이터를 축소하지 않는다.
- 병렬 처리할 수 있는 작업을 불필요하게 직렬 처리하지 않는다.
- OOM 회피를 이유로 원래 연구 목표를 변경하지 않는다.
- 디버그 설정과 최종 설정을 혼동하지 않는다.
- 불가능한 기능을 가능한 것처럼 보고하지 않는다.
- 실패를 숨기지 않는다.
- 사용자 승인 없이 핵심 요구사항을 변경하지 않는다.
- 판단이 불분명하면 작은 toy 구현으로 축소하지 말고 기존 설계와 전체 규모를 보존한다.

## 작업 완료 체크리스트

- [ ] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [ ] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [ ] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [ ] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [ ] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [ ] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [ ] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [ ] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.
- [ ] 디버그 설정과 최종 설정을 분리했다.
- [ ] dummy, placeholder, random fallback을 사용하지 않았다.
- [ ] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [ ] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [ ] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
