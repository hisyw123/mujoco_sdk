# 현재 실행 구조: 기존 C++ MuJoCo 3.13.1

물리 엔진은 기존 `../build-greeny/lib/libmujoco.so.3.13.1`을 링크한다.
새 MuJoCo/ONNX/libtorch를 설치하지 않았다. 기존 greeny_homing 실행파일과 원본 모델은 수정하지 않는다.
이식 실행파일은 별도 `build/augmented_sim`이며 greeny_homing 프로세스나 shared-memory 제어기에
합쳐진 것은 아니다.

| 역할 | 현재 구현 |
|---|---|
| 합성 수집, PD, 적분, CSV 기록 | C++ augmented_sim collect |
| replay 라벨 계산, rollout | C++ augmented_sim |
| GRU 추론 | C++ Eigen, PyTorch r/z/n 게이트와 같은 식 |
| 뷰어 | C++ GLFW + MuJoCo renderer, 회전/줌 카메라 |
| GRU 지도학습 | 기존 Python/PyTorch train.py |
| 실행 옵션, 메타데이터, 통계/그림 | Python wrapper/evaluate.py |

현재 C++ GUI는 마우스 왼쪽 드래그 회전/휠 줌을 지원하는 관찰용 뷰어다.
Python passive viewer가 아니며 MuJoCo simulate 전체 설정 패널이나 기존 GREENY 제어 패널은 아니다.
GUI는 힘/상태를 변경하지 않는다. 창 닫기는 headless 계속 실행, 작업 완료 시 자동 닫기.

## 빌드

```bash
cd /home/dongju/mujoco/raionActuatorNet
source .venv/bin/activate
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ldd build/augmented_sim
```

이미 설치된 Eigen3/yaml-cpp/GLFW/OpenGL 개발 패키지를 사용한다.
Python 3.8.1 백엔드는 레거시 비교용으로 보존했지만 기본 명령은 이를 호출하지 않는다.

## 모델 선택

```bash
python run.py models
python run.py inspect --robot rough-bipedal
python run.py inspect --robot rough-quadrupedal
```

- `test`: 기존 2관절 검증 모델.
- `rough-bipedal`: model/RoughTerrain-3/RoughTerrain-3_bipedal.xml의 스탠드용 복사본.
- `rough-quadrupedal`: 같은 폴더 quadrupedal XML의 스탠드용 복사본.
- `--xml /absolute/custom.xml`: 사용자가 준비한 독립 고정베이스 모델 직접 지정.

RoughTerrain 원본은 자유베이스·접촉·연동 그리퍼를 포함한다. 현재 라벨 계산 전제를 유지하기 위해
선택 시 `models/<robot>/stand27_<sourcehash>.xml`을 생성한다.

1. root freejoint 제거, root body의 원래 world pose 유지.
2. 양쪽 그리퍼 slide joint를 0 위치에서 고정. 링크/질량/관성은 남긴다.
3. gripper equality, native actuator, sensor를 복사본에서 제거.
4. nominal keyframe의 base/gripper 좌표만 제거하여 27 hinge 좌표를 유지.
5. 원본 inertial/armature/damping/frictionloss/관절 제한은 보존한다.
   원본에 이미 있는 balanceinertia=true도 유지한다(새 보정 정책을 추가하지 않음).
6. runtime에서 접촉 비활성화, Euler + 외부 sampled PD 사용.

이것은 **자유롭게 서거나 걷는 원본 장면이 아닌, 스탠드 식별용 변형**이다.
각 XML 옆 JSON에 원본 경로/해시/변경 목록을 기록한다.
mesh/texture 경로는 기존 모델 폴더를 절대경로로 참조하므로 원본 자산을 이동하지 말 것.

## 수집 → 학습 → 평가

아래 gain/진폭은 소프트웨어 검사용이며 하드웨어 승인 설정이 아니다.
관절별 값은 `--joint-config /path/joints.json`으로 설정한다. 세 단계에 같은 설정을 사용한다.
기존 출력이 있으면 run 이름을 바꾼다.

```bash
# 60초 분량 데이터 수집. GUI 실행 속도에 따라 실제 소요 시간은 더 길 수도 있다.
python run.py collect --robot rough-bipedal --run runs/bipedal_01 \
  --seconds 60 --dt .001 --kp 100 --kd 5 --amplitude .02 --viewer

# 기존 GRU 방식으로 학습. 라벨은 C++ 3.13.1에서 새로 계산.
python run.py train --robot rough-bipedal --run runs/bipedal_01 \
  --kp 100 --kd 5 --epochs 30

# C++ GRU로 추론/물리 실행. Python은 지표와 그림 생성.
python run.py evaluate --robot rough-bipedal --run runs/bipedal_01 \
  --kp 100 --kd 5 --viewer
```

quadrupedal은 `--robot rough-quadrupedal`, 다른 run 이름으로 바꾸면 된다.
더 빠르게 처리하려면 `--viewer`를 빼고, GUI 없이 실시간 대기를 원하면 수집에 `--realtime`을 추가한다.
학습에는 GUI가 없다. 실제 로봇 로그는 기존 scripts/train.py의 CSV 입력 방식으로 사용할 수 있다.

## 왜 빨리 끝났는가

- 이전 사용자 gui_demo: 실제 로그 범위 **20초**, 물리 dt 1ms, **20,001행**.
- stride 2이므로 학습 라벨 **10,000개**, GRU hidden64/1layer, 실제 학습 **30 epochs**.
- 학습은 시간을 재생하는 것이 아니라 GPU/CPU에서 배치 계산한다. 20초×30만큼 기다리지 않는다.
- headless 수집/추론도 벽시계 대기를 하지 않는다. 샘플을 생략해서 빠른 것이 아니다.
- duration은 `--seconds`, 물리 간격은 `--dt`, 화면 속도는 `--viewer-speed`로 따로 정한다.
- 소스 모델 초기 로딩, mesh 컴파일, 그림 생성은 시뮬레이션 시간과 별개의 비용이다.

이번 측정:

| 작업 | 데이터 시간 | 실제 시간 | 비고 |
|---|---:|---:|---|
| 2관절 headless 수집 | 20s | 약 0.082s | 물리+CSV 루프만 |
| bipedal 27관절 headless 수집 | 3s | 약 0.154s | 모델 로딩 제외 |
| quadrupedal 27관절 headless 수집 | 3s | 약 0.161s | 모델 로딩 제외 |
| bipedal native GUI 수집 | 3s | 약 9.88s | 고해상도 메시 렌더링 포함, 실시간보다 느림 |

GUI는 목표 속도에 맞춰 대기하지만 렌더링/계산이 느리면 실시간 속도를 보장하지 않는다.
C++은 simulated_seconds/wall_seconds/rows를 출력한다.
collect.py는 모델 검사/로딩 포함 command_wall_seconds도 meta에 저장한다.
train.py는 완료 epochs/windows/prediction_samples/wall_seconds를 config.json에 저장하며,
evaluate.py도 실제 실행 시간을 출력한다.

## 수집 방식의 명시적 변경

Python → C++ RNG가 달라 같은 seed의 위상/노이즈 수열이 이전과 정확히 같지는 않다.
phase는 수집 메타에 남는다. 첫 keyframe이 있으면 그 자세를 사용하고, 목표가 관절 제한을 넘지 않도록
center를 내부로 이동하고 amplitude를 줄일 수 있다. 실제 사용한 center/amplitude도 meta에 남는다.
노이즈는 관측값에만 추가하며 true feedback/true state는 덮어쓰지 않는다.
run.py의 검증 기본값은 torque scale=.9, extra damping=.1로 합성 불일치를 만든다.
기존 actuator의 ctrl/force 제한은 외부 qfrc_applied에 적용되지 않는다. 토크 포화는 아직 미구현이다.

## 검증 결과 및 한계

- 기존 13개 + 직접 native 6개 = 19개 테스트 통과.
- native 다층 GRU/세 feature 조합과 PyTorch parity 검사 통과.
- 2관절 전체 학습/평가: C++ vs PyTorch FF 최대 차이 약 1.4e-9 Nm.
- bipedal 27관절: 3초 수집 → 3 epoch 학습 → C++ 추론/평가/그림까지 성공.
  FF parity 약 5.96e-8 Nm. 같은 로그에서 위치 RMSE 감소 약76.3%, 속도 약19.0%.
  짧은 합성 연결 검증일 뿐 일반화/실기 성능 입증은 아니다.
- bipedal GUI/headless 수집 CSV가 바이트 단위로 일치. 원본 XML 해시 불변 확인.
- quadrupedal은 native 로딩/3초 수집까지 검증. 별도 학습 성능 평가는 아직 하지 않았다.
- 새 엔진 계약과 이전 3.8.1 모델 계약이 다르므로 기본 evaluate는 이전 모델을 거부한다.
  기존 가중치는 삭제/덮어쓰지 않았다. native 모델에는 actuator_net_cpp.json이 추가된다.
- 실기 데이터 수집기, 보행/접촉/그리퍼 학습, C++ GREENY shared-memory 연동은 미구현이다.

```bash
python -m unittest discover -s tests -v
python scripts/audit_changes.py
```
