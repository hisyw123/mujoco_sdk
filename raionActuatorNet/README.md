# raionActuatorNet — 원본 학습 방식을 유지하는 MuJoCo 이식

> **현재 기본 실행은 C++ MuJoCo 3.13.1입니다.**
> 최신 실행법과 모델 선택은 [NATIVE_CPP.md](NATIVE_CPP.md)를 사용하십시오.
> 이 문서 아래의 Python 3.8.1 백엔드/뷰어 설명과 과거 검증 결과는 이전 단계의 기록입니다.
> collect.py는 이제 C++ 실행 래퍼이고, train/evaluate도 C++ replay/rollout을 호출합니다.
> 기존 3.8.1 학습 모델은 엔진 계약이 달라 재학습해야 합니다. 기존 결과는 보존했습니다.

실제 구현 위치: `/home/dongju/mujoco/raionActuatorNet` (대소문자 주의).
원본 기준: `/home/dongju/Downloads/actuator-net-gym-articulated-main`.
`reference/raisim/`은 원본 코드 보존본이며 실행 대상이 아니다.
변경 내역은 **[patch note.txt](patch%20note.txt)**, 파일별 diff/SHA256은 `audit/`에 기록한다.

## 유지한 것과 바꾼 것

- 원본 CSV: elapsedTime, gc_j, gv_j, ptarget_j, 선택적 gf_j.
- 원본 Hann smoothing, 중복 타임스탬프 첫 행 유지, 실제 로그 dt, stride/offset/steps/fixed-dt.
- 원본 전체 관절 GRU, 입력 선택 pos_err/vel/torque, 정규화, 0.33 s 창, 0.06 s warmup,
  매 4번째 0.3 s 검증 블록, Adam, cosine schedule, gradient clipping, early stop, TorchScript.
- 물리 백엔드만 MuJoCo Python API로 재구현. libtorch C++ 빌드나 RaiSim 라이선스는 필요 없다.
- Raisim 내장 implicit PD와 **같은 수치 알고리즘이 아니다**. 매 물리 스텝에
  `kp*(target-q)-kd*v+ff`를 qfrc_applied로 넣고 Euler 설정의 mj_step을 수행한다.
  다른 엔진에서 생성한 라벨/가중치는 재사용 성능을 보장하지 않는다.
- 원본에는 실기 수집 코드가 없다. 추가한 collect.py는 CSV 계약을 따르는 **합성 수집기**다.
  실제 하드웨어 통신·제어 루프와 동일한 수집기를 이식했다고 주장하지 않는다.
- 기존 greeny_actuator_net의 별도 train/val 파일 분할 방식은 이번 이식의 기본값으로 사용하지 않는다.

## GUI로 움직임 보기

수집, 직접 rollout, evaluate에 `--viewer`를 추가하면 MuJoCo 창이 열린다.
기본값은 headless이며 `--viewer-speed 1`은 실시간, `--viewer-speed 0.5`는 느린 재생이다.
물리 dt와 학습 데이터는 화면 속도와 무관하다. 창은 종료 시 자동으로 닫힌다.
중간에 창을 닫으면 작업은 headless로 끝까지 진행하며, 중단은 터미널 Ctrl+C를 사용한다.

```bash
cd /home/dongju/mujoco/raionActuatorNet
source .venv/bin/activate
python scripts/collect.py --urdf tests/pendulum.xml --out runs/gui_demo/log.csv \
  --seconds 20 --kp 10 --kd .5 --torque-scale .9 --extra-damping .1 --viewer
```

이 화면은 실제 Raion 로봇이 아니라 2관절 테스트 모델이다. 기존 평가 명령에도
`--viewer`를 붙이면 baseline → augmented → oracle → (지정 시) fixed-dt 롤아웃을
순서대로 보여준다. 학습 및 라벨 생성 replay에는 뷰어를 켜지 않는다.

뷰어에는 별도 복제한 모델/상태를 보내므로 마우스 조작/GUI 설정이 학습용 물리를 변경하지 않는다.
실제 조작 제어 UI가 아니라 관찰용 화면이다. 데스크톱 DISPLAY와 OpenGL이 필요하며
SSH/디스플레이 없는 환경에서는 headless를 사용한다.

## 실행 환경

이 PC에서는 전용 `.venv`를 만들었으며 기존 mj conda 환경의 MuJoCo/PyTorch를 사용한다.
pandas만 이 가상환경 안에 추가했다. 다른 PC는 requirements.txt를 별도 환경에 설치한다.

```bash
cd /home/dongju/mujoco/raionActuatorNet
source .venv/bin/activate
python -m unittest discover -s tests -v
```

## 먼저 실행해 볼 전체 경로 — 실제 로봇이 아닌 2관절 테스트

출력 디렉터리가 이미 있으면 새 run 이름을 사용한다. 기존 결과를 덮어쓰지 않는다.

```bash
python scripts/collect.py --urdf tests/pendulum.xml --out runs/example/log.csv \
  --seconds 6 --dt .001 --kp 10 --kd .5 --torque-scale .9 --extra-damping .1

python scripts/train.py runs/example/log.csv --out runs/example/model \
  --urdf tests/pendulum.xml --kp 10 --kd .5 --stride 2 --smooth 1 \
  --epochs 30 --batch 16

python scripts/evaluate.py runs/example/log.csv --model runs/example/model \
  --out runs/example/eval --urdf tests/pendulum.xml --kp 10 --kd .5 \
  --stride 2 --smooth 1 --fixed-dt .002
```

검증된 실행 결과는 `runs/verified/`에 있으며 테스트용 3 epoch 학습이다.
실제 로봇 성능 검증도, 일반화 성능 검증도 아니다.

## 실제 로그를 사용하는 방법

```bash
python scripts/train.py /path/log1.csv /path/log2.csv --out runs/robot/model \
  --urdf /path/fixed_base_robot.xml --joint-config /path/joints.json \
  --stride 4 --smooth 11
python scripts/evaluate.py /path/test.csv --model runs/robot/model --out runs/robot/eval \
  --urdf /path/fixed_base_robot.xml --joint-config /path/joints.json \
  --stride 4 --smooth 11
```

`--urdf`는 기존 CLI 호환 이름이며 MJCF XML도 허용한다. 기본 kp=100, kd=1은 원본 기본값이지
Raion 로봇에 승인된 게인이 아니다. joints.json은 **CSV의 관절 순서**로 모든 관절을 명시한다.

```json
{"joints": [{"name": "joint_a", "kp": 10, "kd": 0.5},
            {"name": "joint_b", "kp": 10, "kd": 0.5}]}
```

미지정하면 MuJoCo 컴파일 순서를 CSV 순서로 간주한다. URDF XML에 쓰인 순서와 항상 같지는 않다.
실제 데이터의 순서 확인 전 학습하지 말 것. config.json의 simulator에 모델 해시/관절명/게인을 저장하며
평가 시 불일치를 거부한다. 합성 수집의 meta.json이 있으면 CSV 해시/순서/게인도 확인한다.
기존 실기 CSV는 meta 없이 허용하되 사용자가 순서와 게인을 확인해야 한다.
기존 replay CSV도 원본처럼 입력할 수 있지만 생성 이력을 자동 입증할 수 없으므로
**RaiSim replay가 아닌 이 백엔드로 다시 만든 replay만 사용할 것**.

직접 라벨 생성/롤아웃:

```bash
python scripts/mujoco_sim.py replay /path/log.csv /path/replay.csv \
  --urdf /path/robot.xml --joint-config /path/joints.json --stride 4
python scripts/mujoco_sim.py rollout /path/log.csv /path/rollout.csv \
  --urdf /path/robot.xml --joint-config /path/joints.json --stride 4 \
  --model runs/robot/model/actuator_net_jit.pt
```

## 수집기 의미 및 한계

collect.py는 정현파+1초 smoothstep 시작 램프, seed별 위상, 고정 dt를 사용한다.
`--noise-q/--noise-v`는 로그 관측에만 더한다. 피드백은 노이즈 없는 내부 상태이며 target은 깨끗하다.
`--torque-scale/--extra-damping`은 실제 동역학 측에만 적용하는 인위적인 불일치다.
noise만 추가한 실험을 실제 로봇의 액추에이터 특성이라고 해석하지 말 것.
gf는 해당 구간에 적용할 일반화 힘이며 truth_gc/truth_gv와 command도 별도 기록한다.
마지막 행은 종료 상태여서 그 행의 토크는 적분하지 않는다.
관절별 궤적/노이즈, 외부 목표 CSV 재생, 하드웨어 수집기, 토크 지연/포화 모델은 아직 구현하지 않았다.

## 반드시 알고 사용할 제한

1. 고정 베이스, 스탠드 실험용. 접촉과 XML 내 native actuator를 비활성화한다.
   joint limit/passive 모델은 유지한다. free/ball/equality/tendon 및 URDF mimic은 거부한다.
2. 원본 잔차 `M(q_start)*(v_measured_end-v_nominal_end)/T`는 근사 라벨이다.
   Oracle은 미래 로그로 만든 비교 조건이지 보장된 성능 상한이 아니다.
3. 원본 블록 검증과 겹치는 창을 유지했다. 인접 시계열의 상관과 상태 문맥 공유 가능성은 남아 있다.
   홀드아웃 틱 롤아웃 지표를 독립적인 미관측 궤적 평가와 동일시하면 안 된다.
4. fixed-dt의 네트워크 주기는 원본처럼 정수 stride로 근사한다. 원래 예측률과 정확히 같지 않을 수 있다.
5. Python 백엔드는 이식/검증용이며 실시간 지연, C++ 배포 성능을 검증하지 않았다.
6. 원본 지표 키/그래프의 rad, Nm 표기는 회전 관절 기준이다. slide 혼합 모델에서는 단위를 구분해 해석해야 한다.
7. 27관절 실제 로봇 연결은 미완료. 이전 제공 URDF는 회전 27+그리퍼 2, mimic 및 arm3 관성 문제였다.
   현재 다른 폴더의 로봇 모델을 임의로 고정 베이스화하거나 관성을 보정하지 않았다.

## 이후 변경 기록

학습하지 않은 궤적 검사 결과: [generalization summary](runs/generalization_gui_demo/summary.md).
`scripts/check_generalization.py`는 기존 gui_demo 가중치를 고정하고 네 가지 새 합성 궤적을
수집/평가한다. 학습은 하지 않는다. 결과 폴더가 존재하면 덮어쓰지 않고 거부한다.
설정·가중치 전후 해시·각 조건의 전체 지표는 해당 폴더의 manifest.json에 남는다.

작업마다 patch note.txt에 날짜, 구현 위치, 기능, 원본 대비 차이, 검증 결과와 미완료 항목을 추가한 뒤 실행:

```bash
python scripts/audit_changes.py
```

원본 보존본은 수정하지 않고, 새 timestamp 폴더에 manifest와 unified diff를 남긴다.
