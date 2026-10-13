포터블 통합 빌드: 저장소 루트 README.md 및 scripts/build_sdk.sh 참조.
현재 run.sh는 build-sdk의 MuJoCo와 실행 파일을 함께 빌드합니다.
아래 기존 검증 기록의 build-greeny 및 /home/dongju 경로는 과거 환경입니다.

RoughTerrain3 MuJoCo ONNX inference
=================================

실행:
  cd /home/dongju/mujoco/RoughTerrain3
  ./run.sh
  ./run.sh --diagnostics --depth-view
  ./run.sh --flat-ground --command 0.3 0 0
  ./run.sh --flat-ground --headless --hold-seconds 40 --command 0.3 0 0
  ./run.sh --validate-contract

기본 정책: policy/model_10000.onnx
원본: /home/dongju/greeny/logs/rsl_rl/RoughTerrain3_cat_ppo/2026-10-06_22-30-56_cat_soft_contacts/exported/model_10000.onnx
PT 없이 ONNX Runtime CPU 추론. 학습 정책의 복사본이므로 학습 파일 변경과 독립적이다.
policy/SOURCE.json에 출처와 SHA256을 기록했다. greeny의 500 iteration 자동 ONNX
export는 학습 프로젝트 기능이며 이 실행 스크립트가 학습/export를 수행하지는 않는다.
다른 정책을 사용할 때는 해당 학습 run의 초기 자세/게인/주기와 배포 설정을 함께
검토해야 한다. --policy만 교체해 설정이 섞이지 않도록 checkpoint SHA256을 검사한다.

현재 범위:
- greeny의 창, 카메라, 깊이 미리보기, 조이스틱, 진단 그래프, 관절 궤적 기능 재사용.
- 기존 8관절 + history 정책을 RoughTerrain3 12관절/45관측 정책으로 교체.
- 상위 제어기 -> runtime/MotorHandler -> MujocoDrive 계층으로 분리.
- RobotPC 원본 수정 없음. 시뮬레이션 SharedMemory 용량 32, 실제 바인딩 28.
  RobotPC MAX_DOF=26 메모리와 바이너리 호환된다는 의미가 아니다.
- ARM/HRI 7채널 매핑, 그리퍼 rad↔m 변환, WebRTC 연결은 이번 범위에서 보류.
  이후 실제 ArmCommandHandler 연결 시 runtime/MotorHandler에 어댑터를 추가한다.
- 팔 arm1_J~arm6_J와 R_gripper_J는 연결된 XML의 nominal keyframe 자세를 유지.
  L_gripper_J는 XML equality mimic에 따른다. 다리 초기값은 학습 설정을 사용한다.
- Off/Estop gate는 전체 모터에 적용된다. 제어기의 command_ready=true로 우회 불가.
  Estop은 프로세스 재시작 전까지 유지되며 Backspace 초기화로 해제되지 않는다.
  HRI 통신 자체는 연결하지 않았으므로 0.3/3초 통신 timeout 상태기계는 아직 미적용.

정책 계약:
  float32 obs[1,45] -> actions[1,12]; actor normalization은 ONNX 안에 포함.
  0:3   [vx,vy,wz] * [2,2,0.25]
  3:6   몸체 좌표계 angular velocity * 0.25
  6:9   몸체 좌표계 projected gravity (정상 직립 [0,0,-1])
  9:21  q - 학습 nominal q, 아래 action 순서
  21:33 qd * 0.05
  33:45 직전 clipped action, 초기 0
  action 순서:
    LHP_J LHR_J LHY_J LKP_J LAP_J LAR_J
    RHP_J RHR_J RHY_J RKP_J RAP_J RAR_J
  q_target = 학습 nominal q + 0.25 * clip(action,-100,100)
  policy_dt=0.02s (50Hz). position/velocity/torque는 rad, rad/s, N·m.
  슬라이더는 m, m/s, N. 하위 PD는 XML 물리 step마다 갱신(현재 0.0005s).
  RobotPC SDK의 실시간 1kHz thread를 에뮬레이션한 것은 아니다.
  RL 실행에는 가상 몸체 지지력, 관측 noise, 임의 push, 자동 episode reset 없음.
  기존 homing 모드는 시뮬레이션용 자세 지지력을 사용하므로 RL 검증과 구분한다.

키:
  W/S: vx ±0.05, A/D: wz ±0.05, X: 속도 명령 0
  Space: 일시정지, Backspace: 수동 초기화, Escape: 종료
  H: 기존 homing, R: homing holding 상태에서 RL 시작
  Ctrl+O: 전체 출력 Off, Ctrl+R: 출력 재허용, Ctrl+E: latched Estop
  F1: 정보, F2: 깔끔한 화면, F3/F4/F5: 카메라 전환/선택/자유시점
  F9: 진단 그래프, F12: depth 창
  조이스틱 left-Y: vx, right-X: wz, Start: RL, Back: homing
  선택 정책은 bipedal 전용이므로 wheel posture 요청은 실행하지 않는다.

검증:
  cmake --build build -j4
  ctest --test-dir build --output-on-failure
  tests/controller_contract_test.cc:
    관측 순서/스케일/회전 중력, 이전 clipped action, 50Hz, 비정상 입력 차단,
    전체 Off/Estop, 설정-체크포인트 불일치 거부, Python/C++ ONNX 65개 입력 비교.
  tests/make_onnx_reference.py: Python ORT 기준값 생성 (numpy/onnxruntime 필요).
  headless 종료코드 0: 요청 구간 동안 finite이고 아래 넘어짐 기준 미충족.
  넘어짐 기준: base_z<0.45m 또는 upright_cos<0.5 (선택 CaT 학습 설정).
  종료코드 0만으로 속도 추종 정확도나 하드웨어 안전을 검증한 것은 아니다.
  --flat-ground는 현재 XML의 stairs_*, speed_bump_* 지형을 실행 메모리에서 숨기고 충돌 제외.
  원래 계단 환경은 이 옵션 없이 실행한다.
  최종 결과는 verification.txt를 참조.

주요 파일:
  app/main.cc                       실행 옵션/창/키 입력/시뮬레이션 진행
  controllers/rl_controller.{h,cc}   관측 생성/50Hz 추론/다리 타깃/기타 관절 유지
  policy/onnx_policy_runner.{h,cc}   ONNX 입출력 및 내장 메타데이터 검사/추론
  runtime/motor_handler.h           상하위 모터 경계/전체 출력 gate/Estop latch
  motor/src/mujoco_drive.cc          XML joint-actuator 바인딩/피드백/PD/effort 제한
  motor/include/shared_memory.h     시뮬레이션 전용 관측·명령 저장소
  motor/config/rough_terrain_28dof.yaml 28개 모터 순서/학습 PD/nominal
  config/rl_policy.yaml             학습 run 출처/주기/초기 몸체 자세/action 계약
  config/robots.yaml                연결 XML 선택
  runtime/homing_controller.*       기존 homing
  runtime/joint_trajectory.h        기존 fixed-base 관절 궤적
  runtime/depth_view.h              depth 미리보기 (실제 OAK calibration 아님)
  CMakeLists.txt, run.sh            독립 빌드/실행 (기존 build-greeny 삭제 없음)

의존성:
  /home/dongju/mujoco/include
  /home/dongju/mujoco/build-greeny/lib/libmujoco.so (기존 3.13.1 빌드)
  /home/dongju/mujoco/third_party/onnxruntime
  시스템 glfw3, yaml-cpp, nlohmann-json
  기존 XML/mesh 경로를 참조하며 원본 모델 파일을 수정하지 않는다.
