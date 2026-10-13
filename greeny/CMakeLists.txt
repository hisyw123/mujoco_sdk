add_executable(
  greeny_homing
  app/main.cc
  controllers/rl_controller.cc
  motor/src/motor_config.cc
  motor/src/mujoco_drive.cc
  policy/policy_runner.cc
  runtime/homing_controller.cc
  runtime/joystick_controller.cc
  runtime/robot_catalog.cc
  runtime/wheel_torque_limiter.cc
)

find_package(yaml-cpp REQUIRED)

option(MUJOCO_GREENY_ENABLE_ONNX "Build the GREENY ONNX Runtime policy backend" OFF)
set(GREENY_ONNXRUNTIME_ROOT "" CACHE PATH "Path to an ONNX Runtime installation")

if(MUJOCO_GREENY_ENABLE_ONNX)
  find_path(
    GREENY_ONNXRUNTIME_INCLUDE_DIR
    onnxruntime_cxx_api.h
    HINTS "${GREENY_ONNXRUNTIME_ROOT}/include"
    REQUIRED
  )
  find_library(
    GREENY_ONNXRUNTIME_LIBRARY
    onnxruntime
    HINTS "${GREENY_ONNXRUNTIME_ROOT}/lib"
    REQUIRED
  )
  target_sources(greeny_homing PRIVATE policy/onnx_policy_runner.cc)
  target_include_directories(greeny_homing PRIVATE ${GREENY_ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(greeny_homing PRIVATE ${GREENY_ONNXRUNTIME_LIBRARY})
  target_compile_definitions(
    greeny_homing
    PRIVATE
      GREENY_HAS_ONNXRUNTIME=1
      GREENY_DEFAULT_POLICY_PATH="${CMAKE_CURRENT_SOURCE_DIR}/policy/policy_1_14000.onnx"
  )
  set_target_properties(
    greeny_homing
    PROPERTIES BUILD_RPATH "${GREENY_ONNXRUNTIME_ROOT}/lib"
  )
endif()

target_compile_features(greeny_homing PRIVATE cxx_std_17)
target_include_directories(greeny_homing PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_definitions(
  greeny_homing
  PRIVATE
    GREENY_DEFAULT_MOTOR_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/motor/config/motors.yaml"
    GREENY_DEFAULT_ROBOT_CATALOG="${CMAKE_CURRENT_SOURCE_DIR}/config/robots.yaml"
)
target_link_libraries(
  greeny_homing
  PRIVATE
    mujoco::mujoco
    glfw
    yaml-cpp
)

if(DEFINED MUJOCO_SAMPLE_COMPILE_OPTIONS)
  target_compile_options(greeny_homing PRIVATE ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
endif()
if(DEFINED MUJOCO_SAMPLE_LINK_OPTIONS)
  target_link_options(greeny_homing PRIVATE ${MUJOCO_SAMPLE_LINK_OPTIONS})
endif()
