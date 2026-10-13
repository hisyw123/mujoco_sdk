cmake_minimum_required(VERSION 3.18)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(RoughTerrain3 LANGUAGES CXX)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(DEFAULT_MUJOCO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(MUJOCO_ROOT "${DEFAULT_MUJOCO_ROOT}" CACHE PATH "MuJoCo source root")
set(ONNXRUNTIME_ROOT "${MUJOCO_ROOT}/third_party/onnxruntime" CACHE PATH "ONNX Runtime C++ SDK")
find_package(yaml-cpp REQUIRED)
if(NOT TARGET glfw)
  find_package(glfw3 REQUIRED)
endif()
find_package(nlohmann_json REQUIRED)
find_package(Threads REQUIRED)
if(TARGET mujoco::mujoco)
  set(MUJOCO_LIB mujoco::mujoco)
else()
  find_library(MUJOCO_LIB mujoco HINTS "${MUJOCO_ROOT}/build-sdk/lib" "${MUJOCO_ROOT}/build-greeny/lib" REQUIRED)
endif()
find_library(ORT_LIB onnxruntime HINTS "${ONNXRUNTIME_ROOT}/lib" REQUIRED)
add_library(roughterrain3_core STATIC
 controllers/rl_controller.cc motor/src/motor_config.cc motor/src/mujoco_drive.cc
 policy/policy_runner.cc policy/onnx_policy_runner.cc runtime/homing_controller.cc
 runtime/joystick_controller.cc runtime/robot_catalog.cc runtime/wheel_torque_limiter.cc)
get_filename_component(PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
target_include_directories(roughterrain3_core PUBLIC "${PARENT_DIR}" "${MUJOCO_ROOT}/include" "${ONNXRUNTIME_ROOT}/include")
target_link_libraries(roughterrain3_core PUBLIC "${MUJOCO_LIB}" "${ORT_LIB}" yaml-cpp glfw Threads::Threads nlohmann_json::nlohmann_json)
add_executable(roughterrain3_sim app/main.cc)
target_link_libraries(roughterrain3_sim PRIVATE roughterrain3_core)
target_compile_definitions(roughterrain3_sim PRIVATE
 GREENY_HAS_ONNXRUNTIME=1
 GREENY_DEFAULT_POLICY_PATH="${CMAKE_CURRENT_SOURCE_DIR}/policy/model_10000.onnx"
 GREENY_DEFAULT_MOTOR_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/motor/config/rough_terrain_28dof.yaml"
 GREENY_DEFAULT_ROBOT_CATALOG="${CMAKE_CURRENT_SOURCE_DIR}/config/robots.yaml")
include(CTest)
if(BUILD_TESTING)
 add_executable(controller_contract_test tests/controller_contract_test.cc)
 target_link_libraries(controller_contract_test PRIVATE roughterrain3_core)
 add_test(NAME controller_contract COMMAND controller_contract_test
   "${CMAKE_CURRENT_SOURCE_DIR}/config/rl_policy.yaml"
   "${MUJOCO_ROOT}/model/RoughTerrain-3/RoughTerrain-3_bipedal.xml"
   "${CMAKE_CURRENT_SOURCE_DIR}/policy/model_10000.onnx"
   "${CMAKE_CURRENT_SOURCE_DIR}/tests/onnx_reference.json")
endif()
