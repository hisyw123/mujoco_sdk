// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>


#include "greeny/controllers/rl_controller.h"
#include "greeny/motor/include/motor_config.h"
#include "greeny/motor/include/mujoco_drive.h"
#include "greeny/policy/policy_runner.h"
#if defined(GREENY_HAS_ONNXRUNTIME)
#include "greeny/policy/onnx_policy_runner.h"
#endif
#include "greeny/runtime/homing_controller.h"
#include "greeny/runtime/joystick_controller.h"
#include "greeny/runtime/robot_catalog.h"
#include "greeny/runtime/joint_trajectory.h"
#include "greeny/runtime/depth_view.h"

namespace {

struct ModelDeleter {
  void operator()(mjModel* model) const { mj_deleteModel(model); }
};

struct DataDeleter {
  void operator()(mjData* data) const { mj_deleteData(data); }
};

using ModelPtr = std::unique_ptr<mjModel, ModelDeleter>;
using DataPtr = std::unique_ptr<mjData, DataDeleter>;

enum class StartupMode {
  kJointTrajectory,
  kHoming,
  kRlStand,
  kRlWheel,
};

mjModel* model = nullptr;
mjData* data = nullptr;
greeny::HomingController* homing = nullptr;
greeny::RlController* rl_controller = nullptr;
greeny::MujocoDrive* robot_io = nullptr;
greeny::SharedMemory common_state;
greeny::JointTrajectory* joint_trajectory = nullptr;
bool rl_active = false;
StartupMode startup_mode = StartupMode::kRlStand;

mjvCamera camera;
mjvCamera free_camera;
int preferred_camera = -1;
mjvOption visual_options;
mjvPerturb perturb;
mjvScene scene;
mjrContext render_context;
mjvFigure action_figure;
mjvFigure state_figure;
mjvFigure contact_figure;
mjvFigure torque_figure;
bool diagnostics_enabled = false;
bool depth_enabled = false;
bool diagnostics_initialized = false;

// View state.
bool show_overlay = true;  // F1
bool clean_view = false;   // F2

std::string CameraName() {
  if (camera.type != mjCAMERA_FIXED) return "free";
  const char* name = mj_id2name(model, mjOBJ_CAMERA, camera.fixedcamid);
  return name ? name : "camera " + std::to_string(camera.fixedcamid);
}

void SetCameraView(int id) {
  if (camera.type == mjCAMERA_FREE) free_camera = camera;
  if (id >= 0 && id < model->ncam) {
    camera.type = mjCAMERA_FIXED;
    camera.fixedcamid = id;
    preferred_camera = id;
  } else {
    camera = free_camera;
  }
  std::printf("[GREENY] Camera: %s\n", CameraName().c_str());
}

int plot_motor = 0;
bool plot_all = false;
int plot_kind = 0;  // position, velocity, actuator effort
std::vector<mjvFigure> joint_figures;
std::vector<int> plot_joints;
std::vector<int> plot_actuators;
double last_plot_time = -1.0;
double plot_delta = 0.0;
void InitializeDiagnostics();

bool paused = false;
greeny::JoystickController joystick;
bool left_pressed = false;
bool middle_pressed = false;
bool right_pressed = false;
double previous_mouse_x = 0.0;
double previous_mouse_y = 0.0;
double previous_left_click_time = -1.0;
double previous_left_click_x = 0.0;
double previous_left_click_y = 0.0;

constexpr double kDoubleClickSeconds = 0.25;
constexpr double kDoubleClickDistance = 5.0;


void ResetSimulation() {
  if (joint_trajectory) {
    joint_trajectory->ResetData(model, data);
    mj_forward(model, data);
    robot_io->ReadFeedback(model, data);
    joint_trajectory->Reset(&common_state);
    rl_active = false;
    return;
  }
  if (startup_mode == StartupMode::kHoming) {
    mj_resetData(model, data);
    mj_forward(model, data);
    robot_io->ReadFeedback(model, data);
    homing->Reset(&common_state);
    rl_active = false;
    return;
  }

  const int posture = startup_mode == StartupMode::kRlWheel ? 1 : 0;
  const auto pose = rl_controller->PoseForPosture(posture);
  const std::vector<double> joint_positions(pose.begin(), pose.end());
  const std::array<double, 3> base_position = {0.0, 0.0, 0.9};
  const std::array<double, 4> base_quaternion =
      posture == 0
          ? std::array<double, 4>{1.0, 0.0, 0.0, 0.0}
          : std::array<double, 4>{0.7071068, 0.0, 0.7071068, 0.0};
  robot_io->ResetPose(model, data, joint_positions, base_position,
                      base_quaternion);
  robot_io->ReadFeedback(model, data);
  rl_controller->ResetForPosture(posture, &common_state);
  rl_active = true;
}

const char* ControlStateName(greeny::ControlState state) {
  switch (state) {
    case greeny::ControlState::kBoot:
      return "BOOT";
    case greeny::ControlState::kHoming:
      return "HOMING";
    case greeny::ControlState::kHolding:
      return "HOLDING";
    case greeny::ControlState::kRlTransition:
      return "RL TRANSITION";
    case greeny::ControlState::kRlControl:
      return "RL CONTROL";
    case greeny::ControlState::kFault:
      return "FAULT";
  }
  return "UNKNOWN";
}

void UpdateController() {
  if (joint_trajectory) {
    joint_trajectory->Update(&common_state);
  } else if (rl_active) {
    rl_controller->Update(&common_state);
  } else {
    homing->Update(&common_state);
  }
}

bool IsApplicationKey(int key) {
  return key == GLFW_KEY_H || key == GLFW_KEY_R || key == GLFW_KEY_W ||
         key == GLFW_KEY_S || key == GLFW_KEY_A || key == GLFW_KEY_D ||
         key == GLFW_KEY_X || key == GLFW_KEY_Z;
}

bool ToggleVisualization(int key, int mods) {
  if (key == GLFW_KEY_F6) {
    visual_options.frame = (visual_options.frame + 1) % mjNFRAME;
    return true;
  }
  if (key == GLFW_KEY_F7) {
    visual_options.label = (visual_options.label + 1) % mjNLABEL;
    return true;
  }
  if (key == GLFW_KEY_F8) {
    scene.flags[mjRND_WIREFRAME] = !scene.flags[mjRND_WIREFRAME];
    return true;
  }

  if (key >= GLFW_KEY_1 && key <= GLFW_KEY_6) {
    const int group = key - GLFW_KEY_1;
    if ((mods & GLFW_MOD_SHIFT) != 0) {
      visual_options.sitegroup[group] = !visual_options.sitegroup[group];
    } else {
      visual_options.geomgroup[group] = !visual_options.geomgroup[group];
    }
    return true;
  }

  // Movement and controller keys retain their application behavior. Their
  // official MuJoCo visualization shortcuts remain available with Alt.
  if (IsApplicationKey(key) && (mods & GLFW_MOD_ALT) == 0) {
    return false;
  }
  for (int i = 0; i < mjNVISFLAG; ++i) {
    const char* shortcut = mjVISSTRING[i][2];
    if (shortcut[0] && key == static_cast<unsigned char>(shortcut[0])) {
      visual_options.flags[i] = !visual_options.flags[i];
      return true;
    }
  }
  for (int i = 0; i < mjNRNDFLAG; ++i) {
    const char* shortcut = mjRNDSTRING[i][2];
    if (shortcut[0] && key == static_cast<unsigned char>(shortcut[0])) {
      scene.flags[i] = !scene.flags[i];
      return true;
    }
  }
  return false;
}

void Keyboard(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (action != GLFW_PRESS) {
    return;
  }

  if (key == GLFW_KEY_F1) {
    show_overlay = !show_overlay;
    return;
  }

  if (key == GLFW_KEY_F12) {
    depth_enabled = !depth_enabled;
    return;
  }

  if (key == GLFW_KEY_F3) {
    const int next = camera.type == mjCAMERA_FIXED ? camera.fixedcamid + 1 : 0;
    SetCameraView(next);
    return;
  }

  if (key == GLFW_KEY_F4) {
    SetCameraView(preferred_camera);
    return;
  }

  if (key == GLFW_KEY_F5) {
    SetCameraView(-1);
    return;
  }

  if (key == GLFW_KEY_F2) {
    clean_view = !clean_view;

    if (clean_view) {
      show_overlay = false;
      visual_options.geomgroup[3] = 0;
      visual_options.flags[mjVIS_INERTIA] = 0;
      visual_options.flags[mjVIS_CONTACTPOINT] = 0;
      visual_options.flags[mjVIS_CONTACTFORCE] = 0;
      visual_options.flags[mjVIS_JOINT] = 0;
      visual_options.flags[mjVIS_COM] = 0;
      scene.flags[mjRND_WIREFRAME] = 0;
    } else {
      show_overlay = true;
    }
    return;
  }

  if (key == GLFW_KEY_ESCAPE) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  } else if (key == GLFW_KEY_SPACE) {
    paused = !paused;
  } else if (key == GLFW_KEY_BACKSPACE) {
    ResetSimulation();
    if (diagnostics_initialized) InitializeDiagnostics();
  } else if (key == GLFW_KEY_F9) {
    diagnostics_enabled = !diagnostics_enabled;
    if (diagnostics_enabled) InitializeDiagnostics();
  } else if (joint_trajectory && key == GLFW_KEY_F10) {
    plot_all = !plot_all;
    diagnostics_enabled = true;
    InitializeDiagnostics();
  } else if (joint_trajectory && key == GLFW_KEY_F11) {
    plot_kind = (plot_kind + 1) % 3;
    if (diagnostics_initialized) InitializeDiagnostics();
  } else if (joint_trajectory &&
             (key == GLFW_KEY_LEFT_BRACKET || key == GLFW_KEY_RIGHT_BRACKET)) {
    const int count = static_cast<int>(common_state.motor_count.load());
    if (count > 0) {
      plot_motor = (plot_motor + count + (key == GLFW_KEY_RIGHT_BRACKET ? 1 : -1)) % count;
      if (diagnostics_initialized) InitializeDiagnostics();
    }
  } else if (ToggleVisualization(key, mods)) {
    return;
  } else if (joint_trajectory) {
    return;
  } else if (key == GLFW_KEY_H) {
    robot_io->ReadFeedback(model, data);
    homing->Reset(&common_state);
    rl_active = false;
    paused = false;
  } else if (key == GLFW_KEY_R &&
             common_state.control_state.load(std::memory_order_acquire) ==
                 greeny::ControlState::kHolding) {
    rl_controller->Reset(&common_state);
    rl_active = true;
    paused = false;
  } else if (key == GLFW_KEY_W) {
    common_state.lin_vel_target[0] = std::min(
        1.0, common_state.lin_vel_target[0] + 0.05);
  } else if (key == GLFW_KEY_S) {
    common_state.lin_vel_target[0] = std::max(
        -1.0, common_state.lin_vel_target[0] - 0.05);
  } else if (key == GLFW_KEY_A) {
    common_state.ang_vel_target[0] = std::min(
        0.5, common_state.ang_vel_target[0] + 0.05);
  } else if (key == GLFW_KEY_D) {
    common_state.ang_vel_target[0] = std::max(
        -0.5, common_state.ang_vel_target[0] - 0.05);
  } else if (key == GLFW_KEY_X) {
    common_state.lin_vel_target.fill(0.0);
    common_state.ang_vel_target.fill(0.0);
    if (rl_active) rl_controller->RequestPosture(0, &common_state);
  } else if (key == GLFW_KEY_Z && rl_active) {
    rl_controller->RequestPosture(1, &common_state);
  }
}

greeny::JoystickSample ReadJoystick() {
  greeny::JoystickSample sample;
  GLFWgamepadstate gamepad{};
  sample.connected =
      glfwJoystickIsGamepad(GLFW_JOYSTICK_1) == GLFW_TRUE &&
      glfwGetGamepadState(GLFW_JOYSTICK_1, &gamepad) == GLFW_TRUE;
  if (!sample.connected) {
    return sample;
  }

  sample.forward_axis = -gamepad.axes[GLFW_GAMEPAD_AXIS_LEFT_Y];
  // Match the keyboard convention: A/left is positive yaw.
  sample.yaw_axis = -gamepad.axes[GLFW_GAMEPAD_AXIS_RIGHT_X];
  sample.left_bumper =
      gamepad.buttons[GLFW_GAMEPAD_BUTTON_LEFT_BUMPER] == GLFW_PRESS;
  sample.right_bumper =
      gamepad.buttons[GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER] == GLFW_PRESS;
  sample.stand_button = gamepad.buttons[GLFW_GAMEPAD_BUTTON_X] == GLFW_PRESS;
  sample.wheel_button = gamepad.buttons[GLFW_GAMEPAD_BUTTON_Y] == GLFW_PRESS;
  sample.engage_button = gamepad.buttons[GLFW_GAMEPAD_BUTTON_START] == GLFW_PRESS;
  sample.homing_button = gamepad.buttons[GLFW_GAMEPAD_BUTTON_BACK] == GLFW_PRESS;
  return sample;
}

void UpdateJoystick() {
  if (joint_trajectory) return;
  const greeny::JoystickEvents events =
      joystick.Update(ReadJoystick(), &common_state);
  if (events.request_homing) {
    robot_io->ReadFeedback(model, data);
    homing->Reset(&common_state);
    rl_active = false;
    paused = false;
  } else if (events.request_engage &&
             common_state.control_state.load(std::memory_order_acquire) ==
                 greeny::ControlState::kHolding) {
    rl_controller->Reset(&common_state);
    rl_active = true;
    paused = false;
  }
  if (events.request_stand) {
    common_state.lin_vel_target.fill(0.0);
    common_state.ang_vel_target.fill(0.0);
    if (rl_active) rl_controller->RequestPosture(0, &common_state);
  } else if (events.request_wheel && rl_active) {
    rl_controller->RequestPosture(1, &common_state);
  }
}

void SelectBody(GLFWwindow* window, double x, double y) {
  int width = 0;
  int height = 0;
  glfwGetWindowSize(window, &width, &height);
  if (width <= 0 || height <= 0) {
    return;
  }

  mjtNum selected_point[3];
  int selected_geom = -1;
  int selected_flex = -1;
  int selected_skin = -1;
  const int selected_body =
      mjv_select(model, data, &visual_options,
                 static_cast<mjtNum>(width) / height, x / width,
                 1.0 - y / height, &scene, selected_point, &selected_geom,
                 &selected_flex, &selected_skin);

  if (selected_body > 0) {
    perturb.select = selected_body;
    perturb.flexselect = selected_flex;
    perturb.skinselect = selected_skin;

    mjtNum offset[3];
    mju_sub3(offset, selected_point, data->xpos + 3 * selected_body);
    mju_mulMatTVec(perturb.localpos, data->xmat + 9 * selected_body,
                   offset, 3, 3);
  } else {
    perturb.select = 0;
    perturb.flexselect = -1;
    perturb.skinselect = -1;
  }
  perturb.active = 0;
}

void MouseButton(GLFWwindow* window, int button, int action, int mods) {
  left_pressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
  middle_pressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
  right_pressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
  glfwGetCursorPos(window, &previous_mouse_x, &previous_mouse_y);

  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS &&
      (mods & GLFW_MOD_CONTROL) == 0) {
    const double now = glfwGetTime();
    const double dx = previous_mouse_x - previous_left_click_x;
    const double dy = previous_mouse_y - previous_left_click_y;
    const bool double_click =
        previous_left_click_time >= 0.0 &&
        now - previous_left_click_time <= kDoubleClickSeconds &&
        dx * dx + dy * dy <=
            kDoubleClickDistance * kDoubleClickDistance;

    previous_left_click_time = now;
    previous_left_click_x = previous_mouse_x;
    previous_left_click_y = previous_mouse_y;

    if (double_click) {
      SelectBody(window, previous_mouse_x, previous_mouse_y);
      return;
    }
  }

  const bool control = (mods & GLFW_MOD_CONTROL) != 0;
  if (action == GLFW_PRESS && control && perturb.select > 0 &&
      (button == GLFW_MOUSE_BUTTON_LEFT ||
       button == GLFW_MOUSE_BUTTON_RIGHT)) {
    mjv_initPerturb(model, data, &scene, &perturb);
    perturb.active =
        button == GLFW_MOUSE_BUTTON_RIGHT
            ? mjPERT_TRANSLATE
            : mjPERT_ROTATE;
  } else if (action == GLFW_RELEASE &&
             (button == GLFW_MOUSE_BUTTON_LEFT ||
              button == GLFW_MOUSE_BUTTON_RIGHT)) {
    perturb.active = 0;
  }
}

void MouseMove(GLFWwindow* window, double x, double y) {
  if (!left_pressed && !middle_pressed && !right_pressed) {
    return;
  }

  const double dx = x - previous_mouse_x;
  const double dy = y - previous_mouse_y;
  previous_mouse_x = x;
  previous_mouse_y = y;

  int width = 0;
  int height = 0;
  glfwGetWindowSize(window, &width, &height);
  if (height <= 0) {
    return;
  }

  const bool shift =
      glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;

  mjtMouse action;
  if (right_pressed) {
    action = shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  } else if (left_pressed) {
    action = shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  } else {
    action = mjMOUSE_ZOOM;
  }

  if (perturb.active) {
    mjv_movePerturb(model, data, action, dx / height, -dy / height,
                    &scene, &perturb);
  } else if (camera.type == mjCAMERA_FREE) {
    mjv_moveCamera(model, action, dx / height, dy / height, &camera);
  }
}

void Scroll(GLFWwindow* window, double x_offset, double y_offset) {
  if (camera.type == mjCAMERA_FREE) {
    mjv_moveCamera(model, mjMOUSE_ZOOM, 0.0, -0.05 * y_offset, &camera);
  }
}

ModelPtr LoadModel(const char* filename, bool fixed_base, double base_height) {
  char error[1024] = "Could not load binary model";
  if (fixed_base) {
    // Compile a private in-memory variant; source XML stays unchanged.
    std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)> spec(
        mj_parseXML(filename, nullptr, error, sizeof(error)), mj_deleteSpec);
    if (!spec) throw std::runtime_error(error);
    mjsElement* root = mjs_findElement(spec.get(), mjOBJ_JOINT, "root");
    mjsBody* base = mjs_findBody(spec.get(), "base_link");
    if (!root || !base || mjs_getParent(root) != base ||
        mjs_asJoint(root)->type != mjJNT_FREE)
      throw std::runtime_error("fixed-base mode requires base_link with free joint root");
    // Keyframes use the free-base qpos/qvel layout in the XML. Remove only
    // the root coordinates before compiling the fixed-base variant.
    if (mjs_firstElement(spec.get(), mjOBJ_KEY)) {
      ModelPtr original(mj_compile(spec.get(), nullptr));
      if (!original) throw std::runtime_error(mjs_getError(spec.get()));
      const int root_id = mj_name2id(original.get(), mjOBJ_JOINT, "root");
      const auto remove_root = [](mjDoubleVec* values, int address, int width) {
        int count = 0;
        const double* raw = mjs_getDouble(values, &count);
        if (!count) return;  // Omitted keyframe fields retain their defaults.
        if (address + width > count) throw std::runtime_error("invalid root keyframe layout");
        std::vector<double> adjusted(raw, raw + count);
        adjusted.erase(adjusted.begin() + address, adjusted.begin() + address + width);
        mjs_setDouble(values, adjusted.data(), static_cast<int>(adjusted.size()));
      };
      for (mjsElement* e = mjs_firstElement(spec.get(), mjOBJ_KEY); e;
           e = mjs_nextElement(spec.get(), e)) {
        mjsKey* key = mjs_asKey(e);
        remove_root(key->qpos, original->jnt_qposadr[root_id], 7);
        remove_root(key->qvel, original->jnt_dofadr[root_id], 6);
      }
    }
    if (mjs_delete(spec.get(), root)) throw std::runtime_error("cannot remove free joint");
    base->pos[2] = base_height;
    mjModel* fixed = mj_compile(spec.get(), nullptr);
    if (!fixed) throw std::runtime_error(mjs_getError(spec.get()));
    // Keep the XML collision masks and exclusions in fixed-base mode too.
    return ModelPtr(fixed);
  }
  mjModel* loaded = nullptr;
  const std::size_t length = std::strlen(filename);
  if (length > 4 && std::strcmp(filename + length - 4, ".mjb") == 0) {
    loaded = mj_loadModel(filename, nullptr);
  } else {
    loaded = mj_loadXML(filename, nullptr, error, sizeof(error));
  }
  if (!loaded) {
    throw std::runtime_error(std::string("failed to load model: ") + error);
  }
  return ModelPtr(loaded);
}

struct AppOptions {
  std::string model_path;
  std::string camera_name;
  std::string robot_name;
  std::string catalog_path = GREENY_DEFAULT_ROBOT_CATALOG;
#if defined(GREENY_HAS_ONNXRUNTIME)
  std::string policy_path = GREENY_DEFAULT_POLICY_PATH;
#else
  std::string policy_path;
#endif
  std::string runner_config_path;
  std::string motor_config_path = GREENY_DEFAULT_MOTOR_CONFIG;
  double homing_duration = 3.0;
  double headless_hold_duration = 5.0;
  bool headless = false;
  bool diagnostics = false;
  bool depth_view = false;
  std::string plot_joint;
  bool plot_all = false;
  int plot_kind = 0;
  StartupMode mode = StartupMode::kRlStand;
  int target_posture = -1;
  bool list_robots = false;
  bool validate_contract = false;
  std::string trajectory_config;
  std::string trajectory_log;
  double base_height = 1.5;
};

StartupMode ParseMode(const std::string& text) {
  if (text == "joint-trajectory") return StartupMode::kJointTrajectory;
  if (text == "homing") return StartupMode::kHoming;
  if (text == "rl" || text == "rl-stand" || text == "stand") {
    return StartupMode::kRlStand;
  }
  if (text == "rl-wheel" || text == "wheel" || text == "roll") {
    return StartupMode::kRlWheel;
  }
  throw std::invalid_argument("mode must be rl, homing, rl-wheel, or joint-trajectory");
}

double ParsePositive(const std::string& name, const char* value) {
  std::size_t parsed = 0;
  const std::string text(value);
  const double result = std::stod(text, &parsed);
  if (parsed != text.size() || !std::isfinite(result) || !(result > 0.0)) {
    throw std::invalid_argument(name + " must be positive");
  }
  return result;
}

AppOptions ParseOptions(int argc, const char** argv) {
  AppOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    if (argument == "--catalog" && i + 1 < argc) {
      options.catalog_path = argv[++i];
    } else if (argument == "--robot" && i + 1 < argc) {
      options.robot_name = argv[++i];
    } else if (argument == "--list-robots") {
      options.list_robots = true;
    }
  }
  if (!options.robot_name.empty()) {
    const greeny::RobotProfile profile =
        greeny::LoadRobotProfile(options.catalog_path, options.robot_name);
    options.model_path = profile.model_path;
    options.motor_config_path = profile.motor_contract_path;
    options.runner_config_path = profile.policy_contract_path;
    options.trajectory_config = profile.trajectory_contract_path;
    options.mode = ParseMode(profile.default_mode);
  }

  bool duration_seen = false;
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    if ((argument == "--catalog" || argument == "--robot") && i + 1 < argc) {
      ++i;
    } else if (argument == "--list-robots") {
      continue;
    } else if (argument == "--validate-contract") {
      options.validate_contract = true;
    } else if (argument == "--trajectory-config" && i + 1 < argc) {
      options.trajectory_config = argv[++i];
    } else if (argument == "--trajectory-log" && i + 1 < argc) {
      options.trajectory_log = argv[++i];
    } else if (argument == "--base-height" && i + 1 < argc) {
      options.base_height = ParsePositive("base height", argv[++i]);
    } else if (argument == "--headless") {
      options.headless = true;
    } else if (argument == "--camera" && i + 1 < argc) {
      options.camera_name = argv[++i];
    } else if (argument == "--diagnostics") {
      options.diagnostics = true;
    } else if (argument == "--depth-view") {
      options.depth_view = true;
    } else if (argument == "--plot-joint" && i + 1 < argc) {
      options.plot_joint = argv[++i];
      options.diagnostics = true;
    } else if (argument == "--plot-all") {
      options.plot_all = true;
      options.diagnostics = true;
    } else if (argument == "--plot-kind" && i + 1 < argc) {
      const std::string kind = argv[++i];
      if (kind == "position") options.plot_kind = 0;
      else if (kind == "velocity") options.plot_kind = 1;
      else if (kind == "torque") options.plot_kind = 2;
      else throw std::invalid_argument("--plot-kind must be position, velocity, or torque");
    } else if (argument == "--start-rl") {
      options.mode = StartupMode::kRlStand;
    } else if (argument == "--mode" && i + 1 < argc) {
      options.mode = ParseMode(argv[++i]);
    } else if (argument == "--policy" && i + 1 < argc) {
      options.policy_path = argv[++i];
    } else if (argument == "--motor-config" && i + 1 < argc) {
      options.motor_config_path = argv[++i];
    } else if (argument == "--runner-config" && i + 1 < argc) {
      options.runner_config_path = argv[++i];
    } else if (argument == "--target-posture" && i + 1 < argc) {
      const std::string posture(argv[++i]);
      if (posture == "stand") {
        options.target_posture = 0;
      } else if (posture == "wheel" || posture == "roll") {
        options.target_posture = 1;
      } else {
        throw std::invalid_argument(
            "target posture must be stand, wheel, or roll");
      }
    } else if (argument == "--hold-seconds" && i + 1 < argc) {
      options.headless_hold_duration = ParsePositive("hold duration", argv[++i]);
    } else if (argument.rfind("--", 0) == 0) {
      throw std::invalid_argument("unknown or incomplete option: " + argument);
    } else if (options.model_path.empty()) {
      options.model_path = argument;
    } else if (!duration_seen) {
      options.homing_duration = ParsePositive("homing duration", argv[i]);
      duration_seen = true;
    } else {
      throw std::invalid_argument("unexpected argument: " + argument);
    }
  }
  if (!options.list_robots && options.model_path.empty()) {
    throw std::invalid_argument(
        "select a robot with --robot NAME or provide MODEL_PATH");
  }
  if (options.mode == StartupMode::kJointTrajectory && options.trajectory_config.empty())
    throw std::invalid_argument("joint-trajectory requires --trajectory-config YAML");
  if (options.mode != StartupMode::kJointTrajectory &&
      (!options.trajectory_config.empty() || !options.trajectory_log.empty()))
    throw std::invalid_argument("trajectory options require --mode joint-trajectory");
  return options;
}

bool StateIsFinite(const mjModel* model, const mjData* data) {
  for (int i = 0; i < model->nq; ++i) {
    if (!std::isfinite(data->qpos[i])) {
      return false;
    }
  }
  for (int i = 0; i < model->nv; ++i) {
    if (!std::isfinite(data->qvel[i])) {
      return false;
    }
  }
  return true;
}

struct ContactDiagnostics {
  int contact_count = 0;
  int wheel_stair_contacts = 0;
  double maximum_penetration = 0.0;
  double wheel_stair_penetration = 0.0;
  double maximum_force = 0.0;
  double wheel_stair_force = 0.0;
};

ContactDiagnostics ReadContactDiagnostics() {
  ContactDiagnostics result;
  result.contact_count = data->ncon;
  for (int i = 0; i < data->ncon; ++i) {
    const mjContact& contact = data->contact[i];
    const double penetration = std::max(0.0, -contact.dist);
    mjtNum contact_force[6]{};
    mj_contactForce(model, data, i, contact_force);
    const double normal_force = std::abs(contact_force[0]);
    result.maximum_penetration =
        std::max(result.maximum_penetration, penetration);
    result.maximum_force = std::max(result.maximum_force, normal_force);

    const char* first = mj_id2name(model, mjOBJ_GEOM, contact.geom[0]);
    const char* second = mj_id2name(model, mjOBJ_GEOM, contact.geom[1]);
    const std::string first_name = first ? first : "";
    const std::string second_name = second ? second : "";
    const auto is_wheel = [](const std::string& name) {
      return name == "LWP_collision" || name == "RWP_collision";
    };
    const auto is_stair = [](const std::string& name) {
      return name.find("_level_") != std::string::npos ||
             name.find("stair_") != std::string::npos;
    };
    if ((is_wheel(first_name) && is_stair(second_name)) ||
        (is_stair(first_name) && is_wheel(second_name))) {
      ++result.wheel_stair_contacts;
      result.wheel_stair_penetration =
          std::max(result.wheel_stair_penetration, penetration);
      result.wheel_stair_force =
          std::max(result.wheel_stair_force, normal_force);
    }
  }
  return result;
}

void InitializeDiagnosticFigure(mjvFigure* figure, const char* title,
                                const char* const* line_names,
                                int line_count) {
  mjv_defaultFigure(figure);
  figure->flg_legend = 1;
  figure->flg_extend = 1;
  figure->flg_ticklabel[0] = 1;
  figure->flg_ticklabel[1] = 1;
  figure->figurergba[3] = 0.75f;
  figure->gridsize[0] = 5;
  figure->gridsize[1] = 5;
  std::snprintf(figure->title, sizeof(figure->title), "%s", title);
  std::snprintf(figure->xlabel, sizeof(figure->xlabel), "seconds");
  for (int i = 0; i < line_count; ++i) {
    std::snprintf(figure->linename[i], sizeof(figure->linename[i]), "%s",
                  line_names[i]);
  }
}

void InitializeDiagnostics() {
  diagnostics_initialized = true;
  last_plot_time = -1.0;
  if (joint_trajectory && plot_all) {
    const int count = static_cast<int>(common_state.motor_count.load());
    joint_figures.resize(count);
    const char* names[] = {"target", "measured"};
    const char* effort_names[] = {"command", "output"};
    for (int i = 0; i < count; ++i) {
      const bool slide = model->jnt_type[plot_joints[i]] == mjJNT_SLIDE;
      const char* unit = plot_kind == 0 ? (slide ? "m" : "rad") :
                         plot_kind == 1 ? (slide ? "m/s" : "rad/s") : (slide ? "N" : "Nm");
      const std::string title = common_state.motors[i].name + " (" + unit + ")";
      auto& figure = joint_figures[i];
      InitializeDiagnosticFigure(&figure, title.c_str(), plot_kind == 2 ? effort_names : names, 2);
      figure.flg_legend = 0;  // Shared legend above the grid.
      figure.xlabel[0] = '\0';  // Time unit is shown once in the grid header.
      figure.figurergba[3] = 1.0f;
      figure.gridsize[0] = figure.gridsize[1] = 3;
      figure.linergb[0][0] = 1.0f; figure.linergb[0][1] = 0.65f; figure.linergb[0][2] = 0.1f;
      figure.linergb[1][0] = 0.1f; figure.linergb[1][1] = 0.8f; figure.linergb[1][2] = 1.0f;
    }
    return;
  }
  if (joint_trajectory) {
    const auto& motor = common_state.motors[plot_motor];
    const bool slide = model->jnt_type[plot_joints[plot_motor]] == mjJNT_SLIDE;
    const char* position_names[] = {"target", "measured"};
    const char* torque_names[] = {"command (clamped)", "actuator output"};
    const char* contact_names[] = {"max pen mm", "max force kN", "contacts"};
    const std::string pos = motor.name + (slide ? " position (m)" : " position (rad)");
    const std::string vel = motor.name + (slide ? " velocity (m/s)" : " velocity (rad/s)");
    const std::string force = motor.name + (slide ? " actuator force (N)" : " actuator torque (Nm)");
    InitializeDiagnosticFigure(&action_figure, pos.c_str(), position_names, 2);
    InitializeDiagnosticFigure(&state_figure, vel.c_str(), position_names, 2);
    InitializeDiagnosticFigure(&torque_figure, force.c_str(), torque_names, 2);
    InitializeDiagnosticFigure(&contact_figure, "Contacts", contact_names, 3);
    return;
  }
  const char* action_names[] = {"a0", "a1", "a2", "a3",
                                "a4", "a5", "a6", "a7"};
  const char* state_names[] = {"max joint qd", "max torque",
                               "base speed", "base gyro"};
  const char* contact_names[] = {"wheel pen mm", "max pen mm",
                                 "wheel force kN", "contacts"};
  InitializeDiagnosticFigure(&action_figure, "Policy actions", action_names, 8);
  InitializeDiagnosticFigure(&state_figure, "State spikes", state_names, 4);
  InitializeDiagnosticFigure(&contact_figure, "Contact diagnostics",
                             contact_names, 4);
  action_figure.flg_symmetric = 1;
}

void AppendFigureSample(mjvFigure* figure, const double* values,
                        int line_count) {
  constexpr int kSamples = 600;  // 10 seconds at the 60 Hz render rate.
  for (int line = 0; line < line_count; ++line) {
    const int points = std::min(kSamples, figure->linepnt[line] + 1);
    for (int i = points - 1; i > 0; --i) {
      figure->linedata[line][2 * i] = figure->linedata[line][2 * (i - 1)] -
                                      static_cast<float>(plot_delta);
      figure->linedata[line][2 * i + 1] =
          figure->linedata[line][2 * (i - 1) + 1];
    }
    figure->linepnt[line] = points;
    figure->linedata[line][0] = 0.0f;
    figure->linedata[line][1] = static_cast<float>(values[line]);
  }
}

void UpdateDiagnosticFigures(const ContactDiagnostics& contacts) {
  if (data->time < last_plot_time) InitializeDiagnostics();
  if (last_plot_time >= 0 && data->time - last_plot_time < 1.0 / 60.0) return;
  plot_delta = last_plot_time < 0 ? 0.0 : data->time - last_plot_time;
  last_plot_time = data->time;
  if (joint_trajectory && plot_all) {
    for (std::size_t i = 0; i < joint_figures.size(); ++i) {
      const auto& motor = common_state.motors[i];
      const int j = plot_joints[i], a = plot_actuators[i];
      double values[2];
      if (plot_kind == 0) {
        values[0] = motor.q_target; values[1] = data->qpos[model->jnt_qposadr[j]];
      } else if (plot_kind == 1) {
        values[0] = motor.qd_target; values[1] = data->qvel[model->jnt_dofadr[j]];
      } else {
        values[0] = data->ctrl[a]; values[1] = data->actuator_force[a];
      }
      AppendFigureSample(&joint_figures[i], values, 2);
    }
    return;
  }
  if (joint_trajectory) {
    const auto& motor = common_state.motors[plot_motor];
    const int j = plot_joints[plot_motor];
    const int a = plot_actuators[plot_motor];
    const double q[] = {motor.q_target, data->qpos[model->jnt_qposadr[j]]};
    const double qd[] = {motor.qd_target, data->qvel[model->jnt_dofadr[j]]};
    // The joint-trajectory drive uses direct motor actuators (gain=1).
    const double effort[] = {data->ctrl[a], data->actuator_force[a]};
    const double c[] = {1000.0 * contacts.maximum_penetration,
                        contacts.maximum_force / 1000.0,
                        static_cast<double>(contacts.contact_count)};
    AppendFigureSample(&action_figure, q, 2);
    AppendFigureSample(&state_figure, qd, 2);
    AppendFigureSample(&torque_figure, effort, 2);
    AppendFigureSample(&contact_figure, c, 3);
    return;
  }
  double actions[8]{};
  const auto& raw_action = rl_controller->raw_action();
  for (int i = 0; i < 8; ++i) actions[i] = raw_action[i];

  double max_qd = 0.0;
  double max_torque = 0.0;
  for (std::size_t i = 0; i < common_state.motor_count.load(); ++i) {
    max_qd = std::max(max_qd, std::abs(common_state.motors[i].qd));
    max_torque =
        std::max(max_torque, std::abs(common_state.motors[i].torque));
  }
  const auto norm3 = [](const std::array<double, 3>& value) {
    return std::sqrt(value[0] * value[0] + value[1] * value[1] +
                     value[2] * value[2]);
  };
  const double states[] = {max_qd, max_torque,
                           norm3(common_state.base_linear_velocity),
                           norm3(common_state.base_angular_velocity)};
  const double contact_values[] = {
      1000.0 * contacts.wheel_stair_penetration,
      1000.0 * contacts.maximum_penetration,
      contacts.wheel_stair_force / 1000.0,
      static_cast<double>(contacts.contact_count)};
  AppendFigureSample(&action_figure, actions, 8);
  AppendFigureSample(&state_figure, states, 4);
  AppendFigureSample(&contact_figure, contact_values, 4);
}

void RenderDiagnosticFigures(const mjrRect& viewport, const mjrContext* context) {
  if (joint_trajectory && plot_all) {
    const int count = static_cast<int>(joint_figures.size());
    if (!count) return;
    const int columns = std::min(5, count);
    const int rows = (count + columns - 1) / columns;
    const int top = std::min(44, viewport.height / 5);
    const int available = viewport.height - top;
    for (int i = 0; i < count; ++i) {
      const int col = i % columns, row = i / columns;
      const int left = col * viewport.width / columns;
      const int right = (col + 1) * viewport.width / columns;
      const int bottom = available - (row + 1) * available / rows;
      const int upper = available - row * available / rows;
      mjrRect panel = {viewport.left + left, viewport.bottom + bottom, right-left, upper-bottom};
      mjr_figure(panel, &joint_figures[i], context);
    }
    const char* kinds[] = {"POSITION", "VELOCITY", "TORQUE / FORCE"};
    char header[512];
    std::snprintf(header, sizeof(header), "%d motors | %s | t=%.2f%s | contacts=%d\n"
        "Orange: %s | Cyan: %s | F11: metric | F10: single | F9: hide | Space: pause | Backspace: reset",
        count, kinds[plot_kind], data->time, paused ? " PAUSED" : "", data->ncon,
        plot_kind == 2 ? "clamped command" : "target", plot_kind == 2 ? "actuator output" : "measured");
    mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport, "ALL JOINTS (time: s)", header, context);
    return;
  }
  const int width = viewport.width;
  const int height = viewport.height / (joint_trajectory ? 4 : 3);
  mjrRect panel = {viewport.width - width, 0, width, height};
  mjr_figure(panel, &contact_figure, context);
  panel.bottom += height;
  if (joint_trajectory) {
    mjr_figure(panel, &torque_figure, context);
    panel.bottom += height;
  }
  mjr_figure(panel, &state_figure, context);
  panel.bottom += height;
  mjr_figure(panel, &action_figure, context);
}

}  // namespace

int main(int argc, const char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "Usage: greeny_homing (--robot NAME | MODEL_PATH) "
                 "[HOMING_DURATION_SECONDS] [--headless] [--camera CAMERA_NAME] [--depth-view] "
                 "[--hold-seconds VALUE] [--motor-config motors.yaml] "
                 "[--runner-config runner_config.yaml] "
                 "[--catalog robots.yaml] [--list-robots] "
                 "[--validate-contract] "
                 "[--policy MODEL.onnx] "
                 "[--mode rl|homing|rl-wheel|joint-trajectory] "
                 "[--trajectory-config YAML] [--trajectory-log CSV] [--base-height METERS] "
                 "[--diagnostics] [--plot-joint MOTOR_NAME] [--plot-all] [--plot-kind position|velocity|torque] "
                 "[--target-posture stand|wheel]\n");
    return EXIT_FAILURE;
  }

  try {
    const AppOptions options = ParseOptions(argc, argv);
    if (options.list_robots) {
      for (const std::string& name :
           greeny::ListRobotProfiles(options.catalog_path)) {
        std::printf("%s\n", name.c_str());
      }
      return EXIT_SUCCESS;
    }
    const bool trajectory_mode = options.mode == StartupMode::kJointTrajectory;
    ModelPtr owned_model = LoadModel(options.model_path.c_str(), trajectory_mode,
                                    options.base_height);
    model = owned_model.get();
    const int initial_camera = options.camera_name.empty() ? -1 :
        mj_name2id(model, mjOBJ_CAMERA, options.camera_name.c_str());
    if (!options.camera_name.empty() && initial_camera < 0) {
      throw std::invalid_argument("unknown camera: " + options.camera_name);
    }
    DataPtr owned_data(mj_makeData(model));
    if (!owned_data) {
      throw std::runtime_error("failed to allocate mjData");
    }
    data = owned_data.get();
    const greeny::MotorConfig motor_config =
        greeny::LoadMotorConfig(options.motor_config_path);
    const greeny::RlConfig rl_config = (trajectory_mode || options.runner_config_path.empty())
                                            ? greeny::RlConfig{}
                                            : greeny::LoadRlConfig(
                                                  options.runner_config_path);
    greeny::MujocoDrive owned_robot_io(model, motor_config, &common_state);
    robot_io = &owned_robot_io;
    plot_joints.resize(motor_config.motors.size());
    plot_actuators.resize(motor_config.motors.size());
    bool plot_found = options.plot_joint.empty();
    for (const auto& spec : motor_config.motors) {
      plot_joints[spec.joint_index] = mj_name2id(model, mjOBJ_JOINT, spec.joint_name.c_str());
      plot_actuators[spec.joint_index] = mj_name2id(model, mjOBJ_ACTUATOR, spec.actuator_name.c_str());
      if (spec.name == options.plot_joint || spec.joint_name == options.plot_joint) {
        plot_motor = spec.joint_index;
        plot_found = true;
      }
    }
    if (!plot_found) throw std::runtime_error("unknown plot joint: " + options.plot_joint);
    if (!options.plot_joint.empty() && !trajectory_mode)
      throw std::runtime_error("--plot-joint requires joint-trajectory mode");
    if (options.plot_all && !trajectory_mode)
      throw std::runtime_error("--plot-all requires joint-trajectory mode");
    plot_all = options.plot_all;
    plot_kind = options.plot_kind;
    std::unique_ptr<greeny::JointTrajectory> owned_trajectory;
    if (trajectory_mode) {
      owned_trajectory = std::make_unique<greeny::JointTrajectory>(
          model, motor_config, options.trajectory_config);
      joint_trajectory = owned_trajectory.get();
    }
    if (options.validate_contract) {
      std::unordered_set<int> bound_actuators;
      for (const auto& spec : motor_config.motors) {
        bound_actuators.insert(
            mj_name2id(model, mjOBJ_ACTUATOR, spec.actuator_name.c_str()));
      }
      std::printf(
          "contract OK: robot=%s model=%s motors=%zu/%lld policy_contract=%s\n",
          options.robot_name.empty() ? "<direct-model>"
                                     : options.robot_name.c_str(),
          options.model_path.c_str(), motor_config.motors.size(),
          static_cast<long long>(model->nu),
          options.runner_config_path.empty()
              ? "<runtime-default>"
              : options.runner_config_path.c_str());
      if (bound_actuators.size() != static_cast<std::size_t>(model->nu)) {
        std::printf("unbound actuators:");
        for (int actuator = 0; actuator < model->nu; ++actuator) {
          if (bound_actuators.count(actuator)) continue;
          const char* name = mj_id2name(model, mjOBJ_ACTUATOR, actuator);
          std::printf(" %s", name ? name : "<unnamed>");
        }
        std::printf("\n");
      }
      return EXIT_SUCCESS;
    }

    greeny::HomingConfig config;
    config.duration = options.homing_duration;
    greeny::HomingController owned_homing(motor_config, config);
    homing = &owned_homing;

    std::unique_ptr<greeny::PolicyRunner> owned_policy;
    if (trajectory_mode || options.policy_path.empty()) {
      owned_policy = std::make_unique<greeny::ZeroPolicyRunner>();
    } else {
#if defined(GREENY_HAS_ONNXRUNTIME)
      owned_policy =
          std::make_unique<greeny::OnnxPolicyRunner>(options.policy_path);
#else
      throw std::runtime_error(
          "--policy requires a build with MUJOCO_GREENY_ENABLE_ONNX=ON");
#endif
    }
    greeny::RlController owned_rl(owned_policy.get(), rl_config);
    rl_controller = &owned_rl;
    startup_mode = options.mode;
    ResetSimulation();
    diagnostics_enabled = options.diagnostics;
    if (diagnostics_enabled) InitializeDiagnostics();
    std::ofstream trajectory_log;
    if (!options.trajectory_log.empty()) {
      trajectory_log.open(options.trajectory_log);
      if (!trajectory_log) throw std::runtime_error("cannot open trajectory CSV");
      trajectory_log << "time,motor,q_target,q,qd_target,qd,torque\n";
    }
    const auto log_trajectory = [&]() {
      if (!trajectory_log.is_open()) return;
      for (const auto& spec : motor_config.motors) {
        const auto& motor = common_state.motors[spec.joint_index];
        trajectory_log << common_state.time << ',' << spec.name << ','
                       << motor.q_target << ',' << motor.q << ','
                       << motor.qd_target << ',' << motor.qd << ',' << motor.torque << '\n';
      }
    };
    if (rl_active && options.target_posture >= 0) {
      rl_controller->RequestPosture(options.target_posture, &common_state);
    }

    if (options.headless) {
      const double stop_time =
          ((rl_active || trajectory_mode) ? 0.0 : config.duration) + options.headless_hold_duration;
      const std::uint64_t maximum_steps = static_cast<std::uint64_t>(
          std::ceil(2.0 * stop_time / model->opt.timestep));
      std::uint64_t step_count = 0;
      int trajectory_peak_contacts = 0;
      double trajectory_peak_penetration = 0.0;
      double trajectory_peak_contact_force = 0.0;
      while (data->time < stop_time) {
        if (++step_count > maximum_steps) {
          throw std::runtime_error(
              "simulation time failed to advance; model may be unstable");
        }
        // External wrenches are one-step commands. Clear them before the
        // controller publishes the next base wrench.
        mju_zero(data->xfrc_applied, 6 * model->nbody);
        robot_io->ReadFeedback(model, data);
        UpdateController();
        log_trajectory();
        if (common_state.control_state.load(std::memory_order_acquire) ==
            greeny::ControlState::kFault) {
          throw std::runtime_error("RL controller fault: " +
                                   rl_controller->last_error());
        }
        robot_io->WriteCommand(model, data);
        mj_step(model, data);
        if (diagnostics_enabled) UpdateDiagnosticFigures(ReadContactDiagnostics());
        if (trajectory_mode) {
          const ContactDiagnostics contacts = ReadContactDiagnostics();
          trajectory_peak_contacts =
              std::max(trajectory_peak_contacts, contacts.contact_count);
          trajectory_peak_penetration = std::max(
              trajectory_peak_penetration, contacts.maximum_penetration);
          trajectory_peak_contact_force = std::max(
              trajectory_peak_contact_force, contacts.maximum_force);
        }
        if (!StateIsFinite(model, data)) {
          throw std::runtime_error("simulation produced a non-finite state");
        }
      }
      robot_io->ReadFeedback(model, data);
      if (trajectory_mode) {
        std::printf(
            "joint-trajectory: t=%.3f s, fixed base z=%.6f m, "
            "peak_contacts=%d, peak_penetration=%.3f mm, "
            "peak_contact_force=%.3f N\n",
            data->time, common_state.base_position[2],
            trajectory_peak_contacts, 1000.0 * trajectory_peak_penetration,
            trajectory_peak_contact_force);
        for (const auto& spec : motor_config.motors) {
          const auto& motor = common_state.motors[spec.joint_index];
          std::printf("  %s: target=%.5f measured=%.5f rad\n",
                      spec.name.c_str(), motor.q_target, motor.q);
        }
        const int left_gripper =
            mj_name2id(model, mjOBJ_JOINT, "L_gripper_J");
        const int right_gripper =
            mj_name2id(model, mjOBJ_JOINT, "R_gripper_J");
        if (left_gripper >= 0 && right_gripper >= 0) {
          std::printf(
              "  gripper mimic: L=%.5f R=%.5f delta=%.6f m\n",
              data->qpos[model->jnt_qposadr[left_gripper]],
              data->qpos[model->jnt_qposadr[right_gripper]],
              data->qpos[model->jnt_qposadr[left_gripper]] -
                  data->qpos[model->jnt_qposadr[right_gripper]]);
        }
        // Success here means finite execution, not a tracking accuracy verdict.
        return EXIT_SUCCESS;
      }
      const double roll = common_state.base_rpy[0];
      const double pitch = common_state.base_rpy[1];
      const double joint_error = homing->max_position_error(common_state);
      std::printf("GREENY %s: t=%.3f s, homing=%.0f%%, base_z=%.6f m, "
                  "roll=%.3f rad, pitch=%.3f rad, max_leg_error=%.4f rad",
                  ControlStateName(common_state.control_state.load()),
                  common_state.time, 100.0 * homing->progress(common_state),
                  common_state.base_position[2], roll, pitch, joint_error);
      if (rl_controller->supports_posture_transition()) {
        std::printf(", posture_alpha=%.3f, transition=%s",
                    common_state.posture_command_alpha,
                    common_state.transition_active.load() ? "active" : "done");
      } else {
        std::printf(", policy=basic-rl");
      }
      std::printf("\n");
      const auto errors = homing->position_errors(common_state);
      std::printf("joint errors:");
      for (std::size_t i = 0; i < motor_config.motors.size(); ++i) {
        std::printf(" %s=%+.4f", motor_config.motors[i].name.c_str(),
                    errors[motor_config.motors[i].joint_index]);
      }
      std::printf("\n");
      const bool standing = common_state.base_position[2] > 0.4 &&
                            std::abs(roll) < 0.2 &&
                            std::abs(pitch) < 0.2 && joint_error < 0.45;
      const bool expected_state =
          rl_active ? common_state.control_state.load() ==
                          greeny::ControlState::kRlControl
                    : homing->phase() == greeny::HomingController::Phase::kHolding;
      return expected_state && standing ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    if (!glfwInit()) {
      throw std::runtime_error("failed to initialize GLFW");
    }

    // GLFW window호출전 
    glfwWindowHint(GLFW_SAMPLES, 4);


    GLFWwindow* window = glfwCreateWindow(1200, 900,
                                           "GREENY Homing", nullptr, nullptr);
    if (!window) {
      throw std::runtime_error("failed to create GLFW window");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    mjv_defaultCamera(&camera);
    mjv_defaultOption(&visual_options);

    // Current MJCF: group 3 is collision geometry.
    visual_options.geomgroup[3] = 0;

    mjv_defaultPerturb(&perturb);
    mjv_defaultScene(&scene);
    mjr_defaultContext(&render_context);
    mjv_makeScene(model, &scene, 2000);
    mjr_makeContext(model, &render_context, mjFONTSCALE_150);

    camera.lookat[0] = model->stat.center[0];
    camera.lookat[1] = model->stat.center[1];
    camera.lookat[2] = model->stat.center[2];
    camera.distance = 1.5 * model->stat.extent;
    camera.azimuth = 135.0;
    camera.elevation = -20.0;
    free_camera = camera;
    preferred_camera = mj_name2id(model, mjOBJ_CAMERA, "cam_a");
    if (preferred_camera < 0 && model->ncam > 0) preferred_camera = 0;
    if (initial_camera >= 0) {
      SetCameraView(initial_camera);
    }

    glfwSetKeyCallback(window, Keyboard);
    glfwSetCursorPosCallback(window, MouseMove);
    glfwSetMouseButtonCallback(window, MouseButton);
    glfwSetScrollCallback(window, Scroll);

    GLFWwindow* plot_window = nullptr;
    mjrContext plot_context;
    mjr_defaultContext(&plot_context);
    greeny::DepthView depth_view;
    depth_enabled = options.depth_view;

    while (!glfwWindowShouldClose(window)) {
      if (plot_window && glfwWindowShouldClose(plot_window)) {
        diagnostics_enabled = false;
        glfwSetWindowShouldClose(plot_window, GLFW_FALSE);
      }
      glfwMakeContextCurrent(window);
      UpdateJoystick();
      if (!paused) {
        const mjtNum frame_start = data->time;
        while (data->time - frame_start < 1.0 / 60.0) {
          mju_zero(data->xfrc_applied, 6 * model->nbody);
          mjv_applyPerturbPose(model, data, &perturb, 0);
          mjv_applyPerturbForce(model, data, &perturb);
          robot_io->ReadFeedback(model, data);
          UpdateController();
          log_trajectory();
          robot_io->WriteCommand(model, data);
          mj_step(model, data);
        }
      } else {
        mjv_applyPerturbPose(model, data, &perturb, 1);
        mj_forward(model, data);
        robot_io->ReadFeedback(model, data);
      }

      mjrRect viewport = {0, 0, 0, 0};
      glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
      mjv_updateScene(model, data, &visual_options, &perturb, &camera, mjCAT_ALL,
                      &scene);
      mjr_render(viewport, &scene, &render_context);

      const char* selected_name =
          perturb.select > 0 ? mj_id2name(model, mjOBJ_BODY, perturb.select) : nullptr;
      const ContactDiagnostics contacts = ReadContactDiagnostics();
      if (diagnostics_enabled) UpdateDiagnosticFigures(contacts);
      char status[1024];
      std::snprintf(status, sizeof(status),
                    "%s%s | %.0f%% | t = %.3f s | selected: %s\n"
                    "policy: %s | input: %s | vx %.2f | wz %.2f\n"
                    "contacts: %d | max penetration: %.2f mm | "
                    "wheel-stair: %d / %.2f mm\n"
                    "vis: inertia %d  contact %d  force %d  joint %d  COM %d\n"
                    "Double-left: select  Ctrl-left: rotate  Ctrl-right: move\n"
                    "R: RL  H: homing  W/S: velocity  A/D: yaw\n"
                    "Z/Y: wheel posture  X: stand posture + stop\n"
                    "Gamepad: left-Y velocity  right-X yaw  Start: RL  Back: homing\n"
                    "I: inertia  C/F: contact/force  J: joints  M: COM\n"
                    "F1: HUD  F2: clean  F3: next camera  F4: camera view  F5: free view\n"
                    "F6: frame  F7: label  F8: wireframe  1-6: geom groups\n"
                    "F12: depth window  Space: pause  Backspace: reset",
                    trajectory_mode ? "FIXED BASE / JOINT TRAJECTORY" :
                        ControlStateName(common_state.control_state.load()),
                    paused ? " (PAUSED)" : "",
                    100.0 * homing->progress(common_state), common_state.time,
                    selected_name ? selected_name : "none",
                    rl_controller->policy_name(),
                    joystick.connected() ? "gamepad" : "keyboard",
                    common_state.lin_vel_target[0],
                    common_state.ang_vel_target[0],
                    contacts.contact_count,
                    1000.0 * contacts.maximum_penetration,
                    contacts.wheel_stair_contacts,
                    1000.0 * contacts.wheel_stair_penetration,
                    visual_options.flags[mjVIS_INERTIA],
                    visual_options.flags[mjVIS_CONTACTPOINT],
                    visual_options.flags[mjVIS_CONTACTFORCE],
                    visual_options.flags[mjVIS_JOINT],
                    visual_options.flags[mjVIS_COM]);
      if (show_overlay && !trajectory_mode) {
        const std::string view_status = std::string(status) + "\nCamera: " + CameraName();
        mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport, "GREENY", view_status.c_str(),
                    &render_context);
      }
      if (show_overlay && trajectory_mode) {
        const std::string trajectory_status =
            std::string("FIXED BASE / JOINT TRAJECTORY") +
            (paused ? " (PAUSED)" : "") +
            "\nConfig: " + options.trajectory_config +
            "\nSpace: pause  Backspace: restart trajectory  Escape: exit"
            "\nLocomotion keys and gamepad commands are disabled."
            "\nCamera: " + CameraName() +
            "\nF1: HUD  F2: clean  F3: next camera"
            "\nF4: camera view  F5: free view (restore mouse position)"
            "\nF9: plots  F10: all/single  F11: metric  [: previous  ]: next"
            "\nF12: depth window";
        mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport, "GREENY",
                    trajectory_status.c_str(), &render_context);
      }
      glfwSwapBuffers(window);
      if (diagnostics_enabled) {
        if (!plot_window) {
          plot_window = glfwCreateWindow(1100, 760, "GREENY Joint Plots", nullptr, nullptr);
          if (!plot_window) throw std::runtime_error("failed to create plot window");
          glfwSetWindowSizeLimits(plot_window, 750, 540, GLFW_DONT_CARE, GLFW_DONT_CARE);
          glfwSetKeyCallback(plot_window, Keyboard);
          glfwMakeContextCurrent(plot_window);
          glfwSwapInterval(0);
          // Separate GL resources and smaller text for the compact plot grid.
          mjr_makeContext(model, &plot_context, mjFONTSCALE_50);
        }
        if (!glfwGetWindowAttrib(plot_window, GLFW_VISIBLE)) glfwShowWindow(plot_window);
        glfwMakeContextCurrent(plot_window);
        mjrRect plot_viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(plot_window, &plot_viewport.width, &plot_viewport.height);
        if (plot_viewport.width > 0 && plot_viewport.height > 0) {
          mjr_setBuffer(mjFB_WINDOW, &plot_context);
          const float background[4] = {0.06f, 0.06f, 0.07f, 1.0f};
          mjr_rectangle(plot_viewport, background[0], background[1], background[2], background[3]);
          RenderDiagnosticFigures(plot_viewport, &plot_context);
          glfwSwapBuffers(plot_window);
        }
      } else if (plot_window && glfwGetWindowAttrib(plot_window, GLFW_VISIBLE)) {
        glfwHideWindow(plot_window);
      }
      depth_view.Update(model, data, depth_enabled);
      glfwPollEvents();
    }

    depth_view.Close();
    if (plot_window) {
      glfwMakeContextCurrent(plot_window);
      mjr_freeContext(&plot_context);
      glfwDestroyWindow(plot_window);
    }
    glfwMakeContextCurrent(window);
    mjv_freeScene(&scene);
    mjr_freeContext(&render_context);
    glfwDestroyWindow(window);
#if defined(__APPLE__) || defined(_WIN32)
    glfwTerminate();
#endif
  } catch (const std::exception& error) {
    std::fprintf(stderr, "greeny_homing: %s\n", error.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
