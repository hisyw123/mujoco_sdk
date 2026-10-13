// Copyright 2026

#include "greeny/controllers/rl_controller.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <yaml-cpp/yaml.h>

namespace greeny {
namespace {

template <typename T, std::size_t N>
std::array<T, N> ReadArray(const YAML::Node& node, const char* name) {
  if (!node || !node.IsSequence() || node.size() != N) {
    throw std::runtime_error(std::string(name) + " must contain " +
                             std::to_string(N) + " values");
  }
  std::array<T, N> values{};
  for (std::size_t i = 0; i < N; ++i) values[i] = node[i].as<T>();
  return values;
}

std::array<float, 3> ProjectedGravity(
    const std::array<double, 4>& quaternion) {
  const double w = quaternion[0];
  const double x = quaternion[1];
  const double y = quaternion[2];
  const double z = quaternion[3];
  const double norm = std::sqrt(w * w + x * x + y * y + z * z);
  if (!(norm > 0.0) || !std::isfinite(norm)) {
    throw std::runtime_error("invalid base quaternion");
  }
  const double inverse_norm = 1.0 / norm;
  const double nw = w * inverse_norm;
  const double nx = x * inverse_norm;
  const double ny = y * inverse_norm;
  const double nz = z * inverse_norm;

  // R(body->world)^T * [0, 0, -1].
  return {
      static_cast<float>(-2.0 * (nx * nz - nw * ny)),
      static_cast<float>(-2.0 * (ny * nz + nw * nx)),
      static_cast<float>(-(1.0 - 2.0 * (nx * nx + ny * ny))),
  };
}

}  // namespace

RlConfig LoadRlConfig(const std::string& path) {
  const YAML::Node root = YAML::LoadFile(path);
  RlConfig config;
  config.default_position =
      ReadArray<double, kJointCount>(root["default_joint_angle"],
                                     "default_joint_angle");
  if (root["default_transition_time"]) {
    config.posture_transition_duration =
        root["default_transition_time"].as<double>();
  }
  if (root["policy_dt"]) config.policy_period = root["policy_dt"].as<double>();
  if (root["history_length"]) {
    config.history_length = root["history_length"].as<int>();
  }
  if (root["history_stride"]) {
    config.history_stride = root["history_stride"].as<int>();
  }
  if (root["action_clip"]) config.action_clip = root["action_clip"].as<float>();
  if (root["scale"]) {
    const YAML::Node scale = root["scale"];
    if (scale["lin_vel"]) config.linear_velocity_scale = scale["lin_vel"].as<float>();
    if (scale["ang_vel"]) config.angular_velocity_scale = scale["ang_vel"].as<float>();
    if (scale["dof_pos"]) config.joint_position_scale = scale["dof_pos"].as<float>();
    if (scale["action_scale"]) {
      config.action_scale = ReadArray<float, kJointCount>(scale["action_scale"],
                                                          "scale.action_scale");
    }
  }
  const std::string style = root["joint_idx_style"].as<std::string>();
  const YAML::Node mapping = root["joint_idx_conversion_" + style];
  config.input_order =
      ReadArray<int, kJointCount>(mapping["input"], "joint input mapping");
  config.output_order =
      ReadArray<int, kJointCount>(mapping["output"], "joint output mapping");

  // The deployed transition controller builds the roll endpoint in policy
  // order from default_joint_angle + offset_default_pos_AB, then maps it back
  // to robot order.  default_joint_angle_task_roll is legacy metadata and is
  // not the target actually used by OnnxRunner.
  if (root["offset_default_pos_AB"]) {
    config.posture_transition_enabled = true;
    const auto offset = ReadArray<double, kJointCount>(
        root["offset_default_pos_AB"], "offset_default_pos_AB");
    std::array<double, kJointCount> policy_endpoint{};
    for (int policy_index = 0; policy_index < kJointCount; ++policy_index) {
      policy_endpoint[policy_index] =
          config.default_position[config.input_order[policy_index]] +
          offset[policy_index];
    }
    for (int robot_index = 0; robot_index < kJointCount; ++robot_index) {
      config.wheel_position[robot_index] =
          policy_endpoint[config.output_order[robot_index]];
    }
  } else if (root["default_joint_angle_task_roll"]) {
    config.posture_transition_enabled = true;
    config.wheel_position = ReadArray<double, kJointCount>(
        root["default_joint_angle_task_roll"],
        "default_joint_angle_task_roll");
  }
  return config;
}

RlController::RlController(PolicyRunner* runner, RlConfig config)
    : runner_(runner), config_(std::move(config)) {
  if (!runner_) {
    throw std::invalid_argument("RlController requires a PolicyRunner");
  }
  if (!(config_.policy_period > 0.0) || !(config_.transition_duration > 0.0) ||
      !(config_.posture_transition_duration > 0.0) ||
      config_.history_length <= 0 || config_.history_stride <= 0) {
    throw std::invalid_argument("invalid RL timing or history configuration");
  }
}

void RlController::Reset(SharedMemory* state) {
  ResetForPosture(0, state);
  engagement_transition_active_ = true;
  state->control_state.store(ControlState::kRlTransition,
                             std::memory_order_release);
}

void RlController::ResetForPosture(int posture, SharedMemory* state) {
  if (!state || !state->motor_data_ready.load(std::memory_order_acquire) ||
      state->motor_count.load(std::memory_order_acquire) != kJointCount) {
    throw std::invalid_argument("cannot reset RL without valid measured state");
  }
  if (posture != 0 && posture != 1) {
    throw std::invalid_argument("initial posture must be 0 or 1");
  }
  if (posture == 1 && !config_.posture_transition_enabled) {
    throw std::logic_error("RL config has no wheel posture");
  }
  state->DisableCommand();
  reset_time_ = state->time;
  last_policy_time_ = reset_time_ - config_.policy_period;
  raw_action_.fill(0.0f);
  scaled_action_.fill(0.0);
  posture_alpha_ = static_cast<double>(posture);
  policy_target_ = PoseForPosture(posture);
  policy_target_[static_cast<int>(JointIndex::kLeftWheelPitch)] =
      state->motors[static_cast<int>(JointIndex::kLeftWheelPitch)].q;
  policy_target_[static_cast<int>(JointIndex::kRightWheelPitch)] =
      state->motors[static_cast<int>(JointIndex::kRightWheelPitch)].q;
  history_counter_ = 0;
  transition_start_alpha_ = posture_alpha_;
  transition_phase_ = 0.0;
  transition_target_ = posture;
  posture_transition_active_ = false;
  state->posture_command_alpha = posture_alpha_;
  state->transition_phase = 0.0;
  state->transition_target.store(posture, std::memory_order_release);
  state->transition_active.store(false, std::memory_order_release);
  last_error_.clear();

  observation_ = BuildObservation(*state);
  const int command_count = config_.posture_transition_enabled ? 4 : 3;
  const std::vector<float> first_frame(observation_.begin() + command_count,
                                       observation_.end());
  history_.assign(config_.history_length, first_frame);
  initialized_ = true;
  engagement_transition_active_ = false;
  state->control_state.store(ControlState::kRlControl,
                             std::memory_order_release);
}

std::array<double, kJointCount> RlController::PoseForPosture(
    int posture) const {
  if (posture == 0) return config_.default_position;
  if (posture == 1 && config_.posture_transition_enabled) {
    return config_.wheel_position;
  }
  throw std::invalid_argument("requested posture is not configured");
}

void RlController::RequestPosture(int target, SharedMemory* state) {
  if (!config_.posture_transition_enabled) {
    return;
  }
  if (target != 0 && target != 1) {
    throw std::invalid_argument("posture target must be 0 or 1");
  }
  if (std::abs(posture_alpha_ - target) < 1e-9) return;
  transition_start_alpha_ = posture_alpha_;
  transition_phase_ = 0.0;
  transition_target_ = target;
  posture_transition_active_ = true;
  state->lin_vel_target.fill(0.0);
  state->ang_vel_target.fill(0.0);
  state->transition_target.store(target, std::memory_order_release);
  state->transition_active.store(true, std::memory_order_release);
}

void RlController::Update(SharedMemory* state) {
  if (!state || !state->motor_data_ready.load(std::memory_order_acquire)) {
    throw std::invalid_argument("RL update requires valid SharedMemory feedback");
  }
  if (!initialized_) {
    throw std::logic_error("Reset must be called before RL Update");
  }
  if (state->control_state.load(std::memory_order_acquire) ==
      ControlState::kFault) {
    state->DisableCommand();
    return;
  }

  try {
    if (state->time < reset_time_) {
      Reset(state);
    }
    if (state->time - last_policy_time_ + 1e-12 >=
        config_.policy_period) {
      RunPolicy(*state);
      AdvancePostureTransition(state);
      last_policy_time_ = state->time;
    }

    const double elapsed = std::max(0.0, state->time - reset_time_);
    const double transition_blend = engagement_transition_active_
                                        ? MinimumJerk(std::clamp(
                                              elapsed /
                                                  config_.transition_duration,
                                              0.0, 1.0))
                                        : 1.0;
    if (transition_blend >= 1.0) engagement_transition_active_ = false;
    WriteCommand(state, transition_blend);
    state->control_state.store(
        engagement_transition_active_ ? ControlState::kRlTransition
                                      : ControlState::kRlControl,
        std::memory_order_release);
  } catch (const std::exception& error) {
    SetFault(state, error.what());
  }
}

std::vector<float> RlController::BuildObservation(
    const SharedMemory& state) const {
  std::vector<float> observation;
  observation.reserve(config_.posture_transition_enabled ? 32 : 31);
  const bool transitioning = posture_transition_active_;
  observation.push_back(static_cast<float>(transitioning ? 0.0 : state.lin_vel_target[0]) *
                        config_.linear_velocity_scale);
  observation.push_back(
      config_.posture_transition_enabled
          ? static_cast<float>(transitioning ? 0.0 : state.lin_vel_target[1]) *
                config_.linear_velocity_scale
          : 0.0f);
  observation.push_back(static_cast<float>(transitioning ? 0.0 : state.ang_vel_target[0]) *
                        config_.angular_velocity_scale);
  if (config_.posture_transition_enabled) {
    observation.push_back(static_cast<float>(posture_alpha_));
  }

  auto gravity = ProjectedGravity(state.quat);
  std::array<double, 3> angular_velocity = state.gyro;
  if (config_.posture_transition_enabled) {
    const double theta = 0.5 * 3.14159265358979323846 * posture_alpha_;
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    gravity = {static_cast<float>(c * gravity[0] + s * gravity[2]),
               gravity[1],
               static_cast<float>(-s * gravity[0] + c * gravity[2])};
    angular_velocity = {
        c * angular_velocity[0] + s * angular_velocity[2],
        angular_velocity[1],
        -s * angular_velocity[0] + c * angular_velocity[2]};
  }
  observation.insert(observation.end(), gravity.begin(), gravity.end());
  for (double value : angular_velocity) {
    observation.push_back(static_cast<float>(value) *
                          config_.angular_velocity_scale);
  }

  const auto active_default = ActiveDefaultPosition();
  for (int policy_index = 0; policy_index < 6; ++policy_index) {
    const int joint = config_.input_order[policy_index];
    const double delta = state.motors[joint].q -
                         active_default[joint];
    observation.push_back(static_cast<float>(delta) *
                          config_.joint_position_scale);
  }
  for (int policy_index = 0; policy_index < 6; ++policy_index) {
    const int joint = config_.input_order[policy_index];
    observation.push_back(
        static_cast<float>(state.motors[joint].qd) * 0.05f);
  }
  for (int policy_index = 6; policy_index < 8; ++policy_index) {
    const int joint = config_.input_order[policy_index];
    observation.push_back(
        static_cast<float>(state.motors[joint].qd) * 0.5f);
  }
  for (int i = 0; i < kJointCount; ++i) {
    observation.push_back(raw_action_[i] * (i < 6 ? 0.25f : 1.0f));
  }

  const std::size_t expected = config_.posture_transition_enabled ? 32 : 31;
  if (observation.size() != expected) {
    throw std::logic_error("internal GREENY observation size mismatch");
  }
  return observation;
}

std::vector<float> RlController::FlattenHistory() const {
  std::vector<float> flattened;
  flattened.reserve(config_.history_length * RlConfig::kHistoryFrameSize);
  for (const auto& frame : history_) {
    flattened.insert(flattened.end(), frame.begin(), frame.end());
  }
  return flattened;
}

void RlController::RunPolicy(const SharedMemory& state) {
  observation_ = BuildObservation(state);
  const std::vector<float> action = runner_->Infer(observation_, FlattenHistory());
  if (action.size() != kJointCount) {
    throw std::runtime_error("policy returned an action with size other than 8");
  }

  std::array<double, kJointCount> scaled_policy_order{};
  for (int i = 0; i < kJointCount; ++i) {
    if (!std::isfinite(action[i])) {
      throw std::runtime_error("policy returned NaN or infinity");
    }
    raw_action_[i] = std::clamp(action[i], -config_.action_clip,
                                config_.action_clip);
    scaled_policy_order[i] = raw_action_[i] * config_.action_scale[i];
  }
  const auto active_default = ActiveDefaultPosition();
  for (int joint = 0; joint < kJointCount; ++joint) {
    scaled_action_[joint] = scaled_policy_order[config_.output_order[joint]];
    policy_target_[joint] = active_default[joint] + scaled_action_[joint];
  }
  const int left_wheel = static_cast<int>(JointIndex::kLeftWheelPitch);
  const int right_wheel = static_cast<int>(JointIndex::kRightWheelPitch);
  policy_target_[left_wheel] =
      state.motors[left_wheel].q + scaled_action_[left_wheel];
  policy_target_[right_wheel] =
      state.motors[right_wheel].q + scaled_action_[right_wheel];

  if (++history_counter_ >= config_.history_stride) {
    const int command_count = config_.posture_transition_enabled ? 4 : 3;
    history_.push_back(std::vector<float>(observation_.begin() + command_count,
                                          observation_.end()));
    while (static_cast<int>(history_.size()) > config_.history_length) {
      history_.pop_front();
    }
    history_counter_ = 0;
  }
}

void RlController::AdvancePostureTransition(SharedMemory* state) {
  if (!posture_transition_active_) return;
  transition_phase_ = std::clamp(
      transition_phase_ + config_.policy_period /
                              config_.posture_transition_duration,
      0.0, 1.0);
  const double blend = MinimumJerk(transition_phase_);
  posture_alpha_ = transition_start_alpha_ +
                   blend * (static_cast<double>(transition_target_) -
                            transition_start_alpha_);
  state->posture_command_alpha = posture_alpha_;
  state->transition_phase = transition_phase_;
  if (transition_phase_ >= 1.0) {
    posture_alpha_ = static_cast<double>(transition_target_);
    posture_transition_active_ = false;
    state->posture_command_alpha = posture_alpha_;
    state->transition_active.store(false, std::memory_order_release);
  }

  // The action was inferred from the pre-advance observation, matching the
  // deployed controller, but is applied around the newly advanced default.
  const auto active_default = ActiveDefaultPosition();
  for (int joint = 0; joint < kJointCount; ++joint) {
    policy_target_[joint] = active_default[joint] + scaled_action_[joint];
  }
  const int left_wheel = static_cast<int>(JointIndex::kLeftWheelPitch);
  const int right_wheel = static_cast<int>(JointIndex::kRightWheelPitch);
  policy_target_[left_wheel] =
      state->motors[left_wheel].q + scaled_action_[left_wheel];
  policy_target_[right_wheel] =
      state->motors[right_wheel].q + scaled_action_[right_wheel];
}

std::array<double, kJointCount> RlController::ActiveDefaultPosition() const {
  std::array<double, kJointCount> result{};
  for (int i = 0; i < kJointCount; ++i) {
    result[i] = config_.default_position[i] +
                posture_alpha_ *
                    (config_.wheel_position[i] - config_.default_position[i]);
  }
  return result;
}

void RlController::WriteCommand(SharedMemory* state,
                                double transition_blend) const {
  state->simulation_base_wrench.fill(0.0);
  for (int i = 0; i < kJointCount; ++i) {
    MotorSharedState& motor = state->motors[i];
    motor.q_target = policy_target_[i];
    motor.qd_target = 0.0;
    motor.torque_target = 0.0;
    motor.p_gain = config_.kp[i];
    motor.d_gain = config_.kd[i];
  }

  // Fade the homing-only orientation support while policy control takes over.
  const double support = 1.0 - transition_blend;
  const auto& quaternion = state->quat;
  const auto& angular_velocity = state->base_angular_velocity;
  const double sign = quaternion[0] < 0.0 ? -1.0 : 1.0;
  state->simulation_base_wrench[3] = support *
      (-2.0 * config_.orientation_support_kp * sign * quaternion[1] -
       config_.orientation_support_kd * angular_velocity[0]);
  state->simulation_base_wrench[4] = support *
      (-2.0 * config_.orientation_support_kp * sign * quaternion[2] -
       config_.orientation_support_kd * angular_velocity[1]);
  state->command_ready.store(true, std::memory_order_release);
}

void RlController::SetFault(SharedMemory* state, const std::string& message) {
  last_error_ = message;
  state->DisableCommand();
  state->control_state.store(ControlState::kFault, std::memory_order_release);
}

double RlController::MinimumJerk(double unit_time) {
  const double u = std::clamp(unit_time, 0.0, 1.0);
  return u * u * u * (10.0 + u * (-15.0 + 6.0 * u));
}

}  // namespace greeny
