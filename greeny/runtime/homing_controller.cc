// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0

#include "greeny/runtime/homing_controller.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace greeny {

HomingController::HomingController(const MotorConfig& motor_config,
                                   HomingConfig config)
    : config_(std::move(config)),
      motor_config_(motor_config),
      start_position_(motor_config.motors.size()),
      target_position_(motor_config.motors.size()) {
  if (!(config_.duration > 0.0) || !std::isfinite(config_.duration)) {
    throw std::invalid_argument("homing duration must be finite and positive");
  }
}

void HomingController::Reset(SharedMemory* state) {
  if (!state || !state->motor_data_ready.load(std::memory_order_acquire)) {
    throw std::invalid_argument("cannot reset homing without valid measured state");
  }

  state->DisableCommand();
  start_time_ = state->time;
  for (const MotorSpec& spec : motor_config_.motors) {
    const int i = spec.joint_index;
    start_position_[i] = state->motors[i].q;
    target_position_[i] = spec.continuous ? start_position_[i]
                                          : spec.home_position;
  }

  // LWP and RWP are continuous wheel joints. Hold the positions observed at
  // reset instead of rotating them toward a globally fixed angle.
  phase_ = Phase::kHoming;
  state->control_state.store(ControlState::kHoming, std::memory_order_release);
}

void HomingController::Update(SharedMemory* state) {
  if (!state || !state->motor_data_ready.load(std::memory_order_acquire)) {
    throw std::invalid_argument("homing update requires valid SharedMemory feedback");
  }
  if (phase_ == Phase::kNotInitialized) {
    throw std::logic_error("Reset must be called before Update");
  }

  // A UI reset or keyframe can move simulation time backwards.
  if (state->time < start_time_) {
    Reset(state);
  }

  const double elapsed = std::max(0.0, state->time - start_time_);
  const double unit_time = std::clamp(elapsed / config_.duration, 0.0, 1.0);
  const double blend = MinimumJerk(unit_time);
  const double blend_rate = MinimumJerkDerivative(unit_time) / config_.duration;

  state->simulation_base_wrench.fill(0.0);
  for (const MotorSpec& spec : motor_config_.motors) {
    const int i = spec.joint_index;
    const double delta = target_position_[i] - start_position_[i];
    const double desired_position = start_position_[i] + blend * delta;
    const double desired_velocity = blend_rate * delta;
    MotorSharedState& motor = state->motors[i];
    motor.q_target = desired_position;
    motor.qd_target = desired_velocity;
    motor.torque_target = 0.0;
    motor.p_gain = spec.p_gain;
    motor.d_gain = spec.d_gain;
  }

  // Homing alone cannot balance an inverted-pendulum robot. Until a balancing
  // controller is connected, apply a physical roll/pitch spring-damper torque
  // to the base. The MuJoCo adapter is solely responsible for applying it.
  const auto& quaternion = state->quat;
  const auto& angular_velocity = state->base_angular_velocity;
  const double quaternion_sign = quaternion[0] < 0.0 ? -1.0 : 1.0;
  state->simulation_base_wrench[3] =
      -2.0 * config_.orientation_support_kp * quaternion_sign * quaternion[1] -
      config_.orientation_support_kd * angular_velocity[0];
  state->simulation_base_wrench[4] =
      -2.0 * config_.orientation_support_kp * quaternion_sign * quaternion[2] -
      config_.orientation_support_kd * angular_velocity[1];

  state->command_ready.store(true, std::memory_order_release);
  if (unit_time >= 1.0) {
    phase_ = Phase::kHolding;
    state->control_state.store(ControlState::kHolding, std::memory_order_release);
  }
}

double HomingController::progress(const SharedMemory& state) const {
  if (!state.motor_data_ready.load(std::memory_order_acquire) ||
      phase_ == Phase::kNotInitialized) {
    return 0.0;
  }
  return std::clamp((state.time - start_time_) / config_.duration, 0.0, 1.0);
}

std::vector<double> HomingController::position_errors(
    const SharedMemory& state) const {
  std::vector<double> errors(motor_config_.motors.size(), 0.0);
  if (!state.motor_data_ready.load(std::memory_order_acquire) ||
      phase_ == Phase::kNotInitialized) {
    return errors;
  }
  for (const MotorSpec& spec : motor_config_.motors) {
    errors[spec.joint_index] =
        target_position_[spec.joint_index] - state.motors[spec.joint_index].q;
  }
  return errors;
}

double HomingController::max_position_error(const SharedMemory& state) const {
  const auto errors = position_errors(state);
  double maximum = 0.0;
  for (const MotorSpec& spec : motor_config_.motors) {
    if (!spec.continuous) {
      maximum = std::max(maximum, std::abs(errors[spec.joint_index]));
    }
  }
  return maximum;
}

double HomingController::MinimumJerk(double unit_time) {
  const double u = std::clamp(unit_time, 0.0, 1.0);
  return u * u * u * (10.0 + u * (-15.0 + 6.0 * u));
}

double HomingController::MinimumJerkDerivative(double unit_time) {
  const double u = std::clamp(unit_time, 0.0, 1.0);
  return 30.0 * u * u * (1.0 - u) * (1.0 - u);
}

const char* PhaseName(HomingController::Phase phase) {
  switch (phase) {
    case HomingController::Phase::kNotInitialized:
      return "NOT INITIALIZED";
    case HomingController::Phase::kHoming:
      return "HOMING";
    case HomingController::Phase::kHolding:
      return "HOLDING";
  }
  return "UNKNOWN";
}

}  // namespace greeny
