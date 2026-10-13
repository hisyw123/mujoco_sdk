// Copyright 2026
// Licensed under the Apache License, Version 2.0.

#include "RoughTerrain3/runtime/joystick_controller.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace roughterrain3 {

JoystickController::JoystickController(double max_forward_velocity,
                                       double max_yaw_velocity,
                                       double deadzone)
    : max_forward_velocity_(max_forward_velocity),
      max_yaw_velocity_(max_yaw_velocity),
      deadzone_(deadzone) {
  if (!(max_forward_velocity_ > 0.0) || !(max_yaw_velocity_ > 0.0)) {
    throw std::invalid_argument("joystick velocity limits must be positive");
  }
  if (deadzone_ < 0.0 || deadzone_ >= 1.0) {
    throw std::invalid_argument("joystick deadzone must be in [0, 1)");
  }
}

double JoystickController::ApplyDeadzone(double value) const {
  value = std::clamp(value, -1.0, 1.0);
  if (std::abs(value) <= deadzone_) {
    return 0.0;
  }
  const double magnitude = (std::abs(value) - deadzone_) / (1.0 - deadzone_);
  return std::copysign(magnitude, value);
}

JoystickEvents JoystickController::Update(const JoystickSample& sample,
                                          SharedMemory* state) {
  if (!state) {
    throw std::invalid_argument("joystick shared state must not be null");
  }

  JoystickEvents events;
  events.connected = sample.connected && !connected_;
  events.disconnected = !sample.connected && connected_;

  if (!sample.connected) {
    // Only clear motion on an actual disconnect.  This leaves keyboard control
    // untouched on systems which never had a gamepad attached.
    if (connected_) {
      state->lin_vel_target.fill(0.0);
      state->ang_vel_target.fill(0.0);
    }
    state->LB_pressed.store(false, std::memory_order_release);
    state->RB_pressed.store(false, std::memory_order_release);
    connected_ = false;
    previous_ = {};
    return events;
  }

  
  // max_forward_velocity_ * ApplyDeadzone(sample.forward_axis);
  state->lin_vel_target[0] =  max_forward_velocity_ *
    std::max(0.0, ApplyDeadzone(sample.forward_axis));
  state->lin_vel_target[1] = 0.0;
  state->lin_vel_target[2] = 0.0;
  state->ang_vel_target[0] =
      max_yaw_velocity_ * ApplyDeadzone(sample.yaw_axis);
  state->LB_pressed.store(sample.left_bumper, std::memory_order_release);
  state->RB_pressed.store(sample.right_bumper, std::memory_order_release);

  events.request_stand = sample.stand_button && !previous_.stand_button;
  events.request_wheel = sample.wheel_button && !previous_.wheel_button;
  events.request_engage = sample.engage_button && !previous_.engage_button;
  events.request_homing = sample.homing_button && !previous_.homing_button;

  connected_ = true;
  previous_ = sample;
  return events;
}

}  // namespace roughterrain3
