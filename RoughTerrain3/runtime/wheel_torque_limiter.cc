// Copyright 2026

#include "RoughTerrain3/runtime/wheel_torque_limiter.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace roughterrain3 {

WheelTorqueLimiter::WheelTorqueLimiter() : WheelTorqueLimiter(Config{}) {}

WheelTorqueLimiter::WheelTorqueLimiter(Config config)
    : config_(std::move(config)) {
  if (config_.bus_voltage <= config_.voltage_offset ||
      config_.resistance <= 0.0 || config_.torque_constant <= 0.0 ||
      config_.gear_ratio <= 0.0 || config_.effective_kv <= 0.0 ||
      config_.peak_joint_torque <= 0.0) {
    throw std::invalid_argument("invalid wheel torque limiter configuration");
  }
  torque_voltage_coefficient_ =
      config_.resistance / (config_.torque_constant * config_.gear_ratio);
  velocity_voltage_coefficient_ = config_.gear_ratio / config_.effective_kv;
}

double WheelTorqueLimiter::Limit(double joint_velocity,
                                 double requested_torque) const {
  const double effective_voltage =
      config_.bus_voltage - config_.voltage_offset;
  const double voltage_lower =
      (-effective_voltage - velocity_voltage_coefficient_ * joint_velocity) /
      torque_voltage_coefficient_;
  const double voltage_upper =
      (effective_voltage - velocity_voltage_coefficient_ * joint_velocity) /
      torque_voltage_coefficient_;
  const double lower = std::max(voltage_lower, -config_.peak_joint_torque);
  const double upper = std::min(voltage_upper, config_.peak_joint_torque);
  if (lower <= upper) {
    return std::clamp(requested_torque, lower, upper);
  }
  if (joint_velocity > 0.0) {
    return -config_.peak_joint_torque;
  }
  if (joint_velocity < 0.0) {
    return config_.peak_joint_torque;
  }
  return 0.0;
}

}  // namespace roughterrain3
