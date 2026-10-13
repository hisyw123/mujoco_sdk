// Copyright 2026

#ifndef MUJOCO_ROUGHTERRAIN3_MOTOR_INCLUDE_MOTOR_CONFIG_H_
#define MUJOCO_ROUGHTERRAIN3_MOTOR_INCLUDE_MOTOR_CONFIG_H_

#include <string>
#include <vector>

namespace roughterrain3 {

struct MotorSpec {
  std::string name;
  std::string joint_name;
  std::string actuator_name;
  int joint_index = -1;
  double p_gain = 0.0;
  double d_gain = 0.0;
  double home_position = 0.0;
  bool continuous = false;
  bool wheel = false;
};

struct MotorConfig {
  std::vector<MotorSpec> motors;
};

MotorConfig LoadMotorConfig(const std::string& path);

}  // namespace roughterrain3

#endif  // MUJOCO_ROUGHTERRAIN3_MOTOR_INCLUDE_MOTOR_CONFIG_H_
