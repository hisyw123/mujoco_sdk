// Copyright 2026

#ifndef MUJOCO_ROUGHTERRAIN3_RUNTIME_ROBOT_CATALOG_H_
#define MUJOCO_ROUGHTERRAIN3_RUNTIME_ROBOT_CATALOG_H_

#include <string>
#include <vector>

namespace roughterrain3 {

struct RobotProfile {
  std::string name;
  std::string model_path;
  std::string motor_contract_path;
  std::string policy_contract_path;
  std::string trajectory_contract_path;
  std::string default_mode;
};

RobotProfile LoadRobotProfile(const std::string& catalog_path,
                              const std::string& robot_name);
std::vector<std::string> ListRobotProfiles(const std::string& catalog_path);

}  // namespace roughterrain3

#endif  // MUJOCO_ROUGHTERRAIN3_RUNTIME_ROBOT_CATALOG_H_
