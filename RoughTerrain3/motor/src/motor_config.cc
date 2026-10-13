// Copyright 2026

#include "RoughTerrain3/motor/include/motor_config.h"

#include <yaml-cpp/yaml.h>

#include <stdexcept>
#include <unordered_set>

#include "RoughTerrain3/motor/include/shared_memory.h"

namespace roughterrain3 {

MotorConfig LoadMotorConfig(const std::string& path) {
  const YAML::Node root = YAML::LoadFile(path);
  const YAML::Node motors = root["motors"];
  if (!motors || !motors.IsSequence() || motors.size() == 0) {
    throw std::runtime_error("motor config must contain a non-empty motors sequence");
  }
  if (motors.size() > kMaxDof) {
    throw std::runtime_error("motor count exceeds SharedMemory capacity");
  }

  MotorConfig config;
  std::unordered_set<std::string> names;
  std::unordered_set<int> indices;
  config.motors.reserve(motors.size());
  for (std::size_t i = 0; i < motors.size(); ++i) {
    const YAML::Node node = motors[i];
    MotorSpec spec;
    spec.name = node["name"].as<std::string>();
    spec.joint_name = node["joint_name"]
                          ? node["joint_name"].as<std::string>()
                          : spec.name;
    spec.actuator_name = node["actuator_name"]
                             ? node["actuator_name"].as<std::string>()
                             : spec.name;
    spec.joint_index = node["joint_index"]
                           ? node["joint_index"].as<int>()
                           : static_cast<int>(i);
    spec.p_gain = node["p_gain"].as<double>();
    spec.d_gain = node["d_gain"].as<double>();
    spec.home_position = node["home_position"]
                             ? node["home_position"].as<double>()
                             : 0.0;
    spec.continuous = node["continuous"] && node["continuous"].as<bool>();
    spec.wheel = node["wheel"] && node["wheel"].as<bool>();

    if (spec.joint_index < 0 ||
        spec.joint_index >= static_cast<int>(motors.size())) {
      throw std::runtime_error("joint_index must be dense and within motor count");
    }
    if (!names.insert(spec.name).second || !indices.insert(spec.joint_index).second) {
      throw std::runtime_error("duplicate motor name or joint_index in motor config");
    }
    config.motors.push_back(std::move(spec));
  }
  return config;
}

}  // namespace roughterrain3
