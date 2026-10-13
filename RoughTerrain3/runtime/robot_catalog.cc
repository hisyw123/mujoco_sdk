// Copyright 2026

#include "RoughTerrain3/runtime/robot_catalog.h"

#include <filesystem>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace roughterrain3 {
namespace {

std::string ResolvePath(const std::filesystem::path& catalog,
                        const YAML::Node& profile, const char* key) {
  const YAML::Node value = profile[key];
  if (!value || !value.IsScalar()) {
    throw std::runtime_error(std::string("robot profile requires ") + key);
  }
  std::filesystem::path path(value.as<std::string>());
  if (path.is_relative()) {
    path = catalog.parent_path() / path;
  }
  return std::filesystem::absolute(path).lexically_normal().string();
}

std::string ResolveOptionalPath(const std::filesystem::path& catalog,
                                const YAML::Node& profile, const char* key) {
  if (!profile[key]) return {};
  return ResolvePath(catalog, profile, key);
}

YAML::Node LoadRobots(const std::string& catalog_path) {
  const YAML::Node root = YAML::LoadFile(catalog_path);
  const YAML::Node robots = root["robots"];
  if (!robots || !robots.IsMap() || robots.size() == 0) {
    throw std::runtime_error(
        "robot catalog must contain a non-empty robots map");
  }
  return robots;
}

}  // namespace

RobotProfile LoadRobotProfile(const std::string& catalog_path,
                              const std::string& robot_name) {
  const YAML::Node robots = LoadRobots(catalog_path);
  const YAML::Node profile = robots[robot_name];
  if (!profile || !profile.IsMap()) {
    throw std::runtime_error("unknown robot profile: " + robot_name);
  }

  const std::filesystem::path catalog =
      std::filesystem::absolute(catalog_path).lexically_normal();
  return RobotProfile{
      robot_name,
      ResolvePath(catalog, profile, "model"),
      ResolvePath(catalog, profile, "motor_contract"),
      ResolveOptionalPath(catalog, profile, "policy_contract"),
      ResolveOptionalPath(catalog, profile, "trajectory_contract"),
      profile["default_mode"] ? profile["default_mode"].as<std::string>() : "rl",
  };
}

std::vector<std::string> ListRobotProfiles(const std::string& catalog_path) {
  const YAML::Node robots = LoadRobots(catalog_path);
  std::vector<std::string> names;
  names.reserve(robots.size());
  for (const auto& entry : robots) {
    names.push_back(entry.first.as<std::string>());
  }
  return names;
}

}  // namespace roughterrain3
