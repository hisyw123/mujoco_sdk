// Copyright 2026

#ifndef MUJOCO_ROUGHTERRAIN3_POLICY_POLICY_RUNNER_H_
#define MUJOCO_ROUGHTERRAIN3_POLICY_POLICY_RUNNER_H_

#include <string>
#include <vector>

namespace roughterrain3 {

class PolicyRunner {
 public:
  virtual ~PolicyRunner() = default;

  virtual std::vector<float> Infer(
      const std::vector<float>& observation,
      const std::vector<float>& observation_history) = 0;
  virtual const char* Name() const = 0;
};

// Dependency-free runner for testing the complete RL control path before a
// model/runtime is selected. It produces a neutral twelve-joint action.
class ZeroPolicyRunner final : public PolicyRunner {
 public:
  std::vector<float> Infer(
      const std::vector<float>& observation,
      const std::vector<float>& observation_history) override;
  const char* Name() const override { return "zero-policy-template"; }
};

}  // namespace roughterrain3

#endif  // MUJOCO_ROUGHTERRAIN3_POLICY_POLICY_RUNNER_H_
