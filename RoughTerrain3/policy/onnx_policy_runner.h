// Copyright 2026

#ifndef MUJOCO_ROUGHTERRAIN3_POLICY_ONNX_POLICY_RUNNER_H_
#define MUJOCO_ROUGHTERRAIN3_POLICY_ONNX_POLICY_RUNNER_H_

#include <memory>
#include <string>

#include "RoughTerrain3/policy/policy_runner.h"

namespace roughterrain3 {

class OnnxPolicyRunner final : public PolicyRunner {
 public:
  explicit OnnxPolicyRunner(const std::string& model_path,
                            const std::string& expected_checkpoint_sha256 = "");
  ~OnnxPolicyRunner() override;

  std::vector<float> Infer(
      const std::vector<float>& observation,
      const std::vector<float>& observation_history) override;
  const char* Name() const override { return name_.c_str(); }

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  std::string name_;
};

}  // namespace roughterrain3

#endif  // MUJOCO_ROUGHTERRAIN3_POLICY_ONNX_POLICY_RUNNER_H_
