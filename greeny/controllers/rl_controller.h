// Copyright 2026

#ifndef MUJOCO_GREENY_CONTROLLERS_RL_CONTROLLER_H_
#define MUJOCO_GREENY_CONTROLLERS_RL_CONTROLLER_H_

#include <array>
#include <deque>
#include <string>
#include <vector>

#include "greeny/controllers/controller.h"
#include "greeny/policy/greeny_policy_layout.h"
#include "greeny/policy/policy_runner.h"

namespace greeny {

struct RlConfig {
  static constexpr int kHistoryFrameSize = 28;

  std::array<double, kJointCount> default_position = {
      0.0, -1.00881, 2.01762, 0.0,
      0.0, 1.00881, -2.01762, 0.0,
  };
  std::array<double, kJointCount> kp = {
      80.0, 80.0, 80.0, 15.0, 80.0, 80.0, 80.0, 15.0,
  };
  std::array<double, kJointCount> kd = {
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
  };
  std::array<float, kJointCount> action_scale = {
      0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 1.0f, 1.0f,
  };
  std::array<int, kJointCount> input_order = {0, 4, 1, 5, 2, 6, 3, 7};
  std::array<int, kJointCount> output_order = {0, 2, 4, 6, 1, 3, 5, 7};

  double policy_period = 0.02;  // 50 Hz
  double transition_duration = 0.5;
  int history_length = 10;
  int history_stride = 3;
  float linear_velocity_scale = 2.0f;
  float angular_velocity_scale = 0.25f;
  float joint_position_scale = 1.0f;
  float action_clip = 100.0f;
  double orientation_support_kp = 400.0;
  double orientation_support_kd = 40.0;

  // Optional morphology transition contract.  When enabled the current
  // observation has four command values (vx, vy, wz, posture alpha), while
  // history remains the same 28-value proprioceptive frame.
  bool posture_transition_enabled = false;
  std::array<double, kJointCount> wheel_position = {
      0.0, 1.22, 1.45, 0.0,
      0.0, -1.22, -1.45, 0.0,
  };
  double posture_transition_duration = 1.0;
};

RlConfig LoadRlConfig(const std::string& path);

class RlController final : public Controller {
 public:
  explicit RlController(PolicyRunner* runner, RlConfig config = {});

  void Reset(SharedMemory* state) override;
  void ResetForPosture(int posture, SharedMemory* state);
  void Update(SharedMemory* state) override;

  // 0 is the configured default pose, 1 is wheel_position.
  void RequestPosture(int target, SharedMemory* state);
  int target_posture() const { return transition_target_; }
  bool supports_posture_transition() const {
    return config_.posture_transition_enabled;
  }
  std::array<double, kJointCount> PoseForPosture(int posture) const;

  const std::vector<float>& observation() const { return observation_; }
  const std::array<float, kJointCount>& raw_action() const { return raw_action_; }
  const std::string& last_error() const { return last_error_; }
  const char* policy_name() const { return runner_->Name(); }

 private:
  std::vector<float> BuildObservation(const SharedMemory& state) const;
  std::vector<float> FlattenHistory() const;
  void RunPolicy(const SharedMemory& state);
  void AdvancePostureTransition(SharedMemory* state);
  std::array<double, kJointCount> ActiveDefaultPosition() const;
  void WriteCommand(SharedMemory* state, double transition_blend) const;
  void SetFault(SharedMemory* state, const std::string& message);

  static double MinimumJerk(double unit_time);

  PolicyRunner* runner_;
  RlConfig config_;
  std::array<float, kJointCount> raw_action_{};
  std::array<double, kJointCount> scaled_action_{};
  std::array<double, kJointCount> policy_target_{};
  std::vector<float> observation_;
  std::deque<std::vector<float>> history_;
  double reset_time_ = 0.0;
  double last_policy_time_ = 0.0;
  int history_counter_ = 0;
  bool initialized_ = false;
  bool engagement_transition_active_ = false;
  double posture_alpha_ = 0.0;
  double transition_start_alpha_ = 0.0;
  double transition_phase_ = 0.0;
  int transition_target_ = 0;
  bool posture_transition_active_ = false;
  std::string last_error_;
};

}  // namespace greeny

#endif  // MUJOCO_GREENY_CONTROLLERS_RL_CONTROLLER_H_
