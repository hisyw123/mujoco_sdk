// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUJOCO_GREENY_RUNTIME_HOMING_CONTROLLER_H_
#define MUJOCO_GREENY_RUNTIME_HOMING_CONTROLLER_H_

#include <vector>

#include "greeny/controllers/controller.h"
#include "greeny/motor/include/motor_config.h"

namespace greeny {

// Configuration matching the last non-RL controller in mujoco_GREENY_deploy.
// Wheel targets are replaced with their positions at Reset(), so homing does
// not command an arbitrary wheel rotation.
struct HomingConfig {
  // Virtual roll/pitch support used while no balancing policy is present.
  // This will be faded out when a balancing controller is added later.
  double orientation_support_kp = 400.0;
  double orientation_support_kd = 40.0;
  double duration = 3.0;
};

class HomingController : public Controller {
 public:
  enum class Phase { kNotInitialized, kHoming, kHolding };

  HomingController(const MotorConfig& motor_config, HomingConfig config = {});

  // Captures the current joint positions and starts a new homing trajectory.
  void Reset(SharedMemory* state) override;

  // Reads feedback and publishes lower-controller targets through SharedMemory.
  void Update(SharedMemory* state) override;

  Phase phase() const { return phase_; }
  double progress(const SharedMemory& state) const;
  std::vector<double> position_errors(const SharedMemory& state) const;
  double max_position_error(const SharedMemory& state) const;
  const HomingConfig& config() const { return config_; }

 private:
  static double MinimumJerk(double unit_time);
  static double MinimumJerkDerivative(double unit_time);

  HomingConfig config_;
  MotorConfig motor_config_;
  std::vector<double> start_position_;
  std::vector<double> target_position_;
  double start_time_ = 0.0;
  Phase phase_ = Phase::kNotInitialized;
};

const char* PhaseName(HomingController::Phase phase);

}  // namespace greeny

#endif  // MUJOCO_GREENY_RUNTIME_HOMING_CONTROLLER_H_
