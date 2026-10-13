// Copyright 2026

#ifndef MUJOCO_ROUGHTERRAIN3_MOTOR_INCLUDE_MUJOCO_DRIVE_H_
#define MUJOCO_ROUGHTERRAIN3_MOTOR_INCLUDE_MUJOCO_DRIVE_H_

#include <vector>

#include <mujoco/mujoco.h>

#include "RoughTerrain3/motor/include/motor_config.h"
#include "RoughTerrain3/motor/include/shared_memory.h"
#include "RoughTerrain3/runtime/wheel_torque_limiter.h"

namespace roughterrain3 {

// Simulation peer of the REAL GVdrive.  It is the only control component that
// translates between mjData and the common SharedMemory motor contract.
class MujocoDrive {
 public:
  MujocoDrive(const mjModel* model, MotorConfig config, SharedMemory* shared_memory);

  void ReadFeedback(const mjModel* model, const mjData* data);
  void WriteCommand(const mjModel* model, mjData* data) const;

  // Simulation-only state initialization used to reproduce the exact policy
  // reset condition. Joint positions are in SharedMemory/robot order.
  void ResetPose(const mjModel* model, mjData* data,
                 const std::vector<double>& joint_positions,
                 const std::array<double, 3>& base_position,
                 const std::array<double, 4>& base_quaternion) const;

  const MotorConfig& config() const { return config_; }

 private:
  struct Binding {
    int shared_index = -1;
    int qpos_address = -1;
    int dof_address = -1;
    int actuator_id = -1;
    bool wheel = false;
  };

  static double ClampControl(const mjModel* model, int actuator_id, double value);

  MotorConfig config_;
  SharedMemory* shared_memory_ = nullptr;
  std::vector<Binding> bindings_;
  WheelTorqueLimiter wheel_torque_limiter_;
  int base_body_id_ = -1;
  int orientation_sensor_address_ = -1;
  int gyro_sensor_address_ = -1;
};

}  // namespace roughterrain3

#endif  // MUJOCO_ROUGHTERRAIN3_MOTOR_INCLUDE_MUJOCO_DRIVE_H_
