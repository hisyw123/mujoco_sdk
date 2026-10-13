// Copyright 2026
// Simulation state only. NOT binary-compatible with RobotPC SharedMemory.hpp.

#ifndef MUJOCO_ROUGHTERRAIN3_MOTOR_INCLUDE_SHARED_MEMORY_H_
#define MUJOCO_ROUGHTERRAIN3_MOTOR_INCLUDE_SHARED_MEMORY_H_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

namespace roughterrain3 {

// The REAL source currently says 26 while its own comment totals 27.  Keep
// simulation capacity; RobotPC integration uses an explicit adapter, never ABI aliasing.
inline constexpr std::size_t kMaxDof = 32;

enum class ControlState : int {
  kBoot,
  kHoming,
  kHolding,
  kRlTransition,
  kRlControl,
  kFault,
};

struct MotorSharedState {
  std::string name;

  double q = 0.0;
  double qd = 0.0;
  double torque = 0.0;

  double q_target = 0.0;
  double qd_target = 0.0;
  double torque_target = 0.0;

  double p_gain = 0.0;
  double d_gain = 0.0;
};

struct FeedbackFrame {
  std::uint64_t sequence = 0;
  std::size_t dof = 0;
  std::array<double, kMaxDof> q{};
  std::array<double, kMaxDof> qd{};
  std::array<double, kMaxDof> torque{};
};

struct CommandFrame {
  std::uint64_t sequence = 0;
  std::size_t dof = 0;
  std::array<double, kMaxDof> q_target{};
  std::array<double, kMaxDof> qd_target{};
  std::array<double, kMaxDof> torque_target{};
  std::array<double, kMaxDof> p_gain{};
  std::array<double, kMaxDof> d_gain{};
};

struct SharedMemory {
  std::atomic<std::size_t> motor_count{0};
  std::array<MotorSharedState, kMaxDof> motors;
  std::atomic<bool> motor_data_ready{false};
  std::atomic<bool> command_ready{false};

  // IMU/operator fields use explicit units for a future hardware adapter.
  std::array<double, 4> quat{1.0, 0.0, 0.0, 0.0};  // w, x, y, z
  std::array<double, 3> gyro{};                     // body frame, rad/s
  std::atomic<bool> imu_ready{false};
  std::atomic<bool> RB_pressed{false};
  std::atomic<bool> LB_pressed{false};
  std::atomic<int> task_mode{0};
  std::array<double, 3> lin_vel_target{};
  std::array<double, 1> ang_vel_target{};

  static constexpr std::size_t kCameraWidth = 800;
  static constexpr std::size_t kCameraHeight = 600;
  static constexpr std::size_t kCameraChannels = 3;
  static constexpr std::size_t kCameraFrameSize =
      kCameraWidth * kCameraHeight * kCameraChannels;
  // Preserve the exact names used by the existing RobotPC camera code.
  static constexpr std::size_t CAMERA_WIDTH = kCameraWidth;
  static constexpr std::size_t CAMERA_HEIGHT = kCameraHeight;
  static constexpr std::size_t CAMERA_CHANNELS = kCameraChannels;
  static constexpr std::size_t CAMERA_FRAME_SIZE = kCameraFrameSize;
  std::array<std::uint8_t, kCameraFrameSize> camera_frame{};
  std::atomic<bool> new_frame_available{false};
  std::mutex camera_mtx;

  // Simulation controller lifecycle. These fields are not RobotPC ABI fields.
  std::atomic<ControlState> control_state{ControlState::kBoot};
  std::atomic<bool> transition_active{false};
  std::atomic<int> transition_target{0};
  double transition_phase = 0.0;
  double posture_command_alpha = 0.0;

  // Timestamp/snapshot metadata supplied by the selected drive backend.
  double time = 0.0;
  std::uint64_t sequence = 0;

  // Base telemetry is needed by locomotion policies.  Hardware fills it from
  // the estimator; MujocoDrive fills it from mjData.
  std::array<double, 3> base_position{};
  std::array<double, 3> base_rpy{};
  std::array<double, 3> base_linear_velocity{};
  std::array<double, 3> base_angular_velocity{};
  std::array<double, 3> base_linear_velocity_body{};
  std::array<double, 3> base_angular_velocity_body{};

  // Simulation-only aid used during pre-policy homing.  REAL backends ignore
  // it; keeping it explicit prevents a simulated support force entering the
  // motor command contract.
  std::array<double, 6> simulation_base_wrench{};

  void DisableCommand() {
    command_ready.store(false, std::memory_order_release);
    simulation_base_wrench.fill(0.0);
  }
};

}  // namespace roughterrain3

#endif  // MUJOCO_ROUGHTERRAIN3_MOTOR_INCLUDE_SHARED_MEMORY_H_
