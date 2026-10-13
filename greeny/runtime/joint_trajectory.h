// Copyright 2026
#ifndef MUJOCO_GREENY_RUNTIME_JOINT_TRAJECTORY_H_
#define MUJOCO_GREENY_RUNTIME_JOINT_TRAJECTORY_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <mujoco/mujoco.h>
#include "greeny/motor/include/motor_config.h"
#include "greeny/motor/include/shared_memory.h"

namespace greeny {
// Smooth periodic position references through the normal drive/PD boundary.
// q = q_initial + amplitude * (1 - cos(2*pi*t/period)) / 2.
class JointTrajectory {
 public:
  JointTrajectory(const mjModel* model, const MotorConfig& config,
                  const std::string& path) : motors_(config.motors) {
    const auto root = YAML::LoadFile(path);
    if (root["initial_keyframe"]) {
      const auto name = root["initial_keyframe"].as<std::string>();
      keyframe_ = mj_name2id(model, mjOBJ_KEY, name.c_str());
      if (keyframe_ < 0) throw std::runtime_error("missing initial keyframe: " + name);
    }
    const mjtNum* reference = keyframe_ < 0 ? model->qpos0 :
        model->key_qpos + keyframe_ * model->nq;
    for (const auto& motor : motors_) {
      const int j = mj_name2id(model, mjOBJ_JOINT, motor.joint_name.c_str());
      if (j < 0) throw std::runtime_error("missing trajectory joint: " + motor.name);
      initial_[motor.joint_index] = reference[model->jnt_qposadr[j]];
    }
    const auto joints = root["joints"];
    if (!joints || !joints.IsSequence() || joints.size() == 0)
      throw std::runtime_error("trajectory requires a non-empty joints sequence");
    std::unordered_set<std::string> seen;
    for (const auto& node : joints) {
      const auto name = node["motor"].as<std::string>();
      const double amplitude = node["amplitude"].as<double>();
      const double period = node["period"].as<double>();
      if (!seen.insert(name).second || !std::isfinite(amplitude) ||
          !std::isfinite(period) || period <= 0)
        throw std::runtime_error("invalid or duplicate trajectory: " + name);
      int index = -1;
      for (const auto& motor : motors_) {
        if (motor.name != name) continue;
        index = motor.joint_index;
        const int j = mj_name2id(model, mjOBJ_JOINT, motor.joint_name.c_str());
        if (j < 0) throw std::runtime_error("missing trajectory joint: " + name);
        const double q = initial_[index];
        if (model->jnt_limited[j] &&
            (std::min(q, q + amplitude) < model->jnt_range[2*j] ||
             std::max(q, q + amplitude) > model->jnt_range[2*j+1]))
          throw std::runtime_error("trajectory exceeds joint range: " + name);
      }
      if (index < 0) throw std::runtime_error("unknown trajectory motor: " + name);
      tracks_.push_back({index, amplitude, period});
    }
  }
  void ResetData(const mjModel* model, mjData* data) const {
    if (keyframe_ >= 0) mj_resetDataKeyframe(model, data, keyframe_);
    else mj_resetData(model, data);
  }
  void Reset(SharedMemory* state) {
    state->DisableCommand();
    start_ = state->time;
    state->control_state.store(ControlState::kHolding);
  }
  void Update(SharedMemory* state) const {
    state->simulation_base_wrench.fill(0.0);
    for (const auto& spec : motors_) {
      auto& motor = state->motors[spec.joint_index];
      motor.q_target = initial_[spec.joint_index];
      motor.qd_target = motor.torque_target = 0;
      motor.p_gain = spec.p_gain;
      motor.d_gain = spec.d_gain;
    }
    for (const auto& track : tracks_) {
      auto& motor = state->motors[track.index];
      const double omega = 6.283185307179586 / track.period;
      const double phase = omega * (state->time - start_);
      motor.q_target += track.amplitude * (1 - std::cos(phase)) / 2;
      motor.qd_target = track.amplitude * omega * std::sin(phase) / 2;
    }
    state->command_ready.store(true, std::memory_order_release);
  }
 private:
  struct Track { int index; double amplitude; double period; };
  std::vector<MotorSpec> motors_;
  std::vector<Track> tracks_;
  std::array<double, kMaxDof> initial_{};
  double start_ = 0;
  int keyframe_ = -1;
};
}  // namespace greeny
#endif
