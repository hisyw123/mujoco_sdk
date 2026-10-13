// Copyright 2026

#include "RoughTerrain3/motor/include/mujoco_drive.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace roughterrain3 {

MujocoDrive::MujocoDrive(const mjModel* model, MotorConfig config,
                         SharedMemory* shared_memory)
    : config_(std::move(config)), shared_memory_(shared_memory) {
  if (!model || !shared_memory_) {
    throw std::invalid_argument("MujocoDrive requires model and SharedMemory");
  }
  base_body_id_ = mj_name2id(model, mjOBJ_BODY, "base_link");
  if (base_body_id_ < 0) {
    throw std::runtime_error("missing base body: base_link");
  }
  const int orientation_sensor =
      mj_name2id(model, mjOBJ_SENSOR, "orientation");
  const int gyro_sensor =
      mj_name2id(model, mjOBJ_SENSOR, "angular_velocity");
  if (orientation_sensor >= 0 && model->sensor_dim[orientation_sensor] == 4) {
    orientation_sensor_address_ = model->sensor_adr[orientation_sensor];
  }
  if (gyro_sensor >= 0 && model->sensor_dim[gyro_sensor] == 3) {
    gyro_sensor_address_ = model->sensor_adr[gyro_sensor];
  }

  bindings_.reserve(config_.motors.size());
  for (const MotorSpec& spec : config_.motors) {
    const int joint_id = mj_name2id(model, mjOBJ_JOINT, spec.joint_name.c_str());
    const int actuator_id =
        mj_name2id(model, mjOBJ_ACTUATOR, spec.actuator_name.c_str());
    if (joint_id < 0 || actuator_id < 0) {
      throw std::runtime_error("model has no joint/actuator binding for motor " +
                               spec.name);
    }
    if (model->jnt_type[joint_id] != mjJNT_HINGE &&
        model->jnt_type[joint_id] != mjJNT_SLIDE) {
      throw std::runtime_error(
          "motor joint must be a hinge or slide: " + spec.joint_name);
    }
    bindings_.push_back({spec.joint_index, model->jnt_qposadr[joint_id],
                         model->jnt_dofadr[joint_id], actuator_id, spec.wheel});
    MotorSharedState& motor = shared_memory_->motors[spec.joint_index];
    motor.name = spec.name;
    motor.p_gain = spec.p_gain;
    motor.d_gain = spec.d_gain;
  }
  shared_memory_->motor_count.store(config_.motors.size(),
                                    std::memory_order_release);
}

void MujocoDrive::ReadFeedback(const mjModel* model, const mjData* data) {
  if (!model || !data) {
    throw std::invalid_argument("ReadFeedback requires model and data");
  }
  shared_memory_->time = data->time;
  ++shared_memory_->sequence;
  for (const Binding& binding : bindings_) {
    MotorSharedState& motor = shared_memory_->motors[binding.shared_index];
    motor.q = data->qpos[binding.qpos_address];
    motor.qd = data->qvel[binding.dof_address];
    motor.torque = data->actuator_force[binding.actuator_id];
  }

  const mjtNum* position = data->xpos + 3 * base_body_id_;
  const mjtNum* quaternion = data->xquat + 4 * base_body_id_;
  std::copy_n(position, 3, shared_memory_->base_position.begin());
  if (orientation_sensor_address_ >= 0) {
    std::copy_n(data->sensordata + orientation_sensor_address_, 4,
                shared_memory_->quat.begin());
  } else {
    std::copy_n(quaternion, 4, shared_memory_->quat.begin());
  }

  mjtNum rotation[9];
  mju_quat2Mat(rotation, quaternion);
  shared_memory_->base_rpy[0] = std::atan2(rotation[7], rotation[8]);
  shared_memory_->base_rpy[1] =
      std::asin(std::clamp(-rotation[6], -1.0, 1.0));
  shared_memory_->base_rpy[2] = std::atan2(rotation[3], rotation[0]);

  // mjOBJ_BODY is the inertial frame (ximat), not the base_link frame.
  // XBODY preserves the policy/telemetry body axes and base origin.
  mjtNum velocity[6];
  mj_objectVelocity(model, data, mjOBJ_XBODY, base_body_id_, velocity, 0);
  std::copy_n(velocity, 3, shared_memory_->base_angular_velocity.begin());
  std::copy_n(velocity + 3, 3, shared_memory_->base_linear_velocity.begin());
  mj_objectVelocity(model, data, mjOBJ_XBODY, base_body_id_, velocity, 1);
  std::copy_n(velocity, 3,
              shared_memory_->base_angular_velocity_body.begin());
  std::copy_n(velocity + 3, 3,
              shared_memory_->base_linear_velocity_body.begin());
  if (gyro_sensor_address_ >= 0) {
    std::copy_n(data->sensordata + gyro_sensor_address_, 3,
                shared_memory_->gyro.begin());
  } else {
    shared_memory_->gyro = shared_memory_->base_angular_velocity_body;
  }

  shared_memory_->imu_ready.store(true, std::memory_order_release);
  shared_memory_->motor_data_ready.store(true, std::memory_order_release);
}

void MujocoDrive::WriteCommand(const mjModel* model, mjData* data) const {
  if (!model || !data) {
    throw std::invalid_argument("WriteCommand requires model and data");
  }
  const bool enabled =
      shared_memory_->command_ready.load(std::memory_order_acquire);
  for (const Binding& binding : bindings_) {
    const MotorSharedState& motor =
        shared_memory_->motors[binding.shared_index];
    double effort = 0.0;
    if (enabled) {
      effort = motor.p_gain * (motor.q_target - motor.q) +
               motor.d_gain * (motor.qd_target - motor.qd) +
               motor.torque_target;
      if (binding.wheel) {
        effort = wheel_torque_limiter_.Limit(motor.qd, effort);
      }
    }
    data->ctrl[binding.actuator_id] =
        ClampControl(model, binding.actuator_id, effort);
  }

  mjtNum* wrench = data->xfrc_applied + 6 * base_body_id_;
  for (int i = 0; i < 6; ++i) {
    wrench[i] += enabled ? shared_memory_->simulation_base_wrench[i] : 0.0;
  }
}

void MujocoDrive::ResetPose(
    const mjModel* model, mjData* data,
    const std::vector<double>& joint_positions,
    const std::array<double, 3>& base_position,
    const std::array<double, 4>& base_quaternion) const {
  if (!model || !data || joint_positions.size() != bindings_.size()) {
    throw std::invalid_argument(
        "ResetPose requires model, data, and one position per motor");
  }
  const int key = mj_name2id(model, mjOBJ_KEY, "nominal");
  if (key >= 0) mj_resetDataKeyframe(model, data, key);
  else mj_resetData(model, data);
  const int root_joint_id = mj_name2id(model, mjOBJ_JOINT, "root");
  if (root_joint_id < 0 || model->jnt_type[root_joint_id] != mjJNT_FREE) {
    throw std::runtime_error("policy pose reset requires free joint: root");
  }
  const int root_qpos = model->jnt_qposadr[root_joint_id];
  std::copy(base_position.begin(), base_position.end(), data->qpos + root_qpos);
  std::copy(base_quaternion.begin(), base_quaternion.end(),
            data->qpos + root_qpos + 3);
  for (const Binding& binding : bindings_) {
    data->qpos[binding.qpos_address] =
        joint_positions[static_cast<std::size_t>(binding.shared_index)];
  }
  mju_zero(data->qvel, model->nv);
  mju_zero(data->ctrl, model->nu);
  mju_zero(data->xfrc_applied, 6 * model->nbody);
  shared_memory_->DisableCommand();
  mj_forward(model, data);
}

double MujocoDrive::ClampControl(const mjModel* model, int actuator_id,
                                 double value) {
  if (!model->actuator_ctrllimited[actuator_id]) return value;
  return std::clamp(value, model->actuator_ctrlrange[2 * actuator_id],
                    model->actuator_ctrlrange[2 * actuator_id + 1]);
}

}  // namespace roughterrain3
