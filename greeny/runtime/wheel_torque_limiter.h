// Copyright 2026

#ifndef MUJOCO_GREENY_RUNTIME_WHEEL_TORQUE_LIMITER_H_
#define MUJOCO_GREENY_RUNTIME_WHEEL_TORQUE_LIMITER_H_

namespace greeny {

// Projects a wheel torque into the motor voltage/torque operating region.
// This belongs to the actuator adapter, not to a learned policy.
class WheelTorqueLimiter {
 public:
  struct Config {
    double bus_voltage = 80.0;
    double voltage_offset = 3.254742547;
    double resistance = 0.14;
    double torque_constant = 0.281;
    double gear_ratio = 7.15;
    double effective_kv = 2.63835;
    double peak_joint_torque = 90.8;
  };

  WheelTorqueLimiter();
  explicit WheelTorqueLimiter(Config config);
  double Limit(double joint_velocity, double requested_torque) const;

 private:
  Config config_;
  double torque_voltage_coefficient_ = 0.0;
  double velocity_voltage_coefficient_ = 0.0;
};

}  // namespace greeny

#endif  // MUJOCO_GREENY_RUNTIME_WHEEL_TORQUE_LIMITER_H_
