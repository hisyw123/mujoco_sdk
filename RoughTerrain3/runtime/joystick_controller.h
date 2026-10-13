// Copyright 2026
// Licensed under the Apache License, Version 2.0.

#ifndef MUJOCO_ROUGHTERRAIN3_RUNTIME_JOYSTICK_CONTROLLER_H_
#define MUJOCO_ROUGHTERRAIN3_RUNTIME_JOYSTICK_CONTROLLER_H_

#include "RoughTerrain3/motor/include/shared_memory.h"

namespace roughterrain3 {

// Device-independent snapshot produced by the GLFW adapter in the app.  Keeping
// GLFW out of this class makes the command mapping usable by the REAL runtime.
struct JoystickSample {
  bool connected = false;
  double forward_axis = 0.0;  // positive is forward
  double yaw_axis = 0.0;      // positive is counter-clockwise
  bool left_bumper = false;
  bool right_bumper = false;
  bool stand_button = false;
  bool wheel_button = false;
  bool engage_button = false;
  bool homing_button = false;
};

struct JoystickEvents {
  bool connected = false;
  bool disconnected = false;
  bool request_stand = false;
  bool request_wheel = false;
  bool request_engage = false;
  bool request_homing = false;
};

class JoystickController {
 public:
  JoystickController(double max_forward_velocity = 1.0,
                     double max_yaw_velocity = 0.5,
                     double deadzone = 0.15);

  // Publishes continuous commands and returns rising-edge button events.
  JoystickEvents Update(const JoystickSample& sample, SharedMemory* state);

  bool connected() const { return connected_; }

 private:
  double ApplyDeadzone(double value) const;

  double max_forward_velocity_;
  double max_yaw_velocity_;
  double deadzone_;
  bool connected_ = false;
  JoystickSample previous_;
};

}  // namespace roughterrain3

#endif  // MUJOCO_ROUGHTERRAIN3_RUNTIME_JOYSTICK_CONTROLLER_H_
