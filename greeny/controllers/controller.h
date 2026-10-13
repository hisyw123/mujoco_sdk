// Copyright 2026

#ifndef MUJOCO_GREENY_CONTROLLERS_CONTROLLER_H_
#define MUJOCO_GREENY_CONTROLLERS_CONTROLLER_H_

#include "greeny/motor/include/shared_memory.h"

namespace greeny {

class Controller {
 public:
  virtual ~Controller() = default;
  virtual void Reset(SharedMemory* state) = 0;
  virtual void Update(SharedMemory* state) = 0;
};

}  // namespace greeny

#endif  // MUJOCO_GREENY_CONTROLLERS_CONTROLLER_H_
