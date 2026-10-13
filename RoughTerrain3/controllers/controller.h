// Copyright 2026

#ifndef MUJOCO_ROUGHTERRAIN3_CONTROLLERS_CONTROLLER_H_
#define MUJOCO_ROUGHTERRAIN3_CONTROLLERS_CONTROLLER_H_

#include "RoughTerrain3/motor/include/shared_memory.h"

namespace roughterrain3 {

class Controller {
 public:
  virtual ~Controller() = default;
  virtual void Reset(SharedMemory* state) = 0;
  virtual void Update(SharedMemory* state) = 0;
};

}  // namespace roughterrain3

#endif  // MUJOCO_ROUGHTERRAIN3_CONTROLLERS_CONTROLLER_H_
