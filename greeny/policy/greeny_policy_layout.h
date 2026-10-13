// Copyright 2026

#ifndef MUJOCO_GREENY_POLICY_GREENY_POLICY_LAYOUT_H_
#define MUJOCO_GREENY_POLICY_GREENY_POLICY_LAYOUT_H_

#include <array>

#include "greeny/motor/include/shared_memory.h"

namespace greeny {

// This is a policy-plugin contract, not a runtime-wide robot limit.
inline constexpr int kJointCount = 8;

enum class JointIndex : int {
  kLeftHipYaw = 0,
  kLeftHipPitch,
  kLeftKneePitch,
  kLeftWheelPitch,
  kRightHipYaw,
  kRightHipPitch,
  kRightKneePitch,
  kRightWheelPitch,
};

inline constexpr std::array<const char*, kJointCount> kJointNames = {
    "LHY_J", "LHP_J", "LKP_J", "LWP_J",
    "RHY_J", "RHP_J", "RKP_J", "RWP_J",
};

}  // namespace greeny

#endif  // MUJOCO_GREENY_POLICY_GREENY_POLICY_LAYOUT_H_
