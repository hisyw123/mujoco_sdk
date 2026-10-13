#pragma once

// Source-compatible include path for RobotPC/GVdrive code.  New code uses the
// namespaced types; these aliases let the existing REAL source keep its names.
#include "greeny/motor/include/shared_memory.h"

inline constexpr std::size_t MAX_DOF = greeny::kMaxDof;
using MotorSharedState = greeny::MotorSharedState;
using FeedbackFrame = greeny::FeedbackFrame;
using CommandFrame = greeny::CommandFrame;
using SharedMemory = greeny::SharedMemory;
