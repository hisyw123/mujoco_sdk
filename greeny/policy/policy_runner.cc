// Copyright 2026

#include "greeny/policy/policy_runner.h"

#include <stdexcept>

namespace greeny {

std::vector<float> ZeroPolicyRunner::Infer(
    const std::vector<float>& observation,
    const std::vector<float>& observation_history) {
  if ((observation.size() != 31 && observation.size() != 32) ||
      observation_history.size() != 280) {
    throw std::invalid_argument(
        "GREENY policy expects observation 31/32 and history 280");
  }
  return std::vector<float>(8, 0.0f);
}

}  // namespace greeny
