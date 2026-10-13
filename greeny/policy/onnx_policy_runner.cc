// Copyright 2026

#include "greeny/policy/onnx_policy_runner.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <onnxruntime_cxx_api.h>

namespace greeny {

class OnnxPolicyRunner::Impl {
 public:
  explicit Impl(const std::string& model_path)
      : environment(ORT_LOGGING_LEVEL_WARNING, "greeny"),
        options(),
        session(nullptr) {
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = Ort::Session(environment, model_path.c_str(), options);

    if (session.GetInputCount() != 2 || session.GetOutputCount() < 1) {
      throw std::runtime_error("GREENY ONNX model must have two inputs and one output");
    }
    for (std::size_t i = 0; i < 2; ++i) {
      auto name = session.GetInputNameAllocated(i, allocator);
      input_name_storage.emplace_back(name.get());
    }
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    output_name_storage = output_name.get();
    for (std::size_t i = 0; i < input_names.size(); ++i) {
      input_names[i] = input_name_storage[i].c_str();
    }
    output_names[0] = output_name_storage.c_str();
  }

  Ort::Env environment;
  Ort::SessionOptions options;
  Ort::Session session;
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> input_name_storage;
  std::string output_name_storage;
  std::array<const char*, 2> input_names{};
  std::array<const char*, 1> output_names{};
};

OnnxPolicyRunner::OnnxPolicyRunner(const std::string& model_path)
    : impl_(std::make_unique<Impl>(model_path)), name_("onnx:" + model_path) {}

OnnxPolicyRunner::~OnnxPolicyRunner() = default;

std::vector<float> OnnxPolicyRunner::Infer(
    const std::vector<float>& observation,
    const std::vector<float>& observation_history) {
  if ((observation.size() != 31 && observation.size() != 32) ||
      observation_history.size() != 280) {
    throw std::invalid_argument(
        "GREENY policy expects observation 31/32 and history 280");
  }

  const std::array<int64_t, 2> observation_shape = {
      1, static_cast<int64_t>(observation.size())};
  const std::array<int64_t, 2> history_shape = {
      1, static_cast<int64_t>(observation_history.size())};
  const Ort::MemoryInfo memory =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // ORT does not mutate input buffers, but its C++ API takes a non-const
  // pointer for externally owned tensors.
  auto* observation_data = const_cast<float*>(observation.data());
  auto* history_data = const_cast<float*>(observation_history.data());
  std::array<Ort::Value, 2> inputs = {
      Ort::Value::CreateTensor<float>(memory, observation_data,
                                      observation.size(), observation_shape.data(),
                                      observation_shape.size()),
      Ort::Value::CreateTensor<float>(memory, history_data,
                                      observation_history.size(), history_shape.data(),
                                      history_shape.size())};

  auto outputs = impl_->session.Run(
      Ort::RunOptions{nullptr}, impl_->input_names.data(), inputs.data(), inputs.size(),
      impl_->output_names.data(), impl_->output_names.size());
  const std::size_t count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
  if (count != 8) {
    throw std::runtime_error("GREENY ONNX model output must contain 8 actions");
  }
  const float* values = outputs[0].GetTensorData<float>();
  return std::vector<float>(values, values + count);
}

}  // namespace greeny
