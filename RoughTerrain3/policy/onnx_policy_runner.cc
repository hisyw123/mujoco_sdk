#include "RoughTerrain3/policy/onnx_policy_runner.h"
#include "RoughTerrain3/controllers/rl_controller.h"
#include <array>
#include <cmath>
#include <stdexcept>
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
namespace roughterrain3 {
class OnnxPolicyRunner::Impl {
 public:
 Ort::Env environment{ORT_LOGGING_LEVEL_WARNING,"RoughTerrain3"};
 Ort::SessionOptions options;
 Ort::Session session{nullptr};
 Ort::AllocatorWithDefaultOptions allocator;
 explicit Impl(const std::string& path,const std::string& checkpoint){
  options.SetIntraOpNumThreads(1);options.SetInterOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session=Ort::Session(environment,path.c_str(),options);
  if(session.GetInputCount()!=1||session.GetOutputCount()!=1)throw std::runtime_error("ONNX requires exactly obs -> actions");
  auto in=session.GetInputNameAllocated(0,allocator),out=session.GetOutputNameAllocated(0,allocator);
  if(std::string(in.get())!="obs"||std::string(out.get())!="actions")throw std::runtime_error("ONNX input/output names mismatch");
  auto it=session.GetInputTypeInfo(0),ot=session.GetOutputTypeInfo(0);
  auto i=it.GetTensorTypeAndShapeInfo(),o=ot.GetTensorTypeAndShapeInfo();
  if(i.GetShape()!=std::vector<int64_t>{1,45}||o.GetShape()!=std::vector<int64_t>{1,12}||
     i.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||o.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
   throw std::runtime_error("ONNX requires float32 obs[1,45] -> actions[1,12]");
  auto metadata=session.GetModelMetadata();
  auto raw=metadata.LookupCustomMetadataMapAllocated("baseline_contract",allocator);
  if(!raw)throw std::runtime_error("ONNX missing embedded baseline_contract");
  auto c=nlohmann::json::parse(raw.get());
  if(!checkpoint.empty() && c.at("checkpoint").at("sha256")!=checkpoint)
   throw std::runtime_error("ONNX checkpoint does not match saved deployment config");
  if(c.at("format")!="baseline_policy_v1"||c.at("observation_semantics")!="flattened_actor_groups_before_actor_normalizer"||
     c.at("action_semantics")!="raw_deterministic_policy_action"||!c.at("actor_normalization_in_graph").get<bool>()||
     c.at("io_spec").at("actor_keys")!=nlohmann::json::array({"policy"}))
   throw std::runtime_error("unsupported ONNX semantic contract");
  const auto& e=c.at("io_spec").at("environment");
  if(e.at("action_names").size()!=12||e.at("action_scales").size()!=12||e.at("action_clip")!=100.0)
   throw std::runtime_error("ONNX action contract mismatch");
  for(int j=0;j<12;++j)if(e.at("action_names")[j]!=kActionNames[j]||e.at("action_scales")[j]!=0.25)
   throw std::runtime_error("ONNX joint order/scale mismatch");
  const nlohmann::json fields={{"commands",{0,3}},{"angular_velocity",{3,6}},{"gravity",{6,9}},
   {"joint_position",{9,21}},{"joint_velocity",{21,33}},{"actions",{33,45}}};
  if(e.at("observation_fields")!=fields)throw std::runtime_error("ONNX observation layout mismatch");
  if(e.contains("cat_task") && e.at("cat_task").at("step_dt")!=0.02)
   throw std::runtime_error("ONNX control period mismatch");
 }
};
OnnxPolicyRunner::OnnxPolicyRunner(const std::string& path,const std::string& checkpoint):impl_(std::make_unique<Impl>(path,checkpoint)),name_("onnx:"+path){}
OnnxPolicyRunner::~OnnxPolicyRunner()=default;
std::vector<float> OnnxPolicyRunner::Infer(const std::vector<float>& obs,const std::vector<float>& history){
 if(obs.size()!=45||!history.empty())throw std::invalid_argument("expected 45 observations and no history");
 for(float v:obs)if(!std::isfinite(v))throw std::invalid_argument("nonfinite ONNX input");
 const std::array<int64_t,2> shape{1,45};
 auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
 auto input=Ort::Value::CreateTensor<float>(memory,const_cast<float*>(obs.data()),obs.size(),shape.data(),shape.size());
 const char* in="obs";const char* out="actions";
 auto outputs=impl_->session.Run(Ort::RunOptions{nullptr},&in,&input,1,&out,1);
 const float* values=outputs[0].GetTensorData<float>();
 std::vector<float> result(values,values+12);
 for(float v:result)if(!std::isfinite(v))throw std::runtime_error("nonfinite ONNX output");
 return result;
}
}
