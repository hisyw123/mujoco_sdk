#include "RoughTerrain3/controllers/rl_controller.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <yaml-cpp/yaml.h>
namespace roughterrain3 {
RlConfig LoadRlConfig(const std::string& path) {
 const auto y=YAML::LoadFile(path); RlConfig c;
 c.checkpoint_sha256=y["checkpoint_sha256"].as<std::string>();
 c.motors=LoadMotorConfig((std::filesystem::path(path).parent_path()/y["motor_contract"].as<std::string>()).string());
 c.policy_period=y["policy_dt"].as<double>(); c.action_clip=y["action_clip"].as<double>();
 if(c.motors.motors.size()!=kMotorCount || c.policy_period!=0.02 || c.action_clip!=100)
  throw std::runtime_error("unsupported RoughTerrain3 control contract");
 if(y["action_names"].size()!=kActionCount || y["action_scales"].size()!=kActionCount)
  throw std::runtime_error("expected 12 named action channels");
 for(int i=0;i<kActionCount;++i){
  if(y["action_names"][i].as<std::string>()!=kActionNames[i]) throw std::runtime_error("action order mismatch");
  c.action_scale[i]=y["action_scales"][i].as<double>();
  if(c.action_scale[i]!=0.25) throw std::runtime_error("unsupported action scale");
 }
 c.base_position=y["base_position"].as<std::array<double,3>>();
 c.base_quaternion=y["base_quaternion"].as<std::array<double,4>>();
 return c;
}
RlController::RlController(PolicyRunner* runner,RlConfig config):runner_(runner),config_(std::move(config)) {
 indices_.fill(-1);
 for(int j=0;j<kActionCount;++j) for(const auto& m:config_.motors.motors)
  if(m.joint_name==kActionNames[j]) indices_[j]=m.joint_index;
}
std::array<double,kMotorCount> RlController::PoseForPosture(int posture) const {
 if(posture!=0) throw std::runtime_error("selected ONNX supports bipedal locomotion only");
 std::array<double,kMotorCount> pose{};
 for(const auto& m:config_.motors.motors) pose.at(m.joint_index)=m.home_position;
 return pose;
}
void RlController::Reset(SharedMemory* s){ResetForPosture(0,s);}
void RlController::ResetForPosture(int posture,SharedMemory* s){
 targets_=PoseForPosture(posture);
 if(s->motor_count.load()!=kMotorCount || config_.motors.motors.size()!=kMotorCount)
  throw std::runtime_error("policy requires all 28 actuator feedback channels");
 for(const auto& m:config_.motors.motors) if(s->motors[m.joint_index].name!=m.name)
  throw std::runtime_error("policy/motor storage order mismatch: "+m.name);
 for(int i:indices_) if(i<0) throw std::runtime_error("missing policy joint");
 raw_action_.fill(0);previous_action_.fill(0); observation_.clear();
 inference_count_=0; next_policy_time_=s->time; initialized_=true; faulted_=false;last_error_.clear();
 s->simulation_base_wrench.fill(0);s->control_state.store(ControlState::kRlControl);
 s->transition_active.store(false);
}
void RlController::RequestPosture(int target,SharedMemory*){
 if(target!=0) std::fprintf(stderr,"[RoughTerrain3] wheel posture unavailable in the selected bipedal ONNX\n");
}
void RlController::Update(SharedMemory* s){
 if(faulted_){s->DisableCommand();return;}
 try{
  if(!initialized_)Reset(s);
  if(!s->motor_data_ready.load() || !s->imu_ready.load())throw std::runtime_error("feedback not ready");
  if(s->time+1e-9>=next_policy_time_){
   observation_.assign(45,0);
   observation_[0]=2*s->lin_vel_target[0];observation_[1]=2*s->lin_vel_target[1];
   observation_[2]=0.25*s->ang_vel_target[0];
   for(int j=0;j<3;++j)observation_[3+j]=0.25*s->gyro[j];
   const auto& q=s->quat;const double n=q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3];
   if(!std::isfinite(n)||std::abs(n-1)>1e-3)throw std::runtime_error("invalid wxyz quaternion");
   observation_[6]=2*(q[0]*q[2]-q[1]*q[3]);
   observation_[7]=-2*(q[2]*q[3]+q[0]*q[1]);
   observation_[8]=2*(q[1]*q[1]+q[2]*q[2])-1;
   const auto nominal=PoseForPosture(0);
   for(int j=0;j<kActionCount;++j){const int i=indices_[j];
    observation_[9+j]=s->motors[i].q-nominal[i];
    observation_[21+j]=s->motors[i].qd*0.05;
    observation_[33+j]=previous_action_[j];
   }
   for(float v:observation_)if(!std::isfinite(v))throw std::runtime_error("nonfinite observation");
   const auto a=runner_->Infer(observation_,{});
   if(a.size()!=kActionCount)throw std::runtime_error("expected 12 actions");
   for(int j=0;j<kActionCount;++j){
    if(!std::isfinite(a[j]))throw std::runtime_error("nonfinite action");
    raw_action_[j]=a[j];previous_action_[j]=std::clamp(a[j],-float(config_.action_clip),float(config_.action_clip));
    targets_[indices_[j]]=nominal[indices_[j]]+config_.action_scale[j]*previous_action_[j];
   }
   ++inference_count_;next_policy_time_+=config_.policy_period;
  }
  for(const auto& spec:config_.motors.motors){auto& m=s->motors[spec.joint_index];
   m.q_target=targets_[spec.joint_index];m.qd_target=0;m.torque_target=0;
   m.p_gain=spec.p_gain;m.d_gain=spec.d_gain;
  }
  s->simulation_base_wrench.fill(0);s->command_ready.store(true);
 }catch(const std::exception& e){last_error_=e.what();faulted_=true;s->DisableCommand();s->control_state.store(ControlState::kFault);}
}
}
