#include "RoughTerrain3/controllers/rl_controller.h"
#include "RoughTerrain3/policy/onnx_policy_runner.h"
#include "RoughTerrain3/runtime/motor_handler.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
using namespace roughterrain3;
void Check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void Near(double a,double b,double tol=1e-5){Check(std::abs(a-b)<tol,"numeric mismatch");}
struct Probe final:PolicyRunner {
 std::vector<float> output=std::vector<float>(12,0);
 std::vector<float> Infer(const std::vector<float>&,const std::vector<float>&)override{return output;}
 const char* Name()const override{return "test";}
};
int main(int argc,char** argv){try{
 Check(argc==5,"usage: test config model onnx golden.json");
 auto cfg=LoadRlConfig(argv[1]);
 char error[1024]{};
 std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model(mj_loadXML(argv[2],nullptr,error,1024),mj_deleteModel);
 Check(bool(model),error);
 std::unique_ptr<mjData,decltype(&mj_deleteData)> data(mj_makeData(model.get()),mj_deleteData);
 auto s=std::make_unique<SharedMemory>();MotorHandler motor(model.get(),cfg.motors,s.get());
 Probe p;RlController controller(&p,cfg);
 const auto pose=controller.PoseForPosture(0);
 motor.ResetPose(model.get(),data.get(),{pose.begin(),pose.end()},cfg.base_position,cfg.base_quaternion);
 for(int i=0;i<3;++i){data->qvel[i]=.2*(i+1);data->qvel[3+i]=.1*(i+1);}
 mj_forward(model.get(),data.get());motor.ReadFeedback(model.get(),data.get());
 for(int i=0;i<3;++i){Near(s->base_linear_velocity_body[i],.2*(i+1));Near(s->base_angular_velocity_body[i],.1*(i+1));}
 controller.Reset(s.get());
 s->lin_vel_target={.3,-.2,0};s->ang_vel_target={.4};s->gyro={1,2,3};
 s->quat={std::sqrt(.5),std::sqrt(.5),0,0};
 for(int i=0;i<12;++i){s->motors[16+i].q=pose[16+i]+.01*(i+1);s->motors[16+i].qd=2*(i+1);p.output[i]=i%2?-200:200;}
 controller.Update(s.get());const auto obs=controller.observation();
 Near(obs[0],.6);Near(obs[1],-.4);Near(obs[2],.1);
 Near(obs[3],.25);Near(obs[4],.5);Near(obs[5],.75);
 Near(obs[6],0);Near(obs[7],-1);Near(obs[8],0);
 for(int i=0;i<12;++i){Near(obs[9+i],.01*(i+1));Near(obs[21+i],.1*(i+1));Near(obs[33+i],0);Near(s->motors[16+i].q_target,pose[16+i]+(i%2?-25:25));}
 for(int i=0;i<16;++i)Near(s->motors[i].q_target,pose[i]);
 s->time=.019;controller.Update(s.get());Check(controller.inference_count()==1,"50 Hz scheduling");
 s->time=.02;controller.Update(s.get());Check(controller.inference_count()==2,"50 Hz scheduling");
 for(int i=0;i<12;++i)Near(controller.observation()[33+i],i%2?-100:100);
 motor.WriteCommand(model.get(),data.get());
 bool nonzero=false;for(int i=0;i<model->nu;++i)nonzero|=data->ctrl[i]!=0;Check(nonzero,"enabled output missing");
 motor.disableOutput();controller.Update(s.get());motor.WriteCommand(model.get(),data.get());
 for(int i=0;i<model->nu;++i)Near(data->ctrl[i],0);
 Check(motor.enableOutput(),"normal re-enable failed");motor.latchEstop();Check(!motor.enableOutput(),"estop latch cleared");
 controller.Reset(s.get());controller.Update(s.get());motor.WriteCommand(model.get(),data.get());
 for(int i=0;i<model->nu;++i)Near(data->ctrl[i],0);
 s->time=.04;s->motors[16].q=std::numeric_limits<double>::quiet_NaN();controller.Update(s.get());
 Check(s->control_state.load()==ControlState::kFault&&!s->command_ready.load(),"invalid feedback did not disable outputs");
 OnnxPolicyRunner onnx(argv[3],cfg.checkpoint_sha256);
 bool wrong_checkpoint=false;try{OnnxPolicyRunner bad(argv[3],"wrong");}catch(const std::exception&){wrong_checkpoint=true;}
 Check(wrong_checkpoint,"mismatched deployment config accepted");
 std::ifstream f(argv[4]);nlohmann::json cases;f>>cases;double max_error=0;
 for(const auto& c:cases){auto result=onnx.Infer(c.at("obs").get<std::vector<float>>(),{});for(int i=0;i<12;++i){double expected=c.at("actions")[i];max_error=std::max(max_error,std::abs(result[i]-expected));Near(result[i],expected,1e-4);}}
 bool rejected=false;try{onnx.Infer(std::vector<float>(31,0),{});}catch(const std::exception&){rejected=true;}Check(rejected,"old 8-DOF input accepted");
 std::cout<<"PASS: observation order/scales/gravity, previous clipped actions, 50 Hz, all-motor output gates, latched Estop, nonfinite fault, ONNX parity "<<cases.size()<<" probes max_error="<<max_error<<"\n";
 return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
