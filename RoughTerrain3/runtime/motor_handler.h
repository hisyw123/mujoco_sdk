#pragma once
#include <atomic>
#include "RoughTerrain3/motor/include/mujoco_drive.h"
namespace roughterrain3 {
// RobotPC layering: upper controller -> MotorHandler -> drive backend.
// This is a simulation adapter, not the 7-channel HRI MotorHandler ABI.
// Only output gates may be called from another thread. Physics/state access
// remains on the simulation thread; future HRI feedback needs snapshots.
class MotorHandler {
 public:
 MotorHandler(const mjModel* model, MotorConfig config, SharedMemory* state)
     : state_(state), drive_(model,std::move(config),state) {}
 void ReadFeedback(const mjModel* model,const mjData* data){drive_.ReadFeedback(model,data);}
 void WriteCommand(const mjModel* model,mjData* data) const {
  if(!outputEnabled())state_->DisableCommand();
  drive_.WriteCommand(model,data);
 }
 void ResetPose(const mjModel* model,mjData* data,const std::vector<double>& q,
                const std::array<double,3>& pos,const std::array<double,4>& quat)const{
  drive_.ResetPose(model,data,q,pos,quat);
 }
 void disableOutput(){enabled_.store(false);}
 bool enableOutput(){if(estop_.load())return false;enabled_.store(true);return true;}
 void latchEstop(){estop_.store(true);enabled_.store(false);}
 bool estopLatched()const{return estop_.load();}
 bool outputEnabled()const{return enabled_.load()&&!estop_.load();}
 private:
 SharedMemory* state_;
 MujocoDrive drive_;
 std::atomic<bool> enabled_{true},estop_{false};
};
}
