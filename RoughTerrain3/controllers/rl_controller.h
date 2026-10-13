// Derived from greeny controller interface; RoughTerrain3 policy contract.
#pragma once
#include <array>
#include <string>
#include <vector>
#include "RoughTerrain3/controllers/controller.h"
#include "RoughTerrain3/motor/include/motor_config.h"
#include "RoughTerrain3/policy/policy_runner.h"
namespace roughterrain3 {
inline constexpr int kActionCount = 12;
inline constexpr int kMotorCount = 28;
inline constexpr std::array<const char*, kActionCount> kActionNames = {
 "LHP_J","LHR_J","LHY_J","LKP_J","LAP_J","LAR_J",
 "RHP_J","RHR_J","RHY_J","RKP_J","RAP_J","RAR_J"};
struct RlConfig {
 MotorConfig motors;
 std::string checkpoint_sha256;
 std::array<double, 3> base_position{0,0,1.2};
 std::array<double, 4> base_quaternion{1,0,0,0};
 std::array<double, kActionCount> action_scale{};
 double policy_period=0.02, action_clip=100;
};
RlConfig LoadRlConfig(const std::string& path);
class RlController final : public Controller {
 public:
 explicit RlController(PolicyRunner* runner, RlConfig config = {});
 void Reset(SharedMemory* state) override;
 void ResetForPosture(int posture, SharedMemory* state);
 void Update(SharedMemory* state) override;
 void RequestPosture(int target, SharedMemory* state);
 int target_posture() const { return 0; }
 bool supports_posture_transition() const { return false; }
 std::array<double,kMotorCount> PoseForPosture(int posture) const;
 const std::vector<float>& observation() const { return observation_; }
 const std::array<float,kActionCount>& raw_action() const { return raw_action_; }
 const std::string& last_error() const { return last_error_; }
 const char* policy_name() const { return runner_->Name(); }
 std::uint64_t inference_count() const { return inference_count_; }
 private:
 PolicyRunner* runner_;
 RlConfig config_;
 std::array<int,kActionCount> indices_{};
 std::array<float,kActionCount> raw_action_{}, previous_action_{};
 std::array<double,kMotorCount> targets_{};
 std::vector<float> observation_;
 double next_policy_time_=0;
 bool initialized_=false, faulted_=false;
 std::uint64_t inference_count_=0;
 std::string last_error_;
};
}
