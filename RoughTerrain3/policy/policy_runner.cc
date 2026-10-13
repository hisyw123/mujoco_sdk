#include "RoughTerrain3/policy/policy_runner.h"
#include <stdexcept>
namespace roughterrain3 {
std::vector<float> ZeroPolicyRunner::Infer(const std::vector<float>& obs,const std::vector<float>& history){
 if(obs.size()!=45||!history.empty())throw std::invalid_argument("expected obs[45], no history");
 return std::vector<float>(12,0);
}
}
