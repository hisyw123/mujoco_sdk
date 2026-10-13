// PyTorch GRU gate order r,z,n. Native float inference; no Python/libtorch runtime.
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <filesystem>
struct Net {
  using V=Eigen::VectorXf;
  using R=Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
  std::vector<int> features;
  std::vector<V> h, bi, bh;
  std::vector<R> wi, wh;
  V xm,xs,ym,ys,bias;
  R head;
  int nJ,hidden,layers;
  static V vec(const YAML::Node& x) {
    auto v=x.as<std::vector<float>>();
    return Eigen::Map<V>(v.data(),v.size());
  }
  static R mat(const YAML::Node& x) {
    auto v=x["data"].as<std::vector<float>>();
    int r=x["shape"][0].as<int>(),c=x["shape"][1].as<int>();
    if(r<=0||c<=0||v.size()!=static_cast<size_t>(r*c)) throw std::runtime_error("bad weight shape");
    return Eigen::Map<R>(v.data(),r,c);
  }
  Net(const std::string& path,int n):nJ(n) {
    std::filesystem::path file(path);
    if(file.extension()==".pt") file=file.parent_path()/"actuator_net_cpp.json";
    auto a=YAML::LoadFile(file.string());
    if(a["format"].as<std::string>()!="raion-gru-v1" || a["n_joints"].as<int>()!=n)
      throw std::runtime_error("native GRU format/joint count mismatch");
    features=a["features"].as<std::vector<int>>(); hidden=a["hidden"].as<int>(); layers=a["layers"].as<int>();
    if(hidden<=0||layers<=0||features.empty()) throw std::runtime_error("bad GRU dimensions");
    for(int f:features) if(f<0||f>2) throw std::runtime_error("bad feature code");
    xm=vec(a["x_mean"]);xs=vec(a["x_std"]);ym=vec(a["y_mean"]);ys=vec(a["y_std"]);
    if(xm.size()!=int(features.size())*n||xs.size()!=xm.size()||ym.size()!=n||ys.size()!=n ||
       (xs.array()<=0).any()||(ys.array()<=0).any()) throw std::runtime_error("bad normalization");
    auto w=a["weights"];
    for(int i=0;i<layers;i++) {
      std::string k="gru.";
      wi.push_back(mat(w[k+"weight_ih_l"+std::to_string(i)]));
      wh.push_back(mat(w[k+"weight_hh_l"+std::to_string(i)]));
      bi.push_back(vec(w[k+"bias_ih_l"+std::to_string(i)]["data"]));
      bh.push_back(vec(w[k+"bias_hh_l"+std::to_string(i)]["data"]));
      if(wi.back().rows()!=3*hidden||wi.back().cols()!=(i?hidden:xm.size())||
         wh.back().rows()!=3*hidden||wh.back().cols()!=hidden||bi.back().size()!=3*hidden||bh.back().size()!=3*hidden)
        throw std::runtime_error("incompatible GRU weights");
      h.push_back(V::Zero(hidden));
    }
    head=mat(w["head.weight"]);bias=vec(w["head.bias"]["data"]);
    if(head.rows()!=n||head.cols()!=hidden||bias.size()!=n) throw std::runtime_error("bad GRU head");
  }
  Eigen::VectorXd predict(const Eigen::Matrix<double,3,Eigen::Dynamic>& s) {
    V x(xm.size());
    for(size_t f=0;f<features.size();f++) for(int j=0;j<nJ;j++) x[f*nJ+j]=s(features[f],j);
    x=(x-xm).array()/xs.array();
    for(int i=0;i<layers;i++) {
      V a=wi[i]*x+bi[i],b=wh[i]*h[i]+bh[i];
      V r=(1.f/(1.f+(-(a.head(hidden)+b.head(hidden)).array()).exp())).matrix();
      V z=(1.f/(1.f+(-(a.segment(hidden,hidden)+b.segment(hidden,hidden)).array()).exp())).matrix();
      V candidate=(a.tail(hidden).array()+r.array()*b.tail(hidden).array()).tanh().matrix();
      h[i]=((1.f-z.array())*candidate.array()+z.array()*h[i].array()).matrix();
      x=h[i];
    }
    V y=((head*x+bias).array()*ys.array()+ym.array()).matrix();
    if(!y.allFinite()) throw std::runtime_error("nonfinite network prediction");
    return y.cast<double>();
  }
};
