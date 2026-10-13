// ActuatorNet-augmented MuJoCo simulator for a fixed-base articulated system.
//
//   augmented_sim replay  <log.csv> <out.csv> --urdf robot.urdf [opts]
//       one-step transitions from every logged state -> training targets (err_tau)
//   augmented_sim rollout <log.csv> <out.csv> --urdf robot.urdf [opts]
//       free-running simulation following the logged position targets
//
// opts: --model net_jit.pt  --ff-column err_tau  --kp 100  --kd 1  --stride N
//       --steps N  --fixed-dt D (rollout only)
//
// --stride N decouples the prediction rate from the log rate. replay then measures each transition
// over N ticks instead of one, and rollout calls the network every N ticks and holds its torque in
// between, which is how an RL simulator would run it: physics at the fine step, actuator net at the
// control rate.
//
// --steps N sets how many physics steps make up one prediction interval. The default follows the
// log, one step per logged tick, so physics runs finer than the network. --steps 1 instead advances
// the whole interval in a single step with the target held, which is what a simulator whose physics
// rate equals its control rate does; training with the same setting keeps the targets matched to
// the simulator the network will run in.
//
// --fixed-dt D runs the rollout on its own uniform clock of step D instead of following the log's
// timestamps, which is what a training simulator does. Position targets are zero-order held from the
// log and the reference state is interpolated, so the comparison stays honest; --stride then counts
// D-sized steps, not log ticks.
//
// Input CSV: elapsedTime, gc_j, gv_j, ptarget_j and optionally gf_j, for j = 0..n-1 in joint order.
// The joint count comes from the file and must match the URDF's degrees of freedom.
// By default each integration step uses that row's own dt, so a jittering log is replayed on its
// true time base instead of an averaged one; --steps and --fixed-dt override that.
#include <mujoco/mujoco.h>
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>
#include <random>
#include "net.h"
#include "viewer.h"

namespace {

// feature codes; must match FEATURES in scripts/common.py
enum Feature : int { kPosErr = 0, kVel = 1, kTorque = 2 };
using Step = Eigen::Matrix<double, 3, Eigen::Dynamic>;  // rows: posErr, vel, torque at one sample

struct Opt {
  std::string jointConfig;
  bool viewer=false,realtime=false;
  double speed=1,seconds=20,dt=.001,amplitude=.05,frequency=.5,scale=1,damping=0,noiseQ=0,noiseV=0;
  int seed=0;
  std::string mode, in, out, urdf, model, ffCol;  // ffCol: feed <ffCol>_j from the input CSV as feedforward
  double kp = 100., kd = 1.;
  int stride = 1, offset = 0, steps = 0;   // steps: physics steps per prediction, 0 = one per log tick
  double fixedDt = 0.;   // <= 0: follow the log's own timestamps
};

Opt parseArgs(int argc, char** argv) {
  const char* usage =
      "usage: augmented_sim <replay|rollout> <in.csv> <out.csv> --urdf robot.urdf [--model m.pt] "
      "[--ff-column err_tau] [--kp 100] [--kd 1] [--stride 1] [--offset 0] [--steps N] [--fixed-dt D]";
  if (argc < 4) throw std::runtime_error(usage);
  Opt o;
  o.mode = argv[1]; o.in = argv[2]; o.out = argv[3];
  if (o.mode != "replay" && o.mode != "rollout" && o.mode != "collect" && o.mode != "inspect") throw std::runtime_error(usage);
  for (int i = 4; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&] { if (++i >= argc) throw std::runtime_error(usage); return std::string(argv[i]); };
    if (a == "--urdf") o.urdf = next();
    else if (a == "--joint-config") o.jointConfig=next();
    else if (a == "--viewer") o.viewer=true;
    else if (a == "--realtime") o.realtime=true;
    else if (a == "--viewer-speed") o.speed=std::stod(next());
    else if (a == "--seconds") o.seconds=std::stod(next());
    else if (a == "--dt") o.dt=std::stod(next());
    else if (a == "--amplitude") o.amplitude=std::stod(next());
    else if (a == "--frequency") o.frequency=std::stod(next());
    else if (a == "--torque-scale") o.scale=std::stod(next());
    else if (a == "--extra-damping") o.damping=std::stod(next());
    else if (a == "--noise-q") o.noiseQ=std::stod(next());
    else if (a == "--noise-v") o.noiseV=std::stod(next());
    else if (a == "--seed") o.seed=std::stoi(next());
    else if (a == "--model") o.model = next();
    else if (a == "--ff-column") o.ffCol = next();
    else if (a == "--stride") o.stride = std::stoi(next());
    else if (a == "--offset") o.offset = std::stoi(next());
    else if (a == "--fixed-dt") o.fixedDt = std::stod(next());
    else if (a == "--steps") o.steps = std::stoi(next());
    else if (a == "--kp") o.kp = std::stod(next());
    else if (a == "--kd") o.kd = std::stod(next());
    else throw std::runtime_error("unknown option " + a + "\n" + usage);
  }
  if (o.urdf.empty()) throw std::runtime_error("--urdf is required\n" + std::string(usage));
  for(double v:{o.kp,o.kd,o.speed,o.seconds,o.dt,o.amplitude,o.frequency,o.scale,o.damping,o.noiseQ,o.noiseV,o.fixedDt})
    if(!std::isfinite(v)) throw std::runtime_error("nonfinite option");
  if(o.kp<0||o.kd<0||o.speed<=0||o.dt<=0||o.seconds<o.dt*2||o.noiseQ<0||o.noiseV<0||o.damping<0)
    throw std::runtime_error("invalid gains/time/noise parameters");
  if(o.viewer && o.mode=="replay") throw std::runtime_error("viewer supports collect/rollout only");
  if (o.stride < 1) throw std::runtime_error("--stride must be >= 1");
  if (o.offset < 0 || o.offset >= o.stride) throw std::runtime_error("--offset must be in [0, stride)");
  if (o.steps < 0) throw std::runtime_error("--steps must be >= 1, or 0 to follow the log");
  if (o.steps > 0 && o.fixedDt > 0.)
    throw std::runtime_error("--steps and --fixed-dt both set the step size; use one");
  if (o.fixedDt < 0.) throw std::runtime_error("--fixed-dt must be > 0");
  if (o.fixedDt > 0. && o.mode == "replay")
    throw std::runtime_error("--fixed-dt applies to rollout only; training targets use the log's own timing");
  if (o.fixedDt > 0. && !o.ffCol.empty())
    throw std::runtime_error("--ff-column needs the log's own timing; drop --fixed-dt");
  return o;
}

struct Table {
  std::vector<std::string> names;
  std::map<std::string, int> col;
  std::vector<std::vector<double>> rows;
  int nJ = 0;                         // joints, counted from the gc_* columns
  bool has(const std::string& n) const { return col.count(n) > 0; }
  int c(const std::string& n) const {
    auto it = col.find(n);
    if (it == col.end()) throw std::runtime_error("missing column " + n);
    return it->second;
  }
  double t(size_t row) const { return rows[row][c("elapsedTime")]; }
  double dt(size_t row) const { return t(row + 1) - t(row); }
  Eigen::VectorXd vec(size_t row, const std::string& prefix) const {
    Eigen::VectorXd v(nJ);
    for (int j = 0; j < nJ; ++j) v[j] = rows[row][c(prefix + std::to_string(j))];
    return v;
  }
};

Table readCsv(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  Table t;
  std::string line, cell;
  std::getline(f, line);
  for (std::stringstream ss(line); std::getline(ss, cell, ',');) {
    t.col[cell] = static_cast<int>(t.names.size());
    t.names.push_back(cell);
  }
  while (t.has("gc_" + std::to_string(t.nJ))) t.nJ++;
  if (t.nJ == 0) throw std::runtime_error("no gc_0.. columns in " + path);
  for (const auto& p : {"gv_", "ptarget_"})
    for (int j = 0; j < t.nJ; ++j) t.c(p + std::to_string(j));   // throws if a block is short
  const int te = t.c("elapsedTime");
  double lastT = -1.;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::vector<double> r;
    r.reserve(t.names.size());
    for (std::stringstream ss(line); std::getline(ss, cell, ',');) r.push_back(std::stod(cell));
    if (r.size() != t.names.size()) throw std::runtime_error("ragged row in " + path);
    // some loggers append a row per control loop pass, not per tick; rows repeating elapsedTime are
    // stale re-logs of the same instant, so keep the first of each group
    if (r[te] == lastT) continue;
    if(!std::all_of(r.begin(),r.end(),[](double x){return std::isfinite(x);}) || (!t.rows.empty()&&r[te]<lastT))
      throw std::runtime_error("nonfinite data or non-increasing time");
    lastT = r[te];
    t.rows.push_back(std::move(r));
  }
  if (t.rows.size() < 2) throw std::runtime_error("need at least 2 rows");
  return t;
}

// GRU over the whole joint vector: every joint's history feeds one recurrent state and all joint
// torque errors come out together, so inter-joint coupling is available to the model. The state is
// carried across steps here, exactly as the trainer rolls it forward within a window.
// Native Eigen GRU in net.h.

struct Sim {
  struct ModelDeleter { void operator()(mjModel* p) const { mj_deleteModel(p); } };
  struct DataDeleter { void operator()(mjData* p) const { mj_deleteData(p); } };
  std::unique_ptr<mjModel, ModelDeleter> model;
  std::unique_ptr<mjData, DataDeleter> data;
  int nJ = 0;
  Eigen::VectorXd kp, kd;
  std::vector<int> qa,va;
  std::vector<std::string> names;
  std::unique_ptr<Viewer> view;

  Sim(const Opt& o, int nJoints) : nJ(nJoints) {
    std::ifstream source(o.urdf);
    std::stringstream contents;contents<<source.rdbuf();
    if(contents.str().find("<mimic")!=std::string::npos) throw std::runtime_error("raw URDF mimic unsupported; use an explicit stand model");
    char error[1024] = {};
    model.reset(mj_loadXML(o.urdf.c_str(), nullptr, error, sizeof(error)));
    if (!model) throw std::runtime_error("cannot load URDF " + o.urdf + ": " + error);
    if(nJ<0) nJ=model->nv;
    if (model->nq != nJ || model->nv != nJ)
      throw std::runtime_error("URDF has nq=" + std::to_string(model->nq) +
                               ", nv=" + std::to_string(model->nv) + ", but the log has " +
                               std::to_string(nJ) +
                               " joints. Only a fixed base with 1-DoF joints is supported.");
    for (int j = 0; j < model->njnt; ++j)
      if (model->jnt_type[j] != mjJNT_HINGE && model->jnt_type[j] != mjJNT_SLIDE)
        throw std::runtime_error("only hinge/revolute and slide/prismatic joints are supported");
    if(model->neq||model->ntendon) throw std::runtime_error("use independent fixed-base model; equality/tendon unsupported");
    model->opt.integrator=mjINT_EULER;
    model->opt.disableflags|=mjDSBL_CONTACT|mjDSBL_ACTUATION;
    data.reset(mj_makeData(model.get()));
    if (!data) throw std::runtime_error("MuJoCo could not allocate simulation data");
    kp = Eigen::VectorXd::Constant(nJ, o.kp);
    kd = Eigen::VectorXd::Constant(nJ, o.kd);
    for(int j=0;j<nJ;j++) {
      const char* name=mj_id2name(model.get(),mjOBJ_JOINT,j);
      if(!name) throw std::runtime_error("joint must be named");
      names.push_back(name);qa.push_back(model->jnt_qposadr[j]);va.push_back(model->jnt_dofadr[j]);
    }
    if(!o.jointConfig.empty()) {
      auto joints=YAML::LoadFile(o.jointConfig)["joints"];
      if(int(joints.size())!=nJ) throw std::runtime_error("joint-config size mismatch");
      std::vector<int> used;
      for(int j=0;j<nJ;j++) {
        names[j]=joints[j]["name"].as<std::string>();
        int id=mj_name2id(model.get(),mjOBJ_JOINT,names[j].c_str());
        if(id<0||std::find(used.begin(),used.end(),id)!=used.end()) throw std::runtime_error("invalid/duplicate joint name");
        used.push_back(id);qa[j]=model->jnt_qposadr[id];va[j]=model->jnt_dofadr[id];
        kp[j]=joints[j]["kp"].as<double>();kd[j]=joints[j]["kd"].as<double>();
      }
    }
    if(!kp.allFinite()||!kd.allFinite()||(kp.array()<0).any()||(kd.array()<0).any()) throw std::runtime_error("invalid gains");
    view=std::make_unique<Viewer>(model.get(),o.viewer,o.realtime,o.speed);
    std::cout << nJ << " joints, MuJoCo " << mj_versionString()
              << ", kp=" << o.kp << " kd=" << o.kd << '\n';
  }

  void setState(const Eigen::VectorXd& q, const Eigen::VectorXd& u) {
    mj_resetData(model.get(),data.get());
    for(int j=0;j<nJ;j++){data->qpos[qa[j]]=q[j];data->qvel[va[j]]=u[j];}
    mj_forward(model.get(), data.get());
  }

  void getState(Eigen::VectorXd& q, Eigen::VectorXd& u) const {
    q.resize(nJ);u.resize(nJ);
    for(int j=0;j<nJ;j++){q[j]=data->qpos[qa[j]];u[j]=data->qvel[va[j]];}
  }

  Eigen::MatrixXd massMatrix() const {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dense(nJ, nJ);
    mj_fullM(model.get(), data.get(), dense.data());
    Eigen::MatrixXd ordered(nJ,nJ);
    for(int i=0;i<nJ;i++) for(int j=0;j<nJ;j++) ordered(i,j)=dense(va[i],va[j]);
    return ordered;
  }

  void step(double dt, const Eigen::VectorXd& pTarget, const Eigen::VectorXd& ff) {
    Eigen::VectorXd q,u;getState(q,u);
    torqueStep(dt,kp.cwiseProduct(pTarget-q)-kd.cwiseProduct(u)+ff);
  }
  void torqueStep(double dt,const Eigen::VectorXd& tau) {
    if(!std::isfinite(dt)||dt<=0||!tau.allFinite()) throw std::runtime_error("invalid step/torque");
    model->opt.timestep = dt;
    for(int j=0;j<nJ;j++) data->qfrc_applied[va[j]]=tau[j];
    double before=data->time;
    int warns[mjNWARNING];for(int k=0;k<mjNWARNING;k++) warns[k]=data->warning[k].number;
    mj_step(model.get(), data.get());
    for(int k=0;k<mjNWARNING;k++) if(data->warning[k].number>warns[k]) throw std::runtime_error("MuJoCo warning; unstable settings");
    if(std::abs(data->time-before-dt)>1e-9) throw std::runtime_error("MuJoCo reset detected");
    view->update(model.get(),data.get());
  }
};

struct Csv {
  std::ofstream f;
  int nJ;
  Csv(const std::string& path, int nJoints) : f(path), nJ(nJoints) {
    if (!f) throw std::runtime_error("cannot write " + path);
    f << std::setprecision(17);
  }
  void header(const std::vector<std::string>& blocks) {
    f << "elapsedTime";
    for (auto& b : blocks) for (int j = 0; j < nJ; ++j) f << ',' << b << '_' << j;
    f << '\n';
  }
  void row(double t, const std::vector<const Eigen::VectorXd*>& blocks) {
    f << t;
    for (auto* b : blocks) for (int j = 0; j < nJ; ++j) f << ',' << (*b)[j];
    f << '\n';
  }
};

void report(const std::string& what, const Eigen::VectorXd& sumSq, size_t n) {
  std::cout << what << " RMS per joint:";
  for (int j = 0; j < sumSq.size(); ++j) std::cout << ' ' << std::sqrt(sumSq[j] / n);
  std::cout << "  | all: " << std::sqrt(sumSq.sum() / (n * sumSq.size())) << '\n';
}

Step makeStep(const Eigen::VectorXd& posErr, const Eigen::VectorXd& vel, const Eigen::VectorXd& tau) {
  Step s(3, posErr.size());
  s.row(kPosErr) = posErr.transpose();
  s.row(kVel) = vel.transpose();
  s.row(kTorque) = tau.transpose();
  return s;
}

// One output row per prediction interval: the state and target at the interval start, the state the
// nominal model reaches at the interval end, and the torque that would have closed the gap.
void replay(const Opt& o, const Table& in) {
  const int nJ = in.nJ;
  const size_t S = static_cast<size_t>(o.stride);
  std::unique_ptr<Net> net = o.model.empty() ? nullptr : std::make_unique<Net>(o.model, nJ);
  Sim sim(o, nJ);
  Csv out(o.out, nJ);
  out.header({"gc", "gv", "ptarget", "gf", "sim_gc", "sim_gv", "err_gc", "err_gv", "err_tau", "ff"});
  // --offset shifts which ticks the intervals start on. At stride N the log holds N interleaved
  // sequences at the prediction rate; training on all of them recovers the samples decimation drops.
  const size_t off = static_cast<size_t>(o.offset);
  if(off>=in.rows.size()-1) throw std::runtime_error("offset leaves no complete interval");
  const size_t nInt = (in.rows.size() - 1 - off) / S;
  if (nInt == 0) throw std::runtime_error("--stride is larger than the log");
  Eigen::VectorXd qs(nJ), us(nJ), ff = Eigen::VectorXd::Zero(nJ), sqQ = ff, sqU = ff;
  for (size_t k = 0; k < nInt; ++k) {
    const size_t t0 = off + k * S;
    const Eigen::VectorXd q = in.vec(t0, "gc_"), u = in.vec(t0, "gv_"), p = in.vec(t0, "ptarget_");
    const Eigen::VectorXd tau = in.has("gf_0") ? in.vec(t0, "gf_")
        : Eigen::VectorXd(sim.kp.cwiseProduct(p - q) - sim.kd.cwiseProduct(u));
    if (net) ff = net->predict(makeStep(p - q, u, tau));
    if (!o.ffCol.empty()) ff = in.vec(t0, o.ffCol + "_");
    sim.setState(q, u);
    const Eigen::MatrixXd M = sim.massMatrix();
    // the feedforward torque is held across the interval; the PD target still follows the log tick
    // by tick, because that is what the real controller did
    const double T = in.t(t0 + S) - in.t(t0);
    if (o.steps > 0) {
      // one interval, o.steps equal physics steps, target held: the simulator the net will run in
      for (int i = 0; i < o.steps; ++i) sim.step(T / o.steps, p, ff);
    } else {
      // physics finer than the network: follow the log tick by tick, target and all
      for (size_t i = 0; i < S; ++i) sim.step(in.dt(t0 + i), in.vec(t0 + i, "ptarget_"), ff);
    }
    sim.getState(qs, us);
    const Eigen::VectorXd errQ = in.vec(t0 + S, "gc_") - qs, errU = in.vec(t0 + S, "gv_") - us;
    // Torque that would have produced the measured velocity change over this interval. tau = M*a + h,
    // but both transitions start from the same state, so h(q,u) is identical in each and cancels in
    // the difference; only the mass matrix is left. Measured, see out/report.
    const Eigen::VectorXd errTau = M * errU / T;
    sqQ += errQ.cwiseAbs2(); sqU += errU.cwiseAbs2();
    out.row(in.t(t0), {&q, &u, &p, &tau, &qs, &us, &errQ, &errU, &errTau, &ff});
  }
  report("per-interval err_gc", sqQ, nInt);
  report("per-interval err_gv", sqU, nInt);
}

// One entry per simulator step: when it happens, how long it lasts, the reference state to compare
// against and the position target to drive with. Following the log means one entry per row; a fixed
// timestep means a uniform clock with the reference interpolated onto it.
struct Tick {
  double t, dt;
  Eigen::VectorXd q, u, p;
};

// the log's state at an arbitrary time, linearly interpolated; idx is a forward-moving hint
void interpAt(const Table& in, double tau, size_t& idx, Eigen::VectorXd& q, Eigen::VectorXd& u) {
  const size_t n = in.rows.size();
  while (idx + 1 < n && in.t(idx + 1) <= tau) ++idx;
  const size_t nxt = std::min(idx + 1, n - 1);
  const double span = in.t(nxt) - in.t(idx);
  const double a = span > 0. ? (tau - in.t(idx)) / span : 0.;
  q = (1. - a) * in.vec(idx, "gc_") + a * in.vec(nxt, "gc_");
  u = (1. - a) * in.vec(idx, "gv_") + a * in.vec(nxt, "gv_");
}

struct Plan {
  std::vector<Tick> ticks;
  size_t predEvery = 1;   // network runs every this many ticks
};

Plan schedule(const Opt& o, const Table& in) {
  Plan plan;
  const size_t n = in.rows.size();
  const size_t S = static_cast<size_t>(o.stride);

  if (o.fixedDt > 0.) {                       // uniform clock of its own
    const double t0 = in.t(0), tEnd = in.t(n - 1);
    size_t idx = 0;
    Eigen::VectorXd q, u;
    for (double tau = t0; tau <= tEnd + 1e-12; tau += o.fixedDt) {
      interpAt(in, tau, idx, q, u);
      plan.ticks.push_back({tau, o.fixedDt, q, u, in.vec(idx, "ptarget_")});
    }
    plan.predEvery = S;
    std::cout << "fixed timestep " << o.fixedDt << " s (" << 1. / o.fixedDt << " Hz), "
              << plan.ticks.size() << " steps, network every " << S << " -> "
              << 1. / o.fixedDt / S << " Hz\n";
    return plan;
  }

  if (o.steps > 0) {                          // log's interval durations, o.steps physics steps each
    const size_t nInt = (n - 1) / S;
    if (nInt == 0) throw std::runtime_error("--stride is larger than the log");
    size_t idx = 0;
    Eigen::VectorXd q, u;
    for (size_t k = 0; k < nInt; ++k) {
      const size_t t0 = k * S;
      const double T = in.t(t0 + S) - in.t(t0), d = T / o.steps;
      const Eigen::VectorXd p = in.vec(t0, "ptarget_");   // held across the interval
      for (int i = 0; i < o.steps; ++i) {
        const double tau = in.t(t0) + i * d;
        interpAt(in, tau, idx, q, u);
        plan.ticks.push_back({tau, d, q, u, p});
      }
    }
    const size_t last = nInt * S;
    plan.ticks.push_back({in.t(last), 0., in.vec(last, "gc_"), in.vec(last, "gv_"),
                          in.vec(last, "ptarget_")});
    plan.predEvery = static_cast<size_t>(o.steps);
    std::cout << o.steps << " physics step(s) per prediction over the log's intervals, "
              << plan.ticks.size() << " steps\n";
    return plan;
  }

  plan.ticks.reserve(n);                      // follow the log tick by tick
  for (size_t t = 0; t < n; ++t)
    plan.ticks.push_back({in.t(t), t + 1 < n ? in.dt(t) : 0., in.vec(t, "gc_"), in.vec(t, "gv_"),
                          in.vec(t, "ptarget_")});
  plan.predEvery = S;
  return plan;
}

void rollout(const Opt& o, const Table& in) {
  const int nJ = in.nJ;
  std::unique_ptr<Net> net = o.model.empty() ? nullptr : std::make_unique<Net>(o.model, nJ);
  Sim sim(o, nJ);
  Csv out(o.out, nJ);
  out.header({"gc", "gv", "ptarget", "sim_gc", "sim_gv", "ff"});
  const Plan plan = schedule(o, in);
  const std::vector<Tick>& ticks = plan.ticks;
  const size_t n = ticks.size();
  const size_t S = plan.predEvery;
  Eigen::VectorXd qs(nJ), us(nJ), ff = Eigen::VectorXd::Zero(nJ), sqQ = ff, sqU = ff;
  sim.setState(ticks[0].q, ticks[0].u);
  sim.data->time=ticks[0].t;sim.view->start(ticks[0].t);
  for (size_t t = 0; t < n; ++t) {
    sim.getState(qs, us);
    const Tick& k = ticks[t];
    // the network runs once per interval and its torque is held in between; physics runs every step
    if (t % S == 0 && t + 1 < n) {
      // features come from the sim's own state here, unlike replay
      const Eigen::VectorXd tau = sim.kp.cwiseProduct(k.p - qs) - sim.kd.cwiseProduct(us) + ff;
      if (net) ff = net->predict(makeStep(k.p - qs, us, tau));
      // one schedule entry per log row by default, but --steps decouples them: entry t belongs to
      // prediction interval t / S, which starts at log row (t / S) * stride
      if (!o.ffCol.empty()) ff = in.vec((t / S) * static_cast<size_t>(o.stride), o.ffCol + "_");
    }
    sqQ += (k.q - qs).cwiseAbs2(); sqU += (k.u - us).cwiseAbs2();
    out.row(k.t, {&k.q, &k.u, &k.p, &qs, &us, &ff});
    if (t + 1 < n) sim.step(k.dt, k.p, ff);
  }
  report("rollout gc error", sqQ, n);
  report("rollout gv error", sqU, n);
}

void inspect(const Opt& o) {
  Sim s(o,-1);
  std::ofstream f(o.out);
  f<<std::setprecision(17)<<"{\"backend\":\"mujoco_cpp_explicit_pd_v1\",\"mujoco\":"<<std::quoted(mj_versionString())
   <<",\"joint_names\":[";
  for(int j=0;j<s.nJ;j++)f<<(j?",":"")<<std::quoted(s.names[j]);
  f<<"],\"kp\":[";
  for(int j=0;j<s.nJ;j++)f<<(j?",":"")<<s.kp[j];
  f<<"],\"kd\":[";
  for(int j=0;j<s.nJ;j++)f<<(j?",":"")<<s.kd[j];
  f<<"],\"integrator\":\"Euler\",\"contacts\":false,\"native_actuators\":false}\n";
  mj_saveModel(s.model.get(),(o.out+".mjb").c_str(),nullptr,0);
}

void collect(const Opt& o) {
  Sim s(o,-1);
  int n=s.nJ;
  long ticks=std::lround(o.seconds/o.dt);
  if(std::abs(ticks*o.dt-o.seconds)>1e-9)throw std::runtime_error("seconds must be an integer multiple of dt");
  Eigen::VectorXd center(n),amp=Eigen::VectorXd::Constant(n,std::abs(o.amplitude)),phase(n),q(n),v(n),p(n);
  if(s.model->nkey) mj_resetDataKeyframe(s.model.get(),s.data.get(),0);
  s.getState(center,v);
  // Keep trajectories within joint limits; qpos from first keyframe when supplied.
  for(int j=0;j<n;j++){
    int id=mj_name2id(s.model.get(),mjOBJ_JOINT,s.names[j].c_str());
    if(s.model->jnt_limited[id]){
      double lo=s.model->jnt_range[2*id],hi=s.model->jnt_range[2*id+1],margin=.1*(hi-lo);
      center[j]=std::clamp(center[j],lo+margin,hi-margin);
      amp[j]=std::min(amp[j],.9*std::min(center[j]-lo,hi-center[j]));
    }
  }
  std::mt19937 trajectory(o.seed),sensorQ(o.seed+104729),sensorV(o.seed+130363);
  std::uniform_real_distribution<double> uniform(-3.141592653589793,3.141592653589793);
  std::normal_distribution<double> noise(0.,1.);
  for(int j=0;j<n;j++)phase[j]=uniform(trajectory);
  s.setState(center,Eigen::VectorXd::Zero(n));s.view->start(0);
  Csv out(o.out,n);out.header({"gc","gv","ptarget","gf","truth_gc","truth_gv","command"});
  auto start=std::chrono::steady_clock::now();
  for(long k=0;k<=ticks;k++){
    double t=k*o.dt,u=std::min(1.,t),ramp=3*u*u-2*u*u*u;
    s.getState(q,v);
    for(int j=0;j<n;j++) p[j]=center[j]+amp[j]*ramp*std::sin(2*3.141592653589793*o.frequency*t+phase[j]);
    Eigen::VectorXd mq=q,mv=v;
    for(int j=0;j<n;j++){mq[j]+=o.noiseQ*noise(sensorQ);mv[j]+=o.noiseV*noise(sensorV);}
    Eigen::VectorXd command=s.kp.cwiseProduct(p-q)-s.kd.cwiseProduct(v);
    Eigen::VectorXd tau=o.scale*command-o.damping*v;
    out.row(t,{&mq,&mv,&p,&tau,&q,&v,&command});
    if(k<ticks)s.torqueStep(o.dt,tau);
  }
  double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  std::cout<<"SYNTHETIC collection: simulated_seconds="<<o.seconds<<" wall_seconds="<<wall
    <<" rows="<<ticks+1<<" speedup="<<o.seconds/wall<<"x\n";
  std::ofstream meta(o.out+".collection.json");
  meta<<std::setprecision(17)<<"{\"simulated_seconds\":"<<o.seconds<<",\"wall_seconds\":"<<wall
      <<",\"rows\":"<<ticks+1<<",\"center\":[";
  for(int j=0;j<n;j++)meta<<(j?",":"")<<center[j];
  meta<<"],\"amplitude\":[";
  for(int j=0;j<n;j++)meta<<(j?",":"")<<amp[j];
  meta<<"],\"phase\":[";
  for(int j=0;j<n;j++)meta<<(j?",":"")<<phase[j];
  meta<<"]}\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const Opt o = parseArgs(argc, argv);
    if(std::filesystem::exists(o.out)) throw std::runtime_error("output exists; choose a new path");
    auto parent=std::filesystem::path(o.out).parent_path();if(!parent.empty())std::filesystem::create_directories(parent);
    if(o.mode=="inspect"){inspect(o);return 0;}
    if(o.mode=="collect"){collect(o);return 0;}
    auto started=std::chrono::steady_clock::now();
    const Table in = readCsv(o.in);
    const double span = in.t(in.rows.size() - 1) - in.t(0);
    std::cout << in.rows.size() << " rows, " << span << " s, " << in.nJ << " joints, stride "
              << o.stride << " -> prediction at "
              << (span > 0. ? (in.rows.size() - 1) / span / o.stride : 0.) << " Hz\n";
    if (o.mode == "replay") replay(o, in); else rollout(o, in);
    std::cout<<"simulated_seconds="<<span<<" wall_seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<"\n";
    std::cout << "wrote " << o.out << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
