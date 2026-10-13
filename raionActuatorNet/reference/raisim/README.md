# gym-articulated-actuator-net

A framework for training and deploying an **ActuatorNet** (Hwangbo et al., *Learning agile and
dynamic motor skills for legged robots*, Science Robotics 2019) for one specific, already-built
articulated robot.

The idea: a nominal rigid-body model never matches the machine that was actually manufactured. Given
a recording of the real robot moving under PD control, this framework measures the error between the
simulated and the real state transition at every tick, expresses it as a joint torque, learns to
predict it from the joint history, and injects the prediction back into RaiSim as a feedforward
torque. The result is a simulator tuned to that individual machine.

```
measured robot log
   |-- augmented_sim replay  --> per-tick transition error, as a joint torque (err_tau)
   |-- train.py              --> actuator_net_jit.pt   (TorchScript; normalization inside)
   |-- augmented_sim rollout --> simulator with the net
   `-- evaluate.py           --> plots + metrics.json
```

Nothing is tied to a particular robot. The joint count comes from the log, the kinematics and
inertia from whatever URDF you pass, and the joint names on the plots from that URDF. The one worked
example, a 12-joint quadruped, is in [`out/report/`](out/report/README.md): 54% lower position error
over a free-running simulation, with the caveats that report spells out.

## What your data has to look like

A CSV per recording, one row per control tick, with these columns for `j = 0 .. n-1` in joint order:

| column | meaning |
| --- | --- |
| `elapsedTime` | seconds since the recording started |
| `gc_j` | measured joint position |
| `gv_j` | measured joint velocity |
| `ptarget_j` | position target the PD controller was given on that tick |
| `gf_j` | applied joint torque, optional; only used if you enable the torque feature |

Requirements on the recording:

* **Any PD schedule works.** The framework does not care how the position targets were generated:
  step, sine, chirp, a replayed policy, teleoperation. It only needs the target that was actually
  commanded on each tick, logged alongside the state, and the gains that were in effect
  (`--kp`, `--kd`, constant across joints; edit `Sim` in the C++ for per-joint gains). A schedule
  that excites each joint over a range of speeds and amplitudes will train a better net than a
  single repeated motion.
* **Fixed base.** The robot must be held, on a stand or bolted down, with no contacts. The tool
  injects a `world` link into the URDF at load, and a log whose joint count does not match the
  URDF's generalized coordinates is rejected. A floating base would need the base state in the log.
* **Inertia, not geometry.** Only the `<inertial>` and joint tags affect anything here. The example
  URDF names visual meshes that are not shipped, which costs nothing because nothing in this
  repository renders: its collision bodies are primitives, so even contacts would not need them.
* **Timestamps as recorded.** Jitter is fine and is preserved; see *Calibration*. Rows that repeat
  an `elapsedTime` are treated as stale re-logs of the same instant and only the first is kept,
  which matters for loggers that append a row per control-loop pass rather than per tick.

## Requirements and build

* RaiSim (tested 1.1.8) and libtorch CPU (tested 2.8.0). `CMakeLists.txt` carries this machine's
  paths as fallbacks; elsewhere pass `-DCMAKE_PREFIX_PATH="<raisim>;<libtorch>"`
* Eigen3, CMake >= 3.18, a C++17 compiler
* Python 3.12 with torch, numpy, pandas, matplotlib

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

## Use

```bash
LOG=/path/to/your_log.csv          # several logs can be passed to train.py at once
URDF=/path/to/your_robot.urdf

python scripts/train.py    $LOG --out out/train/run0 --urdf $URDF
python scripts/evaluate.py $LOG --model out/train/run0 --out out/eval/run0 --urdf $URDF
```

Training takes a few minutes on one GPU and is deterministic for a given `--seed` on a given
machine. `out/eval/run0/` then holds `metrics.json`, four per-joint plots and the rollout CSVs.
`scripts/pt_to_jit.py <model dir>` re-exports the TorchScript if weights are edited by hand.

The C++ tool can also be driven directly:

```bash
./build/augmented_sim replay  log.csv out.csv --urdf robot.urdf [--model net_jit.pt]
./build/augmented_sim rollout log.csv out.csv --urdf robot.urdf [--model net_jit.pt]
```

## How it works

**The target.** For every logged tick the simulator is set to the measured state, driven with the
same PD target, and integrated one tick. The velocity mismatch against the next measured sample is
converted to a joint torque:

```
err_tau = M(q) * (u_measured_next - u_sim_next) / dt
```

This is the extra torque the actuators would have needed for the model to follow the machine. It is
a torque rather than a state offset because that is the channel RaiSim exposes, so the correction
stays a physical input and keeps working when contacts are added later. The nonlinearity term of the
equation of motion does **not** appear: both transitions start from the same state, so `h(q, v)` is
identical in each and cancels in the difference. That was measured rather than assumed; see the
report.

**The network.** A GRU over the whole joint vector. The inputs are position error and velocity for
every joint, one recurrent state carries the history, and every joint's torque error comes out
together, so inter-joint coupling is available to the model. Defaults: 64 hidden units, 1 layer,
overlapping 0.33 s windows, the first 0.06 s of each burning in the recurrent state and scored by
neither the training nor the validation loss. Windows are given in seconds so they mean the same
thing at any prediction rate. Validation holds out every fourth 0.3 s block, since adjacent samples
at kHz rates are near-duplicates and a random row split leaks; the block length is recorded in
`config.json` so the evaluator reuses the split the model was trained with.

A per-joint MLP with a one-hot joint id, the shape used in the original paper, was measured against
this and rejected: it fits the one-step target better and is far worse in closed loop. The report
has the numbers.

**Deployment.** The TorchScript file carries the input and output normalization plus the attributes
`features`, `n_joints`, `hidden`, `layers`, so `augmented_sim` reads the layout from the model and
needs no side-channel config. `evaluate.py` re-runs the model in Python over the same features and
checks it against the torque the C++ actually applied.

## Calibration

| flag | default | what it is for |
| --- | --- | --- |
| `--urdf` | required | your robot; its movable joint count must match the log |
| `--kp`, `--kd` | 100, 1 | must match the gains used during collection |
| `--stride N` | 4 | ticks per prediction; the transition error is measured over the same window |
| `--smooth` | 11 | zero-phase low-pass on `gc` and `gv`, in samples; see below |
| `--steps N` | 0 | physics steps per prediction; 0 follows the log, one step per tick |
| `--fixed-dt D` | off | roll out on a uniform clock of step D instead of the log's timestamps |

The `--stride` and `--smooth` defaults are the operating point measured on the example robot: the
network predicts at roughly 450 Hz with a light filter. `--stride` counts log ticks, so on a log
recorded at a rate far from 1.8 kHz it should be scaled to keep the prediction rate near 450 Hz; the
tools print a warning when it lands outside 300 to 900 Hz.

**Timestep.** By default every transition is integrated with its own dt, read from the log
(`world.setTimeStep` per step). No resampling and no averaged dt, so a jittering recording is
replayed on its true time base and the simulated and measured transitions cover the same interval.
That is what makes the training targets trustworthy, so `replay` always works this way.

A deployed simulator has no log to follow, so `rollout --fixed-dt D` runs on a uniform clock of step
D instead: position targets are zero-order held from the log and the reference state is interpolated
onto the clock, and `--stride` then counts D-sized steps rather than log ticks. `evaluate.py
--fixed-dt D` adds that run to the comparison, rescaling the stride so the prediction rate matches.
On the example robot a uniform clock is free: at matched rates it scores the same as the log's own
jittered timestamps, every time it was measured.

`--steps N` is the other half of that question: how many physics steps make up one prediction
interval. The default follows the log, one step per tick, so physics runs finer than the network.
`--steps 1` advances the whole interval in a single step with the target held, which is what a
simulator whose physics rate equals its control rate does, and it applies to `replay` as well, so
the training targets can be built against the same stepping the network will run in. What that is
worth is measured in `out/report`: the step size matters, matching train to deployment barely does.

**Prediction rate.** `--stride N` decouples the network's rate from the log's. Training targets are
then measured over an N-tick window instead of one, and in a rollout the network runs every N ticks
with its torque held in between, while physics keeps stepping at the log's rate. This is how an RL
simulator would run it, and it matters twice over: inference cost falls by N, and the transition
error grows with the window while the measurement noise does not, so the target's signal-to-noise
ratio improves by roughly N. Against that, a prediction error held across a longer window injects
more into the simulation, so both the achievable ceiling and the penalty for an imperfect model
rise. The two effects have opposite signs and the balance point is robot-specific. On the example
robot the network runs four times slower than the log with no loss of accuracy and a quarter of the
added simulation cost, while an eighteen-fold stride is worse than applying no correction at all;
`out/report` has the sweep. `--stride` must match between `train.py` and `evaluate.py` and is
recorded in the model's `config.json`, which the evaluator checks.

At stride N a log holds N interleaved sequences at the prediction rate, and decimation keeps only
one of them. `augmented_sim replay --offset K` starts the intervals at tick K instead, so running
replay for several offsets and passing all the CSVs to `train.py` recovers the samples decimation
drops. Whether that helps is worth checking rather than assuming.

**Measurement noise.** This is the setting most likely to decide whether the framework works on your
data. Encoder velocity often carries more per-sample noise than one control tick of real motion
contains; that noise lands in the training target and in the velocity input at once, so it is both
unlearnable and correlated with the features. `--smooth N` applies a zero-phase symmetric Hann
filter to `gc` and `gv`; being symmetric it adds no lag and commutes with differentiation, so `gv`
stays the derivative of `gc`. Pick N by sweeping it and watching two things: the RMS of `err_tau`
should fall and then flatten, while the *baseline* rollout error stays put, which shows the filter is
removing noise rather than reshaping the trajectory being tracked. On the example robot the raw
target was about 80% noise. How much filtering is needed depends on the prediction rate, which also
cleans the target: 31 samples was right at the log's full rate, 11 at the default 450 Hz.

**Stability.** The predicted torque is a function of position error and velocity, so it adds
stiffness and damping of its own, and either can come out negative. A model that fits well can still
make the rollout ring or diverge, so read the rollout error rather than assuming a good fit deploys
safely.

**Judging a model.** Validation R2 on the torque target is a poor guide; a model that fits it better
can be much worse once it runs in the loop. Use the rollout position error, and read the oracle row
that `evaluate.py` produces: it replays the recorded torque error instead of a prediction and is the
ceiling any predictor could reach on that log. `metrics.json` also reports the rollout error
restricted to held-out ticks, which is the closest thing to an out-of-sample number a single
continuous recording can give.

## Repository layout

```
CMakeLists.txt          RaiSim and libtorch paths, one executable
src/augmented_sim.cpp   replay (build targets) and rollout (run the augmented simulator)
scripts/common.py       log cleaning, feature layout mirrored from the C++, subprocess wrapper
scripts/train.py        GRU trainer, writes TorchScript + config.json + metrics
scripts/evaluate.py     rollouts with baseline / net / oracle, plots, metrics.json
scripts/pt_to_jit.py    re-export TorchScript from saved weights
rsc/                    URDF for the example robot
docs/                   the ActuatorNet paper
out/report/             worked example: trained model, figures, metrics, and what they mean
```

## Limits

* **Fixed base, no contacts.** The torque channel stays valid with contacts, but nothing here has
  been trained or checked with them.
* **Constant PD gains** across joints, taken from the command line. Per-joint gains need a small
  edit in `Sim`.
* **One recording is not enough.** The example was trained on a single 6 s log and shows no gain on
  held-out ticks, which is the expected outcome from that little data. Collect several recordings
  with varied amplitude and speed before trusting a model outside the motion it saw.
