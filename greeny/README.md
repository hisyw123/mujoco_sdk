# REAL-compatible GREENY runtime

The controller side follows the `RobotPC_HRI_v1a` motor boundary. REAL and
MuJoCo use the same `SharedMemory` fields and YAML motor ordering; only the
drive backend changes.

```text
                         shared control code
  HomingController / RlController / OnnxPolicyRunner / YAML configuration
                              |
                         SharedMemory
                      motors[i].q / q_target
                              |
             +----------------+----------------+
             |                                 |
       REAL: GVdrive                    SIM: MujocoDrive
       EtherCAT PDO                     mjData / data->ctrl
```

`motor/include/SharedMemory.hpp` is the source-compatible include path used by
the REAL `GVdrive` code. The implementation lives in the namespaced
`motor/include/shared_memory.h`. `MujocoDrive` publishes motor/IMU feedback and
consumes the same position, velocity, feed-forward torque, gain, and
`command_ready` fields as the real lower controller.

Robot-independent runtime capacity is 32 motors. The eight-joint constants and
joint order are isolated in `policy/greeny_policy_layout.h`; they describe this
ONNX policy plugin and are not a system-wide DOF limit.

## Configuration

`config/robots.yaml` is the robot/scene catalog. Each selectable profile
connects one MuJoCo model to its motor and policy contracts. Adding another
compatible robot or scene therefore does not require changing the runtime:

```yaml
robots:
  my-robot:
    model: ../model/my_robot.xml
    motor_contract: ../motor/config/my_robot_motors.yaml
    policy_contract: my_robot_policy.yaml
```

`motor/config/motors.yaml` defines the model-to-shared-memory binding:

- logical motor name and `joint_index`
- MuJoCo joint and actuator names
- lower-controller P/D gains
- homing position
- continuous/wheel actuator properties

`config/rl_policy.yaml` defines the basic GREENY locomotion policy contract.
It uses 31 current observations and contains no posture-transition input.

`config/transition_policy.yaml` is a separate, optional contract for the
stand-to-wheel policy:

- policy input/output joint mappings
- default stand pose and stand-to-wheel offset
- observation/action scales
- history length/stride and policy period
- transition duration

The wheel endpoint is calculated as in the deployed transition code: the
policy-order `offset_default_pos_AB` is added to the reordered stand pose, then
mapped back to robot order. The legacy `default_joint_angle_task_roll` value is
not used when this offset exists.

## Build

Homing and zero-policy backend:

```sh
cmake -S . -B build-greeny \
  -DMUJOCO_BUILD_GREENY=ON \
  -DMUJOCO_BUILD_TESTS=OFF \
  -DMUJOCO_BUILD_EXAMPLES=OFF \
  -DMUJOCO_BUILD_SIMULATE=ON
cmake --build build-greeny --target greeny_homing -j
```

With ONNX Runtime:

```sh
cmake -S . -B build-greeny \
  -DMUJOCO_BUILD_GREENY=ON \
  -DMUJOCO_BUILD_TESTS=OFF \
  -DMUJOCO_BUILD_EXAMPLES=OFF \
  -DMUJOCO_BUILD_SIMULATE=ON \
  -DMUJOCO_GREENY_ENABLE_ONNX=ON \
  -DGREENY_ONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build-greeny --target greeny_homing -j
```

ONNX-enabled builds use `policy/policy_1_14000.onnx` by default. Pass
`--policy PATH` only when selecting a different policy file.

## Run

List and select catalog entries:

```sh
./build-greeny/bin/greeny_homing --list-robots
./build-greeny/bin/greeny_homing --robot rough-terrain-3 --validate-contract
./build-greeny/bin/greeny_homing --robot greeny-12inch
./build-greeny/bin/greeny_homing --robot rough-terrain-3
```

The helper script accepts the same short profile name:

```sh
./greeny/run.sh rough-terrain-3
```

Passing a direct `MODEL.xml` path remains supported for ad-hoc scenes.

Homing smoke test:

```sh
./build-greeny/bin/greeny_homing MODEL.xml 3.0 \
  --headless --hold-seconds 2 --mode homing \
  --motor-config greeny/motor/config/motors.yaml
```

Basic RL policy:

```sh
./build-greeny/bin/greeny_homing MODEL.xml 3.0 \
  --mode rl \
  --motor-config greeny/motor/config/motors.yaml \
  --runner-config greeny/config/rl_policy.yaml
```

Startup mode is selected explicitly; the default is `rl`:

- `--mode rl`: initialize the exact basic-policy pose and start RL
- `--mode homing`: run physical homing first and wait in `HOLDING`
- `--mode rl-wheel`: available only with the optional transition policy config

In `homing` mode, press `R` after it reaches `HOLDING` to engage basic RL.
With the basic RL config, `Z` and posture transitions are disabled. They are
enabled only when `transition_policy.yaml` is loaded. A transition-path test
can use:

```sh
./build-greeny/bin/greeny_homing MODEL.xml 3.0 \
  --headless --hold-seconds 4 --mode rl-stand \
  --target-posture wheel \
  --runner-config greeny/config/transition_policy.yaml \
  --policy POLICY.onnx
```

The zero-policy backend verifies wiring only; it cannot balance the robot.
Successful ONNX loading/inference proves that the policy contract is connected,
but does not by itself prove physical stability.

## Controls

### Fixed-base joint trajectory test

Select `--mode joint-trajectory` to compile an in-memory model with the
`root` free joint removed from `base_link`. The base is rigidly fixed at
`--base-height` (default 1.5 m); the XML file is not changed. This mode
requires an XML model with that base/root naming convention. Gravity,
world contacts, actuator limits and the existing motor PD control remain
active. Robot self-collision is disabled in the compiled fixed-base model
because the imported collision meshes overlap in their zero pose; leaving it
enabled produces nonphysical contact impulses before trajectory tracking.
The ONNX policy is not loaded in this mode.

```sh
./build-greeny/bin/greeny_homing --robot greeny-12inch \
  --mode joint-trajectory \
  --trajectory-config greeny/config/joint_trajectory.yaml

./build-greeny/bin/greeny_homing --robot rough-terrain-3 \
  --mode joint-trajectory --base-height 1.5 \
  --trajectory-config greeny/config/joint_trajectory.yaml \
  --headless --hold-seconds 4 --trajectory-log /tmp/joint-trajectory.csv

./build-greeny/bin/greeny_homing --robot rough-terrain-3-bipedal \
  --headless --hold-seconds 6 \
  --trajectory-log /tmp/rough-bipedal-28dof.csv
```

The same options can be passed through `./greeny/run.sh ROBOT_NAME`.
Both RoughTerrain profiles default to fixed-base joint-trajectory mode, so
`--mode` and `--trajectory-config` may be omitted when using their catalog
names.
Trajectory YAML entries refer to logical motor names from the motor contract:

```yaml
joints:
  - {motor: LHY_J, amplitude: 0.15, period: 4.0}
  - {motor: RHY_J, amplitude: -0.15, period: 4.0}
```

Each reference is
`q_initial + amplitude * (1 - cos(2*pi*t/period)) / 2`.
Amplitude is a signed excursion in radians from the XML initial angle;
period is in seconds. Position and velocity targets start continuously
with zero velocity. Unknown/duplicate motors, nonfinite parameters,
nonpositive periods and excursions beyond joint limits are rejected.
Other motors in the contract hold their initial angles. The RoughTerrain
profiles bind all 28 independently actuated joints, including `CAM_J`.
`L_gripper_J` follows the commanded
`R_gripper_J` through the XML equality constraint.

Space pauses; Backspace resets the model and restarts the trajectory.
`F3` cycles through the XML cameras and back to the free mouse-controlled view;
the HUD displays the active camera. On the bipedal model the order is
free → debug_camera → cam_a → cam_b → cam_c → free.
`F4` switches directly to the last selected XML camera (initially `cam_a` when
available); `F5` restores the free view with its previous position, rotation,
and zoom. Mouse rotation, panning, and zoom apply in free view. Use `F3` to
choose another camera, then `F4`/`F5` to move between that camera and free view.
To start directly at the onboard camera frame:

```bash
./build-greeny/bin/greeny_homing --robot rough-terrain-3-bipedal --camera cam_a
```

The A/B/C cameras in both RoughTerrain XMLs move with `CAM_J`. Camera A attaches
to the existing OpenGL frame; B/C attach to the optical frames. Their local
camera rotations include the optical-to-MuJoCo conversion and a 180-degree
preview roll correction. The quadrupedal XML additionally accounts for its
frame axes being rebased by +90 degrees about Y relative to bipedal. At the
bipedal 135-degree pose the view points forward/down with the image top upward;
at the quadrupedal nominal 0-degree pose it points forward with the image top
upward. These are body-mounted views, not horizon-stabilized cameras.
Their 60-degree vertical field of view is a preview setting,
not calibrated hardware intrinsics. Press `F1` to hide the HUD for an unobstructed view.

`F12` opens/hides a separate depth window, or start with `--depth-view`:

```bash
./build-greeny/bin/greeny_homing --robot rough-terrain-3-bipedal --depth-view
```

The window renders `cam_a` (or the first XML camera if absent) independently
of the main view at 640x480. In the depth window, `Tab` cycles XML cameras,
`-`/`+` changes the display distance range (initially 5 m), and `F12`/`Esc`
or the close button hides it. Reopen with `F12` in the main window. Red is near,
green is half-range, blue is at or beyond the display range, and black means
no surface within the clipping range. The center readout is camera-axis depth
in metres (`nan` for no surface), not radial distance. Resizing preserves image
aspect ratio. Depth acquisition excludes HUD and main-view debug overlays;
the legend is added afterward. Pausing the main simulation pauses depth motion.

Locomotion keys and gamepad commands are disabled. Optional CSV output records
time, motor, target/measured angle, target/measured velocity and measured
actuator force at each control sample. Reset restarts timestamps at zero.
In headless mode, `--hold-seconds` is the total experiment duration.
Successful exit means finite simulation execution, not successful tracking:
use the CSV to evaluate tracking errors and tune motor gains. The headless
summary also reports peak contact count, penetration and force so that contact
impulses are not mistaken for a controller-only tracking problem.

### Locomotion and homing controls

- `H`: restart homing from the measured pose
- `R`: engage RL after homing
- `Z`: request wheel/roll posture
- `X`: request stand posture and clear velocity commands
- `W` / `S`: forward velocity
- `A` / `D`: yaw velocity
- `Space`: pause
- `Backspace`: reset simulation
- `Escape`: exit

An attached GLFW-compatible gamepad is detected automatically. The standard
gamepad mapping is:

- left stick Y: forward/reverse velocity
- right stick X: yaw velocity
- `X`: stand posture and stop
- `Y`: wheel/roll posture
- `Start`: engage RL after homing
- `Back`: restart homing
- left/right bumper: publish `LB_pressed` / `RB_pressed`

Both sticks use a rescaled 15% deadzone. Disconnecting an active gamepad clears
its velocity and bumper commands; when no gamepad has been connected, keyboard
commands remain unaffected.

For live contact/action diagnostics, run `./greeny/run_contact_diagnostics.sh`.
A separate plot window shows synchronized approximately 10-second plots for raw
policy actions, state spikes, contact penetration, force, and contact count.

For fixed-base joint trajectories, use the built binary from the repository root:

```bash
./build-greeny/bin/greeny_homing --robot rough-terrain-3-bipedal --diagnostics --plot-joint arm2_J
./build-greeny/bin/greeny_homing --robot rough-terrain-3 --diagnostics --plot-joint LSP_J
```

In joint-trajectory mode the four panels show the selected joint's target/measured
position, target/measured velocity, clamped motor command/actuator output, and
contact penetration/force/count. `--plot-joint` also enables diagnostics; it accepts
a motor or joint name in the motor contract. Without it the first motor is selected.
Use `[` / `]` to select the previous/next motor, `F9` to toggle plots, `Space` to
pause, and `Backspace` to restart from the configured keyframe. Switching joints
or restarting clears the plots. Data uses simulation time and stops while paused.
The direct-motor torque plots use Nm for hinge joints and N for slide joints
(the RoughTerrain actuators use unit gear). `--headless` does not show a window.

To display all 28 contracted motors in a five-column grid:

```bash
./build-greeny/bin/greeny_homing --robot rough-terrain-3-bipedal --plot-all
./build-greeny/bin/greeny_homing --robot rough-terrain-3 --plot-all --plot-kind velocity
```

`--plot-all` enables diagnostics in a separate 1100x760 resizable plot window.
The robot viewer remains 1200x900. `F10` switches the plot window between the
grid and single-joint panels; the robot stays visible in its own window.
`F11` cycles position, velocity, and torque/force. `--plot-kind` selects the
initial metric (`position`, `velocity`, or `torque`). Orange is target/clamped
command, cyan is measured/actuator output. Grid cells include the appropriate
units for the gripper. Switching metrics clears history. `F9`, pause, and reset
work in both layouts and either window. Closing the plot window (or Escape in
that window) hides it without stopping the simulation; F9 in the robot window
opens it again. Escape in the robot window exits both windows.
CAM_J is included in the 28-motor grid. Only the passive mimic joint
L_gripper_J is excluded.

The MuJoCo visualization shortcuts remain available; use `Alt` for a
visualization key that conflicts with an application key.
