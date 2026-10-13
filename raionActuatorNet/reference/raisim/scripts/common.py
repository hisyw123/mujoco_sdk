"""Shared helpers: log cleaning, feature layout (mirrors src/augmented_sim.cpp), binary wrapper.

Nothing here is tied to a particular robot: the joint count comes from the CSV's gc_* columns and
joint names come from the URDF when one is given.
"""
import pathlib
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "augmented_sim"
FEATURES = {"pos_err": 0, "vel": 1, "torque": 2}  # codes shared with the Feature enum in the C++
SIM_OPTS = ("urdf", "kp", "kd", "stride", "steps")
MOVABLE = ("revolute", "continuous", "prismatic")


def cols(df, prefix):
    """The df's <prefix>_0 .. <prefix>_n columns, in joint order."""
    out = [c for c in df.columns if c.startswith(prefix + "_") and c[len(prefix) + 1:].isdigit()]
    return sorted(out, key=lambda c: int(c.split("_")[-1]))


def n_joints(df):
    return len(cols(df, "gc"))


def joint_names(urdf, n):
    """Movable joints in URDF order, which is the order raisim uses. Falls back to indices."""
    try:
        names = [j.get("name") for j in ET.parse(urdf).getroot().findall("joint")
                 if j.get("type") in MOVABLE]
        if len(names) == n:
            return names
    except Exception:
        pass
    return [f"joint_{j}" for j in range(n)]


def smooth(x, n):
    """Zero-phase symmetric Hann filter along axis 0, edge-padded. Symmetric, so it adds no lag and
    commutes with differentiation: gv stays the derivative of gc."""
    if n <= 1:
        return x
    n += 1 - n % 2                       # odd window, so the kernel is centred on a sample
    k = np.hanning(n + 2)[1:-1]
    k /= k.sum()
    xp = np.pad(x, ((n // 2, n // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, j], k, "valid") for j in range(x.shape[1])], 1)


def load_csv(path, smooth_n=0):
    """Log -> one row per control tick, on the log's own timestamps.

    Some loggers append a row per control loop pass rather than per tick, so rows repeating
    elapsedTime are stale re-logs of the same instant and only the first of each group is
    kinematically consistent. Timestamps are left exactly as logged, jitter included, because the
    simulator integrates each transition with that transition's own dt.

    smooth_n low-pass filters gc and gv over that many samples. Encoder velocity often carries more
    per-sample noise than one control tick of real motion contains, and that noise lands in both the
    training target and the velocity input, so it has to come out before the transition error means
    anything. See README, "measurement noise".
    """
    df = pd.read_csv(path).drop_duplicates("elapsedTime", keep="first").reset_index(drop=True)
    for p in ("gc", "gv"):
        df[cols(df, p)] = smooth(df[cols(df, p)].to_numpy(dtype=float), smooth_n)
    return df


def build_features(df, features):
    """(T, n_features * n_joints) float32, laid out [feature0 over all joints | feature1 over all
    joints | ...], the same order Net::predict builds in augmented_sim.cpp."""
    src = {
        "pos_err": lambda: df[cols(df, "ptarget")].to_numpy() - df[cols(df, "gc")].to_numpy(),
        "vel": lambda: df[cols(df, "gv")].to_numpy(),
        "torque": lambda: df[cols(df, "gf")].to_numpy(),
    }
    return np.concatenate([src[f]() for f in features], axis=1).astype(np.float32)


def block_mask(t, val_block=0.3):
    """Validation = every 4th contiguous block. Adjacent samples at kHz rates are near-duplicates,
    so a random row split leaks. Keep 4 x val_block off the trajectory period, or every held-out
    block lands on the same phase of a periodic test motion."""
    return np.floor(t / val_block).astype(int) % 4 == 3


def add_sim_args(parser):
    parser.add_argument("--urdf", required=True,
                        help="robot URDF; its movable joint count must match the log's")
    parser.add_argument("--kp", type=float, help="PD position gain used during collection")
    parser.add_argument("--kd", type=float, help="PD velocity gain used during collection")
    parser.add_argument("--stride", type=int, default=4,
                        help="ticks per prediction; the transition error is measured over the same "
                             "window. Decouples the network's rate from the log's. The default suits "
                             "a log near 1.8 kHz; scale it with your log rate to stay near 450 Hz")
    parser.add_argument("--steps", type=int, default=0,
                        help="physics steps per prediction; 0 follows the log, one step per tick. "
                             "Set 1 to train against a simulator whose physics rate equals its "
                             "control rate")
    parser.add_argument("--smooth", type=int, default=11, help="zero-phase low-pass on gc and gv, in samples")


def sim_opts(args):
    return {k: getattr(args, k) for k in SIM_OPTS}


def prediction_rate(df, stride):
    """Hz at which the network will run, and a warning if that is outside the range measured to work.
    See out/report: 450 Hz matches the log rate for accuracy at a quarter of the cost, 225 Hz is
    marginal and 100 Hz is worse than applying no correction at all."""
    rate = 1.0 / float(np.median(np.diff(df["elapsedTime"].to_numpy()))) / max(1, stride)
    if not 300.0 <= rate <= 900.0:
        print(f"WARNING: --stride {stride} on this log predicts at {rate:.0f} Hz. Rates far from "
              f"~450 Hz were measured to work worse; see out/report.")
    return rate


def run_sim(mode, in_csv, out_csv, model=None, **opts):
    """Run `augmented_sim <mode>`; opts map to flags (kp=100 -> --kp 100, steps=1 -> --steps 1)."""
    if not BIN.exists():
        raise FileNotFoundError(f"{BIN} not built; see README")
    cmd = [str(BIN), mode, str(in_csv), str(out_csv)]
    if model:
        cmd += ["--model", str(model)]
    for k, v in opts.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            cmd.append(flag)
        elif v not in (None, False):
            cmd += [flag, str(v)]
    subprocess.run(cmd, check=True)
    return pd.read_csv(out_csv)
