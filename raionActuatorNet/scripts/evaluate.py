#!/usr/bin/env python3
"""Evaluate the ActuatorNet-augmented simulator against a measured robot log.

Runs `augmented_sim` on the cleaned log four ways: free rollout and one-step replay, each with and
without the net. Writes per-joint error-vs-time plots, a running-RMSE plot and metrics.json. The
oracle row reuses approximate recorded residuals; it is NOT a guaranteed performance ceiling.

  python scripts/evaluate.py log.csv --model out/train/run0 --out out/eval/run0 --urdf robot.urdf
"""
import argparse
import json
import pathlib
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
from common import add_viewer_args

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (add_sim_args, block_mask, build_features, cols, joint_names, load_csv,  # noqa: E402
                    n_joints, prediction_rate, run_sim, sim_opts, simulator_contract, check_log_metadata)


def rmse(a, axis=None):
    return np.sqrt(np.mean(np.square(np.asarray(a)), axis=axis))


def plot_series(series, names, ylabel, path):
    """series: name -> (time, values (T, n_joints)); each series carries its own time base."""
    n = len(names)
    ncol = min(3, n)
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 2.5 * nrow), sharex=True, squeeze=False)
    for j, ax in enumerate(axes.flat):
        if j >= n:
            ax.axis("off")
            continue
        for name, (ts, e) in series.items():
            ax.plot(ts, e[:, j], lw=0.7, label=name)
        ax.set_title(names[j])
        ax.grid(alpha=0.3)
    axes[0][0].legend()
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    fig.supylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="measured robot log CSV")
    ap.add_argument("--model", required=True, help="train.py output dir (or a TorchScript file)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixed-dt", type=float,
                    help="also roll out on a uniform clock of this step, as a training simulator "
                         "would, instead of following the log's timestamps")
    add_sim_args(ap)
    add_viewer_args(ap)
    args = ap.parse_args()
    started=time.perf_counter()
    out = pathlib.Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('output directory is not empty; choose a new run')
    out.mkdir(parents=True, exist_ok=True)
    model = pathlib.Path(args.model)
    jit = model / "actuator_net_jit.pt" if model.is_dir() else model
    opts = sim_opts(args)
    roll_opts = dict(opts, viewer=args.viewer, viewer_speed=args.viewer_speed)
    contract = simulator_contract(opts)
    check_log_metadata(args.log, contract)
    val_block = 0.3
    if model.is_dir():
        trained = json.load(open(model / "config.json"))
        if trained.get('simulator') != contract:
            raise ValueError('model/backend/joint mapping/PD gains differ from training or provenance missing; retrain')
        for k in ("stride", "smooth", "steps"):
            if trained.get(k, 0) != getattr(args, k):
                raise SystemExit(f"--{k} is {getattr(args, k)} but the model was trained with "
                                 f"{trained.get(k)}; the two must match")
        val_block = trained.get("val_block", val_block)   # so the held-out split matches training

    log = load_csv(args.log, smooth_n=args.smooth)
    names = joint_names(args.urdf, n_joints(log), args.joint_config)
    clean = out / "log_clean.csv"
    log.to_csv(clean, index=False)

    roll = {"baseline": run_sim("rollout", clean, out / "rollout_baseline.csv", **roll_opts),
            "augmented": run_sim("rollout", clean, out / "rollout_augmented.csv", model=jit, **roll_opts)}
    step = {"baseline": run_sim("replay", clean, out / "replay_baseline.csv", **opts),
            "augmented": run_sim("replay", clean, out / "replay_augmented.csv", model=jit, **opts)}
    # ceiling: the recorded torque error replayed into the rollout, i.e. a perfect predictor. The
    # replay file has one row per prediction interval, so its err_tau is written back onto the
    # full-rate log at the interval start ticks, where rollout reads it.
    oracle_in = log.copy()
    rep = step["baseline"]
    tau = cols(rep, "err_tau")
    oracle_in[tau] = 0.0
    oracle_in.loc[np.arange(len(rep)) * max(1, args.stride), tau] = rep[tau].to_numpy()
    oracle_in.to_csv(out / "oracle_in.csv", index=False)
    roll["oracle"] = run_sim("rollout", out / "oracle_in.csv", out / "rollout_oracle.csv",
                             ff_column="err_tau", **roll_opts)
    (out / "oracle_in.csv").unlink(missing_ok=True)
    # the same model in a fixed-rate simulator: the network runs every --stride steps of --fixed-dt,
    # so the stride is rescaled to keep the prediction rate the log's rollout ran at
    if args.fixed_dt:
        rate = prediction_rate(log, args.stride)
        s_fixed = max(1, round(1.0 / args.fixed_dt / rate))
        # the uniform clock sets the step size, so --steps must not also set it
        roll[f"fixed {1 / args.fixed_dt:.0f} Hz"] = run_sim(
            "rollout", clean, out / "rollout_fixed_dt.csv", model=jit,
            **{**roll_opts, "stride": s_fixed, "steps": 0, "fixed_dt": args.fixed_dt})
        print(f"fixed-dt rollout: {1 / args.fixed_dt:.0f} Hz physics, stride {s_fixed} "
              f"-> {1 / args.fixed_dt / s_fixed:.0f} Hz prediction")

    err = {k: (df["elapsedTime"].to_numpy(),
               df[cols(df, "sim_gc")].to_numpy() - df[cols(df, "gc")].to_numpy(),
               df[cols(df, "sim_gv")].to_numpy() - df[cols(df, "gv")].to_numpy()) for k, df in roll.items()}
    metrics = {"simulator": contract, "rollout": {k: dict(pos_rmse_rad=float(rmse(e[1])), vel_rmse_rad_s=float(rmse(e[2])),
                                   pos_rmse_per_joint_rad=rmse(e[1], 0).tolist()) for k, e in err.items()},
               "one_step": {k: dict(pos_rmse_rad=float(rmse(df[cols(df, "err_gc")])),
                                    vel_rmse_rad_s=float(rmse(df[cols(df, "err_gv")])),
                                    vel_rmse_per_joint=rmse(df[cols(df, "err_gv")], 0).tolist())
                            for k, df in step.items()},
               "joint_names": names}
    # the rollout is one continuous trajectory, so a held-out tick is still reached through trained
    # ones; this says whether the error concentrates on phases the net never fitted, not much more
    for k, e_ in err.items():
        metrics["rollout"][k]["pos_rmse_heldout_ticks_rad"] = float(rmse(e_[1][block_mask(e_[0], val_block)]))
    metrics["improvement"] = {
        f"{m}_{q}": 1 - metrics[m]["augmented"][q] / metrics[m]["baseline"][q]
        for m in ("rollout", "one_step") for q in ("pos_rmse_rad", "vel_rmse_rad_s")}

    if model.is_dir():  # the same features fed to the same model in Python: catches layout bugs
        cfg = json.load(open(model / "config.json"))
        raw = build_features(step["augmented"], cfg["features"])
        with torch.no_grad():
            net = torch.jit.load(str(jit))
            h = torch.zeros(cfg["layers"], 1, cfg["hidden"])
            pred = []
            for r in raw:                       # one step at a time, carrying state, as the C++ does
                y, h = net(torch.tensor(r).view(1, -1), h)
                pred.append(y.numpy()[0])
        metrics["backend_vs_python_ff_max_abs_diff_nm"] = float(
            np.abs(np.stack(pred) - step["augmented"][cols(step["augmented"], "ff")].to_numpy()).max())
        if not np.allclose(np.stack(pred),step["augmented"][cols(step["augmented"],"ff")].to_numpy(),
                           rtol=1e-5,atol=1e-5):
            raise ValueError('C++ exported weights and TorchScript disagree; check/re-export this model')
    metrics['evaluation_seconds_before_plots']=time.perf_counter()-started
    json.dump(metrics, open(out / "metrics.json", "w"), indent=1)

    rows = [(f"rollout {k}", v["pos_rmse_rad"], v["vel_rmse_rad_s"]) for k, v in metrics["rollout"].items()]
    rows += [(f"one-step {k}", v["pos_rmse_rad"], v["vel_rmse_rad_s"]) for k, v in metrics["one_step"].items()]
    print(pd.DataFrame(rows, columns=["", "pos RMSE [rad]", "vel RMSE [rad/s]"]).to_string(index=False))
    print("improvement:", {k: f"{v:.1%}" for k, v in metrics["improvement"].items()})
    if "backend_vs_python_ff_max_abs_diff_nm" in metrics:
        print("MuJoCo backend vs Python feature pipeline max |diff|:", metrics["backend_vs_python_ff_max_abs_diff_nm"], "Nm")

    plot_series({k: (e[0], e[1]) for k, e in err.items()}, names,
                "rollout position error, sim - measured [rad]", out / "position_error.png")
    plot_series({k: (e[0], e[2]) for k, e in err.items()}, names,
                "rollout velocity error, sim - measured [rad/s]", out / "velocity_error.png")
    plot_series({k: (df["elapsedTime"].to_numpy(), df[cols(df, "err_gv")].to_numpy())
                 for k, df in step.items()}, names,
                "one-step velocity error, measured - sim [rad/s]", out / "one_step_error.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    for k, e in err.items():
        ax.plot(e[0], np.sqrt(np.cumsum(np.mean(e[1] ** 2, 1)) / np.arange(1, len(e[0]) + 1)), label=k)
    ax.set(xlabel="time [s]", ylabel="running position RMSE, all joints [rad]", yscale="log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "running_rmse.png", dpi=120)
    print(f"wrote {out}")
    print(f"evaluation total wall seconds (including plots): {time.perf_counter()-started:.3f}")


if __name__ == "__main__":
    main()
