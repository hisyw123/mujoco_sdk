#!/usr/bin/env python3
"""Train the ActuatorNet on one-step transition errors between a measured robot and the nominal model.

Inputs are measured robot logs (or replay CSVs already produced by `augmented_sim replay`, detected
by the err_tau_0 column). For every measured state the C++ tool integrates the nominal RaiSim model
one control tick and reports the torque that would have reproduced the measured velocity change
(err_tau). The net learns err_tau from the joint history; the TorchScript it writes drives
`augmented_sim rollout`.

  python scripts/train.py log_0.csv log_1.csv --out out/train/run0 --urdf robot.urdf

Architecture: a GRU over the whole joint vector, predicting every joint's torque error at once. See
out/report for the measurement against a per-joint MLP.
"""
import argparse
import json
import pathlib
import tempfile
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import (FEATURES, add_sim_args, block_mask, build_features, cols, load_csv, n_joints,
                    prediction_rate, run_sim, sim_opts)


class ActuatorNet(nn.Module):
    """GRU over the whole joint vector: every joint's history feeds one recurrent state and every
    joint's torque error comes out together, so inter-joint coupling is available to the model."""

    def __init__(self, in_dim, hidden, layers, n_out):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, layers)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x, h):            # x: (seq, batch, in_dim), h: (layers, batch, hidden)
        out, h = self.gru(x, h)
        return self.head(out), h


class Deployed(nn.Module):
    """One step of raw features plus the recurrent state in, one torque per joint in Nm out.
    Normalization and the feature layout live in the TorchScript file, so the C++ side needs no
    extra config."""

    def __init__(self, net, cfg):
        super().__init__()
        self.net = net
        for k in ("x_mean", "x_std", "y_mean", "y_std"):
            self.register_buffer(k, torch.tensor(cfg[k]))
        self.register_buffer("features", torch.tensor([FEATURES[f] for f in cfg["features"]], dtype=torch.int64))
        self.n_joints = cfg["n_joints"]
        self.hidden = cfg["hidden"]
        self.layers = cfg["layers"]

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = ((x - self.x_mean) / self.x_std).unsqueeze(0)     # (1, 1, F)
        y, h = self.net(z, h)
        return y[0] * self.y_std + self.y_mean, h


def export_jit(net, cfg, path):
    dep = Deployed(net, cfg).eval()
    torch.jit.script(dep).save(str(path))
    x, h = torch.randn(1, len(cfg["x_mean"])), torch.zeros(cfg["layers"], 1, cfg["hidden"])
    with torch.no_grad():
        assert torch.allclose(torch.jit.load(str(path))(x, h)[0], dep(x, h)[0], atol=1e-6)


def load_replays(inputs, opts, smooth_n):
    """Measured log -> replay CSV via the C++ tool (or pass-through for an existing replay CSV)."""
    for p in inputs:
        if "err_tau_0" in pd.read_csv(p, nrows=0).columns:
            yield pd.read_csv(p)
            continue
        with tempfile.TemporaryDirectory() as tmp:
            clean = pathlib.Path(tmp) / "clean.csv"
            load_csv(p, smooth_n=smooth_n).to_csv(clean, index=False)
            yield run_sim("replay", clean, pathlib.Path(tmp) / "replay.csv", **opts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="measured robot logs (or replay CSVs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", nargs="+", default=["pos_err", "vel"], choices=list(FEATURES))
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--chunk-s", type=float, default=0.33, help="BPTT window in seconds")
    ap.add_argument("--warmup-s", type=float, default=0.06,
                    help="seconds per window spent burning in the recurrent state, scored by neither side")
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--batch", type=int, default=2048, help="windows per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=120)
    ap.add_argument("--val-block", type=float, default=0.3, help="validation = every 4th block of this many seconds")
    ap.add_argument("--seed", type=int, default=0)
    add_sim_args(ap)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    seqs, times, nJ = [], [], None   # per log: (features (T,F), targets (T,nJ), validation mask (T,))
    for df in load_replays(args.inputs, sim_opts(args), args.smooth):
        if nJ is None:
            nJ = n_joints(df)
        elif n_joints(df) != nJ:
            raise ValueError("inputs disagree on the joint count")
        prediction_rate(df, 1)          # the replay CSV is already one row per prediction
        times.append(df["elapsedTime"].to_numpy())
        seqs.append((build_features(df, args.features),
                     df[cols(df, "err_tau")].to_numpy(dtype=np.float32),
                     block_mask(times[-1], args.val_block)))
    x_all = np.concatenate([s[0] for s in seqs])
    y_all = np.concatenate([s[1] for s in seqs])
    v_all = np.concatenate([s[2] for s in seqs])
    x_mean, x_std = x_all[~v_all].mean(0), x_all[~v_all].std(0)
    y_mean, y_std = y_all[~v_all].mean(0), y_all[~v_all].std(0)
    x_std[x_std < 1e-8] = 1.0
    y_std[y_std < 1e-8] = 1.0
    cfg = dict(n_joints=nJ, features=args.features, hidden=args.hidden, layers=args.layers,
               inputs=args.inputs, smooth=args.smooth, stride=args.stride, steps=args.steps,
               val_block=args.val_block, x_mean=x_mean.tolist(), x_std=x_std.tolist(),
               y_mean=y_mean.tolist(), y_std=y_std.tolist())
    print(f"{len(x_all)} ticks over {len(seqs)} log(s), {nJ} joints, {x_all.shape[1]} features, "
          f"{v_all.mean():.0%} validation")

    net = ActuatorNet(x_all.shape[1], args.hidden, args.layers, nJ).to(dev)
    # windows are given in seconds so they mean the same thing at any prediction rate
    dt = float(np.median(np.diff(times[0])))
    chunk = max(8, int(round(args.chunk_s / dt)))
    warmup = min(chunk - 4, max(1, int(round(args.warmup_s / dt))))
    # overlapping contiguous windows; the first `warmup` samples burn in the hidden state
    starts = [(i, s) for i, (x, _, _) in enumerate(seqs)
              for s in range(0, max(1, len(x) - chunk), max(1, chunk // 2))]
    xs = torch.tensor(np.stack([(seqs[i][0][s:s + chunk] - x_mean) / x_std for i, s in starts]), device=dev)
    ys = torch.tensor(np.stack([(seqs[i][1][s:s + chunk] - y_mean) / y_std for i, s in starts]), device=dev)
    val = torch.tensor(np.stack([seqs[i][2][s:s + chunk] for i, s in starts]), device=dev)
    xs, ys, val = xs.transpose(0, 1), ys.transpose(0, 1), val.transpose(0, 1)   # (chunk, window, ...)
    warm = torch.zeros_like(val)
    warm[:warmup] = True        # state burn-in: scored by neither side, trained on by neither
    is_val = torch.tensor(np.array([seqs[i][2][s + warmup:s + chunk].mean() > 0.5
                                    for i, s in starts]), device=dev)
    tr_w, va_w = (~is_val).nonzero().squeeze(1), is_val.nonzero().squeeze(1)
    print(f"{len(starts)} windows of {chunk} samples ({chunk * dt:.2f} s), warmup {warmup}: "
          f"{len(tr_w)} train, {len(va_w)} val")

    def loss_on(w, train):
        h = torch.zeros(args.layers, len(w), args.hidden, device=dev)
        pred, _ = net(xs[:, w], h)
        v, wm = val[:, w], warm[:, w]
        m = (~v & ~wm) if train else (v & ~wm)      # train on kept ticks, score on held-out ticks
        return ((pred - ys[:, w]) ** 2)[m.unsqueeze(-1).expand_as(pred)].mean()

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr / 100)
    best, best_state, stale, hist = np.inf, None, 0, []
    for epoch in range(args.epochs):
        net.train()
        perm = tr_w[torch.randperm(len(tr_w), device=dev)]
        tl = 0.0
        for i in range(0, len(perm), args.batch):
            batch = perm[i:i + args.batch]
            loss = loss_on(batch, True)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(batch)
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = loss_on(va_w, False).item()
        hist.append((epoch + 1, tl / len(perm), vl))
        if vl < best:
            best, stale = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
        if epoch % 20 == 0:
            print(f"epoch {epoch + 1:4d}  train {tl / len(perm):.5f}  val {vl:.5f}")
        if stale >= args.patience:
            print(f"early stop at epoch {epoch + 1}")
            break
    net.load_state_dict(best_state)
    net.cpu().eval()

    with torch.no_grad():   # metrics in Nm, on the real target scale; each log is its own sequence
        pred = np.concatenate([
            net(torch.tensor((sq[0] - x_mean) / x_std).unsqueeze(1),
                torch.zeros(args.layers, 1, args.hidden))[0].squeeze(1).numpy() for sq in seqs])
    pred = pred * y_std + y_mean
    metrics = {}
    for name, m in (("train", ~v_all), ("val", v_all)):
        e = pred[m] - y_all[m]
        metrics[name] = dict(rmse_nm=float(np.sqrt((e ** 2).mean())),
                             r2=float(1 - (e ** 2).sum() / ((y_all[m] - y_all[m].mean(0)) ** 2).sum()),
                             target_rms_nm=float(np.sqrt((y_all[m] ** 2).mean())))
    metrics["val_rmse_per_joint_nm"] = np.sqrt(((pred - y_all)[v_all] ** 2).mean(0)).tolist()
    print(json.dumps({k: v for k, v in metrics.items() if k != "val_rmse_per_joint_nm"}, indent=1))

    torch.save(net.state_dict(), out / "actuator_net.pt")
    export_jit(net, cfg, out / "actuator_net_jit.pt")
    json.dump(dict(cfg, metrics=metrics), open(out / "config.json", "w"), indent=1)
    pd.DataFrame(hist, columns=["epoch", "train_loss", "val_loss"]).to_csv(out / "training_history.csv", index=False)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
