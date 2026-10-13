#!/usr/bin/env python3
"""Re-export a trained model dir (actuator_net.pt + config.json) to TorchScript for augmented_sim.

train.py already writes actuator_net_jit.pt; use this after editing weights or config by hand.

  python scripts/pt_to_jit.py out/train/run0
"""
import json
import pathlib
import sys

import torch

from train import ActuatorNet, export_jit


def main():
    d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out/train/run0")
    cfg = json.load(open(d / "config.json"))
    net = ActuatorNet(len(cfg["x_mean"]), cfg["hidden"], cfg["layers"], cfg["n_joints"])
    net.load_state_dict(torch.load(d / "actuator_net.pt", map_location="cpu", weights_only=True))
    export_jit(net.eval(), cfg, d / "actuator_net_jit.pt")
    print(f"wrote {d / 'actuator_net_jit.pt'}")


if __name__ == "__main__":
    main()
