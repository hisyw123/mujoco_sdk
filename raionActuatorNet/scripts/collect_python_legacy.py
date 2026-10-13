#!/usr/bin/env python3
"""NEW synthetic collector, not the unavailable original real-robot acquisition code."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from mujoco_sim import Sim, frame
from live_viewer import LiveViewer, add_viewer_args


def collect(a):
    out = Path(a.out)
    meta = out.with_suffix('.meta.json')
    if out.exists() or meta.exists():
        raise ValueError('output exists; choose a new path')
    if (not np.isfinite([a.dt,a.seconds,a.amplitude,a.frequency,a.noise_q,a.noise_v,
                          a.torque_scale,a.extra_damping]).all()
            or a.dt <= 0 or a.seconds < 2*a.dt or min(a.noise_q,a.noise_v,a.extra_damping) < 0):
        raise ValueError('invalid collection settings')
    ticks = round(a.seconds/a.dt)
    if not np.isclose(ticks*a.dt, a.seconds, atol=1e-10, rtol=0):
        raise ValueError('seconds must be an integer number of dt ticks')
    s = Sim(a.urdf, a.kp, a.kd, a.joint_config)
    with LiveViewer(s, getattr(a, 'viewer', False), getattr(a, 'viewer_speed', 1.)):
        return _collect(a, s, out, meta, ticks)


def _collect(a, s, out, meta, ticks):
    center = s.m.qpos0[s.qa].copy()
    phases, q_rng, v_rng = [np.random.default_rng(x) for x in np.random.SeedSequence(a.seed).spawn(3)]
    phase = phases.uniform(-np.pi, np.pi, s.n)
    ids = [s.m.joint(name).id for name in s.names]
    for i, j in enumerate(ids):
        if s.m.jnt_limited[j]:
            lo, hi = s.m.jnt_range[j]
            if center[i]-abs(a.amplitude) < lo or center[i]+abs(a.amplitude) > hi:
                raise ValueError(f'{s.names[i]}: target exceeds joint limits; edit model home pose or amplitude')
    s.reset(center, np.zeros(s.n))
    rows = []
    for k in range(ticks+1):
        t = k*a.dt
        q, v = s.state()
        # A smooth one-second ramp avoids a discontinuous initial command.
        u = min(1., t)
        target = center + a.amplitude*(3*u*u-2*u*u*u)*np.sin(2*np.pi*a.frequency*t+phase)
        measured_q = q + q_rng.normal(0, a.noise_q, s.n)
        measured_v = v + v_rng.normal(0, a.noise_v, s.n)
        # Noise is measurement-only; never overwrite true state with sensor noise.
        command = s.pd(target)
        tau = a.torque_scale*command-a.extra_damping*v
        rows.append(np.concatenate(([t], measured_q, measured_v, target, tau, q, v, command)))
        if k < ticks:
            s.torque_step(a.dt, tau)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame(rows, ['gc','gv','ptarget','gf','truth_gc','truth_gv','command'], s.n).to_csv(out, index=False)
    metadata = dict(origin='synthetic_mujoco_not_real_robot', joint_names=s.names,
                    kp=s.kp.tolist(), kd=s.kd.tolist(), options=vars(a),
                    sensor_noise_feedback=False, phase=phase.tolist(),
                    model_sha256=hashlib.sha256(Path(a.urdf).read_bytes()).hexdigest(),
                    csv_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                    last_row='terminal state; gf/command on final row not integrated')
    meta.write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'SYNTHETIC: {len(rows)} rows, {s.n} joints -> {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--urdf', required=True, help='URDF or MJCF XML')
    p.add_argument('--out', required=True); p.add_argument('--joint-config')
    p.add_argument('--kp', type=float, default=100.); p.add_argument('--kd', type=float, default=1.)
    p.add_argument('--dt', type=float, default=.001); p.add_argument('--seconds', type=float, default=6.)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--amplitude', type=float, default=.05); p.add_argument('--frequency', type=float, default=.5)
    p.add_argument('--noise-q', type=float, default=0.); p.add_argument('--noise-v', type=float, default=0.)
    p.add_argument('--torque-scale', type=float, default=1.)
    p.add_argument('--extra-damping', type=float, default=0.)
    add_viewer_args(p)
    collect(p.parse_args())
