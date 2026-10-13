#!/usr/bin/env python3
"""MuJoCo implementation of the original augmented_sim replay/rollout CSV protocol.

PD is sampled explicit torque, NOT RaiSim's built-in implicit PD. No contacts or
native actuators: this backend identifies a fixed-base robot on a stand.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pandas as pd
from live_viewer import LiveViewer, add_viewer_args


class Sim:
    def __init__(self, model, kp=100., kd=1., joint_config=None):
        source = ET.parse(model).getroot()
        if source.tag == 'robot' and source.findall('.//mimic'):
            raise ValueError('URDF mimic is not supported; do not silently discard coupling')
        self.m = mujoco.MjModel.from_xml_path(str(Path(model).resolve()))
        m = self.m
        if not (m.nq == m.nv == m.njnt > 0) or any(
                t not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) for t in m.jnt_type):
            raise ValueError('fixed base with scalar hinge/slide joints required')
        if m.neq or m.ntendon:
            raise ValueError('equality/tendon coupling not supported by this residual formulation')
        names = [m.joint(i).name for i in range(m.njnt)]
        if any(not name for name in names):
            raise ValueError('all joints must be named')
        self.n = m.nv
        self.names = names
        self.kp, self.kd = np.full(self.n, kp), np.full(self.n, kd)
        if joint_config:
            joints = json.loads(Path(joint_config).read_text())['joints']
            self.names = [j['name'] for j in joints]
            if len(self.names) != self.n or set(self.names) != set(names):
                raise ValueError('joint-config must name each model joint exactly once in CSV order')
            self.kp = np.array([j['kp'] for j in joints], dtype=float)
            self.kd = np.array([j['kd'] for j in joints], dtype=float)
        ids = np.array([names.index(n) for n in self.names])
        self.qa, self.va = m.jnt_qposadr[ids], m.jnt_dofadr[ids]
        if not np.isfinite([self.kp, self.kd]).all() or min(self.kp.min(), self.kd.min()) < 0:
            raise ValueError('PD gains must be finite and nonnegative')
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        self.d = mujoco.MjData(m)
        self.viewer = None

    def reset(self, q, v, time=0.):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.qa], self.d.qvel[self.va], self.d.time = q, v, time
        mujoco.mj_forward(self.m, self.d)

    def state(self):
        return self.d.qpos[self.qa].copy(), self.d.qvel[self.va].copy()

    def mass(self):
        dense = np.empty((self.n, self.n))
        try:
            mujoco.mj_fullM(self.m, self.d, dense)
        except TypeError:
            mujoco.mj_fullM(self.m, dense, self.d.qM)
        return dense[np.ix_(self.va, self.va)]

    def pd(self, p):
        q, v = self.state()
        return self.kp*(p-q)-self.kd*v

    def torque_step(self, dt, tau):
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(tau).all():
            raise ValueError('invalid dt or torque')
        self.m.opt.timestep = dt
        before = self.d.time
        warnings = self.d.warning.number.copy()
        self.d.qfrc_applied[:] = 0
        self.d.qfrc_applied[self.va] = tau
        mujoco.mj_step(self.m, self.d)
        if (not np.isfinite(self.d.qpos).all() or not np.isfinite(self.d.qvel).all()
                or np.any(self.d.warning.number > warnings)
                or not np.isclose(self.d.time, before+dt, rtol=0, atol=1e-10)):
            raise RuntimeError('MuJoCo instability/warning; reduce timestep or review model/gains')
        if self.viewer is not None:
            self.viewer.tick()

    def step(self, dt, p, ff):
        self.torque_step(dt, self.pd(p)+ff)


def block(df, prefix, n):
    return df[[f'{prefix}_{i}' for i in range(n)]].to_numpy(dtype=float)


def read_log(path, n):
    df = pd.read_csv(path).drop_duplicates('elapsedTime', keep='first').reset_index(drop=True)
    t = df.elapsedTime.to_numpy(dtype=float)
    if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('need at least two strictly increasing finite timestamps')
    actual = {c for c in df if c.startswith('gc_') and c[3:].isdigit()}
    if actual != {f'gc_{i}' for i in range(n)}:
        raise ValueError('CSV joint count/order does not match model; use --joint-config for mapping')
    for prefix in ('gc', 'gv', 'ptarget'):
        if not np.isfinite(block(df, prefix, n)).all():
            raise ValueError(f'non-finite {prefix}')
    return df


class Net:
    def __init__(self, path, n):
        import torch
        self.torch = torch
        torch.set_num_threads(1)
        self.net = torch.jit.load(str(path), map_location='cpu').eval()
        if self.net.n_joints != n:
            raise ValueError('network joint count mismatch')
        self.h = torch.zeros(self.net.layers, 1, self.net.hidden)

    def predict(self, error, v, tau):
        values = [error, v, tau]
        x = np.concatenate([values[int(i)] for i in self.net.features])
        with self.torch.no_grad():
            y, self.h = self.net(self.torch.tensor(x, dtype=self.torch.float32)[None], self.h)
        return y.numpy()[0].astype(float)


def frame(rows, names, n):
    columns = ['elapsedTime']+[f'{name}_{j}' for name in names for j in range(n)]
    return pd.DataFrame(rows, columns=columns)


def simulate(a):
    s = Sim(a.urdf, a.kp, a.kd, a.joint_config)
    if getattr(a, 'viewer', False) and a.mode != 'rollout':
        raise ValueError('--viewer supports rollout, not training-label replay')
    # Match the viewer clock to logs that do not start at zero.
    s.d.time = float(pd.read_csv(a.input, nrows=1)['elapsedTime'].iloc[0])
    with LiveViewer(s, getattr(a, 'viewer', False), getattr(a, 'viewer_speed', 1.)):
        return _simulate(a, s)


def _simulate(a, s):
    df = read_log(a.input, s.n)
    t = df.elapsedTime.to_numpy()
    q, v, p = [block(df, k, s.n) for k in ('gc', 'gv', 'ptarget')]
    gf = block(df, 'gf', s.n) if 'gf_0' in df else s.kp*(p-q)-s.kd*v
    ff_log = block(df, a.ff_column, s.n) if a.ff_column else None
    net = Net(a.model, s.n) if a.model else None
    ff, rows = np.zeros(s.n), []
    if a.mode == 'replay':
        for k in range(a.offset, len(t)-a.stride, a.stride):
            end = k+a.stride
            s.reset(q[k], v[k], t[k])
            mass = s.mass()
            if net:
                ff = net.predict(p[k]-q[k], v[k], gf[k])
            if ff_log is not None:
                ff = ff_log[k]
            duration = t[end]-t[k]
            if a.steps:
                for _ in range(a.steps):
                    s.step(duration/a.steps, p[k], ff)
            else:
                for i in range(k, end):
                    s.step(t[i+1]-t[i], p[i], ff)
            qs, vs = s.state()
            eq, ev = q[end]-qs, v[end]-vs
            rows.append(np.concatenate(([t[k]], q[k], v[k], p[k], gf[k], qs, vs, eq, ev,
                                        mass@ev/duration, ff)))
        if not rows:
            raise ValueError('no complete prediction interval')
        return frame(rows, ['gc','gv','ptarget','gf','sim_gc','sim_gv','err_gc','err_gv','err_tau','ff'], s.n)
    # Schedule faithfully follows original fixed-dt / steps / log timing rules.
    if a.fixed_dt:
        ts = t[0]+np.arange(int(np.floor((t[-1]-t[0]+1e-12)/a.fixed_dt))+1)*a.fixed_dt
        indices = np.clip(np.searchsorted(t, ts, side='right')-1, 0, len(t)-1)
        targets, every = p[indices], a.stride
    elif a.steps:
        starts = list(range(0, len(t)-a.stride, a.stride))
        if not starts:
            raise ValueError('no complete prediction interval')
        ts = np.array([t[k]+j*(t[k+a.stride]-t[k])/a.steps for k in starts for j in range(a.steps)]
                      +[t[starts[-1]+a.stride]])
        targets = np.array([p[k] for k in starts for _ in range(a.steps)]+[p[starts[-1]+a.stride]])
        every = a.steps
    else:
        ts, targets, every = t, p, a.stride
    qr = np.stack([np.interp(ts, t, q[:,j]) for j in range(s.n)], axis=1)
    vr = np.stack([np.interp(ts, t, v[:,j]) for j in range(s.n)], axis=1)
    s.reset(qr[0], vr[0], ts[0])
    for i, time in enumerate(ts):
        qs, vs = s.state()
        if i % every == 0 and i+1 < len(ts):
            if net:
                ff = net.predict(targets[i]-qs, vs, s.pd(targets[i])+ff)
            if ff_log is not None:
                ff = ff_log[(i//every)*a.stride]
        rows.append(np.concatenate(([time], qr[i], vr[i], targets[i], qs, vs, ff)))
        if i+1 < len(ts):
            s.step(ts[i+1]-time, targets[i], ff)
    return frame(rows, ['gc','gv','ptarget','sim_gc','sim_gv','ff'], s.n)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['replay','rollout'])
    p.add_argument('input'); p.add_argument('output')
    p.add_argument('--urdf', required=True, help='URDF or MJCF XML')
    p.add_argument('--joint-config', help='explicit CSV joint order and per-joint kp/kd JSON')
    p.add_argument('--kp', type=float, default=100.); p.add_argument('--kd', type=float, default=1.)
    p.add_argument('--stride', type=int, default=1); p.add_argument('--steps', type=int, default=0)
    p.add_argument('--offset', type=int, default=0); p.add_argument('--fixed-dt', type=float, default=0.)
    p.add_argument('--model'); p.add_argument('--ff-column')
    add_viewer_args(p)
    a = p.parse_args()
    if (a.stride < 1 or a.steps < 0 or not 0 <= a.offset < a.stride
            or not np.isfinite(a.fixed_dt) or a.fixed_dt < 0):
        p.error('invalid stride/steps/offset/fixed-dt')
    if a.fixed_dt and (a.steps or a.mode == 'replay' or a.ff_column):
        p.error('fixed-dt only supports rollout without steps/ff-column')
    if a.offset and a.mode == 'rollout':
        p.error('offset only supports replay')
    out = Path(a.output)
    if out.exists():
        p.error('output exists; choose a new path')
    result = simulate(a)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f'MuJoCo {mujoco.__version__}: {a.mode} wrote {len(result)} rows -> {out}')


if __name__ == '__main__':
    main()
