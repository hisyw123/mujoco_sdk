#!/usr/bin/env python3
"""CLI/metadata wrapper only. Physics, PD, RNG, CSV capture and GUI run in C++."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from common import BIN, simulator_contract, add_viewer_args


def collect(a):
    started=time.perf_counter()
    out=Path(a.out); meta=out.with_suffix('.meta.json')
    if out.exists() or meta.exists():
        raise ValueError('output exists; choose a new path')
    contract=simulator_contract(vars(a))
    cmd=[str(BIN),'collect','-',str(out)]
    for key in ['urdf','joint_config','kp','kd','dt','seconds','seed','amplitude','frequency',
                'noise_q','noise_v','torque_scale','extra_damping','viewer_speed']:
        value=getattr(a,key,None)
        if value is not None: cmd += ['--'+key.replace('_','-'),str(value)]
    for key in ['viewer','realtime']:
        if getattr(a,key,False): cmd += ['--'+key]
    subprocess.run(cmd,check=True)
    timing=json.loads(Path(str(out)+'.collection.json').read_text())
    metadata=dict(origin='synthetic_mujoco_cpp_not_real_robot',options=vars(a),
                  sensor_noise_feedback=False,simulator=contract,**timing)
    for key in ['joint_names','kp','kd']: metadata[key]=contract[key]
    metadata['csv_sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
    metadata['model_sha256']=hashlib.sha256(Path(a.urdf).read_bytes()).hexdigest()
    metadata['command_wall_seconds']=time.perf_counter()-started
    meta.write_text(json.dumps(metadata,indent=2)+'\n')
    print(f"including model inspection/loading: {metadata['command_wall_seconds']:.3f} wall seconds")


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--urdf',required=True);p.add_argument('--out',required=True)
    p.add_argument('--joint-config')
    p.add_argument('--kp',type=float,default=100.);p.add_argument('--kd',type=float,default=1.)
    p.add_argument('--dt',type=float,default=.001);p.add_argument('--seconds',type=float,default=20.)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--amplitude',type=float,default=.05);p.add_argument('--frequency',type=float,default=.5)
    p.add_argument('--noise-q',type=float,default=0.);p.add_argument('--noise-v',type=float,default=0.)
    p.add_argument('--torque-scale',type=float,default=1.);p.add_argument('--extra-damping',type=float,default=0.)
    p.add_argument('--realtime',action='store_true',help='pace to wall time even without GUI')
    add_viewer_args(p)
    collect(p.parse_args())
