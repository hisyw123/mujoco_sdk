#!/usr/bin/env python3
"""Select a robot, then run C++ collection/physics/inference with Python GRU training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from robot_profiles import CATALOG, prepare
from common import simulator_contract


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['models','inspect','collect','train','evaluate'])
    p.add_argument('--robot',choices=list(CATALOG),default='test')
    p.add_argument('--xml',help='custom independent fixed-base XML, overrides robot selection')
    p.add_argument('--run',default='runs/native_demo')
    p.add_argument('--seconds',type=float,default=20.)
    p.add_argument('--dt',type=float,default=.001)
    p.add_argument('--kp',type=float,default=10.);p.add_argument('--kd',type=float,default=.5)
    p.add_argument('--joint-config')
    p.add_argument('--amplitude',type=float,default=.05);p.add_argument('--frequency',type=float,default=.5)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--stride',type=int,default=2);p.add_argument('--smooth',type=int,default=1)
    p.add_argument('--viewer',action='store_true');p.add_argument('--realtime',action='store_true')
    p.add_argument('--viewer-speed',type=float,default=1.)
    p.add_argument('--torque-scale',type=float,default=.9)
    p.add_argument('--extra-damping',type=float,default=.1)
    a=p.parse_args()
    if a.action=='models':
        for key,path in CATALOG.items(): print(key, '->',path, '(stand27 copy)' if key!='test' else '(test only)')
        return
    model=Path(a.xml).resolve() if a.xml else prepare(a.robot)
    run=Path(a.run).resolve()
    base=['--urdf',str(model),'--kp',str(a.kp),'--kd',str(a.kd)]
    if a.joint_config: base+=['--joint-config',str(Path(a.joint_config).resolve())]
    if a.action=='inspect':
        print('MODEL:',model)
        print(json.dumps(simulator_contract(dict(urdf=model,kp=a.kp,kd=a.kd,
                         joint_config=a.joint_config)),indent=2))
        return
    if a.action=='collect':
        if a.robot!='test' and not a.xml:
            print('STAND-27: fixed base, fixed grippers, no contacts. Gains are TEST values, not approved hardware gains.',flush=True)
        args=['scripts/collect.py',*base,'--out',str(run/'log.csv'),'--seconds',str(a.seconds),
              '--dt',str(a.dt),'--seed',str(a.seed),'--amplitude',str(a.amplitude),'--frequency',str(a.frequency),
              '--torque-scale',str(a.torque_scale),'--extra-damping',str(a.extra_damping)]
        if a.realtime: args+=['--realtime']
    elif a.action=='train':
        args=['scripts/train.py',str(run/'log.csv'),*base,'--out',str(run/'model'),
              '--stride',str(a.stride),'--smooth',str(a.smooth),'--epochs',str(a.epochs),'--batch','16']
    else:
        args=['scripts/evaluate.py',str(run/'log.csv'),*base,'--model',str(run/'model'),
              '--out',str(run/'eval'),'--stride',str(a.stride),'--smooth',str(a.smooth)]
    if a.viewer and a.action!='train': args+=['--viewer','--viewer-speed',str(a.viewer_speed)]
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True)


if __name__=='__main__': main()
