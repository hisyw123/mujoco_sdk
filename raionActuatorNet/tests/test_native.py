"""Native binary regression tests; Python engine is not used here."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import BIN, simulator_contract, build_features, cols
from collect import collect
from export_cpp import export_cpp
from train import ActuatorNet, Deployed


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.xml=ROOT/'tests/pendulum.xml'

    def record(self, name='log', **kw):
        d=dict(urdf=str(self.xml),out=str(self.root/(name+'.csv')),joint_config=None,
               kp=10.,kd=.5,dt=.001,seconds=.1,seed=7,amplitude=.05,frequency=.5,
               noise_q=0.,noise_v=0.,torque_scale=1.,extra_damping=0.)
        d.update(kw);collect(argparse.Namespace(**d))
        return Path(d['out'])

    def native(self, mode, source, name='result', extra=(), success=True):
        output=self.root/(name+'.csv')
        r=subprocess.run([str(BIN),mode,str(source),str(output),'--urdf',str(self.xml),
                          '--kp','10','--kd','.5',*extra],capture_output=True,text=True)
        if success:
            self.assertEqual(r.returncode,0,r.stderr)
            return pd.read_csv(output)
        self.assertNotEqual(r.returncode,0)

    def test_native_engine_contract(self):
        c=simulator_contract(dict(urdf=self.xml,kp=0.,kd=0.))
        self.assertEqual(c['mujoco'],'3.13.1')
        self.assertEqual(c['backend'],'mujoco_cpp_explicit_pd_v1')
        self.assertEqual(c['kp'],[0,0])

    def test_native_zero_residual_and_rollout(self):
        path=self.record()
        r=self.native('replay',path,extra=['--stride','2'])
        self.assertLess(np.abs(r[cols(r,'err_tau')]).to_numpy().max(),1e-10)
        r=self.native('rollout',path,'roll')
        self.assertLess(np.abs(r[cols(r,'sim_gc')].to_numpy()-r[cols(r,'gc')].to_numpy()).max(),1e-12)

    def test_multilayer_gru_three_features_matches_torch(self):
        torch.manual_seed(3)
        cfg=dict(n_joints=2,hidden=7,layers=2,features=['pos_err','vel','torque'],
                 x_mean=[.1]*6,x_std=[.5]*6,y_mean=[.03]*2,y_std=[.2]*2)
        net=ActuatorNet(6,7,2,2).eval()
        export_cpp(net,cfg,self.root/'net.json')
        r=self.native('replay',self.record(),extra=['--model',str(self.root/'net.json'),'--stride','2'])
        dep=Deployed(net,cfg).eval();h=torch.zeros(2,1,7);pred=[]
        with torch.no_grad():
            for x in build_features(r,cfg['features']):
                y,h=dep(torch.tensor(x)[None],h);pred.append(y.numpy()[0])
        self.assertLess(np.max(np.abs(np.array(pred)-r[cols(r,'ff')].to_numpy())),2e-6)

    def test_reverse_mapping_zero_residual(self):
        config=self.root/'joints.json'
        config.write_text(json.dumps(dict(joints=[dict(name='joint_b',kp=0,kd=0),dict(name='joint_a',kp=10,kd=.5)])))
        path=self.record(joint_config=str(config))
        r=self.native('replay',path,extra=['--joint-config',str(config)])
        self.assertLess(np.abs(r[cols(r,'err_tau')]).to_numpy().max(),1e-10)

    def test_reject_time_reversal_and_overwrite(self):
        path=self.record();df=pd.read_csv(path);df.loc[2,'elapsedTime']=-.1;df.to_csv(path,index=False)
        self.native('replay',path,success=False)
        with self.assertRaises(ValueError): self.record()

    def test_fixed_grid_and_substeps(self):
        path=self.record()
        r=self.native('rollout',path,'fixed',['--fixed-dt','.0017'])
        np.testing.assert_allclose(np.diff(r.elapsedTime),.0017,atol=1e-12)
        r=self.native('rollout',path,'steps',['--stride','4','--steps','2'])
        self.assertEqual(len(r),51)
        self.assertAlmostEqual(r.elapsedTime.iloc[-1],.1)


if __name__=='__main__':unittest.main()
