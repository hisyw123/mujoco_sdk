import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from mujoco_sim import Sim, simulate, frame, read_log, block
from collect import collect
from common import simulator_contract, check_log_metadata
from live_viewer import LiveViewer


class BackendTests(unittest.TestCase):
    def test_viewer_mirror_cannot_modify_physics(self):
        class Handle:
            def __init__(self):
                from types import SimpleNamespace
                self.cam = SimpleNamespace(lookat=np.zeros(3), distance=1.)
                self.running = True
            def lock(self):
                from contextlib import nullcontext
                return nullcontext()
            def sync(self): pass
            def is_running(self): return self.running
            def close(self): self.running = False
        s = Sim(self.model)
        s.reset([.1,.2],[0,0])
        h = Handle()
        with patch.dict('os.environ', {'DISPLAY': ':fake'}), patch('mujoco.viewer.launch_passive', return_value=h):
            with LiveViewer(s, True, 1000.) as view:
                view.data.qpos[:] = 9
                view.model.body_mass[:] = 100
                np.testing.assert_array_equal(s.state()[0], [.1,.2])
                self.assertFalse(np.all(s.m.body_mass == 100))
                h.running = False
                view.tick()
                self.assertIsNone(s.viewer)
            self.assertIsNone(s.viewer)

    def test_viewer_rejects_invalid_speed(self):
        for speed in [0,-1,float('nan')]:
            with self.assertRaises(ValueError): LiveViewer(Sim(self.model),False,speed)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = ROOT/'tests/pendulum.xml'

    def log(self, name='log', noise=0., scale=1.):
        out = self.root/f'{name}.csv'
        a = argparse.Namespace(out=str(out), urdf=str(self.model), joint_config=None,
            kp=10., kd=.5, dt=.001, seconds=.1, amplitude=.05, frequency=.5,
            noise_q=noise, noise_v=noise, torque_scale=scale, extra_damping=0., seed=7)
        collect(a)
        return out

    def args(self, path, **kwargs):
        d = dict(urdf=str(self.model), kp=10., kd=.5, joint_config=None, input=str(path),
                 mode='replay', stride=2, steps=0, offset=0, fixed_dt=0., model=None, ff_column=None)
        d.update(kwargs)
        return argparse.Namespace(**d)

    def test_identical_model_has_zero_residual_and_rollout_error(self):
        path = self.log()
        r = simulate(self.args(path))
        self.assertLess(np.abs(block(r,'err_tau',2)).max(), 1e-10)
        r = simulate(self.args(path, mode='rollout'))
        self.assertLess(np.abs(block(r,'sim_gc',2)-block(r,'gc',2)).max(), 1e-12)

    def test_sensor_noise_does_not_change_truth_or_target(self):
        clean, noisy = pd.read_csv(self.log()), pd.read_csv(self.log('noisy',noise=.01))
        for name in ('truth_gc','truth_gv','ptarget','gf'):
            np.testing.assert_array_equal(block(clean,name,2),block(noisy,name,2))
        self.assertGreater(np.abs(block(clean,'gc',2)-block(noisy,'gc',2)).max(),.001)

    def test_physical_mismatch_creates_nonzero_residual(self):
        r = simulate(self.args(self.log(scale=.8)))
        self.assertGreater(np.abs(block(r,'err_tau',2)).max(),1e-6)

    def test_nonuniform_timestamps_follow_each_dt(self):
        s = Sim(self.model, 10., .5)
        t = np.r_[0.,np.cumsum(np.tile([.0004,.0008,.0005],10))]
        s.reset(np.zeros(2),np.zeros(2))
        rows=[]
        for i,time in enumerate(t):
            q,v=s.state(); p=np.array([.1,-.1])*np.cos(time)
            rows.append(np.r_[time,q,v,p])
            if i+1<len(t): s.step(t[i+1]-time,p,np.zeros(2))
        path=self.root/'jitter.csv'
        frame(rows,['gc','gv','ptarget'],2).to_csv(path,index=False)
        r=simulate(self.args(path,stride=3,offset=1))
        self.assertLess(np.abs(block(r,'err_tau',2)).max(),1e-10)
        np.testing.assert_allclose(r.elapsedTime,t[1:len(t)-3:3])

    def test_fixed_dt_reference_interpolation_and_target_hold(self):
        path=self.log(scale=.7); df=pd.read_csv(path)
        r=simulate(self.args(path,mode='rollout',fixed_dt=.0017))
        np.testing.assert_allclose(np.diff(r.elapsedTime),.0017)
        expected=np.interp(r.elapsedTime,df.elapsedTime,df.gc_0)
        np.testing.assert_allclose(r.gc_0,expected)
        idx=np.searchsorted(df.elapsedTime,r.elapsedTime,side='right')-1
        np.testing.assert_allclose(r.ptarget_0,df.ptarget_0.to_numpy()[idx])
        self.assertGreater(np.abs(r.sim_gc_0-r.gc_0).max(),1e-7)

    def test_explicit_substeps_interval_endpoints(self):
        path=self.log()
        r=simulate(self.args(path,mode='rollout',stride=4,steps=2))
        self.assertEqual(len(r),51)
        self.assertAlmostEqual(r.elapsedTime.iloc[-1],.1)

    def test_name_mapping_and_zero_gains(self):
        cfg=self.root/'joints.json'
        cfg.write_text(json.dumps({'joints':[{'name':'joint_b','kp':0,'kd':0},
                                             {'name':'joint_a','kp':10,'kd':.5}]}))
        s=Sim(self.model,joint_config=cfg)
        self.assertEqual(s.qa.tolist(),[1,0])
        s.reset([.1,.2],[.3,.4])
        np.testing.assert_allclose(s.state()[0],[.1,.2])
        self.assertEqual(s.pd(np.zeros(2))[0],0.)
        self.assertGreater(np.linalg.eigvalsh(s.mass()).min(),0)

    def test_collection_metadata_rejects_wrong_gain(self):
        path=self.log()
        c=simulator_contract(dict(urdf=self.model,kp=11.,kd=.5))
        with self.assertRaisesRegex(ValueError,'kp'):
            check_log_metadata(path,c)

    def test_bad_timestamps_rejected(self):
        path=self.log(); df=pd.read_csv(path)
        df.loc[2,'elapsedTime']=-1
        df.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,'timestamps'): read_log(path,2)

    def test_state_reset_clears_previous_torque_and_time(self):
        s=Sim(self.model); s.reset([0,0],[0,0])
        s.step(.001,[.1,.1],np.ones(2)); s.reset([0,0],[0,0],.3)
        np.testing.assert_array_equal(s.d.qfrc_applied,0)
        self.assertEqual(s.d.time,.3)

    def test_free_base_and_equality_rejected(self):
        for name,joint,extra in [('free','<freejoint/>',''),
            ('eq','<joint name="a"/>','<equality><joint joint1="a"/></equality>')]:
            p=self.root/f'{name}.xml'
            p.write_text(f'<mujoco><worldbody><body>{joint}<geom size=".1"/></body></worldbody>{extra}</mujoco>')
            with self.assertRaises(ValueError): Sim(p)


if __name__=='__main__': unittest.main()
