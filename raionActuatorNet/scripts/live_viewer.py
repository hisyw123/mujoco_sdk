"""Optional read-only visual mirror: viewer interaction never changes training physics."""
import copy
import os
import time
import threading

import mujoco
import numpy as np


def add_viewer_args(parser):
    parser.add_argument('--viewer', action='store_true', help='show a MuJoCo window (default: headless)')
    parser.add_argument('--viewer-speed', type=float, default=1., help='playback speed, 1=real time')


class LiveViewer:
    def __init__(self, sim, enabled=False, speed=1.):
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError('viewer-speed must be positive and finite')
        self.sim, self.enabled, self.speed = sim, enabled, speed
        self.handle = None
        self.threads = []

    def __enter__(self):
        if not self.enabled:
            return self
        if os.name == 'posix' and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            raise RuntimeError('--viewer needs a graphical desktop (DISPLAY); omit it for headless')
        import mujoco.viewer
        self.model = copy.copy(self.sim.m)
        self.data = mujoco.MjData(self.model)
        self._copy_state()
        before = set(threading.enumerate())
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        # MuJoCo 3.8 launch_passive owns a daemon render thread but Handle.close()
        # only requests exit. Join that thread before Python/GLFW teardown.
        self.threads = [t for t in threading.enumerate() if t not in before
                        and '_launch_internal' in t.name]
        with self.handle.lock():
            self.handle.cam.lookat[:] = self.model.stat.center
            self.handle.cam.distance = max(1., 2.5*self.model.stat.extent)
        self.wall_start, self.sim_start = time.monotonic(), self.sim.d.time
        self.next_frame = self.sim.d.time
        self.sim.viewer = self
        self.handle.sync()
        print('Viewer: visual mirror only; closing the window continues headless. Speed:', self.speed)
        return self

    def _copy_state(self):
        source = self.sim.d
        self.data.qpos[:] = source.qpos
        self.data.qvel[:] = source.qvel
        self.data.time = source.time
        mujoco.mj_forward(self.model, self.data)

    def tick(self):
        if self.handle is None:
            return
        if not self.handle.is_running():
            self.sim.viewer = None
            self._close()
            print('Viewer closed; continuing headless.')
            return
        now = self.sim.d.time
        if now < self.next_frame:
            return
        delay = self.wall_start+(now-self.sim_start)/self.speed-time.monotonic()
        while delay > 0 and self.handle.is_running():
            time.sleep(min(delay,.02))
            delay = self.wall_start+(now-self.sim_start)/self.speed-time.monotonic()
        if not self.handle.is_running():
            return
        with self.handle.lock():
            self._copy_state()
        self.handle.sync()
        self.next_frame = now+self.speed/60.

    def __exit__(self, *exc):
        self.sim.viewer = None
        self._close()
        return False

    def _close(self):
        if self.handle is not None:
            self.handle.close()
        for thread in self.threads:
            thread.join(timeout=5.)
            if thread.is_alive():
                raise RuntimeError('MuJoCo viewer render thread did not stop')
        self.handle = None
