from collections import namedtuple

import numpy as np


# TODO: Cythonize
class MujocoState(namedtuple('MujocoStateBase', 'qpos qvel qacc')):
    """Represents state from the MuJoCo simulator needed for planning,
       namely position and velocity."""

    @staticmethod
    def from_mjdata(data):
        return MujocoState(data.qpos, data.qvel, data.qacc)

    @staticmethod
    def from_flattened(flattened, sim):
        qpos = flattened[0:sim.model.nq]
        qvel = flattened[sim.model.nq:sim.model.nq + sim.model.nv]
        qacc = flattened[sim.model.nq + sim.model.nv:]
        return MujocoState(qpos, qvel, qacc)

    def set_mjdata(self, data, old_mujoco=False):
        if old_mujoco:
            data.qpos = self.qpos
            data.qvel = self.qvel
            data.qacc = self.qacc
        else:
            data.qpos[:] = self.qpos
            data.qvel[:] = self.qvel
            data.qacc[:] = self.qacc

    def flatten(self):
        return np.concatenate((self.qpos, self.qvel, self.qacc))
