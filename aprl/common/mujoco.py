from collections import namedtuple

import numpy as np


# TODO: Cythonize
class MujocoState(namedtuple('MujocoStateBase', 'qpos qvel')):
    """Represents state from the MuJoCo simulator needed for planning,
       namely position and velocity."""

    @staticmethod
    def from_mjdata(data):
        return MujocoState(data.qpos, data.qvel)

    @staticmethod
    def from_flattened(flattened, sim):
        qpos = flattened[0:sim.model.nq]
        qvel = flattened[sim.model.nq:sim.model.nq + sim.model.nv]
        return MujocoState(qpos, qvel)

    def set_mjdata(self, data):
        data.qpos[:] = self.qpos
        data.qvel[:] = self.qvel

    def flatten(self):
        return np.concatenate((self. qpos, self.qvel))
