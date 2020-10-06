import abc
from collections import namedtuple

import gym
import numpy as np


# TODO: Cythonize
class MujocoState(namedtuple("MujocoStateBase", "qpos qvel")):
    """Represents state from the MuJoCo simulator needed for planning,
    namely position and velocity."""

    @staticmethod
    def from_mjdata(data):
        return MujocoState(data.qpos, data.qvel)

    @staticmethod
    def from_flattened(flattened, sim):
        qpos = flattened[0 : sim.model.nq]
        qvel = flattened[sim.model.nq : sim.model.nq + sim.model.nv]
        return MujocoState(qpos, qvel)

    def set_mjdata(self, data):
        try:
            data.qpos[:] = self.qpos
            data.qvel[:] = self.qvel
        except ValueError:  # older mujoco version
            data.qpos = self.qpos
            data.qvel = self.qvel

    def flatten(self):
        return np.concatenate((self.qpos, self.qvel))


class ResettableEnv(gym.Env, abc.ABC):
    """A Gym environment that can be reset to an arbitrary state."""

    @abc.abstractmethod
    def get_state(self):
        """Returns a serialized representation of the current state."""
        pass

    @abc.abstractmethod
    def set_state(self, x):
        """Restores the environment to a previously saved state.
        :param x: return value of a previous call to get_state()."""
        pass
