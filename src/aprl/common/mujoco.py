from abc import abstractmethod
from collections import namedtuple

import gym
import numpy as np

from aprl.envs.multi_agent import MultiWrapper


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


class ResettableEnv(gym.Env):
    """A Gym environment that can be reset to an arbitrary state."""
    @abstractmethod
    def get_state(self):
        """Returns a serialized representation of the current state."""
        pass

    @abstractmethod
    def set_state(self, x):
        """Restores the environment to a previously saved state.
        :param x: return value of a previous call to get_state()."""
        pass


class OldMujocoResettableWrapper(ResettableEnv, MultiWrapper):
    """Converts a MujocoEnv into a ResettableEnv.

    Note all MuJoCo environments are resettable."""
    def __init__(self, env):
        """Wraps a MujocoEnv, adding get_state and set_state methods.
        :param env: a MujocoEnv. NOTE: it must not be wrapped in a TimeLimit."""
        if hasattr(env, '_max_episode_steps'):
            raise TypeError('Environment must not have a time limit '
                            '(try passing in env.unwrapped instead).')
        gym.Wrapper.__init__(self, env)
        self.sim = env.unwrapped.env_scene

    def get_full_state(self):
        state_dict = {}
        for k, v in type(self.sim.data._wrapped.contents).__dict__['_fields_']:
            if k not in ['contact', 'buffer']:
                state_dict[k] = getattr(self.sim.data, k)
        return state_dict

    def get_state(self, all_data=False):
        """Serializes the qpos and qvel state of the MuJoCo emulator."""
        state = MujocoState.from_mjdata(self.sim.data).flatten()
        if all_data:
            full_state = self.get_full_state()
            try:
                radius = self.env.env.RADIUS
            except AttributeError:
                # this environment does not have a radius.
                radius = None
            return (state, full_state, radius)
        return state

    def set_state(self, x, sim_data=None, radius=None, forward=True):
        """Restores qpos and qvel, calling forward() to derive other values."""
        state = MujocoState.from_flattened(x, self.sim)
        if radius is not None:
            # call this first since it runs forward automatically.
            self.env.env.RADIUS = radius
            self.env.env._set_geom_radius()
        state.set_mjdata(self.sim.data, old_mujoco=True)
        if sim_data is not None:
            # set more than just qacc, qvel, qpos
            for k, v in sim_data.items():
                setattr(self.sim.data, k, v)
        if forward:
            self.sim.model.forward()  # put mjData in consistent state

    def reset(self):
        """See base class."""
        return self.env.reset()

    def step(self, a):
        """See base class."""
        return self.env.step(a)
