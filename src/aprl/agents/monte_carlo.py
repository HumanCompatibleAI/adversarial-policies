"""Monte Carlo receding horizon control."""

from abc import ABC, abstractmethod
from multiprocessing import Pipe, Process

import gym
import numpy as np
from stable_baselines.common.vec_env import CloudpickleWrapper

from aprl.common.mujoco import MujocoState
from aprl.envs.multi_agent import MultiWrapper


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

    def get_state(self):
        """Serializes the qpos and qvel state of the MuJoCo emulator."""
        state = MujocoState.from_mjdata(self.sim.data).flatten()
        return state

    def get_radius(self):
        return self.env.env.RADIUS

    def set_state(self, x, sim_data=None, forward=True):
        """Restores qpos and qvel, calling forward() to derive other values."""
        state = MujocoState.from_flattened(x, self.sim)
        state.set_mjdata(self.sim.data, old_mujoco=True)
        if sim_data is not None:
            # set more than just qacc, qvel, qpos
            self.set_arbitrary_state(sim_data)
        if forward:
            self.sim.model.forward()  # put mjData in consistent state

    def set_arbitrary_state(self, sim_data):
        for k, v in sim_data.items():
            setattr(self.sim.data, k, v)

    def set_radius(self, r):
        self.env.env.RADIUS = r
        self.env.env._set_geom_radius()

    def reset(self):
        """See base class."""
        return self.env.reset()

    def step(self, a):
        """See base class."""
        return self.env.step(a)


class MujocoResettableWrapper(ResettableEnv, gym.Wrapper):
    """Converts a MujocoEnv into a ResettableEnv.

    Note all MuJoCo environments are resettable."""
    def __init__(self, env):
        """Wraps a MujocoEnv, adding get_state and set_state methods.
        :param env: a MujocoEnv. NOTE: it must not be wrapped in a TimeLimit."""
        if hasattr(env, '_max_episode_steps'):
            raise TypeError('Environment must not have a time limit '
                            '(try passing in env.unwrapped instead).')
        gym.Wrapper.__init__(self, env)
        self.sim = env.unwrapped.sim

    def get_state(self):
        """Serializes the qpos and qvel state of the MuJoCo emulator."""
        return MujocoState.from_mjdata(self.sim.data).flatten()

    def set_state(self, x):
        """Restores qpos and qvel, calling forward() to derive other values."""
        state = MujocoState.from_flattened(x, self.sim)
        state.set_mjdata(self.sim.data)
        self.sim.forward()  # put mjData in consistent state

    def reset(self):
        """See base class."""
        return self.env.reset()

    def step(self, a):
        """See base class."""
        return self.env.step(a)


class MonteCarlo(ABC):
    """Selects an action for a ResettableEnv by random search. Randomly samples
     fixed-length sequences of actions. Evaluates each trajectory in the
     environment, resetting the state to the original after each trajectory."""
    @abstractmethod
    def __init__(self, horizon, trajectories):
        """Constructs a MonteCarlo instance for env.
        :param horizon: the length of the trajectories to search over.
        :param trajectories: the number of trajectories to evaluate."""
        self.horizon = horizon
        self.trajectories = trajectories

    @abstractmethod
    def seed(self, seed):
        """Sets a seed for the PRNG for the action sequences.
        :param seed (int): a seed."""
        pass

    @abstractmethod
    def best_action(self, state):
        """Returns the best action out of a random search of action sequences.

        Generates self.trajectories action sequences, each of length
        self.horizon. The cumulative reward of each action sequence is computed,
        starting from state. The function returns the first action and the
        cumulative reward of the action sequences with the largest cumulative
        reward.
        :param state: a value returned by env.get_state().
        :return (action, reward): the best action found and associated reward."""
        pass


class MonteCarloSingle(MonteCarlo):
    """Selects an action for a ResettableEnv by random search.
       See base class for details. This implementation is not parallelized."""
    def __init__(self, env, horizon, trajectories):
        """See base class."""
        super().__init__(horizon, trajectories)
        self.env = env

    def seed(self, seed):
        """Sets a seed for the PRNG for the action sequences.
        :param seed (int): a seed."""
        self.env.action_space.np_random.seed(seed)

    def best_action(self, state):
        """Returns the best action out of a random search of action sequences.
        See base class for details.

        Search takes place in a single environment, which is reset to state
        before evaluating each action sequence."""
        res = []
        for _ in range(self.trajectories):
            self.env.set_state(state)
            us = [self.env.action_space.sample() for _ in range(self.horizon)]
            total_rew = 0
            for u in us:
                _ob, rew, done, _info = self.env.step(u)
                total_rew += rew
                if done:
                    break
            res.append((us[0], total_rew))
        self.env.set_state(state)
        best = max(res, key=lambda x: x[1])
        return best


def _worker(remote, parent_remote, dynamic_fn_wrapper, horizon, trajectories):
    parent_remote.close()
    dynamics = dynamic_fn_wrapper.var()
    dynamics.reset()
    mc = MonteCarloSingle(dynamics, horizon, trajectories)
    try:
        while True:
            cmd, x = remote.recv()
            if cmd == 'seed':
                mc.seed(x)
            elif cmd == 'search':
                best_u, best_r = mc.best_action(x)
                remote.send((best_u, best_r))
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('MonteCarloParallel worker: got KeyboardInterrupt')
    finally:
        dynamics.close()


class MonteCarloParallel(MonteCarlo):
    """Like MonteCarlo, but performs the random search in parallel."""
    # This implementation is inspired by Baselines SubprocVecEnv.
    def __init__(self, env_fns, horizon, trajectories, seed=0):
        """Launch subprocess workers and store configuration parameters.
        :param env_fns (list<()->ResettableEnv>): list of thunks.
        :param horizon (int): length of trajectories to search over.
        :param trajectories (int): minimum number of trajectories to evaluate.
               It will be rounded up to the nearest multiple of len(make_env)."""
        super().__init__(horizon, trajectories)
        nremotes = len(env_fns)
        # Integer ceiling of self.trajectories / nworkers
        traj_per_worker = (self.trajectories - 1) // nremotes + 1

        pipes = [Pipe() for _ in range(nremotes)]
        self.remotes, self.work_remotes = zip(*pipes)
        worker_cfgs = zip(self.work_remotes, self.remotes, env_fns)
        self.ps = []
        for i, (work_remote, remote, dynamic_fn) in enumerate(worker_cfgs):
            args = (work_remote, remote, CloudpickleWrapper(dynamic_fn),
                    horizon, traj_per_worker)
            process = Process(target=_worker, args=args)
            process.daemon = True
            # If the main process crashes, we should not cause things to hang
            process.start()
            self.ps.append(process)
        for remote in self.work_remotes:
            remote.close()

    def seed(self, seed):
        """See base class."""
        for i, remote in enumerate(self.remotes):
            remote.send(('seed', seed + i))

    def best_action(self, state):
        """Returns the best action out of a random search of action sequences."""
        for remote in self.remotes:
            remote.send(('search', state))
        results = [remote.recv() for remote in self.remotes]
        best = max(results, key=lambda x: x[1])
        return best

    def close(self):
        """Shuts down parallel workers."""
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()


def receding_horizon(monte_carlo, env):
    """Receding horizon control
    :param monte_carlo(MonteCarlo): a Monte Carlo controller for env or a clone of env.
    :param env(ResettableEnv): a resettable environment."""
    while True:
        state = env.get_state()
        a, _seq_rew = monte_carlo.best_action(state)
        ob, rew, done, info = env.step(a)
        yield a, ob, rew, done, info
        if done:
            break
