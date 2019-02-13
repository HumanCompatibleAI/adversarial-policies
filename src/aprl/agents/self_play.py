from abc import ABC, abstractmethod

import numpy as np

from aprl.utils import getattr_unwrapped

def e_arr(nenv, multi_space):
    return [np.zeros((nenv, ) + space.shape, dtype=space.dtype.name)
           for space in multi_space.spaces]

class AbstractMultiEnvRunner(ABC):
    """MultiEnv equivalent of AbstractEnvRunner in baselines."""
    def __init__(self, *, env, agents, nsteps):
        self.env = env
        self.agents = agents
        self.nagents = len(agents)
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.obs = e_arr(nenv, env.observation_space)
        obs = env.reset()
        for sobs, eobs in zip(self.obs, obs):
            sobs[:] = eobs
        self.nsteps = nsteps
        self.states = [model.initial_state for model in agents]
        self.dones = np.zeros(nenv, dtype=np.bool)

    @abstractmethod
    def run(self):
        raise NotImplementedError


class SelfPlay(object):
    """Self-play amongst randomly sampled members of a population.

    Assumes symmetric multi-agent environment so that any policy can play as any agent."""

    TRAINING_TYPES = ['best']

    def __init__(self, population_size, training_type, runner_class, env):
        self.population_size = int(population_size)
        self.num_agents = getattr_unwrapped(env, 'num_agents')
        if training_type not in self.TRAINING_TYPES:
            raise NotImplementedError
        self.training_type = training_type
        self.runner_class = runner_class
        self.env = env
        self.nenv = env.num_envs
        self.models = [None for _ in range(population_size)]

    def rollout(self, nsteps):
        # Select num_agents models to play each other.
        players = np.random.choice(self.population_size,
                                   size=self.num_agents,
                                   replace=False)
        if self.training_type == 'best':
            # Use latest version of models
            agents = [self.models[pi] for pi in players]
        else:
            # SOMEDAY: support other training types, e.g. random history
            raise NotImplementedError

        # Generate a rollout
        runner = self.runner_class(env=self.env, agents=agents, nsteps=nsteps)
        trajs, epinfos = runner.run()

        # epinfos contains reward for each player.
        # Extract out the relevant reward for each player, so it is in the
        # standard single-agent baselines format (plus an extra r_joint key).
        agent_epinfos = [
            [
                {
                    'r': epinfo['r{:d}'.format(i)],
                    'r_joint': epinfo['r'],
                    'l': epinfo['l'],
                    't': epinfo['t'],
                }
                for epinfo in epinfos
            ]
            for i in range(self.num_agents)
        ]

        return list(zip(players, agents, trajs, agent_epinfos))

    def learn(self, *args, **kwargs):
        raise NotImplementedError
