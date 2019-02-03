import numpy as np
from baselines.common.runners import AbstractEnvRunner

from aprl.utils import getattr_unwrapped


class AbstractMultiEnvRunner(AbstractEnvRunner):
    def __init__(self, *, env, models, nsteps):
        super().__init__(env=env, model=models[0], nsteps=nsteps)
        self.nmodels = len(models)
        self.models = models
        self.states = [model.initial_state for model in models]
        self.dones = np.zeros(self.nenv, dtype=np.bool)


class SelfPlay(object):
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
            models = [self.models[pi] for pi in players]
        else:
            # SOMEDAY: support other training types, e.g. random history
            raise NotImplementedError

        # Generate a rollout
        runner = self.runner_class(env=self.env, models=models, nsteps=nsteps)
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

        return list(zip(players, models, trajs, agent_epinfos))

    def learn(self, *args, **kwargs):
        raise NotImplementedError
