import numpy as np
import gym
from gym import Env, Wrapper
from gym.spaces import Tuple
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from aprl.utils import getattr_unwrapped


def _vec_space(space, num_agents):
    if isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete([space.n for _ in range(num_agents)])
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return gym.spaces.MultiDiscrete([space.nvec for _ in range(num_agents)])
    elif isinstance(space, gym.shapes.Box):
        low = np.asarray([space.low for _ in range(num_agents)])
        high = np.asarray([space.high for _ in range(num_agents)])
        return gym.spaces.Box(low=low, high=high)


def _check_space(num_agents, agent, full):
    if agent.shape is not None:
        assert full.shape[0] == num_agents
        assert full.shape[1:] == agent.shape


class MultiAgentEnv(Env):
    '''Abstract class for multi-agent environments.
       This differs from the normal gym.Env in two ways:
         + step returns a vector of rewards
         + It has additional attributes num_agents, agent_action_space and
           agent_observation_space. The {action,observation}_space are Tuple's
           of num_agents agent_{action,observation}_space.
       This should really be a different class since it is-not a gym.Env,
       however it's very convenient to have it interoperate with the rest of the
       Gym infrastructure, so we'll abuse this. Sadly there is still no standard
       for multi-agent environments in Gym, issue #934 is working on it.
       '''

    def __init__(self, num_agents, agent_action_space, agent_observation_space,
                 action_space=None, observation_space=None):
        self.num_agents = num_agents
        self.agent_action_space = agent_action_space
        self.agent_observation_space = agent_observation_space

        if action_space is None:
            action_space = _vec_space(agent_action_space, num_agents)
        if observation_space is None:
            observation_space = _vec_space(agent_observation_space, num_agents)

        self.action_space = action_space
        self.action_space.agent_space = agent_action_space
        self.observation_space = observation_space
        self.observation_space.agent_space = agent_observation_space

        _check_space(num_agents, agent_action_space, action_space)
        _check_space(num_agents, agent_observation_space, observation_space)

    def step(self, action_n):
        '''Run one timestep of the environment's dynamics.
           Accepts an action_n of self.num_agents long, each containing
           an action from self.action_space.

           Args:
                action_n (list<object>): actions per agent.
            Returns:
                obs_n (list<object>): observations per agent.
                reward_n (list<float>): reward per agent.
                done (list<boolean>): done per agent.
                info (dict): auxiliary diagnostic info.
        '''
        raise NotImplementedError

    def reset(self):
        '''Resets state of environment.

        Returns: observation (list<object>): per agent.'''
        raise NotImplementedError


class MultiToSingleObs(Wrapper):
    '''Wraps a MultiAgentEnv, changing the action and observation space
       to per-agent dimensions. Note this is inconsistent (i.e. the methods
       no longer take actions or return observations in the declared spaces),
       but is convenient when passing environments to policies that just
       extract the observation and action spaces.'''
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.agent_observation_space
        self.action_space = env.agent_observation_space


class MultiToSingleObsVec(VecEnvWrapper):
    '''Wraps a VecEnv of MultiAgentEnv's, changing the action and observation
       space to per-agent dimensions. See MultiToSingleObs.'''
    def __init__(self, venv):
        observation_space = venv.observation_space.agent_space
        action_space = venv.action_space.agent_space
        super().__init__(venv,
                         observation_space=observation_space,
                         action_space=action_space)

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()


class DummyVecMultiEnv(DummyVecEnv):
    '''Stand-in for DummyVecEnv when applied to MultiEnv's.
       Handles the larger reward size.
       Note SubprocVecEnv works out of the box.'''
    def __init__(self, env_fns):
        super().__init__(env_fns)
        self.num_agents = getattr_unwrapped(self.envs[0], 'num_agents')
        self.buf_rews = np.zeros((self.num_envs, self.num_agents),
                                 dtype=np.float32)

class SubprocVecMultiEnv(SubprocVecEnv):
    '''Stand-in for SubprocVecEnv when applied to MultiEnv's.
       Includes some extra attributes.'''
    def __init__(self, env_fns):
        super().__init__(env_fns)
        env = env_fns[0]()
        self.num_agents = getattr_unwrapped(env[0], 'num_agents')
        env.close()