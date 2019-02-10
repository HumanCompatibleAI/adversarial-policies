from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gym
from gym import Env, Wrapper
import numpy as np

from aprl.utils import getattr_unwrapped


def _vec_space(space, num_agents):
    if isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete([space.n for _ in range(num_agents)])
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return gym.spaces.MultiDiscrete([space.nvec for _ in range(num_agents)])
    elif isinstance(space, gym.spaces.Box):
        low = np.asarray([space.low for _ in range(num_agents)])
        high = np.asarray([space.high for _ in range(num_agents)])
        return gym.spaces.Box(low=low, high=high)


def _check_space(num_agents, agent, full):
    if agent.shape is not None:
        assert full.shape[0] == num_agents
        assert full.shape[1:] == agent.shape


class MultiAgentEnv(Env):
    """Abstract class for multi-agent environments.
       This differs from the normal gym.Env in two ways:
         + step returns a vector of rewards
         + It has additional attributes num_agents, agent_action_space and
           agent_observation_space. The {action,observation}_space are Tuple's
           of num_agents agent_{action,observation}_space.
       This should really be a different class since it is-not a gym.Env,
       however it's very convenient to have it interoperate with the rest of the
       Gym infrastructure, so we'll abuse this. Sadly there is still no standard
       for multi-agent environments in Gym, issue #934 is working on it.
       """

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
        """Run one timestep of the environment's dynamics.
           Accepts an action_n of shape (self.num_agents, ) + self.action_space.

           :param action_n (ndarray): actions per agent.
           :return a tuple containing:
                obs_n (ndarray): observations per agent,
                                 shape (self.num_agents, ) + self.observation_space.
                reward_n (arraylike<float>): reward per agent.
                done (bool): episode over.
                info (dict): auxiliary diagnostic info."""
        raise NotImplementedError

    def reset(self):
        """Resets state of environment.

        Returns: observation (list<object>): per agent."""
        raise NotImplementedError


class MultiWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.agent_action_space = env.agent_action_space
        self.agent_observation_space = env.agent_observation_space


class MultiToSingleObs(Wrapper):
    """Wraps a MultiAgentEnv, changing the action and observation space
       to per-agent dimensions. Note this is inconsistent (i.e. the methods
       no longer take actions or return observations in the declared spaces),
       but is convenient when passing environments to policies that just
       extract the observation and action spaces."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.agent_observation_space
        self.action_space = env.agent_action_space


class MultiToSingleObsVec(VecEnvWrapper):
    """Wraps a VecEnv of MultiAgentEnv's, changing the action and observation
       space to per-agent dimensions. See MultiToSingleObs."""
    def __init__(self, venv):
        observation_space = venv.observation_space.agent_space
        action_space = venv.action_space.agent_space
        super().__init__(venv, observation_space=observation_space, action_space=action_space)

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()


# TODO: Consider if inheritance is correct
class FlattenSingletonEnv(MultiToSingleObs):
    """Wraps a Multi-Agent Environement with one agent and Makes it a Single-Agent Gym environment
    that can actually be thought of as a Gym environment."""

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observations, rewards, done, infos = self.env.step([action])
        return observations[0], rewards[0], done, infos

    def reset(self):
        return self.env.reset()[0]


class CurryEnv(MultiWrapper):
    """Wraps a Multi-Agent Environment fixing one of the players"""

    def __init__(self, env, agent, agent_to_fix=0):
        """
        Take a multi agent environment and fix one of the agents
        :param env: The multi-agent environment
        :param agent: The agent to be fixed
        :param agent_to_fix: The index of the agent that should be fixed
        :return: a new environment which behaves like "env" with the agent at position
                 "agent_to_fix" fixed as "agent"
        """
        super().__init__(env)
        self._env = env
        self._agent_to_fix = agent_to_fix
        self._agent = agent
        self._last_obs = None
        self._last_reward = None

    # TODO: Check if dones are handeled correctly (if you ever have an env in which it matters)
    def step(self, actions):
        action = self._agent.get_action(self._last_obs)
        actions.insert(self._agent_to_fix, action)
        observations, rewards, done, infos = self._env.step(actions)

        self._last_obs = observations.pop(self._agent_to_fix)
        self._last_reward = rewards.pop(self._agent_to_fix)

        return observations, rewards, done, infos

    def reset(self):
        observations = self._env.reset()
        self._last_obs = observations.pop(self._agent_to_fix)

        return observations


class DummyVecMultiEnv(DummyVecEnv):
    """Stand-in for DummyVecEnv when applied to MultiEnv's.
       Handles the larger reward size.
       Note SubprocVecEnv works out of the box."""
    def __init__(self, env_fns):
        super().__init__(env_fns)
        self.num_agents = getattr_unwrapped(self.envs[0], 'num_agents')
        self.buf_rews = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)


class SubprocVecMultiEnv(SubprocVecEnv):
    """Stand-in for SubprocVecEnv when applied to MultiEnv's.
       Includes some extra attributes."""
    def __init__(self, env_fns):
        super().__init__(env_fns)
        env = env_fns[0]()
        self.num_agents = getattr_unwrapped(env, 'num_agents')
        env.close()
