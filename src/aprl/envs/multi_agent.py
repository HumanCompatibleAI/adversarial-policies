import gym
from gym import Env, Wrapper
import numpy as np
from stable_baselines.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from aprl.utils import getattr_unwrapped


class MultiAgentEnv(Env):
    """Abstract class for multi-agent environments.
       This differs from the normal gym.Env in two ways:
         + step returns a tuple of observations, each a numpy array, and a tuple of rewards.
         + It has an additional attribute num_agents.
       Moreover, we guarantee that observation_space and action_space are a Tuple, with the
       i'th element corresponding to the i'th agents observation and action space.

       This should really be a different class since it is-not a gym.Env,
       however it's very convenient to have it interoperate with the rest of the
       Gym infrastructure, so we'll abuse this. Sadly there is still no standard
       for multi-agent environments in Gym, issue #934 is working on it."""
    def __init__(self, num_agents):
        self.num_agents = num_agents
        assert len(self.action_space.spaces) == num_agents
        assert len(self.observation_space.spaces) == num_agents

    def step(self, action_n):
        """Run one timestep of the environment's dynamics.
           Accepts an action_n consisting of a self.num_agents length list.

           :param action_n (list<ndarray>): actions per agent.
           :return a tuple containing:
                obs_n (tuple<ndarray>): observations per agent.
                reward_n (tuple<float>): reward per agent.
                done (bool): episode over.
                info (dict): auxiliary diagnostic info."""
        raise NotImplementedError

    def reset(self):
        """Resets state of environment.
        :return: observation (list<ndarray>): per agent."""
        raise NotImplementedError


class MultiWrapper(Wrapper, MultiAgentEnv):
    def __init__(self, env):
        Wrapper.__init__(self, env)
        MultiAgentEnv.__init__(self, getattr_unwrapped(env, 'num_agents'))


class FakeSingleSpaces(gym.Env):
    """Creates a fake gym.Env that has action and observation spaces corresponding to
       those of agent_id in a MultiEnv env. This is useful for functions that construct
       policy or reward networks given an environment. It will throw an error if reset,
       step or other methods are called."""
    def __init__(self, env, agent_id=0):
        self.observation_space = env.observation_space.spaces[agent_id]
        self.action_space = env.action_space.spaces[agent_id]


class FakeSingleSpacesVec(VecEnv):
    """VecEnv equivalent of FakeSingleSpaces.
    :param venv(VecMultiEnv)
    :return a dummy VecEnv instance."""
    def __init__(self, venv, agent_id=0):
        observation_space = venv.observation_space.spaces[agent_id]
        action_space = venv.action_space.spaces[agent_id]
        super().__init__(venv.num_envs, observation_space, action_space)

    def reset(self):
        # Don't raise an error as some policy loading procedures require an initial observation.
        # Returning None guarantees things will break if the observation is ever actually used.
        return None

    def step_async(self, actions):
        raise NotImplementedError()

    def step_wait(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()


class FlattenSingletonEnv(Wrapper):
    """Adapts a single-agent MultiAgentEnv into a standard Gym Env.

    This is typically used after first applying CurryEnv until there is only one agent left."""
    def __init__(self, env):
        """
        :param env: a MultiAgentEnv.
        :return a single-agent Gym Env.
        """
        assert env.num_agents == 1
        super().__init__(env)
        self.observation_space = env.observation_space.spaces[0]
        self.action_space = env.action_space.spaces[0]

    def step(self, action):
        observations, rewards, done, infos = self.env.step([action])
        return observations[0], rewards[0], done, infos

    def reset(self):
        return self.env.reset()[0]


def flatten_space(tuple_space):
    """Flattens a Tuple of like-spaces into a single bigger space of the appropriate type.
       The spaces do not have to have the same shape, but do need to be of compatible types.
       For example, we can flatten a (Box(10), Box(5)) into Box(15) or a (Discrete(2), Discrete(2))
       into a MultiDiscrete([2, 2]), but cannot flatten a (Box(10), Discrete(2))."""
    unique_types = set(type(space) for space in tuple_space.spaces)
    if len(unique_types) > 1:
        raise TypeError(f"Cannot flatten a space with more than one type: {unique_types}")
    type = unique_types.pop()

    if isinstance(type, gym.spaces.Discrete):
        flat_space = gym.spaces.MultiDiscrete([space.n for space in tuple_space.spaces])
        flatten = unflatten = lambda x: x
    elif isinstance(type, gym.spaces.MultiDiscrete):
        flat_space = gym.spaces.MultiDiscrete([space.nvec for space in tuple_space.spaces])
        flatten = unflatten = lambda x: x
    elif isinstance(type, gym.spaces.Box):
        low = np.concatenate(*[space.low for space in tuple_space.spaces], axis=0)
        high = np.concatenate(*[space.high for space in tuple_space.spaces], axis=0)
        flat_space = gym.spaces.Box(low=low, high=high)

        def flatten(x):
            return np.flatten(x)

        def unflatten(x):
            sizes = [np.prod(space.shape) for space in tuple_space.spaces]
            start = np.cumsum(sizes)
            end = start[1:] + len(x)
            orig = [np.reshape(x[s:e], space.shape)
                    for s, e, space in zip(start, end, tuple_space.spaces)]
            return orig
    else:
        raise NotImplementedError("Unsupported type: f{type}")
    return flat_space, flatten, unflatten


class FlattenMultiEnv(Wrapper):
    """Adapts a MultiAgentEnv into a standard Gym Env by flattening actions and observations.

    This can be used if you wish to perform centralized training and execution
    in a multi-agent RL environment."""
    def __init__(self, env, reward_agg=sum):
        '''
        :param env(MultiAgentEnv): a MultiAgentEnv with any number of agents.
        :param reward_agg(list<float>->float): a function reducing a list of rewards.
        :return a single-agent Gym environment.
        '''
        self.observation_space, self._obs_flatten, _ = flatten_space(env.observation_space)
        self.action_space, _, self._act_unflatten = flatten_space(env.action_space)
        self.reward_agg = reward_agg
        super().__init__(env)

    def step(self, action):
        action = self._act_unflatten(action)
        observations, rewards, done, infos = self.env.step(action)
        return self._obs_flatten(observations), self.reward_agg(rewards), done, infos

    def reset(self):
        return self.env.reset()[0]


class SingleToMulti(Wrapper, MultiAgentEnv):
    """Converts an Env into a MultiAgentEnv with num_agents = 1.

    Consequently observations, actions and rewards are singleton tuples.
    The observation action spaces are singleton Tuple spaces.
    The info dict is nested inside an outer with key 0."""
    def __init__(self, env):
        Wrapper.__init__(self, env)
        self.action_space = gym.spaces.Tuple((self.action_space, ))
        self.observation_space = gym.spaces.Tuple((self.observation_space, ))
        MultiAgentEnv.__init__(self, num_agents=1)

    def step(self, action_n):
        observations, rewards, done, infos = self.env.step(action_n)
        rewards = (rewards,)
        observations = (observations,)
        infos = {0: infos}
        return observations, rewards, done, infos

    def reset(self):
        observations = self.env.reset()
        return (observations,)


class VecMultiEnv(VecEnv):
    """Like a VecEnv, but each environment is a MultiEnv. Adds extra attribute, num_agents.

       Observations and actions are a num_agents-length tuple, with the i'th entry of shape
       (num_envs, ) + {observation,action}_space.spaces[i].shape. Rewards are a ndarray of shape
       (num_agents, num_envs)."""
    def __init__(self, num_envs, num_agents, observation_space, action_space):
        VecEnv.__init__(self, num_envs, observation_space, action_space)
        self.num_agents = num_agents


class VecMultiWrapper(VecEnvWrapper, VecMultiEnv):
    """Like VecEnvWrapper but for VecMultiEnv's."""
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)
        VecMultiEnv.__init__(self, venv.num_envs, venv.num_agents,
                             venv.observation_space, venv.action_space)


def tuple_transpose(xs):
    '''Permutes environment and agent dimension.

    Specifically, VecMultiEnv has an agent-major convention: actions and observations are
    num_agents-length tuples, with the i'th element a num_env-length tuple containing an
    agent & environment specific action/observation. This convention is convenient since we can
    easily mutex the stream to different agents.

    However, it can also be convenient to have an environment-major convention: that is, there is
    a num_envs-length tuple each containing a num_agents-length tuple. In particular, this is the
    most natural internal representation for VecEnv, and is also convenient when sampling from
    the action or observation space of an environment.
    '''
    inner_len = len(xs[0])
    for x in xs:
        assert len(x) == inner_len
    return tuple(tuple([x[i] for x in xs]) for i in range(inner_len))


class _ActionTranspose(VecMultiWrapper):
    def __init__(self, venv):
        super().__init__(venv)

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        actions_per_env = tuple_transpose(actions)
        return self.venv.step_async(actions_per_env)

    def step_wait(self):
        obs, rews, done, info = self.venv.step_wait()
        rews = rews.T
        return obs, rews, done, info


def _make_vec_multi_env(cls):
    def f(env_fns):
        venv = cls(env_fns)
        return _ActionTranspose(venv)
    return f


class _DummyVecMultiEnv(DummyVecEnv, VecMultiEnv):
    """Like DummyVecEnv but implements VecMultiEnv interface.
       Handles the larger reward size.
       Note SubprocVecEnv works out of the box."""
    def __init__(self, env_fns):
        DummyVecEnv.__init__(self, env_fns)
        num_agents = getattr_unwrapped(self.envs[0], 'num_agents')
        VecMultiEnv.__init__(self, self.num_envs, num_agents,
                             self.observation_space, self.action_space)
        self.buf_rews = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)


class _SubprocVecMultiEnv(SubprocVecEnv, VecMultiEnv):
    """Stand-in for SubprocVecEnv when applied to MultiEnv's."""
    def __init__(self, env_fns):
        SubprocVecEnv.__init__(self, env_fns)
        env = env_fns[0]()
        num_agents = getattr_unwrapped(env, 'num_agents')
        env.close()
        VecMultiEnv.__init__(self, self.num_envs, num_agents,
                             self.observation_space, self.action_space)


# TODO: This code is extremely hacky. The best approach is probably to add native support for
# tuples to {Dummy,Subproc}VecEnv. Adding support for handling dict action spaces would be an
# easier alternative and avoid some special casing now. See baselines issue #555.
make_dummy_vec_multi_env = _make_vec_multi_env(_DummyVecMultiEnv)
make_subproc_vec_multi_env = _make_vec_multi_env(_SubprocVecMultiEnv)


def _tuple_pop(input, i):
    output = list(input)
    elt = output.pop(i)
    return tuple(output), elt


def _tuple_replace(input, i, obj):
    output = list(input)
    del output[i]
    output.insert(i, obj)
    return tuple(output)


def _tuple_space_filter(tuple_space, filter_idx):
    filtered_spaces = (space for i, space in enumerate(tuple_space.spaces) if i != filter_idx)
    return gym.spaces.Tuple(tuple(filtered_spaces))


def _tuple_space_replace(tuple_space, replace_idx, replace_space):
    current_spaces = tuple_space.spaces
    old_space = current_spaces.pop(replace_idx)
    if type(old_space) != type(replace_space):
        raise TypeError("Replacement action space has different type than original")
    current_spaces.insert(replace_idx, replace_space)
    return gym.spaces.Tuple(tuple(current_spaces))


def _tuple_space_augment(tuple_space, augment_idx, augment_space):
    current_space = tuple_space.spaces[augment_idx]
    new_low = np.concatenate([current_space.low, augment_space.low])
    new_high = np.concatenate([current_space.high, augment_space.high])

    new_space = gym.spaces.Box(low=new_low, high=new_high)
    return _tuple_space_replace(tuple_space, augment_idx, new_space)


class MergeAgentVecEnv(VecMultiWrapper):
    """Allows merging of two agents into a pseudo-agent by merging their actions"""
    def __init__(self, venv, policy, replace_action_space, merge_agent_idx):
        """Expands one of the players in a VecMultiEnv.
        :param venv(VecMultiEnv): the environments.
        :param policy(Policy): the fixed policy to use at merge_agent_idx
        :param replace_action_space(Box): the action space of the new agent to merge
        :param merge_agent_idx(int): the index of the agent that will be merged with the new agent
        :return: a new VecMultiEnv with the same number of agents. It behaves like venv but
                 with all actions at index merge_agent_idx merged with a fixed policy."""
        super().__init__(venv)

        assert venv.num_agents >= 1  # same as in CurryVecEnv
        self.observation_space = _tuple_space_augment(self.observation_space, merge_agent_idx,
                                                      self.action_space.spaces[merge_agent_idx])
        if replace_action_space.shape != self.action_space.spaces[merge_agent_idx].shape:
            raise ValueError("Replacement action space has different shape than original.")
        self.action_space = _tuple_space_replace(self.action_space, merge_agent_idx,
                                                 replace_action_space)

        self._agent_to_merge = merge_agent_idx
        self._policy = policy
        self._action = None
        self._state = None
        self._obs = None
        self._dones = [False] * venv.num_envs

    def step_async(self, actions):
        actions_copy = list(actions)
        actions_copy[self._agent_to_merge] += self._action
        self.venv.step_async(tuple(actions_copy))

    def step_wait(self):
        observations, rewards, self._dones, infos = self.venv.step_wait()
        observations = self._get_augmented_obs(observations)
        return observations, rewards, self._dones, infos

    def reset(self):
        observations = self._get_augmented_obs(self.venv.reset())
        return observations

    def _get_augmented_obs(self, observations):
        """Augments observations[self._agent_to_merge] with action that self._policy would take
        given its observations. Keeps track of these variables to use in next timestep."""
        self._obs = observations[self._agent_to_merge]
        self._action, self._state = self._policy.predict(self._obs, state=self._state,
                                                         mask=self._dones)

        new_obs = np.concatenate([self._obs, self._action], axis=1)
        return _tuple_replace(observations, self._agent_to_merge, new_obs)


class CurryVecEnv(VecMultiWrapper):
    """Substitutes in a fixed agent for one of the players in a VecMultiEnv."""
    def __init__(self, venv, policy, agent_idx=0):
        """Fixes one of the players in a VecMultiEnv.
        :param env(VecMultiEnv): the environments.
        :param policy(Policy): the policy to use for the agent at agent_idx.
        :param agent_idx(int): the index of the agent that should be fixed.
        :return: a new VecMultiEnv with num_agents decremented. It behaves like env but
                 with all actions at index agent_idx set to those returned by agent."""
        super().__init__(venv)

        assert venv.num_agents >= 1  # allow currying the last agent
        self.num_agents = venv.num_agents - 1
        self.observation_space = _tuple_space_filter(self.observation_space, agent_idx)
        self.action_space = _tuple_space_filter(self.action_space, agent_idx)

        self._agent_to_fix = agent_idx
        self._policy = policy
        self._state = None
        self._obs = None
        self._dones = [False] * venv.num_envs

    def step_async(self, actions):
        action, self._state = self._policy.predict(self._obs, state=self._state, mask=self._dones)
        actions.insert(self._agent_to_fix, action)
        self.venv.step_async(actions)

    def step_wait(self):
        observations, rewards, self._dones, infos = self.venv.step_wait()
        observations, self._obs = _tuple_pop(observations, self._agent_to_fix)
        rewards, _ = _tuple_pop(rewards, self._agent_to_fix)
        return observations, rewards, self._dones, infos

    def reset(self):
        observations = self.venv.reset()
        observations, self._obs = _tuple_pop(observations, self._agent_to_fix)
        return observations


class FlattenSingletonVecEnv(VecEnvWrapper):
    """Adapts a single-agent VecMultiEnv into a standard Baselines VecEnv.

    This is typically used after first applying CurryVecEnv until there is only one agent left."""
    def __init__(self, venv):
        """
        :param venv: a VecMultiEnv.
        :return a single-agent Gym Env.
        """
        assert venv.num_agents == 1
        super().__init__(venv)
        self.observation_space = venv.observation_space.spaces[0]
        self.action_space = venv.action_space.spaces[0]

    def step_async(self, action):
        self.venv.step_async([action])

    def step_wait(self):
        observations, rewards, done, infos = self.venv.step_wait()
        return observations[0], rewards[0], done, infos

    def reset(self):
        return self.venv.reset()[0]
