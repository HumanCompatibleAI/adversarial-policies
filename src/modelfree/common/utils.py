import datetime
import os
from os import path as osp

import gym
from gym import Wrapper
from gym.monitoring import VideoRecorder
import numpy as np
from stable_baselines.common import BaseRLModel
from stable_baselines.common.policies import BasePolicy
import tensorflow as tf

from aprl.common.multi_monitor import MultiMonitor
from aprl.envs.multi_agent import MultiAgentEnv, SingleToMulti


class DummyModel(BaseRLModel):
    """Abstract class for policies pretending to be RL algorithms (models).

    Provides stub implementations that raise NotImplementedError.
    The predict method is left as abstract and must be implemented in base class."""
    def __init__(self, policy, sess):
        """Constructs a DummyModel with given policy and session.
        :param policy: (BasePolicy) a loaded policy.
        :param sess: (tf.Session or None) a TensorFlow session.
        :return an instance of BaseRLModel.
        """
        super().__init__(policy=policy, env=None, requires_vec_env=True, policy_base='Dummy')
        self.sess = sess

    def setup_model(self):
        raise NotImplementedError()

    def learn(self):
        raise NotImplementedError()

    def action_probability(self, observation, state=None, mask=None, actions=None):
        raise NotImplementedError()

    def save(self, save_path):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()


class PolicyToModel(DummyModel):
    """Converts BasePolicy to a BaseRLModel with only predict implemented."""
    def __init__(self, policy):
        """Constructs a BaseRLModel using policy for predictions.
        :param policy: (BasePolicy) a loaded policy.
        :return an instance of BaseRLModel.
        """
        super().__init__(policy=policy, sess=policy.sess)

    def predict(self, observation, state=None, mask=None, deterministic=False):
        if state is None:
            state = self.policy.initial_state
        if mask is None:
            mask = [False for _ in range(self.policy.n_env)]

        actions, _val, states, _neglogp = self.policy.step(observation, state, mask,
                                                           deterministic=deterministic)
        return actions, states


class OpenAIToStablePolicy(BasePolicy):
    """Converts an OpenAI Baselines Policy to a Stable Baselines policy."""
    def __init__(self, old_policy):
        self.old = old_policy
        self.initial_state = old_policy.initial_state
        self.sess = old_policy.sess

    def step(self, obs, state=None, mask=None, deterministic=False):
        stochastic = not deterministic
        return self.old.step(obs, S=state, M=mask, stochastic=stochastic)


class ConstantPolicy(BasePolicy):
    """Policy that returns a constant action."""
    def __init__(self, env, constant):
        assert env.action_space.contains(constant)
        super().__init__(sess=None,
                         ob_space=env.observation_space,
                         ac_space=env.action_space,
                         n_env=env.num_envs,
                         n_steps=1,
                         n_batch=1)
        self.constant = constant
        self.initial_state = None

    def step(self, obs, state=None, mask=None, deterministic=False):
        actions = np.array([self.constant] * self.n_env)
        return actions, None, None, None


class ZeroPolicy(ConstantPolicy):
    """Policy that returns a zero action."""
    def __init__(self, env):
        super().__init__(env, np.zeros(env.action_space.shape))


class RandomPolicy(BasePolicy):
    def __init__(self, env):
        super().__init__(sess=None,
                         ob_space=env.observation_space,
                         ac_space=env.action_space,
                         n_env=env.num_envs,
                         n_steps=1,
                         n_batch=1)
        self.initial_state = None

    def step(self, obs, state=None, mask=None, deterministic=False):
        actions = np.array([self.ac_space.sample() for _ in range(self.n_env)])
        return actions, None, None, None


class VideoWrapper(Wrapper):
    def __init__(self, env, directory):
        super(VideoWrapper, self).__init__(env)
        self.directory = osp.abspath(directory)
        os.makedirs(self.directory, exist_ok=True)
        self.episode_id = 0
        self.video_recorder = None

    def _step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done:
            winners = [i for i, d in info.items() if 'winner' in d]
            metadata = {'winners': winners}
            self._reset_video_recorder(metadata)
        self.video_recorder.capture_frame()
        return obs, rew, done, info

    def _reset(self):
        self._reset_video_recorder()
        self.episode_id += 1
        return self.env.reset()

    def _reset_video_recorder(self, metadata=None):
        if self.video_recorder:
            if metadata is not None:
                self.video_recorder.metadata.update(metadata)
            self.video_recorder.close()
        self.video_recorder = VideoRecorder(
            env=self.env,
            base_path=osp.join(self.directory, 'video.{:06}'.format(self.episode_id)),
            metadata={'episode_id': self.episode_id},
        )


def make_session(graph=None):
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=tf_config)
    return sess


def simulate(venv, policies, render=False):
    """
    Run Environment env with the agents in agents
    :param venv(VecEnv): vector environment.
    :param policies(list<BaseModel>): a policy per agent.
    :param render: true if the run should be rendered to the screen
    :return: streams information about the simulation
    """
    observations = venv.reset()
    dones = [False] * venv.num_envs
    states = [None for _ in policies]

    while True:
        if render:
            venv.render()

        actions = []
        new_states = []
        for policy, obs, state in zip(policies, observations, states):
            act, new_state = policy.predict(obs, state=state, mask=dones)
            actions.append(act)
            new_states.append(new_state)
        actions = tuple(actions)
        states = new_states

        observations, rewards, dones, infos = venv.step(actions)
        yield observations, rewards, dones, infos


def make_env(env_name, seed, i, out_dir, our_idx=None, pre_wrapper=None, post_wrapper=None):
    multi_env = gym.make(env_name)
    if pre_wrapper is not None:
        multi_env = pre_wrapper(multi_env)
    if not isinstance(multi_env, MultiAgentEnv):
        multi_env = SingleToMulti(multi_env)
    multi_env.seed(seed + i)

    if out_dir is not None:
        mon_dir = osp.join(out_dir, 'mon')
        os.makedirs(mon_dir, exist_ok=True)
        multi_env = MultiMonitor(multi_env, osp.join(mon_dir, 'log{}'.format(i)), our_idx)

    if post_wrapper is not None:
        multi_env = post_wrapper(multi_env)

    return multi_env


def make_timestamp():
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    return datetime.datetime.now().strftime(ISO_TIMESTAMP)
