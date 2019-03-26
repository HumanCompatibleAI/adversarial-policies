from collections import defaultdict
import itertools
import os
from os import path as osp

import gym
from gym import Wrapper
from gym.monitoring import VideoRecorder
import numpy as np
from stable_baselines.common import BaseRLModel
from stable_baselines.common.policies import BasePolicy
from stable_baselines.logger import KVWriter
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

    def _get_pretrain_placeholders(self):
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


class ReporterOutputFormat(KVWriter):
    """Key-value logging plugin for Stable Baselines that writes to a Ray Tune StatusReporter."""
    def __init__(self, reporter):
        self.last_kvs = dict()
        self.reporter = reporter

    def writekvs(self, kvs):
        self.last_kvs = kvs
        self.reporter(**kvs)


def make_session(graph=None):
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=tf_config)
    return sess


class TrajectoryRecorder(object):
    def __init__(self, num_envs, num_policies, num_trajectories_to_save):
        self.num_envs = num_envs
        self.num_policies = num_policies
        self.num_trajectories_to_save = num_trajectories_to_save

        self.traj_dicts = [[defaultdict(list) for e in range(num_envs)] for p in range(num_policies)]
        self.full_traj_dicts = [defaultdict(list) for p in range(num_policies)]
        self.num_completed = 0
        self.already_saved = [False for p in range(num_policies)]

    def record_trajectories(self, rewards, actions, dones, prev_observations):
        data_keys = ('rewards', 'actions', 'obs')
        data_vals = (rewards, actions, prev_observations)
        iter_space = itertools.product(enumerate(self.traj_dicts), range(self.num_envs))
        # iterate over both agents over all environments in VecEnv
        for (agent_idx, agent_dicts), env_idx in iter_space:
            for key, val in zip(data_keys, data_vals):
                agent_dicts[env_idx][key].append(val[agent_idx][env_idx])
            if dones[env_idx]:
                ep_len = len(agent_dicts[env_idx]['rewards'])
                ep_starts = [True] + [False] * (ep_len - 1)
                self.full_traj_dicts[agent_idx]['episode_starts'].append(np.array(ep_starts))

                ep_ret = sum(agent_dicts[env_idx]['rewards'])
                self.full_traj_dicts[agent_idx]['episode_returns'].append(np.array([ep_ret]))

                for key in data_keys:
                    # consolidate episode data and append to long-term data dict
                    episode_key_data = np.array(agent_dicts[env_idx][key])
                    self.full_traj_dicts[agent_idx][key].append(episode_key_data)
                agent_dicts[env_idx] = defaultdict(list)
                # so as not to double-count number of episodes
                self.num_completed += int(agent_idx == 0)
                if self.num_completed >= self.num_trajectories_to_save:
                    self.save_trajectories(agent_idx)

    def save_trajectories(self, agent_idx):
        if self.already_saved[agent_idx]:
            print(f'Warning: already saved trajectories for agent {agent_idx}')
        self.full_traj_dicts[agent_idx] = {
            k: np.concatenate(v, axis=0)
            for k, v in self.full_traj_dicts[agent_idx].items()}
        np.savez(f'data/agent_{agent_idx}_traj.npz',
                 **self.full_traj_dicts[agent_idx])
        self.already_saved[agent_idx] = True


def simulate(venv, policies, render=False, record_trajectories=False,
             num_trajectories_to_save=None):
    """
    Run Environment env with the agents in agents
    :param venv(VecEnv): vector environment.
    :param policies(list<BasePolicy>): a policy per agent.
    :param render: true if the run should be rendered to the screen
    :param record_trajectories: whether to save trajectory data in stable_baselines.GAIL format
    :param num_trajectories_to_save: when to save trajectories since this function is a generator
    :return: streams information about the simulation
    """
    observations = venv.reset()
    dones = [False] * venv.num_envs
    states = [None for policy in policies]

    if record_trajectories:
        if num_trajectories_to_save is None:
            raise ValueError("Must set number of trajectories to save in order to save them.")
        recorder = TrajectoryRecorder(venv.num_envs, len(policies), num_trajectories_to_save)

    while True:
        if render:
            venv.render()

        actions = []
        new_states = []
        prev_observations = observations
        for idx, (policy, obs, state) in enumerate(zip(policies, observations, states)):
            act, new_state = policy.predict(obs, state=state, mask=dones)
            actions.append(act)
            new_states.append(new_state)
        actions = tuple(actions)
        states = new_states

        observations, rewards, dones, infos = venv.step(actions)
        if record_trajectories:
            recorder.record_trajectories(rewards, actions, dones, prev_observations)
        yield observations, rewards, dones, infos


def make_env(env_name, seed, i, out_dir, pre_wrapper=None, post_wrapper=None):
    multi_env = gym.make(env_name)
    if pre_wrapper is not None:
        multi_env = pre_wrapper(multi_env)
    if not isinstance(multi_env, MultiAgentEnv):
        multi_env = SingleToMulti(multi_env)
    multi_env.seed(seed + i)

    if out_dir is not None:
        mon_dir = osp.join(out_dir, 'mon')
        os.makedirs(mon_dir, exist_ok=True)
        multi_env = MultiMonitor(multi_env, osp.join(mon_dir, 'log{}'.format(i)))

    if post_wrapper is not None:
        multi_env = post_wrapper(multi_env)

    return multi_env
