from collections import defaultdict
import datetime
import itertools
import os
import shutil
from os import path as osp

import gym
from gym import Wrapper
from gym.monitoring import VideoRecorder
import numpy as np
from stable_baselines.common import BaseRLModel
from stable_baselines.common.policies import BasePolicy
import tensorflow as tf

from aprl.common.multi_monitor import MultiMonitor
from aprl.envs.multi_agent import MultiAgentEnv, SingleToMulti, VecMultiWrapper
from modelfree.transparent import TransparentPolicy


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

        if isinstance(self.policy, TransparentPolicy):
            actions, _val, states, _neglogp, data = self.policy.step(observation, state, mask,
                                                                     deterministic=deterministic)
            return actions, states, data
        else:
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
        if os.path.exists(self.directory):
            shutil.rmtree(self.directory)
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


class TrajectoryRecorder(VecMultiWrapper):
    def __init__(self, venv, save_dir, use_gail_format=False, agent_indices=None):
        VecMultiWrapper.__init__(self, venv)
        self.save_dir = save_dir
        self.use_gail_format = use_gail_format
        if agent_indices is None:
            self.agent_indices = range(self.num_agents)
        elif isinstance(agent_indices, int):
            self.agent_indices = [agent_indices]
        os.makedirs(self.save_dir, exist_ok=True)

        self.traj_dicts = [[defaultdict(list) for e in range(self.num_envs)]
                           for p in self.agent_indices]
        self.full_traj_dicts = [defaultdict(list) for p in self.agent_indices]
        self.prev_obs = None
        self.actions = None

    def step_async(self, actions):
        self.actions = actions
        self.venv.step_async(actions)

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        self.record_traj(self.prev_obs, self.actions, rewards, dones)
        self.prev_obs = observations
        return observations, rewards, dones, infos

    def reset(self):
        observations = self.venv.reset()
        self.prev_obs = observations
        return observations

    def record_transparent_data(self, data, agent_idx):
        # Not traj_dicts[agent_idx] because there may not be a traj_dict for every agent
        agent_dicts = [self.traj_dicts[i] for i in range(len(self.agent_indices)) if
                       self.agent_indices[i] == agent_idx]
        if len(agent_dicts) == 0:
            return
        else:
            agent_dicts = agent_dicts[0]
        for env_idx in range(self.num_envs):
            for key in data.keys():
                agent_dicts[env_idx][key].append(np.squeeze(data[key][env_idx]))

    def record_traj(self, prev_obs, actions, rewards, dones):
        data_keys = ('rewards', 'actions', 'obs')
        data_vals = (rewards, actions, prev_obs)
        # iterate over both agents over all environments in VecEnv
        iter_space = itertools.product(enumerate(self.traj_dicts), range(self.num_envs))
        for (dict_idx, agent_dicts), env_idx in iter_space:
            # in dict number dict_idx, record trajectories for agent number agent_idx
            agent_idx = self.agent_indices[dict_idx]
            for key, val in zip(data_keys, data_vals):
                # data_vals always have data for all agents (use agent_idx not dict_idx)
                agent_dicts[env_idx][key].append(val[agent_idx][env_idx])

            if dones[env_idx]:
                if self.use_gail_format:
                    ep_len = len(agent_dicts[env_idx]['rewards'])
                    # used to index episodes since they are flattened in gail format.
                    ep_starts = [True] + [False] * (ep_len - 1)
                    self.full_traj_dicts[dict_idx]['episode_starts'].append(np.array(ep_starts))

                ep_ret = sum(agent_dicts[env_idx]['rewards'])
                self.full_traj_dicts[dict_idx]['episode_returns'].append(np.array([ep_ret]))

                for key in itertools.chain(data_keys, ('ff_policy', 'ff_value', 'hid')):
                    if key not in agent_dicts[env_idx]:
                        continue
                    # consolidate episode data and append to long-term data dict
                    episode_key_data = np.array(agent_dicts[env_idx][key])
                    self.full_traj_dicts[dict_idx][key].append(episode_key_data)
                agent_dicts[env_idx] = defaultdict(list)

    def save_traj(self):
        for dict_idx, agent_idx in enumerate(self.agent_indices):
            # gail expects array of all episodes flattened together delineated by
            # 'episode_starts' array. To be more efficient, we just keep the additional axis.
            # We use np.asarray instead of np.stack because episodes have heterogenous lengths.
            agg_function = np.concatenate if self.use_gail_format else np.asarray
            dump_dict = {
                k: agg_function(v)
                for k, v in self.full_traj_dicts[dict_idx].items()}
            save_path = os.path.join(self.save_dir, f'agent_{agent_idx}.npz')
            np.savez(save_path, **dump_dict)


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
        for idx, (policy, obs, state) in enumerate(zip(policies, observations, states)):
            if isinstance(policy.policy, TransparentPolicy):
                act, new_state, data = policy.predict(obs, state=state, mask=dones)
                if isinstance(venv, TrajectoryRecorder):
                    venv.record_transparent_data(data, idx)  # e.g. activations, hidden states
            else:
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
