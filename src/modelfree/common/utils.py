from collections import defaultdict
import datetime
import itertools
import os
from os import path as osp
import pickle
import warnings

import gym
from gym import Wrapper
from gym.monitoring import VideoRecorder
import numpy as np
from stable_baselines.common import BaseRLModel
from stable_baselines.common.policies import BasePolicy
from stable_baselines.common.vec_env import VecEnvWrapper
import tensorflow as tf

from aprl.common.multi_monitor import MultiMonitor
from aprl.envs.multi_agent import MultiAgentEnv, SingleToMulti, VecMultiWrapper


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

    def _get_policy_out(self, observation, state, mask, transparent, deterministic=False):
        if state is None:
            state = self.policy.initial_state
        if mask is None:
            mask = [False for _ in range(self.policy.n_env)]

        step_fn = self.policy.step_transparent if transparent else self.policy.step
        return step_fn(observation, state, mask, deterministic=deterministic)

    def predict(self, observation, state=None, mask=None, deterministic=False):
        policy_out = self._get_policy_out(observation, state, mask, transparent=False,
                                          deterministic=deterministic)
        actions, _val, states, _neglogp = policy_out
        return actions, states

    def predict_transparent(self, observation, state=None, mask=None, deterministic=False):
        """Returns same values as predict, as well as a dictionary with transparent data."""
        policy_out = self._get_policy_out(observation, state, mask, transparent=True,
                                          deterministic=deterministic)
        actions, _val, states, _neglogp, data = policy_out
        return actions, states, data


class OpenAIToStablePolicy(BasePolicy):
    """Converts an OpenAI Baselines Policy to a Stable Baselines policy."""

    def __init__(self, old_policy):
        self.old = old_policy
        self.sess = old_policy.sess

    @property
    def initial_state(self):
        return self.old.initial_state

    def step(self, obs, state=None, mask=None, deterministic=False):
        stochastic = not deterministic
        return self.old.step(obs, S=state, M=mask, stochastic=stochastic)

    def proba_step(self, obs, state=None, mask=None):
        raise NotImplementedError()


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

    def step(self, obs, state=None, mask=None, deterministic=False):
        actions = np.array([self.constant] * self.n_env)
        return actions, None, None, None

    def proba_step(self, obs, state=None, mask=None):
        return self.step(obs, state=state, mask=mask)


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

    def step(self, obs, state=None, mask=None, deterministic=False):
        actions = np.array([self.ac_space.sample() for _ in range(self.n_env)])
        return actions, None, None, None

    def proba_step(self, obs, state=None, mask=None):
        raise NotImplementedError()


class VideoWrapper(Wrapper):
    """Creates videos from wrapped environment by called render after each timestep."""
    def __init__(self, env, directory, single_video=True):
        """

        :param env: (gym.Env) the wrapped environment.
        :param directory: the output directory.
        :param single_video: (bool) if True, generates a single video file, with episodes
                             concatenated. If False, a new video file is created for each episode.
                             Usually a single video file is what is desired. However, if one is
                             searching for an interesting episode (perhaps by looking at the
                             metadata), saving to different files can be useful.
        """
        super(VideoWrapper, self).__init__(env)
        self.episode_id = 0
        self.video_recorder = None
        self.single_video = single_video

        self.directory = osp.abspath(directory)

        # Make sure to not put multiple different runs in the same directory,
        # if the directory already exists
        error_msg = "You're trying to use the same directory twice, " \
                    "this would result in files being overwritten"
        assert not os.path.exists(self.directory), error_msg
        os.makedirs(self.directory, exist_ok=True)

    def _step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done:
            winners = [i for i, d in info.items() if 'winner' in d]
            metadata = {'winners': winners}
            self.video_recorder.metadata.update(metadata)
        self.video_recorder.capture_frame()
        return obs, rew, done, info

    def _reset(self):
        self._reset_video_recorder()
        self.episode_id += 1
        return self.env.reset()

    def _reset_video_recorder(self):
        """Called at the start of each episode (by _reset). Always creates a video recorder
           if one does not already exist. When a video recorder is already present, it will only
           create a new one if `self.single_video == False`."""
        if self.video_recorder is not None:
            # Video recorder already started.
            if not self.single_video:
                # We want a new video for each episode, so destroy current recorder.
                self.video_recorder.close()
                self.video_recorder = None

        if self.video_recorder is None:
            # No video recorder -- start a new one.
            self.video_recorder = VideoRecorder(
                env=self.env,
                base_path=osp.join(self.directory, 'video.{:06}'.format(self.episode_id)),
                metadata={'episode_id': self.episode_id},
            )

    def _close(self):
        if self.video_recorder is not None:
            self.video_recorder.close()
            self.video_recorder = None
        super(VideoWrapper, self)._close()


def make_session(graph=None):
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=tf_config)
    return sess


def _filter_dict(d, keys):
    """Filter a dictionary to contain only the specified keys.

    If keys is None, it returns the dictionary verbatim.
    If a key in keys is not present in the dictionary, it gives a warning, but does not fail.

    :param d: (dict)
    :param keys: (iterable) the desired set of keys; if None, performs no filtering.
    :return (dict) a filtered dictionary."""
    if keys is None:
        return d
    else:
        keys = set(keys)
        present_keys = keys.intersection(d.keys())
        missing_keys = keys.difference(d.keys())
        res = {k: d[k] for k in present_keys}
        if len(missing_keys) != 0:
            warnings.warn("Missing expected keys: {}".format(missing_keys), stacklevel=2)
        return res


class TrajectoryRecorder(VecMultiWrapper):
    """Class for recording and saving trajectories in numpy.npz format.
    For each episode, we record observations, actions, rewards and optionally network activations
    for the agents specified by agent_indices.

    :param venv: (VecEnv) environment to wrap
    :param agent_indices: (list,int) indices of agents whose trajectories to record
    :param env_keys: (list,str) keys for environment data to record; if None, record all.
                     Options are 'observations', 'actions' and 'rewards'.
    :param info_keys: (list,str) keys in the info dict to record; if None, record all.
                      This is often used to expose activations from the policy.
    """

    def __init__(self, venv, agent_indices=None, env_keys=None, info_keys=None):
        super().__init__(venv)

        if agent_indices is None:
            self.agent_indices = range(self.num_agents)
        elif isinstance(agent_indices, int):
            self.agent_indices = [agent_indices]
        self.env_keys = env_keys
        self.info_keys = info_keys

        self.traj_dicts = [[defaultdict(list) for _ in range(self.num_envs)]
                           for _ in self.agent_indices]
        self.full_traj_dicts = [defaultdict(list) for _ in self.agent_indices]
        self.prev_obs = None
        self.actions = None

    def step_async(self, actions):
        self.actions = actions
        self.venv.step_async(actions)

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        self.record_timestep_data(self.prev_obs, self.actions, rewards, dones, infos)
        self.prev_obs = observations
        return observations, rewards, dones, infos

    def reset(self):
        observations = self.venv.reset()
        self.prev_obs = observations
        return observations

    def record_extra_data(self, data, agent_idx):
        """Record extra data for the specified agents. `record_timestep_data` will automatically
           record observations, actions, rewards and info dicts. This function is an alternative
           to placing extra information in the info dicts, which can sometimes be more convenient.

           :param data: (dict) treated like an info dict in `record_timestep_data.
           :param agent_idx: (int) index of the agent to record data for."""
        # Not traj_dicts[agent_idx] because there may not be a traj_dict for every agent
        if agent_idx not in self.agent_indices:
            return
        else:
            dict_index = self.agent_indices.index(agent_idx)

        for env_idx in range(self.num_envs):
            for key in data.keys():
                self.traj_dicts[dict_index][env_idx][key].append(np.squeeze(data[key]))

    def record_timestep_data(self, prev_obs, actions, rewards, dones, infos):
        """Record observations, actions, rewards, and optional information from the info dicts
        of one timestep in dict for current episode. Completed episode trajectories are
        collected in a list in preparation for being saved to disk.

        :param prev_obs: (np.ndarray<float>) observations from previous timestep
        :param actions: (np.ndarray<float>) actions taken after observing prev_obs
        :param rewards: (np.ndarray<float>) rewards from actions
        :param dones: ([bool]) whether episode ended (not recorded)
        :param infos: ([dict]) dicts with additional information, e.g. network activations
                               for transparent networks.
        :return: None
        """
        env_data = {
            'observations': prev_obs,
            'actions': actions,
            'rewards': rewards,
        }
        env_data = _filter_dict(env_data, self.env_keys)

        # iterate over both agents over all environments in VecEnv
        iter_space = itertools.product(enumerate(self.traj_dicts), range(self.num_envs))
        for (dict_idx, agent_dicts), env_idx in iter_space:
            # in dict number dict_idx, record trajectories for agent number agent_idx
            agent_idx = self.agent_indices[dict_idx]
            for key, val in env_data.items():
                # data_vals always have data for all agents (use agent_idx not dict_idx)
                agent_dicts[env_idx][key].append(val[agent_idx][env_idx])

            info_dict = infos[env_idx][agent_idx]
            info_dict = _filter_dict(info_dict, self.info_keys)
            for key, val in info_dict.items():
                agent_dicts[env_idx][key].append(val)

            if dones[env_idx]:
                ep_ret = sum(agent_dicts[env_idx]['rewards'])
                self.full_traj_dicts[dict_idx]['episode_returns'].append(np.array([ep_ret]))

                for key, val in agent_dicts[env_idx].items():
                    # consolidate episode data and append to long-term data dict
                    episode_key_data = np.array(val)
                    self.full_traj_dicts[dict_idx][key].append(episode_key_data)
                agent_dicts[env_idx] = defaultdict(list)

    def save(self, save_dir):
        """Save trajectories to save_dir in NumPy compressed-array format, per-agent.

        Our format consists of a dictionary with keys -- e.g. 'observations', 'actions'
        and 'rewards' -- containing lists of NumPy arrays, one for each episode.

        :param save_dir: (str) path to save trajectories; will create directory if needed.
        :return None
        """
        os.makedirs(save_dir, exist_ok=True)

        save_paths = []
        for dict_idx, agent_idx in enumerate(self.agent_indices):
            agent_dicts = self.full_traj_dicts[dict_idx]
            dump_dict = {k: np.asarray(v) for k, v in agent_dicts.items()}

            save_path = os.path.join(save_dir, f'agent_{agent_idx}.npz')
            np.savez(save_path, **dump_dict)
            save_paths.append(save_path)
        return save_paths


def simulate(venv, policies, render=False, record=True):
    """
    Run Environment env with the policies in `policies`.
    :param venv(VecEnv): vector environment.
    :param policies(list<BaseModel>): a policy per agent.
    :param render: (bool) true if the run should be rendered to the screen
    :param record: (bool) true if should record transparent data (if any).
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

        for policy_ind, (policy, obs, state) in enumerate(zip(policies, observations, states)):
            try:
                return_tuple = policy.predict_transparent(obs, state=state, mask=dones)
                act, new_state, transparent_data = return_tuple
                if record:
                    venv.record_extra_data(transparent_data, policy_ind)
            except AttributeError:
                act, new_state = policy.predict(obs, state=state, mask=dones)

            actions.append(act)
            new_states.append(new_state)

        actions = tuple(actions)
        states = new_states

        observations, rewards, dones, infos = venv.step(actions)
        yield observations, rewards, dones, infos


def _apply_wrappers(wrappers, multi_env):
    """Helper method to apply wrappers if they are present. Returns wrapped multi_env"""
    if wrappers is None:
        wrappers = []
    for wrap in wrappers:
        multi_env = wrap(multi_env)
    return multi_env


def make_env(env_name, seed, i, out_dir, our_idx=None, pre_wrappers=None, post_wrappers=None,
             agent_wrappers=None):
    multi_env = gym.make(env_name)

    if agent_wrappers is not None:
        for agent_id in agent_wrappers:
            multi_env.agents[agent_id] = agent_wrappers[agent_id](multi_env.agents[agent_id])

    multi_env = _apply_wrappers(pre_wrappers, multi_env)

    if not isinstance(multi_env, MultiAgentEnv):
        multi_env = SingleToMulti(multi_env)

    if out_dir is not None:
        mon_dir = osp.join(out_dir, 'mon')
        os.makedirs(mon_dir, exist_ok=True)
        multi_env = MultiMonitor(multi_env, osp.join(mon_dir, 'log{}'.format(i)), our_idx)

    multi_env = _apply_wrappers(post_wrappers, multi_env)

    multi_env.seed(seed + i)

    return multi_env


def make_timestamp():
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    return datetime.datetime.now().strftime(ISO_TIMESTAMP)


def add_artifacts(run, dirname, ingredient=None):
    """Convenience function for Sacred to add artifacts inside directory dirname to current run.

    :param run: (sacred.Run) object representing current experiment. Can be captured as `_run`.
    :param dirname: (str) root of directory to save.
    :param ingredient: (sacred.Ingredient or None) optional, ingredient that generated the
                       artifacts. Will be used to tag saved files. This is ignored if ingredient
                       is equal to the currently running experiment.
    :return None"""
    prefix = ""
    if ingredient is not None:
        exp_name = run.experiment_info['name']
        ingredient_name = ingredient.path
        if exp_name != ingredient_name:
            prefix = ingredient_name + "_"

    for root, dirs, files in os.walk(dirname):
        for file in files:
            path = os.path.join(root, file)
            relroot = os.path.relpath(path, dirname)
            name = prefix + relroot.replace('/', '_') + '_' + file
            run.add_artifact(path, name=name)


class DebugVenv(VecEnvWrapper):
    """VecEnvWrapper whose purpose is to record trajectory information for debugging purposes

    :param venv (VecEnv) the environment to wrap
    :param dump_mujoco_state (bool) whether to dump all MjData information (memory intensive)
    """
    def __init__(self, venv, dump_mujoco_state=False):
        super().__init__(venv)
        self.num_agents = self.venv.num_agents
        self.dump_mujoco_state = dump_mujoco_state
        self.debug_file = None
        self.debug_dict = {}

    def step_async(self, actions):
        self.debug_dict['actions'] = actions
        if self.dump_mujoco_state:
            state_data = self.unwrapped.envs[0].env.sim.data
            fields = type(state_data._wrapped.contents).__dict__['_fields_']
            keys = [t[0] for t in fields if t[0] != 'contact']
            for k in keys:
                val = getattr(state_data, k)
                if isinstance(val, np.ndarray) and val.size > 0:
                    self.debug_dict[k] = val

        self.venv.step_async(actions)

    def step_wait(self):
        obs, rew, dones, infos = self.venv.step_wait()
        if self.debug_file is not None:
            self.debug_dict.update({'next_obs': obs, 'rewards': rew})
            pickle.dump(self.debug_dict, self.debug_file)
        self.debug_dict = {}
        return obs, rew, dones, infos

    def reset(self):
        observations = self.venv.reset()
        if self.debug_file is not None:
            self.debug_dict['prev_obs'] = observations
        return observations

    def set_debug_file(self, f):
        """Setter for self.debug_file."""
        self.debug_file = f

    def get_debug_venv(self):
        """Helper method to locate self in a stack of nested VecEnvWrappers"""
        return self
