from collections import defaultdict
import itertools
import os
from os import path as osp
import warnings

import gym
from gym import Wrapper
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np

from aprl.common.multi_monitor import MultiMonitor
from aprl.envs.multi_agent import SingleToMulti, VecMultiWrapper


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

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done:
            winners = [i for i, d in info.items() if 'winner' in d]
            metadata = {'winners': winners}
            self.video_recorder.metadata.update(metadata)
        self.video_recorder.capture_frame()
        return obs, rew, done, info

    def reset(self):
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

    def close(self):
        if self.video_recorder is not None:
            self.video_recorder.close()
            self.video_recorder = None
        super(VideoWrapper, self).close()


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

    if not hasattr(multi_env, 'num_agents'):
        multi_env = SingleToMulti(multi_env)

    if out_dir is not None:
        mon_dir = osp.join(out_dir, 'mon')
        os.makedirs(mon_dir, exist_ok=True)
        multi_env = MultiMonitor(multi_env, osp.join(mon_dir, 'log{}'.format(i)), our_idx)

    multi_env = _apply_wrappers(post_wrappers, multi_env)

    multi_env.seed(seed + i)

    return multi_env
