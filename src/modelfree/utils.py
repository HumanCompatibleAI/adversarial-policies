import os
from os import path as osp

import gym
from gym import Wrapper
from gym.monitoring import VideoRecorder
import numpy as np
from stable_baselines.common.policies import BasePolicy
import tensorflow as tf

from aprl.common.multi_monitor import MultiMonitor


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


class ConstantPolicy(BasePolicy):
    def __init__(self, constant):
        self.constant_action = constant
        self.initial_state = None

    def step(self, obs, state=None, mask=None):
        actions = np.array([self.constant_action] * self.batch_size)
        return actions, None, None, None


class ZeroPolicy(ConstantPolicy):
    def __init__(self, shape):
        super().__init__(np.zeros(shape))


class OldToStable(BasePolicy):
    def __init__(self, old_model):
        self.old = old_model
        self.initial_state = old_model.initial_state

    def step(self, obs, state=None, mask=None):
        return self.old.step(obs, S=state, M=mask)


def simulate(venv, policies, render=False):
    """
    Run Environment env with the agents in agents
    :param venv(VecEnv): vector environment.
    :param policies(list<BasePolicy>): a policy per agent.
    :param render: true if the run should be rendered to the screen
    :return: streams information about the simulation
    """
    observations = venv.reset()
    dones = [False] * venv.num_envs
    states = [None for policy in policies]

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


def make_env(env_name, seed, i, out_dir, pre_wrapper=None, post_wrapper=None):
    multi_env = gym.make(env_name)
    if pre_wrapper is not None:
        multi_env = pre_wrapper(multi_env)
    multi_env.seed(seed + i)

    if out_dir is not None:
        mon_dir = osp.join(out_dir, 'mon')
        os.makedirs(mon_dir, exist_ok=True)
        multi_env = MultiMonitor(multi_env, osp.join(mon_dir, 'log{}'.format(i)))

    if post_wrapper is not None:
        multi_env = post_wrapper(multi_env)

    return multi_env
