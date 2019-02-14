import os
from os import path as osp

from baselines.bench import Monitor
import gym
from gym import Wrapper
from gym.monitoring import VideoRecorder
from gym_compete.policy import Policy
import numpy as np
import tensorflow as tf

from aprl.envs.multi_agent import CurryEnv, FlattenSingletonEnv
from modelfree.gym_compete_conversion import GymCompeteToOurs


class VideoWrapper(Wrapper):
    def __init__(self, env, directory):
        super(VideoWrapper, self).__init__(env)
        self.directory = osp.abspath(directory)
        os.makedirs(self.directory, exist_ok=True)
        self.episode_id = 0
        self.video_recorder = None

    def _step(self, action):
        obs, rew, done, info = self.env.step(action)
        if all(done):
            winners = [i for i, d in enumerate(info) if 'winner' in d]
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
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=tf_config)
    return sess


def make_single_env(env_name, seed, agent_fn, out_dir, env_id=0):
    # TODO: perform the currying at a VecEnv level.
    # This will probably improve performance, but will require making their Agent's stateless.
    g = tf.Graph()
    sess = make_session(g)
    with sess.as_default():
        multi_env = gym.make(env_name)
        multi_env.seed(seed)

        agent = agent_fn(env=multi_env, sess=sess)
        curried_env = CurryEnv(GymCompeteToOurs(multi_env), agent)
        single_env = FlattenSingletonEnv(curried_env)

        # Gym added dtype's to shapes in commit 1c5a463
        # Baselines depends on this, but we have to use an older version of Gym
        # to keep support for MuJoCo 1.31. Monkeypatch shapes to fix this.
        # Note: the environments actually return float64, but this precision is not needed.
        single_env.observation_space.dtype = np.dtype(np.float32)
        single_env.action_space.dtype = np.dtype(np.float32)

        if out_dir is not None:
            mon_dir = osp.join(out_dir, 'mon')
            os.makedirs(mon_dir, exist_ok=True)
            single_env = Monitor(single_env, osp.join(mon_dir, 'log{}'.format(env_id)))
    # TODO: close TF session once env is closed?
    return single_env


class ConstantAgent(object):
    def __init__(self, constant):
        self.constant_action = constant

    def get_action(self, _):
        return self.constant_action

    def reset(self):
        pass


class ZeroAgent(ConstantAgent):
    def __init__(self, shape):
        super().__init__(np.zeros(shape))


class StatefulModel(Policy):
    def __init__(self, env, model, sess):
        self._sess = sess
        self.nenv = env.num_envs
        self.env = env
        self._dones = [False for _ in range(self.nenv)]
        self.model = model
        self._states = model.initial_state

    def get_action(self, observation):
        with self._sess.as_default():
            step_res = self.model.step([observation] * self.nenv, S=self._states, M=self._dones)
            actions, values, self._states, neglocpacs = step_res
            return actions[0]

    def reset(self):
        self._dones = [True] * self.nenv
