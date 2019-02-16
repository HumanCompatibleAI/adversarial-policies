import os
from os import path as osp

from gym import Wrapper
from gym.monitoring import VideoRecorder
from gym_compete.policy import Policy
import numpy as np
import tensorflow as tf


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
    # TODO: why so little parallelism?
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=tf_config)
    return sess


class ConstantPolicy(Policy):
    def __init__(self, constant):
        self.constant_action = constant

    def act(self, observations):
        actions = np.array([self.constant_action] * self.batch_size)
        return actions, {}

    def reset(self, batch_size):
        self.batch_size = batch_size


class ZeroPolicy(ConstantPolicy):
    def __init__(self, shape):
        super().__init__(np.zeros(shape))


class StatefulModel(Policy):
    def __init__(self, model, sess):
        self._sess = sess
        self.model = model
        self._states = model.initial_state

    def act(self, observations):
        with self._sess.as_default():
            step_res = self.model.step(observations, S=self._states, M=self._dones)
            actions, values, self._states, neglocpacs = step_res
            return actions, {}

    def reset(self, batch_size):
        self._states = self.model
        self._dones = [True] * batch_size


def simulate(venv, agents, render=False):
    """
    Run Environment env with the agents in agents
    :param venv: any enviroment following the openai-gym spec
    :param agents: agents that have get-action functions
    :param render: true if the run should be rendered to the screen
    :return: streams information about the simulation
    """
    observations = venv.reset()
    dones = [False] * venv.num_envs

    for agent in agents:
        agent.reset(batch_size=venv.num_envs)

    while True:
        if render:
            venv.render()
        actions = []
        for agent, agent_observation in zip(agents, observations):
            agent_action, _info = agent.act(agent_observation)
            actions.append(agent_action)
        actions = tuple(actions)

        observations, rewards, dones, infos = venv.step(actions)
        if any(dones):  # TODO: this makes no sense -- fix interface
            for agent in agents:
                agent.reset(batch_size=venv.num_envs)

        yield observations, rewards, dones, infos
