from sacred import Experiment
import gym
from gym.core import Wrapper
from gym.monitoring.video_recorder import VideoRecorder
import tensorflow as tf
import numpy as np
import sys
import os
import os.path as osp


from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.a2c import utils

import functools

from sacred.observers import FileStorageObserver

from aprl.envs.multi_agent import FlattenSingletonEnv, CurryEnv

from modelfree.gym_complete_conversion import *
from modelfree.simulation_utils import Agent, simulate


def load_our_mlp(agent_name, env, env_name, sess):
    # TODO DO ANYTHING BUT THIS.  THIS IS VERY DIRTY AND SAD :(
    def make_env(id):
        # TODO: seed (not currently supported)
        # TODO: VecNormalize? (typically good for MuJoCo)
        # TODO: baselines logger?
        # TODO: we're loading identical policy weights into different
        # variables, this is to work-around design choice of Agent's
        # having state stored inside of them.
        sess_inner = make_session()
        with sess_inner.as_default():
            multi_env = env

            attacked_agent = constant_zero_agent(act_dim=8)

            single_env = FlattenSingletonEnv(CurryEnv(TheirsToOurs(multi_env), attacked_agent))

            # TODO: upgrade Gym so don't have to do thi0s
            single_env.observation_space.dtype = np.dtype(np.float32)
        return single_env
        # TODO: close session?

    # TODO DO NOT EVEN READ THE ABOVE CODE :'(
    denv = SubprocVecEnv([functools.partial(make_env, 0)])
    with sess.as_default():
        with sess.graph.as_default():
            model = ppo2.learn(network="mlp", env=denv,
                               total_timesteps=1,
                               seed=0,
                               nminibatches=4,
                               log_interval=1,
                               save_interval=1,
                               load_path=agent_name)

    stateful_model = StatefulModel(denv, model, sess)
    trained_agent = Agent(action_selector=stateful_model.get_action,
                          reseter=stateful_model.reset)

    return trained_agent

def constant_zero_agent(act_dim=8):
    constant_action = np.zeros((act_dim,))

    class Temp:

        def __init__(self, constants):
            self.constants = constants

        def get_action(self, _):
            return self.constants

        def reset(self):
            pass

    return Temp(constant_action)


class Policy(object):
    def reset(self, **kwargs):
        pass

    def act(self, observation):
        # should return act, info
        raise NotImplementedError()


class StatefulModel(Policy):
    def __init__(self, env, model, sess):
        self._sess = sess
        self.nenv = env.num_envs
        self.env = env
        self.dones = [False for _ in range(self.nenv)]
        self.model = model
        self.states = model.initial_state

    def get_action(self, observation):
        with self._sess.as_default():
            actions, values, self.states, neglocpacs = self.model.step([observation] * self.nenv, S=self.states, M=self.dones)
            return actions[0]

    def reset(self):
        self._dones = [True] * self.nenv


#######################################################################################################
#######################################################################################################


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


def get_agent_any_type(agent, agent_type, env, env_name, sess=None):
    agent_loaders = {
        "zoo": load_zoo_agent,
        "our_mlp": load_our_mlp
    }
    return agent_loaders[agent_type](agent, env, env_name, sess=sess)


score_agent_ex = Experiment('score_agent')
score_agent_ex.observers.append(FileStorageObserver.create('my_runs'))


@score_agent_ex.config
def default_score_config():
    agent_a = "agent-zoo/sumo/ants/agent_parameters-v1.pkl"
    agent_a_type = "zoo"
    env = "sumo-ants-v0"
    agent_b_type = "our_mlp"
    agent_b = "outs/20190131_172524 Dummy Exp Name/model.pkl"
    samples = 50
    watch =True
    videos = False
    video_dir = "videos/"


@score_agent_ex.automain
def score_agent(_run, env, agent_a, agent_b, samples, agent_a_type, agent_b_type, watch, videos, video_dir):
    env_object = gym.make(env)

    if videos:
        env_object = VideoWrapper(env_object, video_dir)

    graph_a = tf.Graph()
    sess_a = make_session(graph_a)
    graph_b = tf.Graph()
    sess_b = make_session(graph_b)
    with sess_a:
        with sess_b:
            # TODO seperate tensorflow graphs to get either order of the next two statements to work
            agent_b_object = get_agent_any_type(agent_b, agent_b_type, env_object, env, sess=sess_b)
            agent_a_object = get_agent_any_type(agent_a, agent_a_type, env_object, env, sess=sess_a)

            agents = [agent_a_object, agent_b_object]

            # TODO figure out how to stop the other thread from crashing when I finish
            return get_emperical_score(_run, env_object, agents, samples, render=watch)

