import argparse
import datetime
import functools
import os
import os.path as osp
import pickle

from baselines.a2c import utils
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.bench.monitor import Monitor
from baselines import logger
from baselines.ppo2 import ppo2
import gym
import numpy as np
import tensorflow as tf

from simulation_utils import MultiToSingle, CurryEnv
import main, policy

def mlp_lstm(hiddens, ob_norm=False, layer_norm=False, activation=tf.tanh):
    """Builds MLP for hiddens[:-1] and LSTM for hiddens[-1].
       Based on Baselines LSTM model."""
    def network_fn(X, nenv=1):
        nbatch = X.shape[0]
        nsteps = nbatch // nenv

        h = tf.layers.flatten(X)
        for i in range(len(hiddens) - 1):
            h = utils.fc(h, 'mlp_fc{}'.format(i), nh=hiddens[i], init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)

        nlstm = hiddens[-1]

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, 2*nlstm]) #states

        xs = utils.batch_to_seq(h, nenv, nsteps)
        ms = utils.batch_to_seq(M, nenv, nsteps)

        if layer_norm:
            h5, snew = utils.lnlstm(xs, ms, S, scope='lnlstm', nh=nlstm)
        else:
            h5, snew = utils.lstm(xs, ms, S, scope='lstm', nh=nlstm)

        h = utils.seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)

        return h, {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}

    return network_fn

class StatefulModel(policy.Policy):
    def __init__(self, env, model):
        self.nenv = env.num_envs
        self.env = env
        self.dones = [False for _ in range(self.nenv)]
        self.model = model
        self.states = model.initial_state

    def get_action(self, observation):
        actions, values, self.states, neglocpacs = self.model.step([observation] * self.nenv, S=self.states, M=self.dones)
        return actions[0]

    def reset(self):
        self._dones = [True] * self.nenv

def save_stats(env_wrapper, path):
    venv = env_wrapper.venv
    env_wrapper.venv = None
    with open(path, 'wb') as f:
        serialized = pickle.dump(env_wrapper, f)
    env_wrapper.venv = venv
    return serialized


def restore_stats(path, venv):
    with open(path, 'rb') as f:
        env_wrapper = pickle.load(f)
    env_wrapper.venv = venv
    env_wrapper.num_envs = venv.num_envs
    if hasattr(env_wrapper, 'ret'):
        env_wrapper.ret = np.zeros(env_wrapper.num_envs)
    return env_wrapper

def shape_reward(env, reward_type):
    if reward_type=="default":
        return env
    elif reward_type == "no_shape":
        env.move_reward_weight = 0
        return env

ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Environments for Multi-agent competition")
    p.add_argument("--max-episodes", default=0, help="max number of matches during visualization", type=int)
    p.add_argument('--vector', default=8, help="parallel vector sampling", type=int)
    p.add_argument('--total-timesteps', default=1000000, type=int)
    p.add_argument('--out-dir', default='results', type=str)
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--network', default='our-lstm')
    p.add_argument('--no-normalize', type=bool)
    p.add_argument('exp_name', type=str)
    p.add_argument('--reward', type=str, default="default", help="default, no_shape")
    configs = p.parse_args()

    timestamp = datetime.datetime.now().strftime(ISO_TIMESTAMP)
    out_dir = osp.join(configs.out_dir, '{} {}'.format(timestamp, configs.exp_name))
    os.mkdir(out_dir)
    logger.configure(dir=osp.join(out_dir, 'mon'))

    ant_paths = main.get_trained_sumo_ant_locations()

    ### ENV SETUP ###
    #TODO: upgrade Gym so this monkey-patch isn't needed
    gym.spaces.Dict = type(None)

    def make_env(id):
        #TODO: seed (not currently supported)
        #TODO: VecNormalize? (typically good for MuJoCo)
        #TODO: baselines logger?
        #TODO: we're loading identical policy weights into different
        #variables, this is to work-around design choice of Agent's
        #having state stored inside of them.
        sess = main.make_session()
        with sess.as_default():
            multi_env, policy_type = main.get_env_and_policy_type("sumo-ants")

            multi_env = shape_reward(multi_env, configs.reward)

            attacked_agent = main.load_agent(ant_paths[1], policy_type,
                                             "zoo_ant_policy_{}".format(id), multi_env, 0)
            single_env = MultiToSingle(CurryEnv(multi_env, attacked_agent))
            single_env.spec = gym.envs.registration.EnvSpec('Dummy-v0')

            #TODO: upgrade Gym so don't have to do thi0s
            single_env.observation_space.dtype = np.dtype(np.float32)

            single_env = Monitor(single_env, osp.join(out_dir, 'mon', 'log{}'.format(id)))
        return single_env
        #TODO: close session?
    venv = SubprocVecEnv([functools.partial(make_env, i) for i in range(configs.vector)])

    if not configs.no_normalize:
        venv = VecNormalize(venv)

    sess = main.make_session()
    with sess:
        ### TRAIN AGENT  ####
        network = configs.network
        if configs.network == 'our-lstm':
            network = mlp_lstm([128, 128], layer_norm=True)
        #TODO: speed up construction of mlp_lstm?
        model = ppo2.learn(network=network, env=venv,
                           total_timesteps=configs.total_timesteps,
                           seed=configs.seed,
                           nminibatches=min(4, configs.vector),
                           log_interval=1,
                           save_interval=1)
        model.save(osp.join(out_dir, 'model.pkl'))
        if not configs.no_normalize:
            save_stats(venv, osp.join(out_dir, 'normalize.pkl'))

        ### This just runs a visualization of the results and anounces wins.  Will crash if you can't render
        multi_env, policy_type = main.get_env_and_policy_type("sumo-ants")
        attacked_agent = main.load_agent(ant_paths[1], policy_type,
                                         "zoo_ant_policy_test", multi_env, 0)
        stateful_model = StatefulModel(venv, model)
        trained_agent = main.Agent(action_selector=stateful_model.get_action,
                                   reseter=stateful_model.reset)
        agents = [attacked_agent, trained_agent]
        for _ in range(configs.max_episodes):
            main.anounce_winner(main.simulate(multi_env, agents, render=True))
            for agent in agents:
                agent.reset()
