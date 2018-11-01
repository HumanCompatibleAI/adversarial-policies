import argparse
import datetime
import functools
import os.path as osp

from baselines.a2c import utils
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines import logger
from baselines.ppo2 import ppo2
import gym
import types
import numpy as np
import tensorflow as tf

from simulation_utils import MultiToSingle, CurryEnv
import main, policy

def mlp_lstm(hiddens, layer_norm=False, activation=tf.tanh):
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

ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Environments for Multi-agent competition")
    p.add_argument("--max-episodes", default=0, help="max number of matches during visualization", type=int)
    p.add_argument('--vector', default=8, help="parallel vector sampling", type=int)
    p.add_argument('--total-timesteps', default=1000000, type=int)
    p.add_argument('--out-dir', default='results', type=str)
    p.add_argument('--seed', default=0, type=int)
    configs = p.parse_args()

    logger.configure()

    timestamp = datetime.datetime.now().strftime(ISO_TIMESTAMP)
    out_dir = osp.join(configs.out_dir, timestamp)
    ant_paths = main.get_trained_sumo_ant_locations()

    sess = main.make_session()
    with sess:
        ### ENV SETUP ###
        #TODO: upgrade Gym so this monkey-patch isn't needed
        gym.spaces.Dict = type(None)

        def make_env(id):
            #TODO: seed (not currently supported)
            #TODO: VecNormalize? (typically good for MuJoCo)
            #TODO: baselines logger?
            multi_env, policy_type = main.get_env_and_policy_type("sumo-ants")
            #TODO: we're loading identical policy weights into different
            #variables, this is to work-around design choice of Agent's
            #having state stored inside of them.
            attacked_agent = main.load_agent(ant_paths[1], policy_type,
                                             "zoo_ant_policy_{}".format(id), multi_env, 0)
            single_env = MultiToSingle(CurryEnv(multi_env, attacked_agent))

            #TODO: upgrade Gym so don't have to do this
            single_env.observation_space.dtype = np.dtype(np.float32)
            return single_env
        venv = DummyVecEnv([functools.partial(make_env, i) for i in range(configs.vector)])

        ### TRAIN AGENT  ####
        # Policy class?
        network = mlp_lstm([128, 128], layer_norm=True)
        #TODO: network of reasonable size
        model = ppo2.learn(network='mlp', env=venv,
                           total_timesteps=configs.total_timesteps,
                           seed=configs.seed,
                           nminibatches=min(4, configs.vector))
        model.save(out_dir)
        venv.close()

        #TODO: how to feed in the RNN input?
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
