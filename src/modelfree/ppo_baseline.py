"""Uses PPO to train an attack policy against a fixed victim policy."""

import datetime
import functools
import os
import os.path as osp
import pickle

from baselines import logger
from baselines.a2c import utils
from baselines.bench.monitor import Monitor
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
import gym
from gym_compete.policy import Policy
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
import tensorflow as tf

from aprl.envs.multi_agent import CurryEnv, FlattenSingletonEnv
from modelfree.gym_compete_conversion import TheirsToOurs, load_zoo_agent
from modelfree.simulation_utils import ResettableAgent
from modelfree.utils import make_session


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


def make_single_env(env_name, seed, agent_fn, out_dir):
    # TODO: perform the currying at a VecEnv level.
    # This will probably improve performance, but will require making their Agent's stateless.
    g = tf.Graph()
    sess = make_session(g)
    with sess.as_default():
        multi_env = gym.make(env_name)
        multi_env.seed(seed)

        agent = agent_fn(env=multi_env, sess=sess)
        curried_env = CurryEnv(TheirsToOurs(multi_env), agent)
        single_env = FlattenSingletonEnv(curried_env)

        # Gym added dtype's to shapes in commit 1c5a463
        # Baselines depends on this, but we have to use an older version of Gym
        # to keep support for MuJoCo 1.31. Monkeypatch shapes to fix this.
        # Note: the environments actually return float64, but this precision is not needed.
        single_env.observation_space.dtype = np.dtype(np.float32)
        single_env.action_space.dtype = np.dtype(np.float32)

        if out_dir is not None:
            single_env = Monitor(single_env, osp.join(out_dir, 'mon', 'log{}'.format(id)))
    # TODO: close TF session once env is closed?
    return single_env


def load_our_mlp(agent_name, env, env_name, _, sess):
    # TODO: Find a way of loading a policy without training for one timestep.
    def agent_fn(env, sess):
        return ZeroAgent(env.action_space.shape[0])

    def make_env():
        return make_single_env(env_name=env_name, seed=0, agent_fn=agent_fn, out_dir=None)

    denv = DummyVecEnv([make_env])

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
    trained_agent = ResettableAgent(get_action_in=stateful_model.get_action,
                                    reset_in=stateful_model.reset)

    return trained_agent


def make_zoo_vec_env(env_name, victim, victim_index, no_normalize, seed, out_dir, vector):
    def agent_fn(env, sess):
        return load_zoo_agent(path=victim, env=env, env_name=env_name,
                              index=victim_index, sess=sess)

    def make_env(i):
        return make_single_env(env_name, seed + i, agent_fn, out_dir)

    venv = SubprocVecEnv([functools.partial(make_env, i) for i in range(vector)])

    if not no_normalize:
        venv = VecNormalize(venv)

    return venv


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

        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, 2*nlstm])  # states

        xs = utils.batch_to_seq(h, nenv, nsteps)
        ms = utils.batch_to_seq(M, nenv, nsteps)

        if layer_norm:
            h5, snew = utils.lnlstm(xs, ms, S, scope='lnlstm', nh=nlstm)
        else:
            h5, snew = utils.lstm(xs, ms, S, scope='lstm', nh=nlstm)

        h = utils.seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)

        return h, {'S': S, 'M': M, 'state': snew, 'initial_state': initial_state}

    return network_fn


def save_stats(env_wrapper, path):
    venv = env_wrapper.venv
    env_wrapper.venv = None
    with open(path, 'wb') as f:
        serialized = pickle.dump(env_wrapper, f)
    env_wrapper.venv = venv
    return serialized


def train(env, out_dir="results", seed=1, total_timesteps=1, vector=8, network="our-lstm",
          no_normalize=False, nsteps=2048, load_path=None):
    g = tf.Graph()
    sess = make_session(g)
    model_path = osp.join(out_dir, 'model.pkl')
    with sess:
        if network == 'our-lstm':
            network = mlp_lstm([128, 128], layer_norm=True)
        # TODO: speed up construction of mlp_lstm?
        model = ppo2.learn(network=network, env=env,
                           total_timesteps=total_timesteps,
                           nsteps=nsteps,
                           seed=seed,
                           nminibatches=min(4, vector),
                           log_interval=1,
                           save_interval=1,
                           load_path=load_path)
        model.save(model_path)
        if not no_normalize:
            save_stats(env, osp.join(out_dir, 'normalize.pkl'))
    return osp.join(model_path)


ISO_TIMESTAMP = "%Y%m%d_%H%M%S"


def setup_logger(out_dir="results", exp_name="test"):
    timestamp = datetime.datetime.now().strftime(ISO_TIMESTAMP)
    out_dir = osp.join(out_dir, '{} {}'.format(timestamp, exp_name))
    os.mkdir(out_dir)
    logger.configure(dir=osp.join(out_dir, 'mon'))
    return out_dir


ppo_baseline_ex = Experiment("ppo_baseline")
ppo_baseline_ex.observers.append(FileStorageObserver.create('my_runs'))


@ppo_baseline_ex.config
def default_ppo_config():
    victim = "1"
    victim_type = "zoo"
    env = "multicomp/SumoAnts-v0"
    vectorize = 8
    out_dir = "outs"
    exp_name = "Dummy Exp Name"
    no_normalize = True
    seed = 1
    total_timesteps = 100000
    network = "mlp"
    nsteps = 2048
    load_path = None
    _ = locals()  # quieten flake8 unused variable warning
    del _


@ppo_baseline_ex.automain
def ppo_baseline(_run, env, victim, victim_type, out_dir, exp_name, vectorize,
                 no_normalize, seed, total_timesteps, network, nsteps, load_path):
    # TODO: some bug with vectorizing goalie
    if env == 'kick-and-defend' and vectorize != 1:
        raise Exception("Kick and Defend doesn't work with vecorization above 1")

    out_dir = setup_logger(out_dir, exp_name)
    env = make_zoo_vec_env(env_name=env, victim=victim, victim_index=0, no_normalize=no_normalize,
                           seed=seed, out_dir=out_dir, vector=vectorize)
    res = train(env, out_dir=out_dir, seed=seed, total_timesteps=total_timesteps,
                vector=vectorize, network=network, no_normalize=no_normalize,
                nsteps=nsteps, load_path=load_path)
    env.close()

    return res
