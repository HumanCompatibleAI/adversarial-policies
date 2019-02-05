import os
import os.path as osp
from baselines import logger
import datetime
import gym
from baselines.bench.monitor import Monitor
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.a2c import utils

from aprl.envs.multi_agent import FlattenSingletonEnv, CurryEnv
from modelfree.gym_complete_conversion import TheirsToOurs
from modelfree.utils import make_session

from modelfree.simulation_utils import Agent
import numpy as np
import tensorflow as tf
import pickle
from sacred import Experiment
from sacred.observers import FileStorageObserver
import functools

from modelfree.gym_complete_conversion import load_zoo_policy, get_policy_type_for_agent_zoo

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


def save_stats(env_wrapper, path):
    venv = env_wrapper.venv
    env_wrapper.venv = None
    with open(path, 'wb') as f:
        serialized = pickle.dump(env_wrapper, f)
    env_wrapper.venv = venv
    return serialized


def train(env, out_dir="results", seed=1, total_timesteps=1, vector=8, network="our-lstm", no_normalize=False,
          nsteps=2048, load_path=None):
    g = tf.Graph()
    sess = make_session(g)
    with g.as_default():
        with sess:

            if network == 'our-lstm':
                network = mlp_lstm([128, 128], layer_norm=True)
            # TODO: speed up construction of mlp_lstm?
            model = ppo2.learn(network=network, env=env,
                               total_timesteps=total_timesteps,
                               nsteps= nsteps,
                               seed=seed,
                               nminibatches=min(4, vector),
                               log_interval=1,
                               save_interval=1,
                               load_path=load_path)
            model.save(osp.join(out_dir, 'model.pkl'))
            if not no_normalize:
                save_stats(env, osp.join(out_dir, 'normalize.pkl'))

    env.close()
    sess.close()

    return osp.join(out_dir, 'model.pkl')


def get_env(env_name, victim, victim_type, no_normalize, out_dir, vector):

    #TODO This is nasty, fix
    victim_type = get_policy_type_for_agent_zoo(env_name)

    ### ENV SETUP ###
    # TODO: upgrade Gym so this monkey-patch isn't needed
    gym.spaces.Dict = type(None)

    g = tf.Graph()

    def make_env(id):
        # TODO: seed (not currently supported)
        # TODO: VecNormalize? (typically good for MuJoCo)
        # TODO: baselines logger?
        # TODO: we're loading identical policy weights into different
        # variables, this is to work-around design choice of Agent's
        # having state stored inside of them.
        sess = make_session(g)
        with g.as_default():
            with sess.as_default():

                multi_env = gym.make(env_name)

                policy = load_zoo_policy(victim, victim_type, "zoo_{}_policy_{}".format(env_name, id), multi_env, 0,
                                         sess=sess)

                # TODO remove this trash
                def get_action(observation):
                    return policy.act(stochastic=True, observation=observation)[0]

                single_env = FlattenSingletonEnv(CurryEnv(TheirsToOurs(multi_env), Agent(get_action, policy.reset)))

                #if env_name == 'kick-and-defend':
                   # #attacked_agent = utils.load_agent(trained_agent, policy_type,
                   # #                                  "zoo_{}_policy_{}".format(env_name, id), multi_env, 0)
                   # #single_env = MultiToSingle(CurryEnv(multi_env, attacked_agent))

                  #  single_env = HackyFixForGoalie(single_env)

                # TODO: upgrade Gym so don't have to do thi0s
                single_env.observation_space.dtype = np.dtype(np.float32)

                single_env = Monitor(single_env, osp.join(out_dir, 'mon', 'log{}'.format(id)))
        return single_env
        # TODO: close session?

    venv = SubprocVecEnv([functools.partial(make_env, i) for i in range(vector)])

    if not no_normalize:
        venv = VecNormalize(venv)

    return venv


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
    victim = "agent-zoo/sumo/ants/agent_parameters-v1.pkl"
    victim_type = "zoo"
    env = "sumo-ants-v0"
    vectorize = 8
    out_dir = "outs"
    exp_name = "Dummy Exp Name"
    no_normalize = True
    seed = 1
    total_timesteps = 100000
    network = "mlp"
    nsteps = 2048
    load_path = None


@ppo_baseline_ex.automain
def ppo_baseline(_run, env, victim, victim_type, out_dir, exp_name, vectorize, no_normalize, seed, total_timesteps,
                network, nsteps, load_path):
    #TODO some bug with vectorizing goalie
    if env == 'kick-and-defend' and vectorize != 1:
        raise Exception("Kick and Defend doesn't work with vecorization above 1")

    out_dir = setup_logger(out_dir, exp_name)

    env = get_env(env_name=env, victim=victim, victim_type=victim_type, out_dir=out_dir, no_normalize=no_normalize,
                  vector=vectorize)

    return train(env, out_dir=out_dir, seed=seed, total_timesteps=total_timesteps, vector=vectorize,
         network=network, no_normalize=no_normalize, nsteps=nsteps, load_path=load_path)
