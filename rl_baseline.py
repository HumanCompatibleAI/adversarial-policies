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
import numpy.linalg
from simulation_utils import MultiToSingle, CurryEnv, Gymify, HackyFixForGoalie
import utils
import policy
from utils import DelayedLoadEnv


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


def get_reward_wrapper(rewards=None):
    if rewards is None or not rewards:
        raise(Exception("Gave no reward function for agent.  It has no purpose :( "))

    if "their_win_loss" not in rewards:
        rewards.append("not_their_win_loss")
    else:
        rewards.remove("their_win_loss")
    return functools.partial(shape_reward, rewards)


def shape_reward(rewards=None, env=None):
    if env is None:
        raise(Exception("Env was unspecified, other args were:{}".format([rewards])))

    if rewards is None or not rewards:
        return env

    '''
        Note that when we are using their rewards we modify then recurse and when we are using ours we recurse and then 
        modify.  This is because we have to implement their rewards by modifying the environment while we can implement
        ours with wrappers
    '''

    reward_type = rewards.pop()
    if reward_type == "not_their_win_loss":
        env = env

        return shape_reward(rewards=rewards, env=env)
    elif reward_type == "their_shape":
        env.move_reward_weight = 0
        return shape_reward(rewards=rewards, env=env)

    else:
        env = shape_reward(rewards=rewards, env=env)
        args = reward_type.split("~")
        if len(args) != 4:
            raise (Exception("Unknown reward type {}".format(reward_type)))

        name = args[0]
        shape_style = args[1]
        const = float(args[2])
        cutoff = args[3]

        shapeing_functions = {
            "me_mag": me_mag,
            "me_pos_mag": me_pos_mag,
            "opp_mag": opp_mag,
            "opp_pos_mag": opp_pos_mag,
            "me_pos": me_pos_shape,
            "you_pos": you_pos_shape,
            "opp_goalie_pos_mag": opp_goalie_pos_mag,
            "opp_goalie_mag": opp_goalie_mag
        }

        if name not in shapeing_functions:
            raise (Exception("Unknown reward type {}".format(reward_type)))

        return ShapeingWrapper(env, shapeing_functions[name], shape_style, const, cutoff)


class ShapeingWrapper(object):

    def __init__(self, env, shapeing_fun, shape_style, const, cutoff):
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.shapeing_fun = shapeing_fun
        self.const = const
        self.shape_style = shape_style
        self.last_obs = None
        self.cutoff = cutoff

    def step(self, actions):
        observations, rewards, done, infos = self._env.step(actions)

        delta = self.shapeing_fun(observations, self.last_obs)
        points = np.linalg.norm(delta, ord=float(self.shape_style))

        if self.cutoff == "smooth":
            rewards += points * self.const
        else:
            cutoff_float = float(self.cutoff)
            if points > cutoff_float:
                rewards += self.const

        self.last_obs = observations

        return observations, rewards, done, infos

    def reset(self):
        self.last_obs = None
        observations = self._env.reset()
        return observations


def me_pos_shape(obs, last_obs):
    x_me = obs[0]
    y_me = obs[1]
    z_me = obs[2]

    return [x_me, y_me]


def you_pos_shape(obs, last_obs):

        x_opp = obs[-29]
        y_opp = obs[-28]
        z_opp = obs[-27]

        return [x_opp, y_opp]

def me_pos_mag(obs, last_obs):
    if last_obs is not None:

        last_me_pos = last_obs[0:3]
        cur_me_pos = obs[0:3]
        me_delta = cur_me_pos - last_me_pos

        #multiply by 2 to get units in body diameter
        return me_delta * 2
    return [0]

def me_pos_mag(obs, last_obs):
    if last_obs is not None:

        last_me_pos = last_obs[0:3]
        cur_me_pos = obs[0:3]
        me_delta = cur_me_pos - last_me_pos

        #multiply by 2 to get units in body diameter
        return me_delta * 2
    return [0]

def me_mag(obs, last_obs):
    if last_obs is not None:

        last_me_pos = last_obs[0:30]
        cur_me_pos = obs[0:30]
        me_delta = cur_me_pos - last_me_pos

        return me_delta
    return [0]

#TODO THis probably isnt right?  Shoud be -30:-27?
def opp_pos_mag(obs, last_obs):
    if last_obs is not None:
        last_opp_pos = last_obs[-3:]
        cur_opp_pos = obs[-3:]
        opp_delta = cur_opp_pos - last_opp_pos

        # multiply by 2 to get units in body diameter
        return opp_delta * 2
    return [0]


def opp_mag(obs, last_obs):
    if last_obs is not None:
        last_opp_pos = last_obs[-30:]
        cur_opp_pos = obs[-30:]
        opp_delta = cur_opp_pos - last_opp_pos

        return opp_delta
    return [0]

def opp_goalie_mag(obs, last_obs):
    if last_obs is not None:
        last_opp_pos = last_obs[-24:]
        cur_opp_pos = obs[-24:]
        opp_delta = cur_opp_pos - last_opp_pos

        return opp_delta
    return [0]

def opp_goalie_pos_mag(obs, last_obs):
    if last_obs is not None:
        last_opp_pos = last_obs[-24:-22]
        cur_opp_pos = obs[-24:-22]
        opp_delta = cur_opp_pos - last_opp_pos

        return opp_delta
    return [0]


def train(env, out_dir="results", seed=1, total_timesteps=1, vector=8, network="our-lstm", no_normalize=False):
    sess = utils.make_session()
    with sess:
        ### TRAIN AGENT  ####
        if network == 'our-lstm':
            network = mlp_lstm([128, 128], layer_norm=True)
        # TODO: speed up construction of mlp_lstm?
        model = ppo2.learn(network=network, env=env,
                           total_timesteps=total_timesteps,
                           seed=seed,
                           nminibatches=min(4, vector),
                           log_interval=1,
                           save_interval=1)
        model.save(osp.join(out_dir, 'model.pkl'))
        if not no_normalize:
            save_stats(env, osp.join(out_dir, 'normalize.pkl'))

    env.close()
    sess.close()


def setup_logger(out_dir="results", exp_name="test"):
    timestamp = datetime.datetime.now().strftime(ISO_TIMESTAMP)
    out_dir = osp.join(out_dir, '{} {}'.format(timestamp, exp_name))
    os.mkdir(out_dir)
    logger.configure(dir=osp.join(out_dir, 'mon'))
    return out_dir


def get_env(env_name, no_normalize = False, out_dir="results", vector=8, reward_wrapper=lambda env: env):
    trained_agent = utils.get_trained_agent(env_name)

    ### ENV SETUP ###
    # TODO: upgrade Gym so this monkey-patch isn't needed
    gym.spaces.Dict = type(None)

    def make_env(id):
        # TODO: seed (not currently supported)
        # TODO: VecNormalize? (typically good for MuJoCo)
        # TODO: baselines logger?
        # TODO: we're loading identical policy weights into different
        # variables, this is to work-around design choice of Agent's
        # having state stored inside of them.
        sess = utils.make_session()
        with sess.as_default():
            multi_env, policy_type = utils.get_env_and_policy_type(env_name)

            if env_name == 'kick-and-defend':
                #attacked_agent = utils.load_agent(trained_agent, policy_type,
                #                                  "zoo_{}_policy_{}".format(env_name, id), multi_env, 0)
                #single_env = MultiToSingle(CurryEnv(multi_env, attacked_agent))
                single_env = MultiToSingle(DelayedLoadEnv(multi_env, trained_agent, policy_type,
                                                          "zoo_{}_policy_{}".format(env_name, id), 0, sess))
                single_env = HackyFixForGoalie(single_env)
            else:
                attacked_agent = utils.load_agent(trained_agent, policy_type,
                                                  "zoo_{}_policy_{}".format(env_name, id), multi_env, 0)
                single_env = MultiToSingle(CurryEnv(multi_env, attacked_agent))

            single_env = reward_wrapper(single_env)

            single_env = Gymify(single_env)
            single_env.spec = gym.envs.registration.EnvSpec('Dummy-v0')

            # TODO: upgrade Gym so don't have to do thi0s
            single_env.observation_space.dtype = np.dtype(np.float32)

            single_env = Monitor(single_env, osp.join(out_dir, 'mon', 'log{}'.format(id)))
        return single_env
        # TODO: close session?

    venv = SubprocVecEnv([functools.partial(make_env, i) for i in range(vector)])

    if not no_normalize:
        venv = VecNormalize(venv)

    return venv


def main(configs):

    #TODO some bug with vectorizing goalie
    if configs.env=='kick-and-defend':
        configs.vector=1

    out_dir = setup_logger(configs.out_dir, configs.exp_name)

    reward_wrapper = get_reward_wrapper(configs.reward)

    env = get_env(env_name=configs.env, out_dir=out_dir, no_normalize=configs.no_normalize, vector=configs.vector,
                  reward_wrapper=reward_wrapper)

    train(env, out_dir=out_dir, seed=configs.seed, total_timesteps=configs.total_timesteps, vector=configs.vector,
          network=configs.network, no_normalize=configs.no_normalize)



ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Runs RL against fixed opponent")
    p.add_argument("--env", default="sumo-ants", type=str)
    p.add_argument('--vector', default=8, help="parallel vector sampling", type=int)
    p.add_argument('--total-timesteps', default=1000000, type=int)
    p.add_argument('--out-dir', default='results', type=str)
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--network', default='our-lstm')
    p.add_argument('--no-normalize', type=bool)
    p.add_argument('exp_name', type=str)
    p.add_argument('--reward', action='append', required=True)

    main(p.parse_args())
