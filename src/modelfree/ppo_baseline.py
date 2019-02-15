"""Uses PPO to train an attack policy against a fixed victim policy."""

import datetime
import functools
import os
import os.path as osp
import pickle

from baselines import logger
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from sacred import Experiment
from sacred.observers import FileStorageObserver
import tensorflow as tf

from modelfree.policy_loader import get_agent_any_type
from modelfree.utils import make_session, make_single_env


def make_zoo_vec_env(env_name, victim_path, victim_type, victim_index,
                     no_normalize, seed, out_dir, vector):
    def agent_fn(env, sess):
        return get_agent_any_type(agent=victim_path, agent_type=victim_type, env=env,
                                  env_name=env_name, index=victim_index, sess=sess)

    def make_env(i):
        return make_single_env(env_name, seed + i, agent_fn, out_dir, env_id=i)

    venv = SubprocVecEnv([functools.partial(make_env, i) for i in range(vector)])

    if not no_normalize:
        venv = VecNormalize(venv)

    return venv


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
        model = ppo2.learn(network=network, env=env,
                           total_timesteps=total_timesteps,
                           nsteps=nsteps // vector,
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
    os.makedirs(out_dir, exist_ok=True)
    logger.configure(dir=osp.join(out_dir, 'mon'))
    return out_dir


ppo_baseline_ex = Experiment("ppo_baseline")
ppo_baseline_ex.observers.append(FileStorageObserver.create("data/sacred"))


@ppo_baseline_ex.config
def default_ppo_config():
    victim_path = "1"
    victim_type = "zoo"
    env = "multicomp/SumoAnts-v0"
    vectorize = 8
    out_dir = "data/baselines"
    exp_name = "Dummy Exp Name"
    no_normalize = True
    seed = 1
    total_timesteps = 1000000
    network = "mlp"
    nsteps = 2048
    load_path = None
    _ = locals()  # quieten flake8 unused variable warning
    del _


# TODO: use victim_type
@ppo_baseline_ex.automain
def ppo_baseline(_run, env, victim_path, victim_type, out_dir, exp_name, vectorize,
                 no_normalize, seed, total_timesteps, network, nsteps, load_path):
    # TODO: some bug with vectorizing goalie
    if env == 'kick-and-defend' and vectorize != 1:
        raise Exception("Kick and Defend doesn't work with vecorization above 1")

    out_dir = setup_logger(out_dir, exp_name)
    env = make_zoo_vec_env(env_name=env, victim_path=victim_path, victim_type=victim_type,
                           victim_index=0, no_normalize=no_normalize, seed=seed,
                           out_dir=out_dir, vector=vectorize)
    res = train(env, out_dir=out_dir, seed=seed, total_timesteps=total_timesteps,
                vector=vectorize, network=network, no_normalize=no_normalize,
                nsteps=nsteps, load_path=load_path)
    env.close()

    return res
