"""Uses PPO to train an attack policy against a fixed victim policy."""

import datetime
import os
import os.path as osp
import pickle

from baselines import logger
from baselines.common.vec_env.vec_normalize import VecEnvWrapper, VecNormalize
from baselines.ppo2 import ppo2
from sacred import Experiment
from sacred.observers import FileStorageObserver
import tensorflow as tf

from aprl.envs.multi_agent import CurryVecEnv, FlattenSingletonVecEnv, make_subproc_vec_multi_env
from modelfree.gym_compete_conversion import make_gym_compete_env
from modelfree.policy_loader import get_agent_any_type
from modelfree.utils import make_session


class EmbedVictimWrapper(VecEnvWrapper):
    def __init__(self, multi_env, env_name, victim_path, victim_type, victim_index):
        victim_graph = tf.Graph()
        self.sess = make_session(victim_graph)
        with self.sess.as_default():
            # TODO: multi_env is a VecEnv not an Env, may break loaders?
            victim = get_agent_any_type(agent=victim_path, agent_type=victim_type, env=multi_env,
                                        env_name=env_name, index=victim_index, sess=self.sess)
            curried_env = CurryVecEnv(multi_env, victim, agent_idx=victim_index)
            single_env = FlattenSingletonVecEnv(curried_env)

        super().__init__(single_env)

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()

    def close(self):
        self.sess.close()
        super().close()


def save_stats(env_wrapper, path):
    venv = env_wrapper.venv
    env_wrapper.venv = None
    with open(path, 'wb') as f:
        serialized = pickle.dump(env_wrapper, f)
    env_wrapper.venv = venv
    return serialized


def train(env, out_dir="results", seed=1, total_timesteps=1, vector=8, network="our-lstm",
          normalize=False, batch_size=2048, load_path=None):
    g = tf.Graph()
    sess = make_session(g)
    model_path = osp.join(out_dir, 'model.pkl')
    with sess:
        model = ppo2.learn(network=network, env=env,
                           total_timesteps=total_timesteps,
                           nsteps=batch_size // vector,
                           seed=seed,
                           nminibatches=min(4, vector),
                           log_interval=1,
                           save_interval=1,
                           load_path=load_path)
        model.save(model_path)
        if normalize:
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
    env_name = "multicomp/SumoAnts-v0"   # Gym environment ID
    victim_type = "zoo"             # type supported by policy_loader.py
    victim_path = "1"               # path or other unique identifier
    victim_index = 0                # which agent the victim is (we default to other agent)
    vectorize = 8                   # number of environments to run in parallel
    out_dir = "data/baselines"      # root of directory to store baselines log
    exp_name = "Dummy Exp Name"     # name of experiment
    normalize = False               # normalize observations
    total_timesteps = 4096          # total number of timesteps to train for
    network = "mlp"                 # policy network type
    batch_size = 2048               # batch size
    seed = 1
    load_path = None                # path to load initial policy from
    _ = locals()  # quieten flake8 unused variable warning
    del _


@ppo_baseline_ex.automain
def ppo_baseline(_run, env_name, victim_path, victim_type, victim_index, out_dir, exp_name,
                 vectorize, normalize, seed, total_timesteps, network, batch_size, load_path):
    # TODO: some bug with vectorizing goalie
    if env_name == 'kick-and-defend' and vectorize != 1:
        raise Exception("Kick and Defend doesn't work with vectorization above 1")

    out_dir = setup_logger(out_dir, exp_name)

    def make_env(i):
        return make_gym_compete_env(env_name, seed, i, out_dir)

    multi_env = make_subproc_vec_multi_env([lambda: make_env(i) for i in range(vectorize)])
    single_env = EmbedVictimWrapper(multi_env=multi_env, env_name=env_name,
                                    victim_path=victim_path, victim_type=victim_type,
                                    victim_index=victim_index)
    if normalize:
        single_env = VecNormalize(single_env)
    res = train(single_env, out_dir=out_dir, seed=seed, total_timesteps=total_timesteps,
                vector=vectorize, network=network, normalize=normalize,
                batch_size=batch_size, load_path=load_path)
    single_env.close()

    return res
