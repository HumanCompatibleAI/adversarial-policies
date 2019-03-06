"""Load serialized policies of different types."""

import logging
import os
import pickle

from stable_baselines import PPO2
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
import tensorflow as tf

from aprl.envs.multi_agent import FakeSingleSpacesVec
from modelfree.gym_compete_conversion import load_zoo_agent
from modelfree.utils import (DummyModel, OpenAIToStablePolicy, PolicyToModel, RandomPolicy,
                             ZeroPolicy)

pylog = logging.getLogger('modelfree.policy_loader')


class NormalizeModel(DummyModel):
    def __init__(self, policy, vec_normalize):
        super().__init__(policy, policy.sess)
        self.vec_normalize = vec_normalize

    def predict(self, observation, state=None, mask=None, deterministic=False):
        norm_obs = self.vec_normalize._normalize_observation(observation)
        return self.policy.predict(norm_obs, state, mask, deterministic)


def load_stable_baselines(cls):
    def f(root_dir, env, env_name, index):
        denv = FakeSingleSpacesVec(env, agent_id=index)
        model_path = os.path.join(root_dir, 'model.pkl')
        pylog.info(f"Loading Stable Baselines policy for '{cls}' from '{model_path}'")
        model = cls.load(model_path, env=denv)

        try:
            vec_normalize = VecNormalize(denv, training=False)
            vec_normalize.load_running_average(root_dir)
            model = NormalizeModel(model, vec_normalize)
            pylog.info(f"Loaded normalization statistics from '{root_dir}'")
        except FileNotFoundError:
            # We did not use VecNormalize during training, skip
            pass

        return model

    return f


def load_old_ppo2(root_dir, env, env_name, index):
    try:
        from baselines.ppo2 import ppo2 as ppo2_old
    except ImportError as e:
        msg = "{}. HINT: you need to install (OpenAI) Baselines to use old_ppo2".format(e)
        raise ImportError(msg)

    denv = FakeSingleSpacesVec(env, agent_id=index)
    model_path = os.path.join(root_dir, 'model.pkl')

    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with sess.as_default():
        with graph.as_default():
            pylog.info(f"Loading Baselines PPO2 policy from '{model_path}'")
            policy = ppo2_old.learn(network="mlp", env=denv,
                                    total_timesteps=1, seed=0,
                                    nminibatches=4, log_interval=1, save_interval=1,
                                    load_path=model_path)
    stable_policy = OpenAIToStablePolicy(policy)
    model = PolicyToModel(stable_policy)

    try:
        normalize_path = os.path.join(root_dir, 'normalize.pkl')
        with open(normalize_path, 'rb') as f:
            old_vec_normalize = pickle.load(f)
        vec_normalize = VecNormalize(denv, training=False)
        vec_normalize.obs_rms = old_vec_normalize.ob_rms
        vec_normalize.ret_rms = old_vec_normalize.ret_rms
        model = NormalizeModel(model, vec_normalize)
        pylog.info(f"Loaded normalization statistics from '{normalize_path}'")
    except FileNotFoundError:
        # We did not use VecNormalize during training, skip
        pass

    return model


def load_zero(path, env, env_name, index):
    denv = FakeSingleSpacesVec(env, agent_id=index)
    policy = ZeroPolicy(denv)
    return PolicyToModel(policy)


def load_random(path, env, env_name, index):
    denv = FakeSingleSpacesVec(env, agent_id=index)
    policy = RandomPolicy(denv)
    return PolicyToModel(policy)


AGENT_LOADERS = {
    'zoo': load_zoo_agent,
    'ppo2': load_stable_baselines(PPO2),
    'old_ppo2': load_old_ppo2,
    'zero': load_zero,
    'random': load_random,
}


def load_policy(policy_type, policy_path, env, env_name, index):
    agent_loader = AGENT_LOADERS.get(policy_type)
    if agent_loader is None:
        raise ValueError(f"Unrecognized agent type '{policy_type}'")
    return agent_loader(policy_path, env, env_name, index)
