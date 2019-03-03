"""Load serialized policies of different types."""

from stable_baselines import PPO2
import tensorflow as tf

from aprl.envs.multi_agent import FakeSingleSpacesVec
from modelfree.gym_compete_conversion import load_zoo_agent
from modelfree.utils import OpenAIToStablePolicy, PolicyToModel, RandomPolicy, ZeroPolicy


def load_stable_baselines(cls):
    def f(path, env, env_name, index):
        denv = FakeSingleSpacesVec(env, agent_id=index)
        return cls.load(path, env=denv)
    return f


def load_old_ppo2(path, env, env_name, index):
    try:
        from baselines.ppo2 import ppo2 as ppo2_old
    except ImportError as e:
        msg = "{}. HINT: you need to install (OpenAI) Baselines to use old_ppo2".format(e)
        raise ImportError(msg)

    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with sess.as_default():
        with graph.as_default():
            denv = FakeSingleSpacesVec(env, agent_id=index)
            policy = ppo2_old.learn(network="mlp", env=denv,
                                    total_timesteps=1, seed=0,
                                    nminibatches=4, log_interval=1, save_interval=1,
                                    load_path=path)
    stable_policy = OpenAIToStablePolicy(policy)
    return PolicyToModel(stable_policy)


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
