import os
from os import path as osp
import pickle
import pkgutil

import gym
from gym import Wrapper
from gym_compete.policy import LSTMPolicy, MlpPolicyValue
import numpy as np
import tensorflow as tf

from aprl.common.multi_monitor import MultiMonitor
from aprl.envs.multi_agent import MultiAgentEnv


class GymCompeteToOurs(Wrapper, MultiAgentEnv):
    """This adapts gym_compete.MultiAgentEnv to our eponymous MultiAgentEnv.

       The main differences are that we have a scalar done (episode-based) rather than vector
       (agent-based), and only return one info dict (property of environment not agent)."""
    def __init__(self, env):
        Wrapper.__init__(self, env)
        MultiAgentEnv.__init__(self, num_agents=2)
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        # Gym added dtype's to shapes in commit 1c5a463
        # Baselines depends on this, but we have to use an older version of Gym
        # to keep support for MuJoCo 1.31. Monkeypatch shapes to fix this.
        # Note: the environments actually return float64, but this precision is not needed.
        def set_dtype(tuple_space, dtype):
            tuple_space.dtype = dtype
            for space in tuple_space.spaces:
                space.dtype = dtype

        set_dtype(self.observation_space, np.dtype('float32'))
        set_dtype(self.action_space, np.dtype('float32'))

    def step(self, action_n):
        observations, rewards, dones, infos = self.env.step(action_n)
        done = any(dones)
        infos = {i: v for i, v in enumerate(infos)}
        return observations, rewards, done, infos

    def reset(self):
        return self.env.reset()


def make_gym_compete_env(env_name, seed, i, out_dir):
    multi_env = gym.make(env_name)
    multi_env = GymCompeteToOurs(multi_env)
    multi_env.seed(seed + i)

    if out_dir is not None:
        mon_dir = osp.join(out_dir, 'mon')
        os.makedirs(mon_dir, exist_ok=True)
        multi_env = MultiMonitor(multi_env, osp.join(mon_dir, 'log{}'.format(i)))

    return multi_env


def get_policy_type_for_agent_zoo(env_name):
    """Determines the type of policy gym_complete used in each environment. This is needed because
    we must tell their code how to load their own policies.
    :param env_name: the environment of the policy we want to load
    :return: the type of the gym policy."""
    policy_types = {
        "KickAndDefend-v0": "lstm",
        "RunToGoalHumans-v0": "mlp",
        "RunToGoalAnts-v0": "mlp",
        "YouShallNotPassHumans-v0": "mlp",
        "SumoHumans-v0": "lstm",
        "SumoAnts-v0": "lstm"
    }
    if env_name in policy_types:
        return policy_types[env_name]
    else:
        msg = "Unsupported Environment: {}, choose from {}".format(env_name, policy_types.keys())
        raise ValueError(msg)


def set_from_flat(var_list, flat_params, sess=None):
    shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
    total_size = np.sum([int(np.prod(shape)) for shape in shapes])
    theta = tf.placeholder(tf.float32, [total_size])
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = int(np.prod(shape))
        assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
        start += size
    op = tf.group(*assigns)
    if sess is None:
        sess = tf.get_default_session()
    sess.run(op, {theta: flat_params})


def load_zoo_policy(id, policy_type, scope, env, env_name, index, sess):
    # Construct graph
    if policy_type == 'lstm':
        policy = LSTMPolicy(scope=scope, reuse=False,
                            ob_space=env.observation_space.spaces[index],
                            ac_space=env.action_space.spaces[index],
                            hiddens=[128, 128], normalize=True, sess=sess)
    elif policy_type == 'mlp':
        policy = MlpPolicyValue(scope=scope, reuse=False,
                                ob_space=env.observation_space.spaces[index],
                                ac_space=env.action_space.spaces[index],
                                hiddens=[64, 64], normalize=True, sess=sess)
    else:
        raise NotImplementedError()

    # Load parameters
    dir = os.path.join('agent_zoo', env_name)
    asymmetric_fname = f'agent{index}_parameters-v{id}.pkl'
    symmetric_fname = f'agent_parameters-v{id}.pkl'
    try:  # asymmetric version, parameters tagged with agent id
        params_pkl = pkgutil.get_data('gym_compete', os.path.join(dir, asymmetric_fname))
    except OSError:  # symmetric version, parameters not associated with a specific agent
        params_pkl = pkgutil.get_data('gym_compete', os.path.join(dir, symmetric_fname))

    # Restore parameters
    params = pickle.loads(params_pkl)
    set_from_flat(policy.get_variables(), params, sess=sess)

    return policy


def load_zoo_agent(path, env, env_name, index, sess):
    env_prefix, env_suffix = env_name.split('/')
    assert env_prefix == 'multicomp'
    policy_type = get_policy_type_for_agent_zoo(env_suffix)

    with sess.graph.as_default():
        policy = load_zoo_policy(path, policy_type, "zoo_policy_{}".format(path),
                                 env, env_suffix, index, sess=sess)
        return policy
