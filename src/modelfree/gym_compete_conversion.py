from collections import Counter
import logging
import os
import pickle
import pkgutil

from gym import Wrapper
from gym_compete.policy import LSTMPolicy, MlpPolicyValue
import numpy as np
import tensorflow as tf

from aprl.envs.multi_agent import MultiAgentEnv, VecMultiWrapper
from modelfree.utils import PolicyToModel, make_session

pylog = logging.getLogger('modelfree.gym_compete_conversion')


class GymCompeteToOurs(Wrapper, MultiAgentEnv):
    """This adapts gym_compete.MultiAgentEnv to our eponymous MultiAgentEnv.

       The main differences are that we have a scalar done (episode-based) rather than vector
       (agent-based), and only return one info dict (property of environment not agent)."""
    def __init__(self, env):
        Wrapper.__init__(self, env)
        MultiAgentEnv.__init__(self, num_agents=2)

    def step(self, action_n):
        observations, rewards, dones, infos = self.env.step(action_n)
        done = any(dones)
        infos = {i: v for i, v in enumerate(infos)}
        return observations, rewards, done, infos

    def reset(self):
        return self.env.reset()


def game_outcome(info):
    draw = True
    for i, agent_info in info.items():
        if 'winner' in agent_info:
            return i
    if draw:
        return None


class GameOutcomeMonitor(VecMultiWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.outcomes = []

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rew, dones, infos = self.venv.step_wait()
        for done, info in zip(dones, infos):
            if done:
                self.outcomes.append(game_outcome(info))
        return obs, rew, dones, infos

    def log_callback(self, logger):
        c = Counter()
        c.update(self.outcomes)
        num_games = len(self.outcomes)
        if num_games > 0:
            for agent in range(self.num_agents):
                logger.logkv(f"game_win{agent}", c.get(agent, 0) / num_games)
            logger.logkv("game_tie", c.get(None, 0) / num_games)
        logger.logkv("game_total", num_games)
        self.outcomes = []


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
        "SumoAnts-v0": "lstm",
    }
    if env_name in policy_types:
        return policy_types[env_name]
    else:
        msg = "Unsupported Environment: {}, choose from {}".format(env_name, policy_types.keys())
        raise ValueError(msg)


def set_from_flat(var_list, flat_params, sess):
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
    sess.run(op, {theta: flat_params})


def load_zoo_policy(tag, policy_type, scope, env, env_name, index):
    g = tf.Graph()
    sess = make_session(g)

    with g.as_default():
        with sess.as_default():
            # Construct graph
            kwargs = dict(sess=sess, ob_space=env.observation_space.spaces[index],
                          ac_space=env.action_space.spaces[index], n_env=env.num_envs,
                          n_steps=1, n_batch=env.num_envs, scope=scope, reuse=False,
                          normalize=True)
            if policy_type == 'lstm':
                policy = LSTMPolicy(hiddens=[128, 128], **kwargs)
            elif policy_type == 'mlp':
                policy = MlpPolicyValue(hiddens=[64, 64], **kwargs)
            else:
                raise NotImplementedError()

            # Load parameters
            dir = os.path.join('agent_zoo', env_name)
            asymmetric_fname = f'agent{index+1}_parameters-v{tag}.pkl'
            symmetric_fname = f'agent_parameters-v{tag}.pkl'
            try:  # asymmetric version, parameters tagged with agent id
                path = os.path.join(dir, asymmetric_fname)
                params_pkl = pkgutil.get_data('gym_compete', path)
            except OSError:  # symmetric version, parameters not associated with a specific agent
                path = os.path.join(dir, symmetric_fname)
                params_pkl = pkgutil.get_data('gym_compete', path)
            pylog.info(f"Loaded zoo policy from '{path}'")

            # Restore parameters
            params = pickle.loads(params_pkl)
            set_from_flat(policy.get_variables(), params, sess)

            return PolicyToModel(policy)


def load_zoo_agent(path, env, env_name, index):
    env_aliases = {
        'multicomp/SumoHumansAutoContact-v0': 'multicomp/SumoHumans-v0'
    }
    env_name = env_aliases.get(env_name, env_name)

    env_prefix, env_suffix = env_name.split('/')
    assert env_prefix == 'multicomp'
    policy_type = get_policy_type_for_agent_zoo(env_suffix)
    return load_zoo_policy(path, policy_type, "zoo_policy_{}".format(path),
                           env, env_suffix, index)
