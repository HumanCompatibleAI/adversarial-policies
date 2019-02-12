import os
import pickle
import pkgutil

from gym_compete.policy import LSTMPolicy, MlpPolicyValue
import numpy as np
import tensorflow as tf

from aprl.envs.multi_agent import MultiAgentEnv
from modelfree.simulation_utils import ResettableAgent


class TheirsToOurs(MultiAgentEnv):
    """This class is a wrapper around the multi-agent environments from gym_compete which converts
       it to match our MultiAgentEnv class."""
    def __init__(self, env):
        super().__init__(2, env.action_space.spaces[0], env.observation_space.spaces[0])
        # TODO ask adam about the action/observation spaces.
        # I am worried about asymmetric envs like goalie

        self.spec = env.spec
        self._env = env

    def step(self, action_n):
        observations, rewards, dones, infos = self._env.step(action_n)

        observations = list(observations)
        rewards = list(rewards)
        done = any(dones)
        infos = {i: v for i, v in enumerate(infos)}

        return observations, rewards, done, infos

    def reset(self):
        return list(self._env.reset())


def announce_winner(sim_stream):
    """This function determines the winner of a match in one of the gym_compete environments.
    :param sim_stream: a stream of obs, rewards, dones, infos from one of the gym_compete envs.
    :return: the index of the winning player, or None if it was a tie."""
    for _, _, dones, infos in sim_stream:
        if dones[0]:
            draw = True
            for i in range(len(infos)):
                if 'winner' in infos[i]:
                    return i
            if draw:
                return None

    raise Exception("No Winner or Tie was ever announced")


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

    def get_action(observation):
        with sess.as_default():
            return policy.act(stochastic=True, observation=observation)[0]

    return ResettableAgent(get_action, policy.reset)
