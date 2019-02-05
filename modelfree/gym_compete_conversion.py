from aprl.envs.multi_agent import MultiAgentEnv
from modelfree.simulation_utils import ResettableAgent
import pickle
import numpy as np
import tensorflow as tf
import gym_compete
from modelfree.policy import LSTMPolicy, MlpPolicyValue

#TODO Some of the functions in here are copied from "main" in the multi-agent repo, and we have our own copy of policy


class TheirsToOurs(MultiAgentEnv):
    '''
    This class is a wrapper around the multi-agent environments from gym_compete which converts it to match our
    MultiAgentEnv class.
    '''
    def __init__(self, env):
        super().__init__(2, env.action_space.spaces[0], env.observation_space.spaces[0])
        # TODO ask adam about the action/observation spaces.  I am worried about asymmetric envs like goalie

        self.spec = env.spec
        self._env = env

    def step(self, action_n):
        observations, rewards, dones, infos = self._env.step(action_n)

        observations = list(observations)
        rewards = list(rewards)
        dones = list(dones)
        infos = {i: v for i, v in enumerate(infos)}

        return observations, rewards, dones, infos

    def reset(self):
        return list(self._env.reset())


def anounce_winner(sim_stream):
    '''
    This function determines the winner of a match in one of the gym_compete environments.
    :param sim_stream: a stream of obs, rewards, dones, infos from one of the gym_compete envs.
    :return: the index of the winning player, or None if it was a tie
    '''
    for _, _, dones, infos in sim_stream:
        if dones[0]:
            draw = True
            for i in range(len(infos)):
                if 'winner' in infos[i]:
                    draw = False
                    return i
            if draw:
                return None

    raise Exception("No Winner or Tie was ever announced")


def load_from_file(param_pkl_path):
    with open(param_pkl_path, 'rb') as f:
        params = pickle.load(f)
    return params


def get_policy_type_for_agent_zoo(env_name):
    '''
    Determines the type of policy gym_complete used in each environment.  This is needed because we must tell their code
    how to load their own policies.
    :param env_name: the environment of the policy we want to load
    :return: the type of the gym policy
    '''
    envs_policy_types = {
        "kick-and-defend-v0": "lstm",
        "run-to-goal-humans-v0": "mlp",
        "run-to-goal-ants-v0": "mlp",
        "you-shall-not-pass-humans-v0": "mlp",
        "sumo-humans-v0": "lstm",
        "sumo-ants-v0": "lstm"
    }

    if env_name in envs_policy_types:
        return envs_policy_types[env_name]

    raise Exception("Unsupported Environment: {}, choose from {}".format(env_name, envs_policy_types.keys()))


def set_from_flat(var_list, flat_params, sess = None):
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


def load_zoo_policy(file, policy_type, scope, env, index, sess):
    if policy_type == "lstm":
        policy = LSTMPolicy(scope=scope, reuse=False,
                            ob_space=env.observation_space.spaces[index],
                            ac_space=env.action_space.spaces[index],
                            hiddens=[128, 128], normalize=True, sess=sess)
    else:
        policy = MlpPolicyValue(scope=scope, reuse=False,
                                ob_space=env.observation_space.spaces[index],
                                ac_space=env.action_space.spaces[index],
                                hiddens=[64, 64], normalize=True, sess=sess)
    set_from_flat(policy.get_variables(), load_from_file(param_pkl_path=file), sess=sess)
    return policy


def load_zoo_agent(agent, env, env_name, index, sess):
    policy_type = get_policy_type_for_agent_zoo(env_name)

    with sess.graph.as_default():
            policy = load_zoo_policy(agent, policy_type, "zoo_policy_{}".format(agent), env, index, sess=sess)

    def get_action(observation):
        with sess.as_default():
            return policy.act(stochastic=True, observation=observation)[0]

    return ResettableAgent(get_action, policy.reset)
