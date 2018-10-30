from policy import LSTMPolicy, MlpPolicyValue
import gym
# noinspection PyUnresolvedReferences
import gym_compete
import pickle
import sys
import argparse
import tensorflow as tf
import numpy as np
from simulation_utils import *


def load_from_file(param_pkl_path):
    with open(param_pkl_path, 'rb') as f:
        params = pickle.load(f)
    return params


def set_from_flat(var_list, flat_params):
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
    tf.get_default_session().run(op, {theta: flat_params})


def load_policy(file, policy_type, scope, env, index):
    if policy_type == "lstm":
        policy = LSTMPolicy(scope=scope, reuse=False,
                            ob_space=env.observation_space.spaces[index],
                            ac_space=env.action_space.spaces[index],
                            hiddens=[128, 128], normalize=True)
    else:
        policy = MlpPolicyValue(scope=scope, reuse=False,
                                ob_space=env.observation_space.spaces[index],
                                ac_space=env.action_space.spaces[index],
                                hiddens=[64, 64], normalize=True)
    set_from_flat(policy.get_variables(), load_from_file(param_pkl_path=file))
    return policy


def get_env_and_policy_type(env_name):
    if env_name == "kick-and-defend":
        env = gym.make("kick-and-defend-v0")
        policy_type = "lstm"
    elif env_name == "run-to-goal-humans":
        env = gym.make("run-to-goal-humans-v0")
        policy_type = "mlp"
    elif env_name == "run-to-goal-ants":
        env = gym.make("run-to-goal-ants-v0")
        policy_type = "mlp"
    elif env_name == "you-shall-not-pass":
        env = gym.make("you-shall-not-pass-humans-v0")
        policy_type = "mlp"
    elif env_name == "sumo-humans":
        env = gym.make("sumo-humans-v0")
        policy_type = "lstm"
    elif env_name == "sumo-ants":
        env = gym.make("sumo-ants-v0")
        policy_type = "lstm"
    else:
        print("unsupported environment")
        print("choose from: run-to-goal-humans, run-to-goal-ants, you-shall-not-pass,\
               sumo-humans, sumo-ants, kick-and-defend")
        sys.exit()
    return env, policy_type


class Agent(object):

    def __init__(self, action_selector, reseter, values=None, sess = None):
        """
        Takes policies from their format to mine
        :param actable: a policy in the format used by mult-agent-compeitition
        """
        self._action_selector = action_selector
        self._reseter = reseter
        self._values = values
        self._sess = sess

    def get_action(self, observation):
        action = self._action_selector(observation)
        #print(action)
        return action

    def reset(self):
        return self._reseter()


    def reinti(self):
        def reiniter():
            self._sess.run(self._values)



def load_agent(file, policy_type, scope, env, index):
    policy = load_policy(file, policy_type, scope, env, index)

    def get_action(observation):
        return policy.act(stochastic=True, observation=observation)[0]

    return Agent(get_action, policy.reset)


def make_session():
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    sess = tf.Session(config=tf_config)
    return sess


def run(config):
    env, policy_type = get_env_and_policy_type(config.env)

    ant_paths = get_trained_sumo_ant_locations()

    with(make_session()):
        agent_paths = [ant_paths[1], ant_paths[3]]
        agents = []
        for i in range(2):
            agents.append(load_agent(agent_paths[i], policy_type, "policy" + str(i), env, i))

        for _ in range(config.max_episodes):
            anounce_winner(simulate(env, agents))
            for agent in agents:
                agent.reset()


def get_trained_sumo_ant_locations():
    policy_loc = "multiagent-competition/agent-zoo/sumo/ants/"
    return [policy_loc + "agent_parameters-v1.pkl",
            policy_loc + "agent_parameters-v2.pkl",
            policy_loc + "agent_parameters-v3.pkl",
            policy_loc + "agent_parameters-v4.pkl"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Environments for Multi-agent competition")
    p.add_argument("--env", default="sumo-humans", type=str,
                   help="competitive environment: run-to-goal-humans, run-to-goal-ants,\
                         you-shall-not-pass, sumo-humans, sumo-ants, kick-and-defend")
    p.add_argument("--max-episodes", default=10, help="max number of matches", type=int)

    configs = p.parse_args()

    run(configs)
