import argparse
from simulation_utils import *
from main import *
import numpy as np
from numpy.random import random
import datetime
import pickle
import time

def random_search(env, agent_sampler, samples=5):
    """
    Performs random search for policies from "agent_sampler" which maximize reward on "env"
    :param env: The environment to be optimized for
    :param agent_sampler: The agent sampler to optimize from
    :param samples: how many samples to take
    :return: the best agent that has been found and the utility it achieved
    """
    first_iteration = True
    best_util = 0
    best_agent = None
    for i in range(samples):
        agent = agent_sampler()
        util = utility(simulate_single(env, agent))

        if best_util < util or first_iteration:
            best_util = util
            best_agent = agent

        if i %100 ==0 :
            print("Itteration {} at time {}".format(i,datetime.datetime.now()))

    return best_agent, best_util


def LSTM_agent_sampler(env, sess):
    policy = LSTMPolicy(scope="agent_new", reuse=False,
                        ob_space=env.observation_space.spaces[0],
                        ac_space=env.action_space.spaces[0],
                        hiddens=[128, 128], normalize=True)

    def sampler():

        def get_action(observation):
            return policy.act(stochastic=True, observation=observation)[0]

        sess.run(tf.initialize_variables(policy.get_variables()))

        values = {}
        for var in policy.get_variables():
            values[var] = sess.run(var)


        return Agent(get_action, policy.reset, values=values, sess=sess)

    return sampler


def matrix_agent_sampler(obs_dim=137, act_dim=8, magnitude = 100, mem_dim = 8):
    matrix_act_1 = random((obs_dim + mem_dim, act_dim)) * magnitude -magnitude/2
    matrix_act_2 = random((act_dim, act_dim)) * magnitude -magnitude/2
    matrix_act_3 = random((act_dim, act_dim)) * magnitude-magnitude/2
    matrix_mem = random((obs_dim + mem_dim, mem_dim)) * magnitude-magnitude/2
    mem_bias = random((mem_dim,)) * magnitude-magnitude/2

    class Temp():
        def __init__(self):
            self.start = np.random.random((mem_dim,))
            self.mem = self.start

        def get_action(self, observation):

            action = np.tanh(np.matmul(np.concatenate((observation, self.mem)), matrix_act_1))
            action = np.tanh(np.matmul(action, matrix_act_2))
            action = np.tanh(np.matmul(action, matrix_act_3))
            self.mem = np.tanh(np.matmul(np.concatenate((observation, self.mem)), matrix_mem)+mem_bias)

            print("me action {}".format(action))
            return action

        def reset(self):
            self.mem = self.start

    return Temp()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Environments for Multi-agent competition")
    p.add_argument("--samples", default=1, help="number of samples for random search", type=int)
    p.add_argument("--max-episodes", default=500, help="max number of matches", type=int)
    p.add_argument("--run_other", default=None, help="True if you should run last best", type=str)
    configs = p.parse_args()

    env, policy_type = get_env_and_policy_type("sumo-ants")

    ant_paths = get_trained_sumo_ant_locations()

    sess = make_session()
    with sess:

        attacked_agent = load_agent(ant_paths[1], policy_type, "zoo_ant_policy", env, 0)

        if configs.run_other is not None:
            policy = LSTMPolicy(scope="agent_new", reuse=False,
                                ob_space=env.observation_space.spaces[0],
                                ac_space=env.action_space.spaces[0],
                                hiddens=[128, 128], normalize=True)

            def get_action(observation):
                return policy.act(stochastic=True, observation=observation)[0]


            trained_agent = Agent(get_action, policy.reset)

            with open(configs.run_other, "rb") as file:
                values_from_save = pickle.load(file)

            for key, value in values_from_save.items():
                var = tf.get_default_graph().get_tensor_by_name(key)
                sess.run(tf.assign(var, value))

        else:
            trained_agent, reward = random_search(MultiToSingle(CurryEnv(env, attacked_agent)),
                                                  LSTM_agent_sampler(env, sess), configs.samples)

            print("Got {} reward!".format(reward))
            trained_agent.reinti()

            values_to_save = {}
            for key, value in trained_agent._values.items():
                values_to_save[key.name] = value
            print(values_to_save)
            with open("out_{}.pkl".format(int(round(time.time() * 1000))), "wb") as file:
                pickle.dump(values_to_save, file, pickle.HIGHEST_PROTOCOL)


        agents = [attacked_agent, trained_agent]
        for _ in range(configs.max_episodes):
            anounce_winner(simulate(env, agents, render=True))
            for agent in agents:
                agent.reset()
