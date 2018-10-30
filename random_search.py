import argparse
from simulation_utils import *
from main import *
import numpy as np
from numpy.random import random
import datetime

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


def matrix_agent_sampler(obs_dim=137, act_dim=8, magnitude = 2, mem_size = 10):
    matrix = random((obs_dim, act_dim)) * magnitude

    class Temp():
        def __init__(self):
            self.start = np.random.random((mem_size,))
            self.mem = self.start

        def get_action(self, observation):
            action = np.matmul(observation, matrix)

            action = action - np.mean(action)
            action = action/np.std(action)
            return action

        def reset(self):
            self.mem = self.start

    return Temp()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Environments for Multi-agent competition")
    p.add_argument("--samples", default=1500, help="number of samples for random search", type=int)
    p.add_argument("--max-episodes", default=500, help="max number of matches", type=int)

    configs = p.parse_args()

    env, policy_type = get_env_and_policy_type("sumo-ants")

    ant_paths = get_trained_sumo_ant_locations()

    with(make_session()):

        attacked_agent = load_agent(ant_paths[1], policy_type, "zoo_ant_policy", env, 0)
        trained_agent, reward = random_search(MultiToSingle(CurryEnv(env, attacked_agent)), matrix_agent_sampler, configs.samples)

        print("Got {} reward!".format(reward))

        agents = [attacked_agent, trained_agent]
        for _ in range(configs.max_episodes):
            anounce_winner(simulate(env, agents, render=True))
            for agent in agents:
                agent.reset()
