import argparse
from simulation_utils import *
from main import *
import numpy as np
from numpy.random import random
import datetime
import pickle
import time


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Environments for Multi-agent competition")
    p.add_argument("--max-episodes", default=0, help="max number of matches during visualization", type=int)
    configs = p.parse_args()

    env, policy_type = get_env_and_policy_type("sumo-ants")

    ant_paths = get_trained_sumo_ant_locations()

    sess = make_session()
    with sess:

        attacked_agent = load_agent(ant_paths[1], policy_type, "zoo_ant_policy_1", env, 0)

        env_wit_agent_embedded = MultiToSingle(CurryEnv(env, attacked_agent))


        ### TRAIN AGENT  ####
        #TODO
        trained_agent = load_agent(ant_paths[3], policy_type, "zoo_ant_policy_2", env, 1)
        #####################


        ### Save Agent Probably? ###
        #TODO
        ############################


        ### This just runs a visualization of the results and anounces wins.  Will crash if you can't render
        agents = [attacked_agent, trained_agent]
        for _ in range(configs.max_episodes):
            anounce_winner(simulate(env, agents, render=True))
            for agent in agents:
                agent.reset()
