import argparse
from simulation_utils import *
from main import *
from random_search import *
import numpy as np
from numpy.random import random
import datetime
import pickle
import time

def get_emperical_score(agents, trials, render=False):
    tiecount = 0
    wincount = [0] * len(agents)
    for _ in range(trials):
        result = new_anounce_winner(simulate(env, agents, render=render))
        if result == 1:
            tiecount += 1
        else:
            wincount[result] += 1
        for agent in agents:
            agent.reset()
    return tiecount, wincount

#I copied this over to avoid possible merge conflicts, the other one should be removed T
def new_anounce_winner(sim_stream):
    for _, _, dones, infos in sim_stream:

        if dones[0]:
            draw = True
            for i in range(len(infos)):
                if 'winner' in infos[i]:
                    draw = False
                    print("Winner: Agent {}, Scores: {}, Total Episodes: {}".format(i, 1,1))
                    return i
            if draw:
                print("Game Tied: Agent {}, Scores: {}, Total Episodes: {}".format(i, 1,1))
                return -1

def get_agent_any_type(type, name, policy_type, env):
    if type == "zoo":
        return load_agent(name, policy_type,"zoo_ant_policy_2", env, 1)
    elif type == "const":
        trained_agent = constant_agent_sampler()
        trained_agent.load(name)
        return trained_agent
    elif type =="lstm":
        policy = LSTMPolicy(scope="agent_new", reuse=False,
                            ob_space=env.observation_space.spaces[0],
                            ac_space=env.action_space.spaces[0],
                            hiddens=[128, 128], normalize=True)

        def get_action(observation):
            return policy.act(stochastic=True, observation=observation)[0]

        trained_agent = Agent(get_action, policy.reset)

        with open(name, "rb") as file:
            values_from_save = pickle.load(file)

        for key, value in values_from_save.items():
            var = tf.get_default_graph().get_tensor_by_name(key)
            sess.run(tf.assign(var, value))

        return trained_agent


def evaluate_agent(attacked_agent, type_in, name, policy_type, env, samples):
    trained_agent = get_agent_any_type(type_in, name, policy_type, env)

    agents = [attacked_agent, trained_agent]
    tiecount, wincounts = get_emperical_score(agents, samples, render=False)

    print("After {} trials the tiecount was {} and the wincounts were {}".format(samples,
                                                                                 tiecount, wincounts))
    return tiecount, wincounts

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Collecting Win/Loss/Tie Statistics against ant_pats[1]")
    p.add_argument("--samples", default=0, help="max number of matches during visualization", type=int)
    p.add_argument("--agent_to_eval", default=None, help="True if you should run last best", type=str)
    p.add_argument("--agent_type", default="zoo", help="Either zoo, const, lstm or matrix", type=str)
    p.add_argument("--all", default=False, help="run evaluation on all of the default agents", type=bool)
    configs = p.parse_args()

    env, policy_type = get_env_and_policy_type("sumo-ants")

    ant_paths = get_trained_sumo_ant_locations()

    sess = make_session()
    with sess:

        #TODO Load Agent should be changed to "load_zoo_agent"
        attacked_agent = load_agent(ant_paths[1], policy_type, "zoo_ant_policy", env, 0)

        if not configs.all:
            evaluate_agent(attacked_agent, configs.agent_type, configs.agent_to_eval, policy_type, env, configs.samples)

        else:
            trained_agents = {"pretrained": {"agent_to_eval": get_trained_sumo_ant_locations()[3],
                                             "agent_type": "zoo"},
                              "random_const": {"agent_to_eval": "out_random_const.pkl",
                                               "agent_type": "const"},
                              "random_lstm": {"agent_to_eval": "out_lstm_rand.pkl",
                                              "agent_type": "lstm"}}
            results = {}
            for key, value in trained_agents.items():
                results[key] = evaluate_agent(attacked_agent, value["agent_type"], value["agent_to_eval"], policy_type, env,
                               configs.samples)

            print()
            print()
            print("{} samples were taken for each agent.".format(configs.samples))
            for key in results.keys():
                print("{} acheived {} Ties and winrates {}".format(key, results[key][0], results[key][1]))


