
import argparse
import numpy as np
import pickle
import os
import os.path as osp
import functools

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.ppo2 import ppo2
import gym
from gym.core import Wrapper
from gym.monitoring.video_recorder import VideoRecorder
import tensorflow as tf

import utils
from random_search import constant_agent_sampler
from rl_baseline import StatefulModel
from simulation_utils import simulate
from utils import load_agent, LSTMPolicy, Agent, Gymify, MultiToSingle, CurryEnv
from utils import get_env_and_policy_type, get_trained_sumo_ant_locations, make_session, get_trained_kicker_locations

from gather_statistics import get_emperical_score, get_agent_any_type


samples = 20

env, pol_type = get_env_and_policy_type('kick-and-defend')


sess = make_session()
with sess:
    attacked_agent = load_agent(get_trained_kicker_locations()[1], pol_type, "attacked", env, 0)

    known_agent = load_agent('agent-zoo/kick-and-defend/defender/agent2_parameters-v1.pkl', 'lstm',
                                              "known_policy", env, 0)

    #TODO Load Agent should be changed to "load_zoo_agent"

    trained_agent = get_agent_any_type('our_mlp', 'rando-ralph', pol_type, env)

    agents = [attacked_agent, known_agent]
    ties, win_loss = get_emperical_score(env, agents, samples, render=True, silent=True)

    print("[MAGIC NUMBER 87623123] In {} trials {} acheived {} Ties and winrates {}".format(samples, 'known agent got',
                                                                                            ties, win_loss))
    agents = [attacked_agent, trained_agent]
    ties, win_loss = get_emperical_score(agents, samples, render=True, silent=True)

    print("[MAGIC NUMBER 87623123] In {} trials {} acheived {} Ties and winrates {}".format(samples, 'rando ralph got',
                                                                                            ties, win_loss))
