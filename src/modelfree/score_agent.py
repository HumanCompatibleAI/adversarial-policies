"""Load two agents for a given environment and perform rollouts, reporting the win-tie-loss."""

import gym
from sacred import Experiment
from sacred.observers import FileStorageObserver
import tensorflow as tf

from modelfree.policy_loader import get_agent_any_type
from modelfree.simulation_utils import simulate
from modelfree.utils import VideoWrapper, make_session


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


def get_empirical_score(_run, env, agents, episodes, render=False):
    result = {
        "ties": 0,
        "wincounts": [0] * len(agents)
    }

    # This tells sacred about the intermediate computation so it
    # updates the result as the experiment is running
    _run.result = result

    for i in range(episodes):
        winner = announce_winner(simulate(env, agents, render=render))
        if winner is None:
            result["ties"] += 1
        else:
            result["wincounts"][winner] += 1
        for agent in agents:
            agent.reset()

    return result


score_agent_ex = Experiment('score_agent')
score_agent_ex.observers.append(FileStorageObserver.create("data/sacred"))


@score_agent_ex.config
def default_score_config():
    agent_a_type = "zoo"
    agent_a_path = "1"
    env_name = "multicomp/SumoAnts-v0"
    agent_b_type = "zoo"
    agent_b_path = "2"
    samples = 5
    render = True
    videos = False
    video_dir = "videos/"
    _ = locals()  # quieten flake8 unused variable warning
    del _


@score_agent_ex.automain
def score_agent(_run, env_name, agent_a_path, agent_b_path, agent_a_type, agent_b_type,
                samples, render, videos, video_dir):
    env = gym.make(env_name)
    if videos:
        env = VideoWrapper(env, video_dir)

    agent_paths = [agent_a_path, agent_b_path]
    agent_types = [agent_a_type, agent_b_type]
    graphs = [tf.Graph() for _ in agent_paths]
    sessions = [make_session(graph) for graph in graphs]
    with sessions[0], sessions[1]:
        zipped = zip(agent_paths, agent_types, sessions)
        agents = [get_agent_any_type(agent, agent_type, env, env_name, i, sess=sess)
                  for i, (agent, agent_type, sess) in enumerate(zipped)]
        return get_empirical_score(_run, env, agents, samples, render=render)
