"""Load two agents for a given environment and perform rollouts, reporting the win-tie-loss."""

import gym
from sacred import Experiment
from sacred.observers import FileStorageObserver
import tensorflow as tf

from modelfree.gym_compete_conversion import announce_winner, load_zoo_agent
from modelfree.ppo_baseline import load_our_mlp
from modelfree.simulation_utils import simulate
from modelfree.utils import VideoWrapper, make_session


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


def get_agent_any_type(agent, agent_type, env, env_name, index, sess=None):
    agent_loaders = {
        "zoo": load_zoo_agent,
        "our_mlp": load_our_mlp
    }
    return agent_loaders[agent_type](agent, env, env_name, index, sess=sess)


score_agent_ex = Experiment('score_agent')
score_agent_ex.observers.append(FileStorageObserver.create('my_runs'))


@score_agent_ex.named_config
def human_score_config():
    agent_a_path = "1"
    agent_a_type = "zoo"
    env_name = "SumoHumans-v0"
    agent_b_type = "our_mlp"
    agent_b_path = "outs/20190208_144744 test-experiments/model.pkl"
    samples = 20
    render = True
    videos = False
    video_dir = "videos/"
    _ = locals()  # quieten flake8 unused variable warning
    del _

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
