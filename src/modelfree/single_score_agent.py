import gym
from sacred import Experiment
from sacred.observers import FileStorageObserver
import tensorflow as tf
from modelfree.simulation_utils import simulate_single
from modelfree.score_agent import get_agent_any_type
from modelfree.gym_compete_conversion import announce_winner, load_zoo_agent
from modelfree.ppo_baseline import load_our_mlp
from modelfree.simulation_utils import simulate
from modelfree.utils import VideoWrapper, make_session


single_score_agent_ex = Experiment('single_score_agent')
single_score_agent_ex.observers.append(FileStorageObserver.create('my_runs'))

def get_mean_score(_run, env, agent, samples, render):
    all_rewards = []
    for _ in range(samples):
        sim_r = 0
        for o, r, d, i in simulate_single(env, agent, render=render):
            sim_r += r
        all_rewards.append(sim_r)

    return sum(all_rewards) / float(samples)

@single_score_agent_ex.named_config
def human_score_config():
    agent_a_path = "1"
    agent_a_type = "zoo"
    env_name = "SumoHumans-v0"
    samples = 20
    render = True
    videos = False
    video_dir = "videos/"
    _ = locals()  # quieten flake8 unused variable warning
    del _


@single_score_agent_ex.config
def default_score_config():
    agent_a_type = "zoo"
    agent_a_path = "1"
    env_name = "multicomp/SumoAnts-v0"
    samples = 5
    render = True
    videos = False
    video_dir = "videos/"
    _ = locals()  # quieten flake8 unused variable warning
    del _


@single_score_agent_ex.automain
def score_agent(_run, env_name, agent_a_path, agent_a_type,
                samples, render, videos, video_dir):
    env = gym.make(env_name)
    if videos:
        env = VideoWrapper(env, video_dir)

    graph = tf.Graph()
    session = make_session(graph)
    with session:
        agent = get_agent_any_type(agent_a_path, agent_a_type, env, env_name, 0, sess=session)
        return get_mean_score(_run, env, agent, samples, render=render)