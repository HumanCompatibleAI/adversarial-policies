"""Load two agents for a given environment and perform rollouts, reporting the win-tie-loss."""

import os.path as osp

from sacred import Experiment
from sacred.observers import FileStorageObserver
import tensorflow as tf

from aprl.envs.multi_agent import make_dummy_vec_multi_env
from modelfree.gym_compete_conversion import make_gym_compete_env
from modelfree.policy_loader import get_agent_any_type
from modelfree.utils import VideoWrapper, make_session, simulate


def announce_winner(sim_stream):
    """This function determines the winner of a match in one of the gym_compete environments.
    :param sim_stream: a stream of obs, rewards, dones, infos from one of the gym_compete envs.
    :return: the index of the winning player, or None if it was a tie."""
    for _, _, dones, infos in sim_stream:
        for done, info in zip(dones, infos):
            if done:
                draw = True
                for i, agent_info in info.items():
                    if 'winner' in agent_info:
                        yield i
                if draw:
                    yield None


def get_empirical_score(_run, env, agents, episodes, render=False):
    result = {
        'ties': 0,
        'wincounts': [0] * len(agents)
    }

    # This tells sacred about the intermediate computation so it
    # updates the result as the experiment is running
    _run.result = result

    for ep, winner in enumerate(announce_winner(simulate(env, agents, render=render))):
        print(winner)
        if winner is None:
            result['ties'] += 1
        else:
            result['wincounts'][winner] += 1
        if ep + 1 >= episodes:
            break

    return result


score_agent_ex = Experiment('score_agent')
score_agent_ex.observers.append(FileStorageObserver.create('data/sacred'))


@score_agent_ex.config
def default_score_config():
    env_name = 'multicomp/SumoAnts-v0'  # Gym env ID
    agent_a_type = 'zoo'   # type supported by policy_loader.py
    agent_a_path = '1'     # path or other unique identifier
    agent_b_type = 'zoo'   # type supported by policy_loader.py
    agent_b_path = '2'     # path or other unique identifier
    vectorize = 1          # number of environments to run in parallel
    episodes = 1           # number of episodes to evaluate
    render = True          # display on screen (warning: slow)
    videos = False         # generate videos
    video_dir = 'videos/'  # video directory
    _ = locals()  # quieten flake8 unused variable warning
    del _


@score_agent_ex.automain
def score_agent(_run, _seed, env_name, agent_a_path, agent_b_path, agent_a_type, agent_b_type,
                vectorize, episodes, render, videos, video_dir):
    def make_env(i):
        env = make_gym_compete_env(env_name, _seed, i, None)
        if videos:
            env = VideoWrapper(env, osp.join(video_dir, str(i)))
        return env
    env_fns = [lambda: make_env(i) for i in range(vectorize)]
    # WARNING: Be careful changing this to subproc!
    # On XDummy, rendering in a subprocess when you have already rendered in the parent causes
    # things to block indefinitely. This does not seem to be an issue on native X servers,
    # but will break our tests and remote rendering.
    venv = make_dummy_vec_multi_env(env_fns)

    agent_paths = [agent_a_path, agent_b_path]
    agent_types = [agent_a_type, agent_b_type]
    graphs = [tf.Graph() for _ in agent_paths]
    sessions = [make_session(graph) for graph in graphs]
    with sessions[0], sessions[1]:
        zipped = zip(agent_paths, agent_types, sessions)
        agents = [get_agent_any_type(agent, agent_type, venv, env_name, i, sess=sess)
                  for i, (agent, agent_type, sess) in enumerate(zipped)]
        score = get_empirical_score(_run, venv, agents, episodes, render=render)
    venv.close()

    return score
