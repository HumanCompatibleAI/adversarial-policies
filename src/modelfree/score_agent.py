"""Load two agents for a given environment and perform rollouts, reporting the win-tie-loss."""

import os.path as osp

from sacred import Experiment
from sacred.observers import FileStorageObserver

from aprl.envs.multi_agent import make_dummy_vec_multi_env, make_subproc_vec_multi_env
from modelfree.gym_compete_conversion import GymCompeteToOurs, game_outcome
from modelfree.policy_loader import load_policy
from modelfree.utils import VideoWrapper, make_env, simulate


def announce_winner(sim_stream):
    """This function determines the winner of a match in one of the gym_compete environments.
    :param sim_stream: a stream of obs, rewards, dones, infos from one of the gym_compete envs.
    :return: the index of the winning player, or None if it was a tie."""
    for _, _, dones, infos in sim_stream:
        for done, info in zip(dones, infos):
            if done:
                yield game_outcome(info)


def get_empirical_score(_run, env, agents, episodes, render=False, record_trajectories=False):
    result = {
        'ties': 0,
        'wincounts': [0] * len(agents)
    }

    # This tells sacred about the intermediate computation so it
    # updates the result as the experiment is running
    _run.result = result
    sim_stream = simulate(env, agents, render=render, record_trajectories=record_trajectories,
                          num_trajectories_to_save=episodes)
    for ep, winner in enumerate(announce_winner(sim_stream)):
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
    record_trajectories = False  # record trajectories in stable_baselines.GAIL format
    num_env = 1            # number of environments to run in parallel
    episodes = 20           # number of episodes to evaluate
    render = True          # display on screen (warning: slow)
    videos = False         # generate videos
    video_dir = 'videos/'  # video directory
    seed = 0
    _ = locals()  # quieten flake8 unused variable warning
    del _


@score_agent_ex.automain
def score_agent(_run, _seed, env_name, agent_a_path, agent_b_path, agent_a_type, agent_b_type,
                record_trajectories, num_env, episodes, render, videos, video_dir):
    pre_wrapper = GymCompeteToOurs if 'multicomp' in env_name else None

    def env_fn(i):
        env = make_env(env_name, _seed, i, None, pre_wrapper=pre_wrapper)
        if videos:
            env = VideoWrapper(env, osp.join(video_dir, str(i)))
        return env
    env_fns = [lambda: env_fn(i) for i in range(num_env)]
    if num_env > 1:
        venv = make_subproc_vec_multi_env(env_fns)
    else:
        venv = make_dummy_vec_multi_env(env_fns)

    if venv.num_agents == 1 and agent_b_path != 'none':
        raise ValueError("Set agent_b_path to 'none' if environment only uses one agent.")

    agent_paths = [agent_a_path, agent_b_path]
    agent_types = [agent_a_type, agent_b_type]
    zipped = list(zip(agent_types, agent_paths))

    agents = [load_policy(policy_type, policy_path, venv, env_name, i)
              for i, (policy_type, policy_path) in enumerate(zipped[:venv.num_agents])]
    score = get_empirical_score(_run, venv, agents, episodes, render=render,
                                record_trajectories=record_trajectories)
    for agent in agents:
        if agent.sess is not None:
            agent.sess.close()
    venv.close()

    return score
