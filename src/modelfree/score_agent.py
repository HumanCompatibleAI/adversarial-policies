"""Load two agents for a given environment and perform rollouts, reporting the win-tie-loss."""

import functools
import os.path as osp

from sacred import Experiment
from sacred.observers import FileStorageObserver

from aprl.envs.multi_agent import make_dummy_vec_multi_env, make_subproc_vec_multi_env
from modelfree.common.policy_loader import load_policy
from modelfree.common.utils import TrajectoryRecorder, VideoWrapper, make_env, simulate
from modelfree.envs.gym_compete import GymCompeteToOurs, game_outcome

score_ex = Experiment('score')


def announce_winner(sim_stream):
    """This function determines the winner of a match in one of the gym_compete environments.
    :param sim_stream: a stream of obs, rewards, dones, infos from one of the gym_compete envs.
    :return: the index of the winning player, or None if it was a tie."""
    for _, _, dones, infos in sim_stream:
        for done, info in zip(dones, infos):
            if done:
                yield game_outcome(info)


def get_empirical_score(_run, env, agents, episodes, render=False):
    """Computes number of wins for each agent and ties.

    :param env: (gym.Env) environment
    :param agents: (list<BaseModel>) agents/policies to execute.
    :param episodes: (int) number of episodes.
    :param render: (bool) whether to render to screen during simulation.
    :return a dictionary mapping from 'winN' to wins for each agent N, and 'ties' for ties."""
    result = {f'win{i}': 0 for i in range(len(agents))}
    result['ties'] = 0

    # This tells sacred about the intermediate computation so it
    # updates the result as the experiment is running
    _run.result = result
    sim_stream = simulate(env, agents, render=render)
    for ep, winner in enumerate(announce_winner(sim_stream)):
        if winner is None:
            result['ties'] += 1
        else:
            result[f'win{winner}'] += 1
        if ep + 1 >= episodes:
            break

    return result


@score_ex.config
def default_score_config():
    env_name = 'multicomp/SumoAnts-v0'  # Gym env ID
    agent_a_type = 'zoo'                # type supported by policy_loader.py
    agent_a_path = '1'                  # path or other unique identifier
    agent_b_type = 'zoo'                # type supported by policy_loader.py
    agent_b_path = '2'                  # path or other unique identifier
    record_traj = False                 # whether to record trajectories
    record_traj_params = {              # parameters for recording trajectories
        'save_dir': 'data/experts',     # directory to save trajectories to
        'agent_indices': None,          # which agent trajectories to save
    }
    num_env = 1                         # number of environments to run in parallel
    episodes = 20                       # number of episodes to evaluate
    render = True                       # display on screen (warning: slow)
    videos = False                      # generate videos
    video_dir = 'data/videos/'          # video directory
    seed = 0
    _ = locals()  # quieten flake8 unused variable warning
    del _


@score_ex.main
def score_agent(_run, _seed, env_name, agent_a_path, agent_b_path, agent_a_type, agent_b_type,
                record_traj, record_traj_params, num_env, episodes, render, videos, video_dir):
    pre_wrapper = GymCompeteToOurs if 'multicomp' in env_name else None

    def env_fn(i):
        env = make_env(env_name, _seed, i, None, pre_wrapper=pre_wrapper)
        if videos:
            env = VideoWrapper(env, osp.join(video_dir, str(i)))
        return env
    env_fns = [functools.partial(env_fn, i) for i in range(num_env)]
    if num_env > 1:
        venv = make_subproc_vec_multi_env(env_fns)
    else:
        venv = make_dummy_vec_multi_env(env_fns)

    if record_traj:
        venv = TrajectoryRecorder(venv, record_traj_params['agent_indices'])

    if venv.num_agents == 1 and agent_b_path != 'none':
        raise ValueError("Set agent_b_path to 'none' if environment only uses one agent.")

    agent_paths = [agent_a_path, agent_b_path]
    agent_types = [agent_a_type, agent_b_type]
    zipped = list(zip(agent_types, agent_paths))

    agents = [load_policy(policy_type, policy_path, venv, env_name, i)
              for i, (policy_type, policy_path) in enumerate(zipped[:venv.num_agents])]
    score = get_empirical_score(_run, venv, agents, episodes, render=render)

    if record_traj:
        venv.save(save_dir=record_traj_params['save_dir'])

    for agent in agents:
        if agent.sess is not None:
            agent.sess.close()
    venv.close()

    return score


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'score'))
    score_ex.observers.append(observer)
    score_ex.run_commandline()


if __name__ == '__main__':
    main()
