"""Load two agents for a given environment and perform rollouts, reporting the win-tie-loss."""

import functools
import glob
import logging
import os
import os.path as osp
import pdb
import re
import tempfile
import warnings

from sacred import Experiment
from sacred.observers import FileStorageObserver

from aprl.envs.multi_agent import make_dummy_vec_multi_env, make_subproc_vec_multi_env
from modelfree.common.policy_loader import load_policy
from modelfree.common.utils import TrajectoryRecorder, VideoWrapper, make_env, simulate
from modelfree.envs.gym_compete import GymCompeteToOurs, game_outcome
from modelfree.training.shaping_wrappers import NoisyAgentWrapper

score_ex = Experiment('score')
score_ex_logger = logging.getLogger('score_agent')


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


def _clean_video_directory_structure(observer_obj):
    """
    A simple utility method to take saved videos within a Sacred run structure and clean
    up the file pathways, so that all videos are organized under a "videos" directory

    :param observer_obj: A Sacred FileStorageObserver object
    :return: None
    """
    basedir = observer_obj.dir
    video_files = glob.glob("{}/*.mp4".format(basedir))
    metadata_files = glob.glob("{}/*metadata.json".format(basedir))
    if len(video_files) == 0:
        return

    new_video_dir = os.path.join(basedir, "videos")
    os.mkdir(new_video_dir)
    new_video_metadata_dir = os.path.join(new_video_dir, "metadata")
    os.mkdir(new_video_metadata_dir)
    for video_file in video_files:
        base_file_name = os.path.basename(video_file)
        os.rename(video_file, os.path.join(new_video_dir, base_file_name))

    for metadata_file in metadata_files:
        base_file_name = os.path.basename(metadata_file)
        os.rename(metadata_file, os.path.join(new_video_metadata_dir, base_file_name))


def _save_video_or_metadata(env_dir, saved_video_path):
    """
    A helper method to pull the logic for pattern matching certain kinds of video and metadata
    files and storing them as sacred artifacts with clearer names

    :param env_dir: The path to a per-environment folder where videos are stored
    :param saved_video_path: The video file to be reformatted and saved as a sacred artifact
    :return: None
    """
    env_number = env_dir.split("/")[-1]
    video_ptn = re.compile(r'video.(\d*).mp4')
    metadata_ptn = re.compile(r'video.(\d*).meta.json')
    video_search_result = video_ptn.match(saved_video_path)
    metadata_search_result = metadata_ptn.match(saved_video_path)

    if video_search_result is not None:
        episode_id = video_search_result.groups()[0]
        sacred_name = "env_{}_episode_{}_recording.mp4".format(env_number, int(episode_id))
    elif metadata_search_result is not None:
        episode_id = metadata_search_result.groups()[0]
        sacred_name = "env_{}_episode_{}_metadata.json".format(env_number, int(episode_id))
    else:
        return

    score_ex.add_artifact(filename=os.path.join(env_dir, saved_video_path),
                          name=sacred_name)


@score_ex.config
def default_score_config():
    env_name = 'multicomp/SumoHumans-v0'  # Gym env ID
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
    video_dir = None                    # directory to store videos in.
    adversary_action_noise = None      # Size of noise ball to add to adversary's actions
    # If video_dir set to None, and videos set to true, videos will store in a
    # tempdir, but will be copied to Sacred run dir in either case

    seed = 0
    _ = locals()  # quieten flake8 unused variable warning
    del _


@score_ex.main
def score_agent(_run, _seed, env_name, agent_a_path, agent_b_path, agent_a_type, agent_b_type,
                record_traj, record_traj_params, num_env, episodes, render, videos, video_dir,
                adversary_action_noise):
    if videos:
        if video_dir is None:
            score_ex_logger.info("No directory provided for saving videos; using a tmpdir instead,"
                                 "but videos will be saved to Sacred run directory")
            tmp_dir = tempfile.TemporaryDirectory()
            video_dir = tmp_dir.name
        else:
            tmp_dir = None
        video_dirs = [osp.join(video_dir, str(i)) for i in range(num_env)]
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

    if adversary_action_noise is not None:
        agents[1] = NoisyAgentWrapper(agents[1], noise_annealer=lambda: adversary_action_noise)

    score = get_empirical_score(_run, venv, agents, episodes, render=render)

    if record_traj:
        venv.save(save_dir=record_traj_params['save_dir'])

    if videos:
        for env_video_dir in video_dirs:
            try:
                for file_path in os.listdir(env_video_dir):
                    _save_video_or_metadata(env_video_dir, file_path)

            except FileNotFoundError:
                warnings.warn("Can't find path {}; no videos from that path added as artifacts"
                              .format(env_video_dir))

        if tmp_dir is not None:
            tmp_dir.cleanup()

    for observer in score_ex.observers:
        if hasattr(observer, 'dir'):
            _clean_video_directory_structure(observer)

    for agent in agents:
        if agent.sess is not None:
            agent.sess.close()

    venv.close()
    return score


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'score'))
    score_ex.observers.append(observer)
    score_ex.run_commandline()
    score_ex_logger.info("Sacred run completed, files stored at {}".format(observer.dir))


if __name__ == '__main__':
    main()
