"""Load two agents for a given environment and perform rollouts, reporting the win-tie-loss."""

import collections
import ctypes
import functools
import glob
import logging
import os
import os.path as osp
import re
import tempfile
import warnings

from PIL import Image, ImageDraw, ImageFont
import gym
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver

from aprl.envs.multi_agent import make_dummy_vec_multi_env, make_subproc_vec_multi_env
from modelfree.common.policy_loader import load_policy
from modelfree.common.utils import TrajectoryRecorder, VideoWrapper, make_env, simulate
from modelfree.configs.multi.common import VICTIM_INDEX
from modelfree.envs.gym_compete import GymCompeteToOurs, env_name_to_canonical, game_outcome

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
    video_dir = None                    # directory to store videos in.
    video_per_episode = False           # False: single file, True: file per episode
    # If video_dir set to None, and videos set to true, videos will store in a
    # tempdir, but will be copied to Sacred run dir in either case

    seed = 0
    _ = locals()  # quieten flake8 unused variable warning
    del _


VICTIM_OPPONENT_COLORS = {
    'victim': (77, 175, 74, 255),
    'opponent': (228, 26, 28, 255),
}


POLICY_TYPE_COLORS = {
    'zoo': (128, 128, 128, 255),  # grey
    'ppo2': (255, 0, 0, 255),  # red
    'zero': (0, 0, 0, 255),  # black
    'random': (0, 0, 255, 255),  # blue
}

# Brewer Accent
PATH_COLORS = {
    '1': (127, 201, 127, 255),
    '2': (190, 174, 212, 255),
    '3': (253, 192, 134, 255),
    '4': (255, 255, 153, 255),
}


def body_color(is_victim, agent_type, agent_path):
    key = 'victim' if is_victim else 'opponent'
    return VICTIM_OPPONENT_COLORS[key]


def head_color(is_victim, agent_type, agent_path):
    return POLICY_TYPE_COLORS[agent_type]


GEOM_MAPPINGS = {
    '*': body_color,
    # Ant
    'torso_geom': head_color,
    # Humanoid
    'torso1': head_color,
    'uwaist': head_color,
    'lwaist': head_color,
    'head': head_color,
}


def set_geom_rgba(model, value):
    """Does what model.geom_rgba = ... should do, but doesn't because of a bug in mujoco-py."""
    val_ptr = np.array(value, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ctypes.memmove(model._wrapped.contents.geom_rgba, val_ptr,
                   model.ngeom * 4 * ctypes.sizeof(ctypes.c_float))


def set_geom_colors(model, patterns):
    names = [name.decode('utf-8') for name in model.geom_names]
    patterns = {re.compile(k): tuple(x / 255 for x in v) for k, v in patterns.items()}

    modified = np.array(model.geom_rgba)
    for row_idx, name in enumerate(names):
        for pattern, color in patterns.items():
            if pattern.match(name):
                modified[row_idx, :] = color

    set_geom_rgba(model, modified)


CAMERA_CONFIG = {
    # From behind
    'KickAndDefend-v0': {'azimuth': 0, 'distance': 10, 'elevation': -23},
    # From side, slightly behind (runner always goes forward, never back)
    'YouShallNotPassHumans-v0': {'azimuth': 110, 'distance': 9, 'elevation': -21},
    # Defaults fine for Sumo
    'SumoHumans-v0': {},
    'SumoAnts-v0': {},
}


class PrettyMujocoWrapper(gym.Wrapper):
    def __init__(self, env, env_name, agent_a_type, agent_a_path, agent_b_type, agent_b_path,
                 font="times", font_size=24, spacing=0.02,
                 color=(0, 0, 0, 255), color_changed=(255, 255, 255, 255)):
        super(PrettyMujocoWrapper, self).__init__(env)

        # Set agent colors
        self.env_name = env_name
        self.victim_index = VICTIM_INDEX[env_name]
        agent_mapping = {
            'agent0': (0 == self.victim_index, agent_a_type, agent_a_path),
            'agent1': (1 == self.victim_index, agent_b_type, agent_b_path)
        }
        color_patterns = {f'{agent_key}/{geom_key}': geom_fn(*agent_val)
                          for geom_key, geom_fn in GEOM_MAPPINGS.items()
                          for agent_key, agent_val in agent_mapping.items()}
        set_geom_colors(self.env.unwrapped.env_scene.model, color_patterns)

        # Text overlay
        self.font = ImageFont.truetype(f'{font}.ttf', font_size)
        self.font_bold = ImageFont.truetype(f'{font}bd.ttf', font_size)
        self.spacing = spacing
        self.color = color
        self.color_changed = color_changed

        # Internal state
        self.result = collections.defaultdict(int)
        self.changed = collections.defaultdict(int)
        self.last_won = None

        self.env.unwrapped.env_scene._get_viewer()  # force viewer to start
        self.camera_setup()

    def camera_setup(self):
        canonical_env_name = env_name_to_canonical(self.env_name)
        camera_cfg = CAMERA_CONFIG[canonical_env_name]
        viewer = self.env.unwrapped.env_scene.viewer

        for k, v in camera_cfg.items():
            setattr(viewer.cam, k, v)

    def _reset(self):
        ob = super(PrettyMujocoWrapper, self)._reset()

        if self.env.unwrapped.env_scene.viewer is not None:
            self.camera_setup()

        return ob

    def _step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done:
            # TODO: code duplication
            winner = game_outcome(info)
            if winner is None:
                k = 'ties'
            else:
                k = f'win{winner}'
            self.result[k] += 1
            self.last_won = k
        return obs, rew, done, info

    def _render(self, mode='human', close=False):
        res = self.env.render(mode, close)
        if mode == 'rgb_array':
            img = Image.fromarray(res)
            draw = ImageDraw.Draw(img)

            width, height = img.size
            ypos = height * 0.9
            texts = collections.OrderedDict([
                (f'win{1 - self.victim_index}', 'Opponent'),
                (f'win{self.victim_index}', 'Victim'),
                ('ties', 'Ties'),
            ])

            to_draw = []
            for k, label in texts.items():
                msg = f'{label} = {self.result[k]}'
                font = self.font
                color = self.color
                if k == self.last_won:
                    font = self.font_bold
                    color = self.color_changed

                to_draw.append((msg, font, color))

            lengths = [font.getsize(msg)[0] + self.spacing * width
                       for (msg, font, color) in to_draw]
            total_length = sum(lengths)
            xpos = (width - total_length) / 2

            for (msg, font, color), length in zip(to_draw, lengths):
                draw.text((xpos, ypos), msg, font=font, fill=color)
                xpos += length

            res = np.array(img)

        return res


@score_ex.main
def score_agent(_run, _seed, env_name, agent_a_path, agent_b_path, agent_a_type, agent_b_type,
                record_traj, record_traj_params, num_env, episodes, render,
                videos, video_dir, video_per_episode):
    if videos:
        assert num_env == 1, "videos requires num_env=1"
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
            env = PrettyMujocoWrapper(env, env_name, agent_a_type, agent_a_path,
                                      agent_b_type, agent_b_path)
            env = VideoWrapper(env, osp.join(video_dir, str(i)), video_per_episode)
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
