"""Generate videos for adversaries and standard baselines."""

import json
import logging
import os
import os.path as osp
import shutil

from sacred import Experiment
from sacred.observers import FileStorageObserver

from modelfree.common import utils
from modelfree.configs.multi.common import VICTIM_INDEX
from modelfree.envs.gym_compete import env_name_to_canonical
from modelfree.multi.score import multi_score_ex

make_videos_ex = Experiment('make_videos', ingredients=[multi_score_ex])
make_videos_logger = logging.getLogger('make_videos')


@make_videos_ex.config
def default_config():
    adversary_path = osp.join('data', 'score_agents',
                              '2019-04-29T14:11:08-07:00_best_adversaries.json')
    ray_upload_dir = 'data'  # where Ray will upload multi.score outputs. 'data' works on baremetal
    score_configs = ['zoo_baseline', 'fixed_baseline', 'adversary_transfer']
    root_dir = 'data/videos'
    exp_name = 'default'
    _ = locals()  # quieten flake8 unused variable warning
    del _


@make_videos_ex.named_config
def debug_config():
    score_configs = ['debug']  # noqa: F841


@make_videos_ex.capture
def generate_videos(adversary_path, score_configs):
    # Sad workaround for Sacred config limitation,
    # see modelfree.configs.multi.score:_get_adversary_paths
    os.putenv('ADVERSARY_PATHS', adversary_path)

    video_dirs = {}
    for config in score_configs:
        run = multi_score_ex.run(named_configs=[config, 'video'])
        exp_id = run.result['exp_id']
        video_dirs[config] = exp_id

    return video_dirs


@make_videos_ex.capture
def extract_videos(out_dir, video_dirs, ray_upload_dir):
    for experiment, video_dir in video_dirs.items():
        experiment_root = osp.join(ray_upload_dir, video_dir)
        # video_root contains one directory for each score_agent trial.
        # These directories have names of form score-<hash>_<id_num>_<k=v>...
        for trial_name in os.listdir(experiment_root):
            # Each trial contains the Sacred output from score_agent.
            sacred_root = osp.join(experiment_root, trial_name, 'data', 'sacred', 'score', '1')

            with open(osp.join(sacred_root, 'config.json'), 'r') as f:
                cfg = json.load(f)

            def agent_key(agent):
                return cfg[agent + '_type'] + '_' + cfg[agent + '_path']

            env_name = cfg['env_name']
            if VICTIM_INDEX[env_name] == 0:
                victim_key = agent_key('agent_a')
                opponent_key = agent_key('agent_b')
            else:
                victim_key = agent_key('agent_b')
                opponent_key = agent_key('agent_a')

            if 'multicomp' in cfg['env_name']:
                env_name = env_name_to_canonical(env_name)
            env_name = env_name.replace('/', '-')  # sanitize

            src_path = osp.join(sacred_root, 'videos', 'env_0_episode_0_recording.mp4')
            new_name = f'{env_name}_victim_{victim_key}_opponent_{opponent_key}.mp4'
            dst_path = osp.join(out_dir, new_name)
            shutil.copy(src_path, dst_path)


@make_videos_ex.main
def make_videos(root_dir, exp_name):
    out_dir = osp.join(root_dir, exp_name, utils.make_timestamp())
    os.makedirs(out_dir)
    video_dirs = generate_videos()
    extract_videos(out_dir=out_dir, video_dirs=video_dirs)


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'make_videos'))
    make_videos_ex.observers.append(observer)
    make_videos_ex.run_commandline()
    make_videos_logger.info("Sacred run completed, files stored at {}".format(observer.dir))


if __name__ == '__main__':
    main()
