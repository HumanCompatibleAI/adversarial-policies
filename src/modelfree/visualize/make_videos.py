"""Generate videos for adversaries and standard baselines."""

import logging
import os
import os.path as osp

from sacred import Experiment
from sacred.observers import FileStorageObserver

from modelfree.common import utils
from modelfree.multi.score import extract_data, run_external

make_videos_ex = Experiment('make_videos')
make_videos_logger = logging.getLogger('make_videos')


@make_videos_ex.config
def default_config():
    adversary_path = osp.join('data', 'aws', 'score_agents',
                              '2019-04-29T14:11:08-07:00_best_adversaries.json')
    ray_upload_dir = 'data'  # where Ray will upload multi.score outputs. 'data' works on baremetal
    score_configs = ['zoo_baseline', 'fixed_baseline', 'adversary_transfer']
    multi_score = {}
    root_dir = 'data/videos'
    exp_name = 'default'
    _ = locals()  # quieten flake8 unused variable warning
    del _


@make_videos_ex.named_config
def debug_config():
    score_configs = ['debug_one_each_type']
    config_updates = {'score': {'episodes': 2}}
    exp_name = 'debug'
    _ = locals()  # quieten flake8 unused variable warning
    del _


@make_videos_ex.capture
def generate_videos(score_configs, multi_score, adversary_path):
    """Uses multi.score to generate videos."""
    return run_external(score_configs, post_named_configs=['video'],
                        config_updates=multi_score, adversary_path=adversary_path)


@make_videos_ex.capture
def extract_videos(out_dir, video_dirs, ray_upload_dir):
    def path_generator(trial_root, env_name, victim_index, victim_type, victim_path,
                       opponent_type, opponent_path):
        src_path = osp.join(trial_root, 'data', 'sacred', 'score', '1',
                            'videos', 'env_0_episode_0_recording.mp4')
        new_name = (f'{env_name}_victim_{victim_type}_{victim_path}'
                    f'_opponent_{opponent_type}_{opponent_path}')
        return src_path, new_name, 'mp4'

    return extract_data(path_generator, out_dir, video_dirs, ray_upload_dir)


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
