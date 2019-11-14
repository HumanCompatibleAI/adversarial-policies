"""Generate videos for adversaries and standard baselines."""

import logging
import os
import os.path as osp

from sacred import Experiment
from sacred.observers import FileStorageObserver

from aprl.common.utils import make_timestamp
from aprl.configs import DATA_LOCATION
from aprl.multi.score import extract_data, run_external
from aprl.visualize import util

make_videos_ex = Experiment('make_videos')
make_videos_logger = logging.getLogger('make_videos')


BASIC_CONFIGS = ['adversary_transfer', 'zoo_baseline', 'fixed_baseline']


@make_videos_ex.config
def default_config():
    adversary_path = osp.join(DATA_LOCATION, 'multi_train', 'paper',
                              'highest_win_policies_and_rates.json')
    ray_upload_dir = 'data'  # where Ray will upload multi.score outputs. 'data' works on baremetal
    score_configs = [(x, ) for x in BASIC_CONFIGS]
    score_configs += [(x, 'mask_observations_of_victim') for x in BASIC_CONFIGS]
    multi_score = {}
    root_dir = 'data/videos'
    exp_name = 'default'
    _ = locals()  # quieten flake8 unused variable warning
    del _


@make_videos_ex.named_config
def slides_config():
    """Generate a subset of videos, with tighter-cropped camera.
       Intended for slideshows/demos."""
    score_configs = [('summary', ), ('summary', 'mask_observations_of_victim')]
    multi_score = {
        'score': {
            'video_params': {
                'annotation_params': {
                    'camera_config': 'close',
                    'short_labels': True,
                }
            }
        }
    }
    exp_name = 'slides'
    _ = locals()  # quieten flake8 unused variable warning
    del _


LOW_RES = {
    'score': {
        'video_params': {
            'annotation_params': {
                'resolution': (640, 480),
                'font_size': 24,
            }
        }
    }
}


@make_videos_ex.named_config
def low_res():
    multi_score = LOW_RES  # noqa: F841


@make_videos_ex.named_config
def debug_config():
    score_configs = [('debug_one_each_type', ),
                     ('debug_one_each_type', 'mask_observations_of_victim')]
    multi_score = dict(LOW_RES)
    multi_score['score']['episodes'] = 2
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
    def path_generator(trial_root, env_sanitized, victim_index, victim_type, victim_path,
                       opponent_type, opponent_path, cfg):
        src_path = osp.join(trial_root, 'data', 'sacred', 'score', '1',
                            'videos', 'env_0_episode_0_recording.mp4')

        victim_suffix = ''
        opponent_suffix = ''
        mask_index = cfg['mask_agent_index']
        if mask_index is not None:
            if mask_index == victim_index:
                victim_suffix = 'M'
            else:
                opponent_suffix == 'M'

        victim = util.abbreviate_agent_config(cfg['env_name'], victim_type, victim_path,
                                              victim_suffix, victim=True)
        opponent = util.abbreviate_agent_config(cfg['env_name'], opponent_type, opponent_path,
                                                opponent_suffix, victim=False)

        new_name = f'{env_sanitized}_victim_{victim}_opponent_{opponent}'
        return src_path, new_name, 'mp4'

    return extract_data(path_generator, out_dir, video_dirs, ray_upload_dir)


@make_videos_ex.main
def make_videos(root_dir, exp_name):
    out_dir = osp.join(root_dir, exp_name, make_timestamp())
    os.makedirs(out_dir)

    video_dirs = generate_videos()
    extract_videos(out_dir=out_dir, video_dirs=video_dirs)


def main():
    observer = FileStorageObserver(osp.join('data', 'sacred', 'make_videos'))
    make_videos_ex.observers.append(observer)
    make_videos_ex.run_commandline()
    make_videos_logger.info("Sacred run completed, files stored at {}".format(observer.dir))


if __name__ == '__main__':
    main()
