"""Hyperparameter search for train.py using Ray Tune."""

import json
import logging
import math
import os
import os.path as osp
import shutil

from ray import tune
from sacred import Experiment
from sacred.observers import FileStorageObserver

from modelfree.configs.multi.score import make_configs
from modelfree.envs import VICTIM_INDEX
from modelfree.envs.gym_compete import env_name_to_canonical
from modelfree.multi import common
from modelfree.multi.score_worker import score_worker
from modelfree.score_agent import score_ex

multi_score_ex = Experiment('multi_score', ingredients=[score_ex])
pylog = logging.getLogger('modelfree.multi_score')

# Load common configs (e.g. upload directories) and define the run command
run = common.make_sacred(multi_score_ex, 'score', score_worker)

# Load named configs for individual experiments (these change a lot, so keep out of this file)
make_configs(multi_score_ex)


@multi_score_ex.config
def default_config(score):
    spec = {  # experiment specification
        'resources_per_trial': {'cpu': math.ceil(score['num_env'] / 2)},
        'config': {},
    }

    save_path = None      # path to save JSON results. If None, do not save.
    exp_name = 'default'  # experiment name

    _ = locals()  # quieten flake8 unused variable warning
    del _


@score_ex.config
def score_config():
    render = False
    videos = False

    _ = locals()  # quieten flake8 unused variable warning
    del _


@multi_score_ex.named_config
def debug_config(score):
    """Try zero-agent and random-agent against pre-trained zoo policies."""
    score = dict(score)
    score['episodes'] = 1
    score['agent_a_type'] = 'zoo'
    score['agent_b_type'] = 'zoo'
    spec = {
        'config': {
            'agent_a_path': tune.grid_search(['1', '2']),
        }
    }
    exp_name = 'debug'
    _ = locals()  # quieten flake8 unused variable warning
    del _


def _remap_keys(d):
    return [{'k': k, 'v': v} for k, v in d.items()]


@multi_score_ex.main
def multi_score(score, save_path):
    f = None
    try:
        if save_path is not None:
            f = open(save_path, 'w')  # open it now so we fail fast if file is unwriteable

        trials, exp_id = run(base_config=score)
        additional_index_keys = score.get('index_keys', [])
        results = {}
        for trial in trials:
            idx = trial.last_result['idx']
            cols = ['env_name', 'agent_a_type', 'agent_a_path', 'agent_b_type', 'agent_b_path']
            cols += additional_index_keys
            key = tuple(idx[col] for col in cols)
            results[key] = trial.last_result['score']

        if f is not None:
            json.dump(_remap_keys(results), f)
    finally:
        if f is not None:
            f.close()

    return {'scores': results, 'exp_id': exp_id}


def run_external(named_configs, post_named_configs, config_updates, adversary_path=None):
    """Run multiple multi_score experiments. Intended for use by external scripts,
       not accessible from commandline.

       :param named_configs: (list<str>) list of named configs, executed one by one
       :param post_named_configs: (list<str>) list of base named configs, applied after the
                                              current config from `named_configs`.
       :param config_updates: (dict) a dict of config options, overriding the named config.
       :param adversary_path: (str or None) path to JSON, needed by adversary_transfer config.
       :return (dict) mapping from named configs to their output directory
    """
    # Sad workaround for Sacred config limitation,
    # see modelfree.configs.multi.score:_get_adversary_paths
    os.environ['ADVERSARY_PATHS'] = adversary_path

    output_dir = {}
    for trial_configs in named_configs:
        configs = list(trial_configs) + list(post_named_configs)
        run = multi_score_ex.run(named_configs=configs,
                                 config_updates=config_updates)
        assert run.status == 'COMPLETED'
        exp_id = run.result['exp_id']
        output_dir[tuple(trial_configs)] = exp_id

    return output_dir


def extract_data(path_generator, out_dir, experiment_dirs, ray_upload_dir):
    """Helper method to extract data from multiple_score experiments."""
    for experiment, experiment_dir in experiment_dirs.items():
        experiment_root = osp.join(ray_upload_dir, experiment_dir)
        # video_root contains one directory for each score_agent trial.
        # These directories have names of form score-<hash>_<id_num>_<k=v>...
        for trial_name in os.listdir(experiment_root):
            # Each trial contains the Sacred output from score_agent.
            # Note Ray Tune is running with a fresh working directory per trial, so Sacred
            # output will always be at score/1.
            trial_root = osp.join(experiment_root, trial_name)

            sacred_config = osp.join(trial_root, 'data', 'sacred', 'score', '1', 'config.json')
            with open(sacred_config, 'r') as f:
                cfg = json.load(f)

            def agent_key(agent):
                return cfg[agent + '_type'], cfg[agent + '_path']

            env_name = cfg['env_name']
            victim_index = VICTIM_INDEX[env_name]
            if victim_index == 0:
                victim_type, victim_path = agent_key('agent_a')
                opponent_type, opponent_path = agent_key('agent_b')
            else:
                victim_type, victim_path = agent_key('agent_b')
                opponent_type, opponent_path = agent_key('agent_a')

            if 'multicomp' in cfg['env_name']:
                env_name = env_name_to_canonical(env_name)
            env_name = env_name.replace('/', '-')  # sanitize

            if opponent_path.startswith('/'):  # is path name
                opponent_root = osp.sep.join(opponent_path.split(osp.sep)[:-3])
                opponent_sacred = osp.join(opponent_root, 'sacred', 'train', '1', 'config.json')

                with open(opponent_sacred, 'r') as f:
                    opponent_cfg = json.load(f)

                opponent_path = opponent_cfg['victim_path']

            src_path, new_name, suffix = path_generator(trial_root=trial_root,
                                                        cfg=cfg,
                                                        env_name=env_name,
                                                        victim_index=victim_index,
                                                        victim_type=victim_type,
                                                        victim_path=victim_path,
                                                        opponent_type=opponent_type,
                                                        opponent_path=opponent_path)
            dst_path = osp.join(out_dir, f'{new_name}.{suffix}')
            shutil.copy(src_path, dst_path)
            dst_config = osp.join(out_dir, f'{new_name}_sacred.json')
            shutil.copy(sacred_config, dst_config)


def main():
    observer = FileStorageObserver(osp.join('data', 'sacred', 'multi_score'))
    multi_score_ex.observers.append(observer)
    multi_score_ex.run_commandline()


if __name__ == '__main__':
    main()
