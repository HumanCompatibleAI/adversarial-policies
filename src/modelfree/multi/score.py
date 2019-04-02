"""Hyperparameter search for train.py using Ray Tune."""

import json
import logging
import os.path as osp

from ray import tune
from sacred import Experiment
from sacred.observers import FileStorageObserver

from modelfree.configs.multi.score import make_configs
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
        'run': 'score',
        # TODO: tune number of actual CPUs required
        'resources_per_trial': {'cpu': score['num_env'] // 2},
    }

    save_path = None      # path to save JSON results. If None, do not save.

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

        trials = run(base_config=score)
        results = {}
        for trial in trials:
            idx = trial.last_result['idx']
            cols = ['env_name', 'agent_a_type', 'agent_a_path', 'agent_b_type', 'agent_b_path']
            key = tuple(idx[col] for col in cols)
            results[key] = trial.last_result['score']

        if f is not None:
            json.dump(_remap_keys(results), f)
    finally:
        if f is not None:
            f.close()

    return results


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'multi_score'))
    multi_score_ex.observers.append(observer)
    multi_score_ex.run_commandline()


if __name__ == '__main__':
    main()
