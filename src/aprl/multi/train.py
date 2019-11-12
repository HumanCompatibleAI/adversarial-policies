"""Hyperparameter search for train.py using Ray Tune."""

import logging
import math
import os.path as osp

from ray import tune
from sacred import Experiment
from sacred.observers import FileStorageObserver

from aprl.configs.multi.train import make_configs
from aprl.multi import common, train_worker
from aprl.train import train_ex

multi_train_ex = Experiment('multi_train', ingredients=[train_ex])
pylog = logging.getLogger('aprl.multi.train')

# Load common configs (e.g. upload directories) and define the run command
run = common.make_sacred(multi_train_ex, 'train_rl', train_worker.train_rl)

# Load named configs for individual experiments (these change a lot, so keep out of this file)
make_configs(multi_train_ex)


@multi_train_ex.config
def default_config(train):
    spec = {  # experiment specification
        'run_kwargs': {
            'resources_per_trial': {'cpu': math.ceil(train['num_env'] / 2)},
        },
        'config': {},
    }

    _ = locals()  # quieten flake8 unused variable warning
    del _


@multi_train_ex.named_config
def debug_config():
    spec = {
        'config': {
            'seed': tune.grid_search([0, 1]),
        },
    }
    exp_name = 'debug'
    _ = locals()  # quieten flake8 unused variable warning
    del _


@multi_train_ex.main
def multi_train(train):
    return run(base_config=train)


def main():
    observer = FileStorageObserver(osp.join('data', 'sacred', 'multi_train'))
    multi_train_ex.observers.append(observer)
    multi_train_ex.run_commandline()


if __name__ == '__main__':
    main()
