"""Helper functions for training.py executed on worker nodes using Ray Tune.

It's important these are all pickleable."""

import os.path as osp

from sacred import observers
from stable_baselines import logger

from aprl.multi import common_worker


class ReporterOutputFormat(logger.KVWriter):
    """Key-value logging plugin for Stable Baselines that writes to a Ray Tune StatusReporter."""
    def __init__(self, reporter):
        self.last_kvs = dict()
        self.reporter = reporter

    def writekvs(self, kvs):
        self.last_kvs = kvs
        self.reporter(**kvs)


def train_rl(base_config, tune_config, reporter):
    """Run a modelfree.training experiment with specified config, logging to reporter.

    :param base_config: (dict) default config
    :param tune_config: (dict) overrides values in base_config
    :param reporter: (ray.tune.StatusReporter) Ray Tune internal logger."""
    common_worker.fix_sacred_capture()

    # train_ex is not pickleable, so we cannot close on it.
    # Instead, import inside the function.
    from aprl.train import train_ex

    config = dict(base_config)
    tune_config = common_worker.flatten_config(tune_config)
    common_worker.update(config, tune_config)
    tune_kv_str = '-'.join([f'{k}={v}' for k, v in tune_config.items()])
    config['exp_name'] = config['exp_name'] + '-' + tune_kv_str

    output_format = ReporterOutputFormat(reporter)
    config['log_output_formats'] = [output_format]

    # We're breaking the Sacred interface by running an experiment from within another experiment.
    # This is the best thing we can do, since we need to run the experiment with varying configs.
    # Just be careful: this could easily break things.
    observer = observers.FileStorageObserver(osp.join('data', 'sacred', 'train'))

    train_ex.observers.append(observer)
    train_ex.run(config_updates=config)
    reporter(done=True, **output_format.last_kvs)
