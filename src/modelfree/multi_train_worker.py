"""Helper functions for multi_train.py executed on worker nodes using Ray Tune.

It's important these are all pickleable."""

import collections
import os.path as osp

from sacred.observers import FileStorageObserver
from stable_baselines.logger import KVWriter


class ReporterOutputFormat(KVWriter):
    """Key-value logging plugin for Stable Baselines that writes to a Ray Tune StatusReporter."""
    def __init__(self, reporter):
        self.last_kvs = dict()
        self.reporter = reporter

    def writekvs(self, kvs):
        self.last_kvs = kvs
        self.reporter(**kvs)


def _flatten_config(config):
    """Take dict with ':'-separated keys and values or tuples of values,
       flattening to single key-value pairs.

       Example: _flatten_config({'a:b': (1, 2), 'c': 3}) -> {'a: 1, 'b': 2, 'c': 3}."""
    new_config = {}
    for ks, vs in config.items():
        ks = ks.split(':')
        if len(ks) == 1:
            vs = (vs, )

        for k, v in zip(ks, vs):
            assert k not in new_config, f"duplicate key '{k}'"
            new_config[k] = v

    return new_config


def _update(d, u):
    """Recursive dictionary update."""
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = _update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def train_rl(base_config, tune_config, reporter):
    """Run a modelfree.train experiment with specified config, logging to reporter.

    :param base_config: (dict) default config
    :param tune_config: (dict) overrides values in base_config
    :param reporter: (ray.tune.StatusReporter) Ray Tune internal logger."""
    # train_ex is not pickleable, so we cannot close on it.
    # Instead, import inside the function.
    from modelfree.train import train_ex

    config = dict(base_config)
    tune_config = _flatten_config(tune_config)
    _update(config, tune_config)
    tune_kv_str = '-'.join([f'{k}={v}' for k, v in tune_config.items()])
    config['exp_name'] = config['exp_name'] + '-' + tune_kv_str

    output_format = ReporterOutputFormat(reporter)
    config['log_output_formats'] = [output_format]

    # We're breaking the Sacred interface by running an experiment from within another experiment.
    # This is the best thing we can do, since we need to run the experiment with varying configs.
    # Just be careful: this could easily break things.
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'train'))
    train_ex.observers.append(observer)
    train_ex.run(config_updates=config)
    reporter(done=True, **output_format.last_kvs)
