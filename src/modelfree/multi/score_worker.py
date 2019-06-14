"""Helper functions for training.py executed on worker nodes using Ray Tune.

It's important these are all pickleable."""

import os.path as osp

from sacred.observers import FileStorageObserver

from modelfree.multi.common_worker import flatten_config, update


def score_worker(base_config, tune_config, reporter):
    """Run a modelfree.training experiment with specified config, logging to reporter.

    :param base_config: (dict) default config
    :param tune_config: (dict) overrides values in base_config
    :param reporter: (ray.tune.StatusReporter) Ray Tune internal logger."""
    # train_ex is not pickleable, so we cannot close on it.
    # Instead, import inside the function.
    from modelfree.score_agent import score_ex

    config = dict(base_config)
    tune_config = flatten_config(tune_config)
    update(config, tune_config)

    # We're breaking the Sacred interface by running an experiment from within another experiment.
    # This is the best thing we can do, since we need to run the experiment with varying configs.
    # Just be careful: this could easily break things.
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'score'))
    score_ex.observers.append(observer)
    run = score_ex.run(config_updates=config)
    index_keys = config.get("index_keys", [])

    idx = {k: v for k, v in config.items()
           if k.startswith('agent') or k == 'env_name' or k in index_keys}

    reporter(done=True, score=run.result, idx=idx)
