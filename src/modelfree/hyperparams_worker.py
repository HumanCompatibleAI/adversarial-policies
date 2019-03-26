"""Helper functions for hyperparams.py executed on worker nodes using Ray Tune.

It's important these are all pickleable."""

import json
import subprocess
import os.path as osp

from sacred.observers import FileStorageObserver


def train_rl(base_config, tune_config, reporter, num_mpi):
    """Run a modelfree.train experiment with specified config, logging to reporter.

    :param base_config: (dict) default config
    :param tune_config: (dict) overrides values in base_config
    :param reporter: (ray.tune.StatusReporter) Ray Tune internal logger."""
    # train_ex is not pickleable, so we cannot close on it.
    # Instead, import inside the function.
    config = dict(base_config)
    config.update(tune_config)

    if num_mpi > 0:
        config_hash = hash(str(config))
        config['exp_name'] = str(config_hash)
        with open(f"config-{config_hash}.json", 'w') as config_file:
            json.dump(config, config_file)
        command_str = f"mpirun --allow-run-as-root -np {num_mpi} python -m modelfree.train" + \
            f" with config-{config_hash}.json > stdout 2>stderr"
        subprocess.call(command_str, shell=True)
        reporter(done=True, **{})
    else:
        from modelfree.train import train_ex
        from modelfree.utils import ReporterOutputFormat

        tune_kv_str = '-'.join([f'{k}={v}' for k, v in tune_config.items()])
        config['exp_name'] = 'dummy' + '-' + tune_kv_str

        output_format = ReporterOutputFormat(reporter)
        config['log_output_formats'] = [output_format]

        # We're breaking the Sacred interface by running an experiment from within another experiment.
        # This is the best thing we can do, since we need to run the experiment with varying configs.
        # Just be careful: this could easily break things.

        observer = FileStorageObserver.create(osp.join('data', 'sacred', 'train'))
        train_ex.observers.append(observer)
        train_ex.run(config_updates=config)
        reporter(done=True, **output_format.last_kvs)
