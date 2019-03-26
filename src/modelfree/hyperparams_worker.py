"""Helper functions for hyperparams.py executed on worker nodes using Ray Tune.

It's important these are all pickleable."""

import json
import os
import os.path as osp

import numpy as np
from sacred.observers import FileStorageObserver
#from stable_baselines.logger import KVWriter


#class ReporterOutputFormat(KVWriter):
#    """Key-value logging plugin for Stable Baselines that writes to a Ray Tune StatusReporter."""
#    def __init__(self, reporter):
#        self.last_kvs = dict()
#        self.reporter = reporter
#
#    def writekvs(self, kvs):
#        self.last_kvs = kvs
#        self.reporter(**kvs)


def short_str(d):
    final_str_list = []
    d_copy = d.copy()
    abbrev_dict = {
        'adversary_entcoeff': 'adv_ent',
        'expert_dataset_path': 'exp_data',
        'hidden_size_adversary': 'hid_adv',
        'timesteps_per_batch': 'tsteps/batch',
        'victim_index': 'vic_idx',
        'victim_path': 'vic_path',
        'd_stepsize': 'd_step',
        'vf_stepsize': 'vf_step'
    }
    for k, v in d_copy.items():
        k_str = abbrev_dict.get(k, str(k))
        if isinstance(v, dict):
            v_str = "{" + f"{short_str(v)}" + "}"
        elif isinstance(v, float):
            v_str = "{:.6f}".format(v)
        else:
            v_str = str(v)
        final_str_list.append(f"{k_str}={v_str}")
    return '-'.join(final_str_list)


def train_rl(base_config, tune_config, reporter):
    """Run a modelfree.train experiment with specified config, logging to reporter.

    :param base_config: (dict) default config
    :param tune_config: (dict) overrides values in base_config
    :param reporter: (ray.tune.StatusReporter) Ray Tune internal logger."""
    # train_ex is not pickleable, so we cannot close on it.
    # Instead, import inside the function.
    # from modelfree.train import train_ex

    config = dict(base_config)
    config.update(tune_config)
    # tune_kv_str = '-'.join([f'{k}={v}' if not isinstance(v, dict) else f'{k}={short_str(v)}'])
    tune_kv_str = short_str(config)
    config['exp_name'] = 'dummy' + '-' + tune_kv_str

    #output_format = ReporterOutputFormat(reporter)
    # config['log_output_formats'] = [output_format]

    # We're breaking the Sacred interface by running an experiment from within another experiment.
    # This is the best thing we can do, since we need to run the experiment with varying configs.
    # Just be careful: this could easily break things.

    # observer = FileStorageObserver.create(osp.join('data', 'sacred', 'train'))
    # train_ex.observers.append(observer)

    # train_ex.run(config_updates=config)

    config_hash = hash(str(config))
    config['exp_name'] = str(config_hash)
    with open(f"config-{config_hash}.json", 'w') as config_file:
        json.dump(config, config_file)
    print('cwd', os.getcwd())
    command_str = f"mpirun --allow-run-as-root -np 4 python -m modelfree.train with config-{config_hash}.json > stdout 2>stderr"
    #command_str = f"mpirun --allow-run-as-root -np 2 python -c 'print(\"Hello\")'"
    #command_str = f"mpirun --allow-run-as-root -np 2 /bin/true"
    #command_str = f"python -c 'print(\"Hello\")'"
    print('running: ', command_str)
    ret = os.system(command_str)
    print('return: ', ret)
    reporter(done=True, **{})
