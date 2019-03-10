"""Hyperparameter search for train.py using Ray Tune."""

import functools
import getpass
import logging
import os
import os.path as osp
import socket
import subprocess
import urllib

import ray
from ray import tune
from sacred import Experiment
from sacred.observers import FileStorageObserver

from modelfree.configs.hyperparams import make_configs
from modelfree.hyperparams_worker import train_rl
from modelfree.logger import make_timestamp
from modelfree.train import train_ex

hyper_ex = Experiment('hyperparams', ingredients=[train_ex])
pylog = logging.getLogger('modelfree.hyperparams')


@hyper_ex.config
def default_config(train):
    spec = {              # hyperparameter search specification
        'run': 'train_rl',
        # TODO: tune number of actual CPUs required
        'resources_per_trial': {'cpu': train['num_env'] // 2},
    }
    exp_name = 'default'  # experiment name

    platform = None       # hosting: 'baremetal' or 'ec2'
    s3_bucket = None      # results storage on 'ec2' platform
    baremetal = {}        # config options for 'baremetal' platform
    ray_server = None     # if None, start cluster on local machine

    _ = locals()  # quieten flake8 unused variable warning
    del _


@hyper_ex.named_config
def debug_config():
    spec = {
        'config': {
            'seed': tune.grid_search([0, 1]),
        },
    }
    exp_name = 'debug'
    _ = locals()  # quieten flake8 unused variable warning
    del _


def _detect_ec2():
    """Auto-detect if we are running on EC2."""
    try:
        EC2_ID_URL = 'http://169.254.169.254/latest/dynamic/instance-identity/document'
        with urllib.request.urlopen(EC2_ID_URL, timeout=3) as f:
            response = f.read().decode()
            if 'availabilityZone' in response:
                return True
            else:
                raise ValueError(f"Received unexpected response from '{EC2_ID_URL}'")
    except urllib.error.URLError:
        return False


@hyper_ex.config
def ec2_config(platform, s3_bucket, exp_name, spec):
    """When running on AWS EC2 cloud.

    If you are not the authors of this project, you will need to override s3_bucket."""
    if platform is None:
        if _detect_ec2():
            platform = 'ec2'

    if platform == 'ec2':
        # We're running on EC2
        if s3_bucket is None:
            s3_bucket = 'adversarial-policies'
        spec['upload_dir'] = f's3://{s3_bucket}/hyper/{exp_name}/{make_timestamp()}'


def _rsync_func(local_dir, remote_uri):
    """rsync data from worker to a remote location (by default the driver)."""
    # SOMEDAY: This function blocks until syncing completes, which is unfortunate.
    # If we instead specified a shell command, ray.tune._LogSyncer would run it asynchronously.
    # But we need to do a two-stage command, creating the directories first, because rsync will
    # balk if destination directory does not exist; so no easy way to do that.
    remote_host, ssh_key, remote_dir = remote_uri.split(':')

    ssh_command = ['ssh', '-o', 'StrictHostKeyChecking=no', '-i', ssh_key]
    ssh_mkdir = ssh_command + [remote_host, 'mkdir', '-p', remote_dir]
    subprocess.run(ssh_mkdir, check=True)

    rsync = ['rsync', '-rlptv', '-e', ' '.join(ssh_command),
             f'{local_dir}/', f'{remote_host}:{remote_dir}']
    subprocess.run(rsync)


@hyper_ex.config
def baremetal_config(platform, baremetal, exp_name, spec):
    """When running on bare-metal hardware (i.e. not in cloud).

    The workers must have permission to rsync to local_out."""
    if platform is None:
        # No platform specified; assume baremetal if no previous config autodetected.
        platform = 'baremetal'

    if platform == 'baremetal':
        baremetal = dict(baremetal)
        if 'ssh_key' not in baremetal:
            baremetal['ssh_key'] = '~/.ssh/adversarial-policies'
        if 'local_host' not in baremetal:
            baremetal['local_host'] = f'{getpass.getuser()}@{socket.getfqdn()}'
        if 'local_dir' not in baremetal:
            baremetal['local_dir'] = osp.abspath(osp.join(os.getcwd(), 'data', 'hyper'))

        spec = dict(spec)
        spec['upload_dir'] = (baremetal['local_host'] + ':' + baremetal['ssh_key'] + ':' +
                              osp.join(baremetal['local_dir'], exp_name, make_timestamp()))
        spec['sync_function'] = tune.function(_rsync_func)


# Load named configs for individual experiments (these change a lot, so keep out of this file)
make_configs(hyper_ex)


@hyper_ex.main
def hyperparameter_search(ray_server, train, spec):
    ray.init(redis_address=ray_server)
    tune.register_trainable('train_rl', functools.partial(train_rl, train))
    return tune.run_experiments({'tune_rl': spec})


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'hyperparams'))
    hyper_ex.observers.append(observer)
    hyper_ex.run_commandline()


if __name__ == '__main__':
    main()
