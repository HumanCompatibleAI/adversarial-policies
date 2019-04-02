"""Configuration that is common between multi.train and multi.score.

In particular, configures sensible defaults for upload directory and Ray server
depending on if running on EC2 or baremetal.
"""

import functools
import getpass
import os
import os.path as osp
import shlex
import socket
import subprocess
import urllib

import ray
from ray import tune

from modelfree.common import utils


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


def _rsync_func(local_dir, remote_uri):
    """rsync data from worker to a remote location (by default the driver)."""
    # SOMEDAY: This function blocks until syncing completes, which is unfortunate.
    # If we instead specified a shell command, ray.tune._LogSyncer would run it asynchronously.
    # But we need to do a two-stage command, creating the directories first, because rsync will
    # balk if destination directory does not exist; so no easy way to do that.
    remote_host, ssh_key, *remainder = remote_uri.split(':')
    remote_dir = ':'.join(remainder)  # remote directory may contain :
    remote_dir = shlex.quote(remote_dir)  # make safe for SSH/rsync call

    ssh_command = ['ssh', '-o', 'StrictHostKeyChecking=no', '-i', ssh_key]
    ssh_mkdir = ssh_command + [remote_host, 'mkdir', '-p', remote_dir]
    subprocess.run(ssh_mkdir, check=True)

    rsync = ['rsync', '-rlptv', '-e', ' '.join(ssh_command),
             f'{local_dir}/', f'{remote_host}:{remote_dir}']
    subprocess.run(rsync)


def make_sacred(ex, worker_name, worker_fn):
    @ex.config
    def default_config():
        spec = {}             # Ray spec
        platform = None       # hosting: 'baremetal' or 'ec2'
        s3_bucket = None      # results storage on 'ec2' platform
        baremetal = {}        # config options for 'baremetal' platform
        ray_server = None     # if None, start cluster on local machine
        upload_root = None    # root of upload_dir
        exp_name = 'default'  # experiment name

        _ = locals()  # quieten flake8 unused variable warning
        del _

    @ex.config
    def ec2_config(platform, s3_bucket, spec):
        """When running on AWS EC2 cloud.

        If you are not the authors of this project, you will need to override s3_bucket."""
        if platform is None:
            if _detect_ec2():
                platform = 'ec2'

        if platform == 'ec2':
            # We're running on EC2
            if s3_bucket is None:
                s3_bucket = 'adversarial-policies'

            spec['upload_dir'] = f's3://{s3_bucket}/'
            ray_server = 'localhost:6379'

        _ = locals()  # quieten flake8 unused variable warning
        del _

    @ex.config
    def baremetal_config(platform, baremetal, spec):
        """When running on bare-metal hardware (i.e. not in cloud).

        The workers must have permission to rsync to local_out."""
        if platform is None:
            # No platform specified; assume baremetal if no previous config autodetected.
            platform = 'baremetal'

        if platform == 'baremetal':
            baremetal = dict(baremetal)
            if 'ssh_key' not in baremetal:
                baremetal['ssh_key'] = '~/.ssh/adversarial-policies'
            if 'host' not in baremetal:
                baremetal['host'] = f'{getpass.getuser()}@{socket.getfqdn()}'
            if 'dir' not in baremetal:
                baremetal['dir'] = osp.abspath(osp.join(os.getcwd(), 'data'))

            spec['upload_dir'] = ':'.join([baremetal['host'],
                                           baremetal['ssh_key'],
                                           baremetal['dir']])
            spec['sync_function'] = tune.function(_rsync_func)

    @ex.capture
    def run(base_config, ray_server, exp_name, spec):
        ray.init(redis_address=ray_server)
        tune.register_trainable(worker_name, functools.partial(worker_fn, base_config))
        exp_id = f'{ex.path}/{exp_name}/{utils.make_timestamp()}'
        result = tune.run_experiments({exp_id: spec})
        ray.shutdown()  # run automatically on exit, but needed here to not break tests
        return result

    return run
