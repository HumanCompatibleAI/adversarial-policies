[![CircleCI](https://circleci.com/gh/HumanCompatibleAI/adversarial-policies.svg?style=svg)](https://circleci.com/gh/HumanCompatibleAI/adversarial-policies)
[![codecov](https://codecov.io/gh/HumanCompatibleAI/adversarial-policies/branch/master/graph/badge.svg)](https://codecov.io/gh/HumanCompatibleAI/adversarial-policies)

Codebase to train, evaluate and analyze adversarial policies: policies attacking a fixed victim
agent in a multi-agent system. See [paper](https://arxiv.org/abs/1905.10615) for more information.

# Installation

The easiest way to install the code is to build the Docker image in the `Dockerfile`.
This will install all necessary binary and Python dependencies. Build the image by:

  ```bash
  $ docker build .
  ```

You can also pull a Docker image for the latest master commit from
`humancompatibleai/adversarial_policies:latest`. Once you have built the image, run it by:

  ```bash
  docker run -it --env MUJOCO_KEY=URL_TO_YOUR_MUJOCO_KEY \
         humancompatibleai/adversarial_policies:latest /bin/bash  # change tag if built locally
  ```

If you want to run outside of Docker (for example, for ease of development), read on.

This codebase uses Python 3.7. The main binary dependencies are MuJoCo (version 1.3.1, for
`gym_compete` environments, and 2.0 for the others). You may also need to install some other
libraries, such as OpenMPI.

Create a virtual environment by running `ci/build_venv.sh`. Activate it
by `. ./venv/bin/activate`. Finally, run `pip install -e .` to install
an editable version of this package.

# Reproducing Results

Note we use [Sacred](https://github.com/IDSIA/sacred) for
experiment configuration.

## Training adversarial policies

`aprl.train` trains a single adversarial policy. By default it will train on `SumoAnts` for
a brief period of time. You can override any of config parameters, defined in `train_config`, at
the command line. For example, to replicate one of the experiments in the paper, run:

  ```bash
  # Train on Sumo Humans for 20M timesteps
  python -m aprl.train with env_name=multicomp/SumoHumans-v0 paper
  ```

`aprl.multi.train` trains multiple adversarial policies, using Ray (see below) for
parallelization. To replicate the results in the paper (there may be slight differences due to 
randomness not captured in the seeding), run `python -m aprl.multi.train with paper`. To run
the hyperparameter sweep, run `python -m aprl.multi.train with hyper`.

You can find results from our training run on s3://adversarial-policies-public/multi_train/paper.
This includes TensorBoard logs, final model weights, checkpoints, and individual policy configs.
Run `experiments/pull_public_s3.sh` to sync this and other data to `data/aws-public/`.

## Evaluating adversarial policies

`aprl.score_agent` evaluates a pair of policies, for example an adversary and a victim.
It outputs the win rate for each agent and the number of ties. It can also render to the screen
or produce videos.

We similarly use `aprl.multi.score` to evaluate multiple pairs of policies in parallel.
To reproduce all the evaluations used in the paper, run the following bash scripts, which call
`aprl.multi.score` internally:
  - `experiments/modelfree/baselines.sh`: fixed baselines (no adversarial policies).
  - `experiments/modelfree/attack_transfer.sh <path-to-trained-adversaries>`. To use our
     pre-trained policies, use the path `data/aws-public/multi_train/paper/20190429_011349`
     after syncing against S3.

## Visualizing Results

Most of the visualization code lives in the `aprl.visualize` package. To reproduce the figures
in the paper, use `paper_config`; for those in the appendix, use `supplementary_config`. So:

```bash
  python -m aprl.visualize.scores with paper_config  # heatmaps in the paper
  python -m aprl.visualize.training with supplementary_config  # training curves in appendix
```

To re-generate all the videos, use `aprl.visualize.make_videos`. We would recommend running
in Docker, in which case it will render using `Xdummy`. This avoids rendering issues with many
graphics drivers.

Note you will likely need to change the default paths in the config to point at your evaluation
results from the previous section, and desired output directory. For example:

  ```bash
  python -m aprl.visualize.scores with tb_dir=<path/to/trained/models> \
                                       transfer_path=<path/to/multi_score/output>
  python -m aprl.visualize.make_videos with adversary_path=<path/to/best_adversaries.json>
  ```

## Additional Analysis

The density modeling can be run by `experiments/aprl/density.sh`, or with custom
configurations via `aprl.density.pipeline`.

The t-SNE visualizations can be replicated with `aprl.tsne.pipeline`.

## Using Ray

Many of the experiments are computationally intensive. You can run them on a single machine, but it
might take several weeks. We use [Ray](https://github.com/ray-project/ray) to run distributed
experiments. We include example configs in `src/aprl/configs/ray/`. To use `aws.yaml` you
will need to, at a minimum, edit the config to use your own AMI (anything with Docker should work)
and private key. Then just run `ray up <path-to-config>` and it will start a cluster. SSH into the
head node, start a shell in Docker, and then follow the above instructions. The script should
automatically detect it is part of a Ray cluster and run on the existing Ray server, rather than
starting a new one.

# Contributions

The codebase follows PEP8, with a 100-column maximum line width. Docstrings should be in reST.

Please run the `ci/code_checks.sh` before committing. This runs several linting steps.
These are also run as a continuous integration check.

I like to use Git commit hooks to prevent bad commits from happening in the first place:
```bash
ln -s ../../ci/code_checks.sh .git/hooks/pre-commit
```
