"""Integration test: run experiments with some small & fast configs.

Only cursory 'smoke' checks -- there are plenty of errors this won't catch."""

import json
import os

import pytest

from modelfree.score_agent import score_agent_ex
from modelfree.policy_loader import AGENT_LOADERS
from modelfree.train_and_score import train_and_score
from modelfree.train import NO_VECENV, RL_ALGOS, train_ex


EXPERIMENTS = [score_agent_ex, train_and_score, train_ex]


@pytest.mark.parametrize('experiment', EXPERIMENTS)
def test_experiment(experiment):
    """Smoke test to check the experiments runs with default config."""
    run = experiment.run()
    assert run.status == 'COMPLETED'


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
SCORE_AGENT_CONFIGS = [
    {'agent_b_type': 'zoo', 'agent_b_path': '2', 'videos': True, 'episodes': 2},
    {'env_name': 'multicomp/KickAndDefend-v0', 'episodes': 1},
]
SCORE_AGENT_CONFIGS += [
    {
        'agent_b_type': rl_algo,
        'agent_b_path': os.path.join(BASE_DIR, 'dummy_sumo_ants', rl_algo),
        'episodes': 1,
    }
    for rl_algo in AGENT_LOADERS.keys() if rl_algo != 'zoo'
]


@pytest.mark.parametrize('config', SCORE_AGENT_CONFIGS)
def test_score_agent(config):
    """Smoke test for score agent to check it runs with some different configs."""
    config = dict(config)
    config['render'] = False  # faster without, test_experiment already tests with render

    run = score_agent_ex.run(config_updates=config)
    assert run.status == 'COMPLETED'

    ties = run.result['ties']
    win_a, win_b = run.result['wincounts']
    assert sum([ties, win_a, win_b]) == run.config['episodes']


def load_json(fname):
    with open(fname) as f:
        return json.load(f)


TRAIN_CONFIGS = [
    {'num_env': 1},
    {'env_name': 'multicomp/KickAndDefend-v0'},
    {'normalize': False},
    {'victim_type': 'ppo2', 'victim_path': os.path.join(BASE_DIR, 'dummy_sumo_ants', 'ppo2')},
    {
        'env_name': 'multicomp/SumoHumans-v0',
        'rew_shape': True,
        'rew_shape_params': {'anneal_frac': 0.1},
    },
    {
        'env_name': 'multicomp/SumoHumans-v0',
        'victim_noise': True,
    },
    {
        'env_name': 'Humanoid-v1',
        'victim_type': 'none',
    },
    {
        'env_name': 'multicomp/SumoHumansAutoContact-v0',
        'rew_shape': True,
        'rew_shape_params': {'metric': 'length', 'min_wait': 100, 'window_size': 100},
    },
    {
        'env_name': 'multicomp/SumoHumans-v0',
        'rew_shape': True,
        'victim_noise': True,
        'victim_noise_params': {'metric': 'sparse', 'min_wait': 100, 'window_size': 100},
    },
    {
        'env_name': 'multicomp/SumoHumansAutoContact-v0',
        'adv_noise_agent_val': 0.1
    }
]
TRAIN_CONFIGS += [{'rl_algo': algo, 'num_env': 1 if algo in NO_VECENV else 8}
                  for algo in RL_ALGOS.keys()]


@pytest.mark.parametrize('config', TRAIN_CONFIGS)
def test_train(config):
    config = dict(config)
    # Use a small number of steps to keep things quick
    config['batch_size'] = 512
    config['total_timesteps'] = 1024
    run = train_ex.run(config_updates=config)
    final_dir = run.result
    assert os.path.isdir(final_dir), "final result not saved"
    assert os.path.isfile(os.path.join(final_dir, 'model.pkl')), "model weights not saved"

