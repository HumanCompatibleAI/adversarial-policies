"""Integration test: run experiments with some small & fast configs.

Only cursory 'smoke' checks -- there are plenty of errors this won't catch."""

import os

import pytest

from modelfree.score_agent import score_agent_ex
from modelfree.ppo_baseline import ppo_baseline_ex


EXPERIMENTS = [score_agent_ex, ppo_baseline_ex]


@pytest.mark.parametrize('experiment', EXPERIMENTS)
def test_experiment(experiment):
    """Smoke test to check the experiments runs with default config."""
    run = experiment.run()
    assert run.status == 'COMPLETED'


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
SCORE_AGENT_CONFIGS = [
    {'agent_b_type': 'zoo', 'agent_b_path': '2', 'videos': True},
    {
        'agent_b_type': 'mlp',
        'agent_b_path': os.path.join(BASE_DIR, 'dummy_sumo_ants.pkl'),
        'episodes': 5
    },
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


PPO_BASELINE_CONFIGS = [
    {'vectorize': 1},
    {'no_normalize': False},
    {'victim_type': 'mlp', 'victim_path': os.path.join(BASE_DIR, 'dummy_sumo_ants.pkl')},
]


@pytest.mark.parametrize('config', PPO_BASELINE_CONFIGS)
def test_ppo_baseline(config):
    config = dict(config)
    config['total_timesteps'] = 4096  # small number of steps to keep things quick
    run = ppo_baseline_ex.run(config_updates=config)
    assert os.path.isfile(run.result), "model weights not saved"
