"""Integration test: run experiments with some small & fast configs.

Only cursory 'smoke' checks -- there are plenty of errors this won't catch."""

import os

import numpy as np
import pytest
from ray.tune.trial import Trial

from modelfree.common.policy_loader import AGENT_LOADERS
from modelfree.multi.score import multi_score_ex
from modelfree.multi.train import multi_train_ex
from modelfree.score_agent import score_ex
from modelfree.train import NO_VECENV, RL_ALGOS, train_ex
from modelfree.train_and_score import train_and_score

EXPERIMENTS = [score_ex, train_and_score, train_ex]


@pytest.mark.parametrize('experiment', EXPERIMENTS)
def test_experiment(experiment):
    """Smoke test to check the experiments runs with default config."""
    run = experiment.run()
    assert run.status == 'COMPLETED'


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
SCORE_AGENT_CONFIGS = [
    {'agent_b_type': 'zoo', 'agent_b_path': '2', 'videos': True, 'episodes': 2},
    {'env_name': 'multicomp/KickAndDefend-v0', 'episodes': 1},
    {
        'record_traj': True,
        'record_traj_params': {'save_dir': 'test_dir'},
    }
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
    if 'episodes' not in config:
        config['episodes'] = 1  # speed up tests
    config['render'] = False  # faster without, test_experiment already tests with render

    run = score_ex.run(config_updates=config)
    assert run.status == 'COMPLETED'

    outcomes = [run.result[k] for k in ['ties', 'win0', 'win1']]
    assert sum(outcomes) == run.config['episodes']

    if config.get('record_traj', False):
        try:
            for i in range(2):
                traj_file_path = os.path.join(config['record_traj_params']['save_dir'],
                                              f'agent_{i}.npz')
                traj_data = np.load(traj_file_path)
                assert set(traj_data.keys()).issuperset(['observations', 'actions', 'rewards'])
                for k, ep_data in traj_data.items():
                    assert len(ep_data) == config['episodes'], f"unexpected array length at '{k}'"
                os.remove(traj_file_path)
        finally:
            os.rmdir(config['record_traj_params']['save_dir'])


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
        'adv_noise_params': {'noise_val': 0.1},
    },
    {
        'rl_algo': 'gail',
        'num_env': 1,
        'expert_dataset_path': os.path.join(BASE_DIR, 'SumoAnts_traj/agent_0.npz'),
    },
    {
        # test TransparentLSTMPolicy
        'transparent_params': ['ff_policy', 'hid'],
    },
    {
        # test TransparentMLPPolicyValue
        'env_name': 'multicomp/YouShallNotPassHumans-v0',
        'transparent_params': ['ff_policy'],
    }

]
TRAIN_CONFIGS += [{'rl_algo': algo, 'num_env': 1 if algo in NO_VECENV else 8}
                  for algo in RL_ALGOS.keys() if algo != 'gail']


@pytest.mark.parametrize('config', TRAIN_CONFIGS)
def test_train(config):
    config = dict(config)
    # Use a small number of steps to keep things quick
    config['batch_size'] = 512
    config['total_timesteps'] = 1024

    run = train_ex.run(config_updates=config)
    assert run.status == 'COMPLETED'

    final_dir = run.result
    assert os.path.isdir(final_dir), "final result not saved"
    assert os.path.isfile(os.path.join(final_dir, 'model.pkl')), "model weights not saved"


MULTI_EXPERIMENTS = [multi_score_ex, multi_train_ex]


def _test_multi(ex):
    multi_config = {
        'spec': {
            'resources_per_trial': {'cpu': 2},  # Travis only has 2 cores
            'upload_dir': None,  # do not upload test results anywhere
            'sync_function': None,  # as above
        },
    }

    run = ex.run(config_updates=multi_config, named_configs=('debug_config',))
    assert run.status == 'COMPLETED'

    return run


def test_multi_score():
    run = _test_multi(multi_score_ex)
    assert isinstance(run.result, dict)


def test_multi_train():
    run = _test_multi(multi_train_ex)

    trials = run.result
    for trial in trials:
        assert isinstance(trial, Trial)
