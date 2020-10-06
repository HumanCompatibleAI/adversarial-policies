"""Integration test: run experiments with some small & fast configs.

Only cursory 'smoke' checks -- there are plenty of errors this won't catch."""

import os
import shutil
import tempfile

import numpy as np
import pytest
import ray
from ray import tune

from aprl.activations.density.pipeline import density_ex
from aprl.activations.tsne.pipeline import tsne_ex
from aprl.multi.score import multi_score_ex
from aprl.multi.train import multi_train_ex
from aprl.policies.loader import AGENT_LOADERS
from aprl.score_agent import score_ex
from aprl.train import NO_VECENV, RL_ALGOS, train_ex

EXPERIMENTS = [score_ex, train_ex]


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_experiment(experiment):
    """Smoke test to check the experiments runs with default config."""
    run = experiment.run()
    assert run.status == "COMPLETED"


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
SCORE_AGENT_CONFIGS = [
    {"agent_b_type": "zoo", "agent_b_path": "2", "videos": True, "episodes": 2},
    {"env_name": "multicomp/KickAndDefend-v0", "episodes": 1},
    {"record_traj": True, "record_traj_params": {"save_dir": "test_dir"}},
    {"noisy_agent_index": 0},
    {"mask_agent_index": 0},
    {"mask_agent_index": 0, "mask_agent_masking_type": "additive_noise", "mask_agent_noise": 1.0},
]
SCORE_AGENT_CONFIGS += [
    {
        "agent_b_type": rl_algo,
        "agent_b_path": os.path.join(BASE_DIR, "dummy_sumo_ants", rl_algo),
        "episodes": 1,
    }
    for rl_algo in AGENT_LOADERS.keys()
    if rl_algo != "zoo"
]


@pytest.mark.parametrize("config", SCORE_AGENT_CONFIGS)
def test_score_agent(config):
    """Smoke test for score agent to check it runs with some different configs."""
    config = dict(config)
    if "episodes" not in config:
        config["episodes"] = 1  # speed up tests
    config["render"] = False  # faster without, test_experiment already tests with render

    run = score_ex.run(config_updates=config)
    assert run.status == "COMPLETED"

    outcomes = [run.result[k] for k in ["ties", "win0", "win1"]]
    assert sum(outcomes) == run.config["episodes"]

    if config.get("record_traj", False):
        try:
            for i in range(2):
                traj_file_path = os.path.join(
                    config["record_traj_params"]["save_dir"], f"agent_{i}.npz"
                )
                traj_data = np.load(traj_file_path)
                assert set(traj_data.keys()).issuperset(["observations", "actions", "rewards"])
                for k, ep_data in traj_data.items():
                    assert len(ep_data) == config["episodes"], f"unexpected array length at '{k}'"
                os.remove(traj_file_path)
        finally:
            os.rmdir(config["record_traj_params"]["save_dir"])


SCORE_AGENT_VIDEO_CONFIGS = {
    "none_dir": {
        "videos": True,
        "video_params": {"save_dir": None},
        "episodes": 1,
        "render": False,
    },
    "specified_dir": {
        "videos": True,
        "video_params": {"save_dir": "specific_video_dir"},
        "episodes": 1,
        "render": False,
    },
}


def test_score_agent_video():
    # Confirm that experiment runs properly saving videos to a temp dir
    none_dir_run = score_ex.run(config_updates=SCORE_AGENT_VIDEO_CONFIGS["none_dir"])
    assert none_dir_run.status == "COMPLETED"

    try:
        # Confirm that the first time you try to save videos to a specified dir, it works properly
        specified_dir_run = score_ex.run(config_updates=SCORE_AGENT_VIDEO_CONFIGS["specified_dir"])
        assert specified_dir_run.status == "COMPLETED"

        # Confirm that the second time you try to save videos to the same specified dir, it fails
        with pytest.raises(AssertionError):
            _ = score_ex.run(config_updates=SCORE_AGENT_VIDEO_CONFIGS["specified_dir"])
    finally:
        shutil.rmtree(SCORE_AGENT_VIDEO_CONFIGS["specified_dir"]["video_params"]["save_dir"])


TRAIN_CONFIGS = [
    {"num_env": 1},
    {"env_name": "multicomp/YouShallNotPassHumans-v0"},
    {"normalize": False},
    {"embed_type": "ppo2", "embed_path": os.path.join(BASE_DIR, "dummy_sumo_ants", "ppo2")},
    {
        "env_name": "multicomp/SumoHumans-v0",
        "rew_shape": True,
        "rew_shape_params": {"anneal_frac": 0.1},
    },
    {"env_name": "multicomp/SumoHumans-v0", "embed_noise": True},
    {"env_name": "Humanoid-v3", "embed_types": [], "embed_paths": []},
    {
        "env_name": "multicomp/SumoHumansAutoContact-v0",
        "rew_shape": True,
        "rew_shape_params": {"metric": "length", "min_wait": 100, "window_size": 100},
    },
    {
        "env_name": "multicomp/SumoHumans-v0",
        "rew_shape": True,
        "embed_noise": True,
        "embed_noise_params": {"metric": "sparse", "min_wait": 100, "window_size": 100},
    },
    {"env_name": "multicomp/SumoHumansAutoContact-v0", "adv_noise_params": {"noise_val": 0.1}},
    {
        # test TransparentLSTMPolicy
        "transparent_params": ["ff_policy", "hid"],
    },
    {
        # test TransparentMLPPolicyValue
        "env_name": "multicomp/YouShallNotPassHumans-v0",
        "transparent_params": ["ff_policy"],
        "batch_size": 32,
    },
    {
        "env_name": "multicomp/SumoHumans-v0",
        "lookback_params": {"lb_num": 2, "lb_path": 1, "lb_type": "zoo"},
        "adv_noise_params": {"noise_val": 0.1},
        "transparent_params": ["ff_policy"],
    },
]
try:
    from stable_baselines import GAIL

    del GAIL
    TRAIN_CONFIGS.append(
        {
            "rl_algo": "gail",
            "num_env": 1,
            "expert_dataset_path": os.path.join(BASE_DIR, "SumoAnts_traj/agent_0.npz"),
        }
    )
except ImportError:  # pragma: no cover
    # skip GAIL test if algorithm not available
    pass
TRAIN_CONFIGS += [
    {"rl_algo": algo, "num_env": 1 if algo in NO_VECENV else 2}
    for algo in RL_ALGOS.keys()
    if algo != "gail"
]


# Choose hyperparameters to minimize resource consumption in tests
TRAIN_SMALL_RESOURCES = {
    "batch_size": 64,
    "total_timesteps": 128,
    "num_env": 2,
}


@pytest.mark.parametrize("config", TRAIN_CONFIGS)
def test_train(config):
    config = dict(config)
    for k, v in TRAIN_SMALL_RESOURCES.items():
        config.setdefault(k, v)

    run = train_ex.run(config_updates=config)
    assert run.status == "COMPLETED"

    final_dir = run.result
    assert os.path.isdir(final_dir), "final result not saved"
    assert os.path.isfile(os.path.join(final_dir, "model.pkl")), "model weights not saved"


def _test_multi(ex, config_updates=None):
    multi_config = {
        "spec": {
            "run_kwargs": {
                "resources_per_trial": {"cpu": 2},  # CI build only has 2 cores
            },
            "sync_config": {
                "upload_dir": None,  # do not upload test results anywhere
                "sync_to_cloud": None,  # as above
            },
        },
        "init_kwargs": {"num_cpus": 2},  # CI build only has 2 cores
    }
    if config_updates:
        multi_config.update(config_updates)

    run = ex.run(config_updates=multi_config, named_configs=("debug_config",))
    assert run.status == "COMPLETED"
    assert ray.state.state.redis_client is None, "ray has not been shutdown"

    return run


def test_multi_score():
    run = _test_multi(multi_score_ex)
    assert "scores" in run.result
    assert "exp_id" in run.result
    assert isinstance(run.result["scores"], dict)


def test_multi_train():
    config_updates = {
        "train": TRAIN_SMALL_RESOURCES,
    }
    run = _test_multi(multi_train_ex, config_updates=config_updates)

    analysis, exp_id = run.result
    assert isinstance(analysis, tune.analysis.ExperimentAnalysis)
    assert isinstance(exp_id, str)


ACTIVATION_EXPERIMENTS = [
    (density_ex, "fit_density_model"),
    (tsne_ex, "tsne_fit_model"),
]


@pytest.mark.parametrize("test_cfg", ACTIVATION_EXPERIMENTS)
def test_activation_pipeline(test_cfg):
    ex, inner_exp_name = test_cfg
    with tempfile.TemporaryDirectory(prefix="test_activation_pipeline") as tmpdir:
        config_updates = {
            "generate_activations": {
                "score_update": {
                    "spec": {
                        "run_kwargs": {
                            "resources_per_trial": {"cpu": 2},  # CI build only has 2 cores
                        },
                        "sync_config": {
                            "upload_dir": os.path.join(tmpdir, "ray"),
                            "sync_to_cloud": (
                                "mkdir -p {target} && " "rsync -rlptv {source}/ {target}"
                            ),
                        },
                    },
                    "init_kwargs": {"num_cpus": 2},  # CI build only has 2 cores
                },
                "ray_upload_dir": os.path.join(tmpdir, "ray"),
            },
            inner_exp_name: {"init_kwargs": {"num_cpus": 2}},  # CI build only has 2 cores
            "output_root": os.path.join(tmpdir, "main"),
        }

        run = ex.run(config_updates=config_updates, named_configs=("debug_config",))
        assert run.status == "COMPLETED"
        os.stat(run.result)  # check output path exists
