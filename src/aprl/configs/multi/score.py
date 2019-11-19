"""Named configs for aprl.multi.score."""

import itertools
import json
import logging
import os.path
from typing import Callable, Iterable, List, NamedTuple, Optional, Tuple

import numpy as np
from ray import tune

from aprl.configs import DATA_LOCATION
from aprl.configs.multi.common import BANSAL_GOOD_ENVS, get_adversary_paths
from aprl.envs import VICTIM_INDEX, gym_compete

AgentConfigGenFn = Callable[[str, int], Iterable[Tuple[str, str]]]


class EnvAgentConfig(NamedTuple):
    env_name: str
    agent_a_type: str
    agent_a_path: str
    agent_b_type: str
    agent_b_path: str


PATHS_AND_TYPES = ":".join(EnvAgentConfig._fields)

logger = logging.getLogger("aprl.configs.multi.score")


def _zoo(env, agent_index):
    """Returns all Zoo policies in `env`."""
    del agent_index
    num_zoo = gym_compete.num_zoo_policies(env)
    return [("zoo", str(i)) for i in range(1, num_zoo + 1)]


def _from_paths(policy_paths):
    """Returns a function that returns policies from `policy_paths`."""

    def helper(env, agent_index):
        """Returns all policies in `env` playing in index `agent_index`."""
        victim_index = 1 - agent_index
        paths = policy_paths.get(env, {}).get(str(victim_index))
        if paths is None:
            logger.warning(f"Missing adversary path in '{env}' for index '{agent_index}'")
            return []
        else:
            return [("ppo2", os.path.join(DATA_LOCATION, path)) for path in paths.values()]

    return helper


def _from_json(json_path):
    """Returns a function that returns policies from the specified JSON."""
    json_path = os.path.join(DATA_LOCATION, json_path)
    if os.path.isdir(json_path):
        json_path = os.path.join(json_path, "highest_win_policies_and_rates.json")
    with open(json_path, "r") as f:
        policy_paths = json.load(f)["policies"]
    return _from_paths(policy_paths)


def _adversary():
    """Returns all adversaries from default JSON."""
    return _from_paths(get_adversary_paths())


def _fixed(env, agent_index):
    """Returns all baseline, environment-independent policies."""
    del env, agent_index
    return [("random", "none"), ("zero", "none")]


def _to_fn(cfg: str) -> AgentConfigGenFn:
    """Converts config of form cfg_type[:path] into a configuration function.

    Supported `cfg`'s are of the format:
        zoo
        fixed
        adversary
        json:/path/to/json
    """
    cfg_type, *rem = cfg.split(":")
    if cfg_type == "zoo":
        assert not rem
        return _zoo
    elif cfg == "fixed":
        assert not rem
        return _fixed
    elif cfg_type == "adversary":
        assert not rem
        return _adversary()
    elif cfg_type == "json":
        assert len(rem) == 1
        return _from_json(rem[0])
    else:
        raise ValueError(f"Unrecognized config type '{cfg_type}'")


def _gen_configs(
    victim_fns: Iterable[AgentConfigGenFn],
    opponent_fns: Iterable[AgentConfigGenFn],
    envs: Optional[Iterable[str]] = None,
) -> List[EnvAgentConfig]:
    """Helper function to generate configs.

    :param victim_fns: list of callables, taking environment name and agent index.
    :param opponent_fns: as above.
    :param envs: optionally, a list of environments to generate configs for.

    :return A list of (env, agent_a_type, agent_a_path, agent_b_type, agent_b_path).
    """
    if envs is None:
        envs = BANSAL_GOOD_ENVS

    configs = []
    for env in envs:
        victim_index = VICTIM_INDEX[env]
        victims = list(itertools.chain(*[fn(env, victim_index) for fn in victim_fns]))
        opponents = list(itertools.chain(*[fn(env, 1 - victim_index) for fn in opponent_fns]))

        for victim_type, victim_path in victims:
            for adversary_type, adversary_path in opponents:
                if victim_index == 0:
                    cfg = EnvAgentConfig(
                        env, victim_type, victim_path, adversary_type, adversary_path
                    )
                elif victim_index == 1:
                    cfg = EnvAgentConfig(
                        env, adversary_type, adversary_path, victim_type, victim_path
                    )
                else:
                    raise ValueError(f"Victim index '{victim_index}' out of range")

                if cfg not in configs:
                    configs.append(cfg)

    return configs


def _make_default_exp_suffix(victims, opponents):
    victims = [x.replace("/", "_") for x in victims]
    opponents = [x.replace("/", "_") for x in opponents]
    return f"{':'.join(victims)}_vs_{':'.join(opponents)}"


def make_configs(multi_score_ex):

    # ### Modifiers ###
    # You can use these with other configs.
    # Note: these set singleton dictionaries `exp_prefix = {k: None}`, where k is the
    # name of the named_config. These dictionaries are then merged by Sacred.
    # The prefixes are then sorted and concatenated to be used as part of the experiment name.
    # Note: only the keys are ever used, the values are ignored.

    # Accuracy

    @multi_score_ex.named_config
    def high_accuracy(score):
        score = dict(score)
        score["episodes"] = 1000
        score["num_env"] = 16
        exp_prefix = {"high_accuracy": None}  # noqa: F841

    @multi_score_ex.named_config
    def medium_accuracy(score):
        score = dict(score)
        score["episodes"] = 100
        score["num_env"] = 16
        exp_prefix = {"medium_accuracy": None}  # noqa: F841

    # Artifacts: activations and/or videos

    @multi_score_ex.named_config
    def save_activations(score):
        score = dict(score)
        score["episodes"] = None
        # Trajectory length varies a lot between environments and opponents; make sure we have
        # a consistent number of data points.
        score["timesteps"] = 20000
        score["record_traj"] = True
        score["transparent_params"] = {"ff_policy": True, "ff_value": True}
        score["record_traj_params"] = {
            "save_dir": "data/trajectories",
        }
        spec = {  # noqa: F841
            "config": {
                "record_traj_params": {
                    "agent_indices": tune.sample_from(
                        lambda spec: VICTIM_INDEX[spec.config[PATHS_AND_TYPES][0]]
                    ),
                }
            }
        }
        exp_prefix = {"activations": None}  # noqa: F841

    @multi_score_ex.named_config
    def video(score):
        score = dict(score)
        score["videos"] = True
        score["num_env"] = 1
        score["episodes"] = None
        score["timesteps"] = 60 * 60  # one minute of video @ 60 fps
        score["video_params"] = {"annotation_params": {"resolution": (1920, 1080), "font_size": 70}}
        exp_prefix = {"video": None}  # noqa: F841

    # Observation masking

    @multi_score_ex.named_config
    def mask_observations_of_victim():
        spec = {  # noqa: F841
            "config": {
                "mask_agent_index": tune.sample_from(
                    lambda spec: VICTIM_INDEX[spec.config[PATHS_AND_TYPES][0]]
                ),
            }
        }
        exp_prefix = {"victim_mask": None}  # noqa: F841

    @multi_score_ex.named_config
    def mask_observations_of_adversary():
        spec = {  # noqa: F841
            "config": {
                "mask_agent_index": tune.sample_from(
                    lambda spec: 1 - VICTIM_INDEX[spec.config[PATHS_AND_TYPES][0]]
                ),
            }
        }
        exp_prefix = {"adversary_mask": None}  # noqa: F841

    @multi_score_ex.named_config
    def mask_observations_with_zeros(score):
        score = dict(score)
        score["mask_agent_masking_type"] = "zeros"
        exp_prefix = {"zero": None}  # noqa: F841

    def _mask_observations_with_additive_noise(score, agent_noise):
        score["index_keys"] = ["mask_agent_masking_type", "mask_agent_noise"]
        score["mask_agent_masking_type"] = "additive_noise"
        return {
            "num_samples": 25,
            "mask_agent_noise": agent_noise,
        }

    @multi_score_ex.named_config
    def mask_observations_with_additive_noise(score):
        score = dict(score)
        spec = _mask_observations_with_additive_noise(  # noqa: F841
            score=score,
            agent_noise=tune.sample_from(lambda _: np.random.lognormal(mean=0.5, sigma=1.5)),
        )
        exp_prefix = {"additive_noise": None}  # noqa: F841

    @multi_score_ex.named_config
    def mask_observations_with_smaller_additive_noise(score):
        score = dict(score)
        spec = _mask_observations_with_additive_noise(  # noqa: F841
            score=score, agent_noise=tune.sample_from(lambda _: np.random.exponential(scale=1.0))
        )
        exp_prefix = {"smaller_additive_noise": None}  # noqa: F841

    # Adding noise to actions

    def _noise_actions(score):
        score["index_keys"] = ["noisy_agent_magnitude", "noisy_agent_index"]
        return {
            "num_samples": 25,
            "config": {
                "noisy_agent_magnitude": tune.sample_from(
                    lambda spec: np.random.lognormal(mean=0.5, sigma=1.5)
                )
            },
        }

    @multi_score_ex.named_config
    def noise_adversary_actions(score):
        score = dict(score)
        spec = _noise_actions(score)
        spec["config"]["noisy_agent_index"] = tune.sample_from(
            lambda spec: 1 - VICTIM_INDEX[spec.config[PATHS_AND_TYPES][0]]
        )
        exp_prefix = {"adversary_action_noise": None}  # noqa: F841

    @multi_score_ex.named_config
    def noise_victim_actions(score):
        score = dict(score)
        spec = _noise_actions(score)
        spec["config"]["noisy_agent_index"] = tune.sample_from(
            lambda spec: VICTIM_INDEX[spec.config[PATHS_AND_TYPES][0]]
        )
        exp_prefix = {"victim_action_noise": None}  # noqa: F841

    # ### Experimental Configs ###
    # These specify which agents to compare in which environments

    # Debugging

    @multi_score_ex.named_config
    def debug_one_each_type(score):
        """One Zoo agent from each environment, plus one opponent of each type.
           Intended for debugging purposes as a quick experiment that is still diverse.."""
        score = dict(score)
        score["episodes"] = 2
        spec = {
            "config": {
                PATHS_AND_TYPES: tune.grid_search(
                    [
                        cfg
                        for cfg in _gen_configs(victim_fns=[_zoo], opponent_fns=[_zoo])
                        if cfg.agent_a_path == "1" and cfg.agent_b_path == "1"
                    ]
                    + [
                        cfg
                        for cfg in _gen_configs(victim_fns=[_zoo], opponent_fns=[_fixed])
                        if cfg.agent_a_path == "1" or cfg.agent_b_path == "1"
                    ]
                    + _gen_configs(
                        victim_fns=[_zoo], opponent_fns=[_from_paths(get_adversary_paths())],
                    )[0:1],
                ),
            },
        }
        exp_suffix = "debug_one_each_type"

        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_score_ex.named_config
    def debug_two_agents(score):
        """Zoo1 and Rand in Kick and Defend. Very minimalistic test case."""
        score = dict(score)
        score["episodes"] = 2
        spec = {
            "config": {
                PATHS_AND_TYPES: tune.grid_search(
                    [EnvAgentConfig("multicomp/KickAndDefend-v0", "zoo", "1", "zoo", "1")]
                    + _gen_configs(
                        victim_fns=[_zoo],
                        opponent_fns=[_fixed],
                        envs=["multicomp/KickAndDefend-v0"],
                    )[0:1]
                ),
            }
        }
        exp_suffix = "debug_two_agents"

        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_score_ex.named_config
    def normal():
        victims = ["zoo"]
        opponents = ["zoo", "fixed", "adversary"]
        exp_suffix = "normal"
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_score_ex.named_config
    def defenses():
        victims = [
            "zoo",
            "json:multi_train/finetune_defense_single_mlp/",
            "json:multi_train/finetune_defense_dual_mlp/",
        ]
        opponents = [
            "zoo",
            "fixed",
            "json:multi_train/paper/",
            "json:multi_train/adv_from_scratch_against_finetune_defense_single_mlp/",
            "json:multi_train/adv_from_scratch_against_finetune_defense_dual_mlp/",
        ]
        envs = ["multicomp/YouShallNotPassHumans-v0"]
        exp_suffix = "defense"
        _ = locals()  # quieten flake8 unused variable warning
        del _

    # Standard experiments

    @multi_score_ex.config
    def default_placeholders():
        exp_name = None
        envs = None
        victims = []
        opponents = []
        exp_prefix = {}
        exp_suffix = None

        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_score_ex.config
    def default_spec(spec, exp_suffix, envs, victims, opponents, exp_prefix):
        """Compare victims to opponents."""
        if "config" not in spec and not victims:
            raise ValueError(
                "You must use a modifier config to specify the " "victim policies to compare."
            )
        if "config" not in spec and not opponents:
            raise ValueError(
                "You must use a modifier config to specify the " "opponent policies to compare."
            )

        if victims and opponents:
            spec = {
                "config": {
                    PATHS_AND_TYPES: tune.grid_search(
                        _gen_configs(
                            victim_fns=[_to_fn(cfg) for cfg in victims],
                            opponent_fns=[_to_fn(cfg) for cfg in opponents],
                            envs=None if envs is None else envs,
                        )
                    ),
                }
            }

            if exp_suffix is None:
                exp_suffix = _make_default_exp_suffix(victims, opponents)

        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_score_ex.config
    def prefix_exp_name(exp_suffix, exp_prefix):
        exp_name = (  # noqa: F841
            f"{':'.join(sorted(exp_prefix.keys()))}-" if exp_prefix else ""
        ) + exp_suffix
