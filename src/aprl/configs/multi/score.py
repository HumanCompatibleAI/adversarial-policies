"""Named configs for aprl.multi.score."""

import logging
import os.path
from typing import Callable, Iterable, List, NamedTuple, Optional, Tuple

import numpy as np
from ray import tune

from aprl.configs.multi.common import BANSAL_GOOD_ENVS, DATA_LOCATION, get_adversary_paths
from aprl.envs import VICTIM_INDEX, gym_compete

AgentConfigGenFn = Callable[[str, int], Iterable[Tuple[str, str]]]


class EnvAgentConfig(NamedTuple):
    env_name: str
    agent_a_type: str
    agent_a_path: str
    agent_b_type: str
    agent_b_path: str


PATHS_AND_TYPES = ':'.join(EnvAgentConfig._fields)

logger = logging.getLogger('aprl.configs.multi.score')


def _zoo(env, agent_index):
    """Returns all Zoo policies in `env`."""
    del agent_index
    num_zoo = gym_compete.num_zoo_policies(env)
    return [('zoo', str(i)) for i in range(1, num_zoo + 1)]


def _adversaries(adversary_paths):
    """Returns a function that returns all adversaries in `adversary_paths`."""
    def helper(env, agent_index):
        """Returns all adversaries in `env` playing in index `agent_index`."""
        victim_index = 1 - agent_index
        paths = adversary_paths.get(env, {}).get(str(victim_index))
        if paths is None:
            logger.warning(f"Missing adversary path in '{env}' for index '{agent_index}'")
            return []
        else:
            return [('ppo2', os.path.join(DATA_LOCATION, path)) for path in paths.values()]

    return helper


def _fixed(env, agent_index):
    """Returns all baseline, environment-independent policies."""
    del env, agent_index
    return [('zero', 'none'), ('random', 'none')]


def _gen_configs(victim_fn: AgentConfigGenFn,
                 adversary_fn: AgentConfigGenFn,
                 envs: Optional[Iterable[str]] = None,
                 ) -> List[EnvAgentConfig]:
    """Helper function to generate configs.

    :param victim_fn: (callable) called with environment name and agent index.
    :param adversary_fn: as above.
    :param envs: optionally, a list of environments to generate configs for.

    :return A list of (env, agent_a_type, agent_a_path, agent_b_type, agent_b_path).
    """
    if envs is None:
        envs = BANSAL_GOOD_ENVS

    configs = []
    for env in envs:
        victim_index = VICTIM_INDEX[env]
        victims = victim_fn(env, victim_index)
        opponents = adversary_fn(env, 1 - victim_index)

        for victim_type, victim_path in victims:
            for adversary_type, adversary_path in opponents:
                if victim_index == 0:
                    cfg = EnvAgentConfig(env, victim_type, victim_path,
                                         adversary_type, adversary_path)
                elif victim_index == 1:
                    cfg = EnvAgentConfig(env, adversary_type, adversary_path,
                                         victim_type, victim_path)
                else:
                    raise ValueError(f"Victim index '{victim_index}' out of range")

                if cfg not in configs:
                    configs.append(cfg)

    return configs


def make_configs(multi_score_ex):

    # ### Modifiers ###
    # You can use these with other configs.

    # Accuracy

    @multi_score_ex.named_config
    def high_accuracy(exp_name, score):
        score = dict(score)
        score['episodes'] = 1000
        score['num_env'] = 16
        exp_name = 'high_accuracy_' + exp_name

    @multi_score_ex.named_config
    def medium_accuracy(exp_name, score):
        score = dict(score)
        score['episodes'] = 100
        score['num_env'] = 16
        exp_name = 'medium_accuracy_' + exp_name
        _ = locals()
        del _

    # Artifacts: activations and/or videos

    @multi_score_ex.named_config
    def save_activations(exp_name, score, spec):
        score = dict(score)
        score['episodes'] = None
        # Trajectory length varies a lot between environments and opponents; make sure we have
        # a consistent number of data points.
        score['timesteps'] = 20000
        score['record_traj'] = True
        score['transparent_params'] = {'ff_policy': True, 'ff_value': True}
        score['record_traj_params'] = {
            'save_dir': 'data/trajectories',
        }
        spec['config']['record_traj_params'] = {
            'agent_indices': tune.sample_from(
                lambda spec: VICTIM_INDEX[spec.config[PATHS_AND_TYPES][0]]
            ),
        }
        exp_name = 'activations_' + exp_name

    @multi_score_ex.named_config
    def video(exp_name, score):
        score = dict(score)
        score['videos'] = True
        score['num_env'] = 1
        score['episodes'] = None
        score['timesteps'] = 60 * 60  # one minute of video @ 60 fps
        score['video_params'] = {
            'annotation_params': {
                'resolution': (1920, 1080),
                'font_size': 70,
            }
        }
        exp_name = 'video_' + exp_name  # noqa: F401

    # Observation masking

    @multi_score_ex.named_config
    def mask_observations_of_victim(exp_name, spec):
        spec['config']['mask_agent_index'] = tune.sample_from(
            lambda spec: VICTIM_INDEX[spec.config[PATHS_AND_TYPES][0]]
        )
        exp_name = 'victim_mask_' + exp_name

    @multi_score_ex.named_config
    def mask_observations_of_adversary(exp_name, spec):
        spec['config']['mask_agent_index'] = tune.sample_from(
            lambda spec: 1 - VICTIM_INDEX[spec.config[PATHS_AND_TYPES][0]]
        )
        exp_name = 'adversary_mask_' + exp_name

    @multi_score_ex.named_config
    def mask_observations_with_zeros(exp_name, score):
        score = dict(score)
        score['mask_agent_masking_type'] = 'zeros'
        exp_name = 'zero_' + exp_name

    def _mask_observations_with_additive_noise(score, spec):
        score['index_keys'] = ['mask_agent_masking_type', 'mask_agent_noise']
        score['mask_agent_masking_type'] = 'additive_noise'
        spec['num_samples'] = 25

    @multi_score_ex.named_config
    def mask_observations_with_additive_noise(exp_name, score, spec):
        score = dict(score)
        _mask_observations_with_additive_noise(score, spec)
        spec['config']['mask_agent_noise'] = tune.sample_from(
            lambda spec: np.random.lognormal(mean=0.5, sigma=1.5)
        )
        exp_name = 'additive_noise_' + exp_name

    @multi_score_ex.named_config
    def mask_observations_with_smaller_additive_noise(exp_name, score, spec):
        score = dict(score)
        _mask_observations_with_additive_noise(score, spec)
        spec['config']['mask_agent_noise'] = tune.sample_from(
            lambda spec: np.random.exponential(scale=1.0)
        )
        exp_name = 'smaller_additive_noise_' + exp_name

    # Adding noise to actions

    def _noise_actions(score, spec):
        score['index_keys'] = ['noisy_agent_magnitude', 'noisy_agent_index']
        spec['num_samples'] = 25
        spec['config']['noisy_agent_magnitude'] = tune.sample_from(
            lambda spec: np.random.lognormal(mean=0.5, sigma=1.5)
        )

    @multi_score_ex.named_config
    def noise_adversary_actions(exp_name, score, spec):
        score = dict(score)
        _noise_actions(score, spec)
        spec['config']['noisy_agent_index'] = tune.sample_from(
            lambda spec: 1 - VICTIM_INDEX[spec.config[PATHS_AND_TYPES][0]]
        )
        exp_name = 'adversary_action_noise_' + exp_name

    @multi_score_ex.named_config
    def noise_victim_actions(exp_name, score, spec):
        score = dict(score)
        _noise_actions(score, spec)
        spec['config']['noisy_agent_index'] = tune.sample_from(
            lambda spec: VICTIM_INDEX[spec.config[PATHS_AND_TYPES][0]]
        )
        exp_name = 'victim_action_noise_' + exp_name

    # ### Experimental Configs ###
    # These specify which agents to compare in which environments

    # Debugging

    @multi_score_ex.named_config
    def debug_one_each_type(score):
        """One Zoo agent from each environment, plus one opponent of each type.
           Intended for debugging purposes as a quick experiment that is still diverse.."""
        score = dict(score)
        score['episodes'] = 2
        spec = {
            'config': {
                PATHS_AND_TYPES: tune.grid_search(
                    [cfg for cfg in _gen_configs(victim_fn=_zoo, adversary_fn=_zoo)
                     if cfg.agent_a_path == '1' and cfg.agent_b_path == '1'] +
                    [cfg for cfg in _gen_configs(victim_fn=_zoo, adversary_fn=_fixed)] +
                    _gen_configs(
                        victim_fn=_zoo, adversary_fn=_adversaries(get_adversary_paths()),
                    )[0:1],
                ),
            },
        }
        exp_name = 'debug_one_each_type'

        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_score_ex.named_config
    def debug_two_agents(score):
        """Zoo1 and Rand in Kick and Defend. Very minimalistic test case."""
        score = dict(score)
        score['episodes'] = 2
        spec = {
            'config': {
                PATHS_AND_TYPES: tune.grid_search(
                    [EnvAgentConfig('multicomp/KickAndDefend-v0', 'zoo', '1', 'zoo', '1')] +
                    _gen_configs(victim_fn=_zoo, adversary_fn=_fixed)[0:1]
                ),
            }
        }
        exp_name = 'debug_two_agents'

        _ = locals()  # quieten flake8 unused variable warning
        del _

    # Standard experiments

    @multi_score_ex.named_config
    def zoo_victim():
        victim_fn = _zoo  # noqa: F841
        victim_name = 'zoo'  # noqa: F841

    @multi_score_ex.named_config
    def zoo_opponent():
        opponent_fn = _zoo  # noqa: F841
        opponent_name = 'zoo'  # noqa: F841

    @multi_score_ex.named_config
    def fixed_opponent():
        opponent_fn = _fixed  # noqa: F841
        opponent_name = 'fixed'  # noqa: F841

    @multi_score_ex.named_config
    def adversary_opponent():
        # TODO(adam): different adversary paths?
        opponent_fn = _adversaries(get_adversary_paths())  # noqa: F841
        opponent_name = 'adversary'  # noqa: F841

    @multi_score_ex.config
    def default_placeholders():
        spec = None
        victim_fn = None
        victim_name = None
        opponent_fn = None
        opponent_name = None

        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_score_ex.config
    def default_spec(spec, victim_fn, victim_name, opponent_fn, opponent_name):
        """Compare victims to opponents."""
        if spec is None:
            if victim_fn is None:
                raise ValueError("You must specify the victims to compare with a modifier.")
            if opponent_fn is None:
                raise ValueError("You must specify the opponents to compare with a modifier.")

            spec = {
                'config': {
                    PATHS_AND_TYPES: tune.grid_search(
                        _gen_configs(victim_fn=victim_fn, adversary_fn=opponent_fn),
                    ),
                }
            }
            exp_name = f'{victim_name}_vs_{opponent_name}'

        _ = locals()  # quieten flake8 unused variable warning
        del _
