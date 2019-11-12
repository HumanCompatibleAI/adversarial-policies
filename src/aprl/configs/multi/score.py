"""Named configs for aprl.multi.score."""

import json
import logging
import os.path

import numpy as np
from ray import tune

from aprl.configs.multi.common import BANSAL_GOOD_ENVS
from aprl.envs import VICTIM_INDEX, gym_compete

logger = logging.getLogger('aprl.configs.multi.score')


def _gen_configs(victim_fn, adversary_fn, agents=None):
    """Helper function to generate configs.

    :param victim_fn: (callable) called with environment name, victim index, victim id,
                      and adversary ID. Should return a victim_type, victim_path tuple.
    :param adversary_fn: (callable) analogous to victim_fn.
    :param agents: (dict or None) optional dict of environments to (victim_ids, adversary_ids)
                   where victim_ids and adversary_ids are iterables. If None, then defaults to
                   {victim,adversary}_ids = range(num_zoo_policies) for each
                   env in BANSAL_GOOD_ENVS.

    :return A list of (env, agent_a_type, agent_a_path, agent_b_type, agent_b_path).
    """
    if agents is None:
        agents = {}
        for env in BANSAL_GOOD_ENVS:
            num_zoo = gym_compete.num_zoo_policies(env)
            agents[env] = (range(1, num_zoo + 1), range(1, num_zoo + 1))

    configs = []
    for env, (victim_ids, adversary_ids) in agents.items():
        victim_index = VICTIM_INDEX[env]
        for victim_id in victim_ids:
            for adversary_id in adversary_ids:
                victim_type, victim_path = victim_fn(env, victim_index, victim_id, adversary_id)
                adversary = adversary_fn(env, victim_index, adversary_id, victim_id)
                if adversary is None:
                    continue
                adversary_type, adversary_path = adversary

                if victim_index == 0:
                    cfg = (env, victim_type, victim_path, adversary_type, adversary_path)
                elif victim_index == 1:
                    cfg = (env, adversary_type, adversary_path, victim_type, victim_path)
                else:
                    raise ValueError(f"Victim index '{victim_index}' out of range")

                if cfg not in configs:
                    configs.append(cfg)

    return configs


def _zoo_identity(_env, _victim_index, our_id, _opponent_id):
    return 'zoo', str(our_id)


def _env_agents(**kwargs):
    return _gen_configs(victim_fn=_zoo_identity, adversary_fn=_zoo_identity, **kwargs)


def _fixed_vs_victim(fixed_type, envs=None):
    def adversary_fn(_env, _victim_index, _our_id, _opponent_id):
        return fixed_type, 'none'

    return _gen_configs(victim_fn=_zoo_identity, adversary_fn=adversary_fn, agents=envs)


def _adversary_vs_victims(adversary_type, adversary_paths, no_transfer=False, **kwargs):
    """Generates configs for adversaries.

    :param adversary_type: (str) the policy type of the adversary.
    :param adversary_paths: (dict) paths to adversaries, loaded by _get_adversary_paths
    :param no_transfer: (bool) when True, only return the adversary trained against that victim;
                                otherwise, returns all adversaries (useful for testing transfer).
    :param kwargs: (dict) passed through to `gen_configs`.
    """
    def adversary_fn(env, victim_index, our_id, opponent_id):
        if no_transfer and our_id != opponent_id:
            return None

        victim_index = str(victim_index)
        our_id = str(our_id)

        path = adversary_paths.get(env, {}).get(victim_index, {}).get(our_id)
        if path is None:
            logger.warning(f"Missing adversary path {env} {victim_index} {our_id}")
            return None
        else:
            return adversary_type, os.path.abspath(path)

    return _gen_configs(victim_fn=_zoo_identity, adversary_fn=adversary_fn, **kwargs)


PATHS_AND_TYPES = 'env_name:agent_a_type:agent_a_path:agent_b_type:agent_b_path'


def _get_adversary_paths():
    # Sacred named_configs execute before configs, so we can't make this a Sacred config param.
    path = os.getenv('ADVERSARY_PATHS')
    if path is None:
        raise ValueError("Specify path to JSON file containing adversaries in ADVERSARY_PATHS "
                         "environment variable. (Run 'experiments/highest_win_rate.py'"
                         "to generate this.)")
    with open(path, 'r') as f:
        return json.load(f)['policies']


def _summary_paths():
    summary_agents = {
        # ([median victim Zoo id], [best Zoo opponent ID]) -- note 0-indexed
        'multicomp/KickAndDefend-v0': ([2], [2]),
        'multicomp/YouShallNotPassHumans-v0': ([1], [1]),
        'multicomp/SumoHumansAutoContact-v0': ([2], [3]),
    }
    adversary_agents = {env: (victim_ids, victim_ids)
                        for env, (victim_ids, _opponent_ids) in summary_agents.items()}
    adversaries = _adversary_vs_victims('ppo2', _get_adversary_paths(),
                                        no_transfer=True, agents=adversary_agents)
    zoo = _env_agents(agents=summary_agents)
    return adversaries + zoo


def make_configs(multi_score_ex):
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

    def _mask_obserations_with_additive_noise(score, spec):
        score['index_keys'] = ['mask_agent_masking_type', 'mask_agent_noise']
        score['mask_agent_masking_type'] = 'additive_noise'
        spec['num_samples'] = 25

    @multi_score_ex.named_config
    def mask_observations_with_additive_noise(exp_name, score, spec):
        score = dict(score)
        _mask_obserations_with_additive_noise(score, spec)
        spec['config']['mask_agent_noise'] = tune.sample_from(
            lambda spec: np.random.lognormal(mean=0.5, sigma=1.5)
        )
        exp_name = 'additive_noise_' + exp_name

    @multi_score_ex.named_config
    def mask_observations_with_smaller_additive_noise(exp_name, score, spec):
        score = dict(score)
        _mask_obserations_with_additive_noise(score, spec)
        spec['config']['mask_agent_noise'] = tune.sample_from(
            lambda spec: np.random.exponential(scale=1.0)
        )
        exp_name = 'smaller_additive_noise_' + exp_name

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

    @multi_score_ex.named_config
    def debug_one_each_type(score):
        """One Zoo agent from each environment, plus one opponent of each type.
           Intended for debugging purposes as a quick experiment that is still diverse.."""
        score = dict(score)
        score['episodes'] = 2
        spec = {
            'config': {
                PATHS_AND_TYPES: tune.grid_search(
                    _env_agents(agents={env: ([1], [1]) for env in BANSAL_GOOD_ENVS}) +
                    _fixed_vs_victim('zero')[0:1] +
                    _fixed_vs_victim('random')[0:1] +
                    _adversary_vs_victims('ppo2', _get_adversary_paths())[0:1]
                ),
            }
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
                    _env_agents(agents={BANSAL_GOOD_ENVS[0]: ([1], [1])}) +
                    _fixed_vs_victim('random')[0:1],
                ),
            }
        }
        exp_name = 'debug_two_agents'

        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_score_ex.named_config
    def zoo_baseline():
        """Try all pre-trained policies from Bansal et al's gym_compete zoo against each other."""
        spec = {
            'config': {
                PATHS_AND_TYPES: tune.grid_search(_env_agents()),
            },
        }
        exp_name = 'zoo_baseline'

        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_score_ex.named_config
    def fixed_baseline():
        """Try zero-agent and random-agent against pre-trained zoo policies."""
        spec = {
            'config': {
                PATHS_AND_TYPES: tune.grid_search(_fixed_vs_victim('random') +
                                                  _fixed_vs_victim('zero')),
            }
        }
        exp_name = 'fixed_baseline'

        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_score_ex.named_config
    def random_baseline():
        """Try random-agent against pre-trained zoo policies."""
        spec = {
            'config': {
                PATHS_AND_TYPES: tune.grid_search(_fixed_vs_victim('random')),
            }
        }
        exp_name = 'random_baseline'

        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_score_ex.named_config
    def adversary_transfer():
        """Do adversarial policies trained on victim X transfer to victim Y?"""
        spec = {
            'config': {
                PATHS_AND_TYPES: tune.grid_search(
                    _adversary_vs_victims('ppo2', _get_adversary_paths())
                ),
            }
        }
        exp_name = 'adversary_transfer'

        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_score_ex.named_config
    def adversary_trained():
        """Try adversaries against the victim they were trained against."""
        spec = {
            'config': {
                PATHS_AND_TYPES: tune.grid_search(
                    _adversary_vs_victims('ppo2', _get_adversary_paths(), no_transfer=True)
                ),
            }
        }
        exp_name = 'adversary_trained'

        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_score_ex.named_config
    def summary():
        """Try adversaries against the victim they were trained against."""
        spec = {
            'config': {
                PATHS_AND_TYPES: tune.grid_search(_summary_paths()),
            }
        }
        exp_name = 'summary'

        _ = locals()  # quieten flake8 unused variable warning
        del _
