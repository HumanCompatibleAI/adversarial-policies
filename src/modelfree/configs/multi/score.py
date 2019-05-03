"""Named configs for modelfree.multi.score."""

import json
import logging
import os.path
import pkgutil

from ray import tune

from modelfree.configs.multi.common import BANSAL_GOOD_ENVS, VICTIM_INDEX
from modelfree.envs import gym_compete

logger = logging.getLogger('modelfree.configs.multi.score')


def _gen_configs(victim_fn, adversary_fn, envs=None):
    if envs is None:
        envs = BANSAL_GOOD_ENVS

    configs = []
    for env in envs:
        num_zoo = gym_compete.num_zoo_policies(env)
        victim_index = VICTIM_INDEX[env]
        for victim_id in range(num_zoo):
            for adversary_id in range(num_zoo):
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
    return 'zoo', str(our_id + 1)


def _env_agents(envs=None):
    return _gen_configs(victim_fn=_zoo_identity, adversary_fn=_zoo_identity, envs=envs)


def _fixed_vs_victim(fixed_type, envs=None):
    def adversary_fn(_env, _victim_index, _our_id, _opponent_id):
        return fixed_type, 'none'

    return _gen_configs(victim_fn=_zoo_identity, adversary_fn=adversary_fn, envs=envs)


def _adversary_vs_victims(adversary_type, adversary_paths, envs=None):
    def adversary_fn(env, victim_index, our_id, _opponent_id):
        victim_index = str(victim_index)
        our_id = str(our_id + 1)
        path = adversary_paths.get(env, {}).get(victim_index, {}).get(our_id)
        if path is None:
            logger.warning(f"Missing adversary path {env} {victim_index} {our_id}")
            return None
        else:
            return adversary_type, os.path.abspath(path)

    return _gen_configs(victim_fn=_zoo_identity, adversary_fn=adversary_fn, envs=envs)


def load_json(path):
    content = pkgutil.get_data('modelfree', path)
    return json.loads(content)


PATHS_AND_TYPES = 'env_name:agent_a_type:agent_a_path:agent_b_type:agent_b_path'


def _get_adversary_paths():
    # Sacred named_configs execute before configs, so we can't make this a Sacred config param.
    path = os.getenv('ADVERSARY_PATHS')
    if path is None:
        raise ValueError("Specify path to JSON file containing adversaries in ADVERSARY_PATHS "
                         "environment variable. (Run 'experiments/modelfree/highest_win_rate.py'"
                         "to generate this.)")
    with open(path, 'r') as f:
        return json.load(f)['policies']


def make_configs(multi_score_ex):
    @multi_score_ex.config
    def high_accuracy(score):
        score = dict(score)
        score['episodes'] = 1000
        score['num_env'] = 16

    @multi_score_ex.named_config
    def save_activations(exp_name, score, spec):
        score = dict(score)
        score['episodes'] = 2  # TODO: check with Cody whether this is a sensible default
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
    def debug(score):
        score = dict(score)
        score['episodes'] = 2
        spec = {
            'config': {
                PATHS_AND_TYPES: tune.grid_search(
                    _env_agents()[0:1] +
                    _fixed_vs_victim('zero')[0:1] +
                    _adversary_vs_victims('ppo2', _get_adversary_paths())[0:1]
                ),
            }
        }
        exp_name = 'debug'

        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_score_ex.named_config
    def zoo_baseline(score):
        """Try all pre-trained policies from Bansal et al's gym_compete zoo against each other."""
        score = dict(score)
        spec = {
            'config': {
                PATHS_AND_TYPES: tune.grid_search(_env_agents()),
            },
        }
        exp_name = 'zoo_baseline'

        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_score_ex.named_config
    def fixed_baseline(score):
        """Try zero-agent and random-agent against pre-trained zoo policies."""
        score = dict(score)
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
    def adversary_transfer(score):
        """Do adversarial policies trained on victim X transfer to victim Y?"""
        score = dict(score)
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
