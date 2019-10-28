import json
import os

from modelfree.envs import gym_compete

BANSAL_ENVS = ['multicomp/' + env for env in gym_compete.POLICY_STATEFUL.keys()]
BANSAL_ENVS += ['multicomp/SumoHumansAutoContact-v0', 'multicomp/SumoAntsAutoContact-v0']
BANSAL_GOOD_ENVS = [  # Environments well-suited to adversarial attacks
    'multicomp/KickAndDefend-v0',
    'multicomp/SumoHumansAutoContact-v0',
    'multicomp/SumoAntsAutoContact-v0',
    'multicomp/YouShallNotPassHumans-v0',
]


def _get_adversary_paths():
    # Sacred named_configs execute before configs, so we can't make this a Sacred config param.
    path = os.getenv('ADVERSARY_PATHS')
    if path is None:
        raise ValueError("Specify path to JSON file containing adversaries in ADVERSARY_PATHS "
                         "environment variable. (Run 'experiments/modelfree/highest_win_rate.py'"
                         "to generate this.)")
    with open(path, 'r') as f:
        return json.load(f)['policies']
