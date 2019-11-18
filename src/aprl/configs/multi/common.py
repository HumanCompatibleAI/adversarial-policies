import json
import os

from aprl.envs import gym_compete

DATA_LOCATION = os.path.abspath(os.environ.get("DATA_LOC", "data"))

BANSAL_ENVS = ["multicomp/" + env for env in gym_compete.POLICY_STATEFUL.keys()]
BANSAL_ENVS += ["multicomp/SumoHumansAutoContact-v0", "multicomp/SumoAntsAutoContact-v0"]
BANSAL_GOOD_ENVS = [  # Environments well-suited to adversarial attacks
    "multicomp/KickAndDefend-v0",
    "multicomp/SumoHumansAutoContact-v0",
    "multicomp/SumoAntsAutoContact-v0",
    "multicomp/YouShallNotPassHumans-v0",
]


def get_adversary_paths():
    """Load adversary paths from ADVERSARY_PATHS environment variable.

    We can't make this a Sacred config param since Sacred named_configs execute before configs.
    """
    path = os.getenv("ADVERSARY_PATHS")
    if path is None:
        raise ValueError(
            "Specify path to JSON file containing adversaries in ADVERSARY_PATHS "
            "environment variable. (Run 'experiments/modelfree/highest_win_rate.py'"
            "to generate this.)"
        )
    with open(path, "r") as f:
        return json.load(f)["policies"]
