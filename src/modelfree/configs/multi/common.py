import collections

from modelfree.envs import gym_compete

BANSAL_ENVS = ['multicomp/' + env for env in gym_compete.POLICY_STATEFUL.keys()]
BANSAL_ENVS += ['multicomp/SumoHumansAutoContact-v0', 'multicomp/SumoAntsAutoContact-v0']
BANSAL_GOOD_ENVS = [  # Environments well-suited to adversarial attacks
    'multicomp/KickAndDefend-v0',
    'multicomp/SumoHumansAutoContact-v0',
    'multicomp/SumoAntsAutoContact-v0',
    'multicomp/YouShallNotPassHumans-v0',
]

VICTIM_INDEX = collections.defaultdict(lambda: 0)
VICTIM_INDEX.update({
    # YouShallNotPass: 1 is the walker, 0 is the blocker agent.
    # An adversarial walker makes little sense, but a blocker can be adversarial.
    'multicomp/YouShallNotPassHumans-v0': 1,
})
