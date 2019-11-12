# flake8: noqa: F401
import collections

from gym.envs.registration import register
from pkg_resources import resource_filename

# Low-dimensional multi-agent environments

register(
    id='aprl/CrowdedLine-v0',
    entry_point='aprl.envs.crowded_line:CrowdedLineEnv',
    max_episode_steps=200,
    reward_threshold=0,
    kwargs={'num_agents': 3},
)

register(
    id='aprl/IteratedMatchingPennies-v0',
    entry_point='aprl.envs.matrix_game:IteratedMatchingPenniesEnv',
    max_episode_steps=200,
    reward_threshold=100,
)

register(
    id='aprl/RockPaperScissors-v0',
    entry_point='aprl.envs.matrix_game:RockPaperScissorsEnv',
    max_episode_steps=200,
    reward_threshold=100,
)

# Our variants of environments from gym_compete

register(
    id='multicomp/SumoHumansAutoContact-v0',
    entry_point='aprl.envs.sumo_auto_contact:SumoAutoContactEnv',
    kwargs={'agent_names': ['humanoid_fighter', 'humanoid_fighter'],
            'scene_xml_path': resource_filename(
                'gym_compete',
                'new_envs/assets/world_body_arena.humanoid_body.humanoid_body.xml'
            ),
            'init_pos': [(-1, 0, 1.4), (1, 0, 1.4)],
            'max_episode_steps': 500,
            'min_radius': 1.5,
            'max_radius': 3.5
            },
)

register(
    id='multicomp/SumoAntsAutoContact-v0',
    entry_point='aprl.envs.sumo_auto_contact:SumoAutoContactEnv',
    kwargs={'agent_names': ['ant_fighter', 'ant_fighter'],
            'scene_xml_path': resource_filename(
                'gym_compete',
                'new_envs/assets/world_body_arena.ant_body.ant_body.xml'
            ),
            'world_xml_path': resource_filename(
                'gym_compete',
                'new_envs/assets/world_body_arena.xml'
            ),
            'init_pos': [(-1, 0, 2.5), (1, 0, 2.5)],
            'max_episode_steps': 500,
            'min_radius': 2.5,
            'max_radius': 4.5
            },
)


# Which index does the victim play in?
# This is really an experiment parameter rather than an environment parameter.
# However, it's used so widely (training, evaluation, figure and video generation) and is
# held fixed across all experiments it's convenient to treat it as if it is static.
VICTIM_INDEX = collections.defaultdict(lambda: 0)
VICTIM_INDEX.update({
    # YouShallNotPass: 1 is the walker, 0 is the blocker agent.
    # An adversarial walker makes little sense, but a blocker can be adversarial.
    'multicomp/YouShallNotPassHumans-v0': 1,
})
