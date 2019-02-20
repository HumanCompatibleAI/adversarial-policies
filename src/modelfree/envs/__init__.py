from gym.envs.registration import register
import os

register(
    id='multicomp/SumoHumansAutoContact-v0',
    entry_point='modelfree.envs.sumo_auto_contact:SumoAutoContactEnv',
    kwargs={'agent_names': ['humanoid_fighter', 'humanoid_fighter'],
            'scene_xml_path': os.path.join(
                os.path.dirname(__file__),
                "world_body_arena.humanoid_body.humanoid_body.xml"
            ),
            'init_pos': [(-1, 0, 1.4), (1, 0, 1.4)],
            'max_episode_steps': 500,
            'min_radius': 1.5,
            'max_radius': 3.5
            },
)
