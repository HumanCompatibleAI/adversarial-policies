import collections
import ctypes
import re

from PIL import Image, ImageDraw, ImageFont
import gym
import numpy as np

from modelfree.configs.multi.common import VICTIM_INDEX
from modelfree.envs.gym_compete import env_name_to_canonical, game_outcome

VICTIM_OPPONENT_COLORS = {
    'Victim': (77, 175, 74, 255),
    'Opponent': (228, 26, 28, 255),
    'Ties': (0, 0, 0, 255),
}


POLICY_TYPE_COLORS = {
    'zoo': (128, 128, 128, 255),  # grey
    'ppo2': (255, 0, 0, 255),  # red
    'zero': (0, 0, 0, 255),  # black
    'random': (0, 0, 255, 255),  # blue
}

# Brewer Accent
PATH_COLORS = {
    '1': (127, 201, 127, 255),
    '2': (190, 174, 212, 255),
    '3': (253, 192, 134, 255),
    '4': (255, 255, 153, 255),
}


def body_color(is_victim, agent_type, agent_path):
    key = 'Victim' if is_victim else 'Opponent'
    return VICTIM_OPPONENT_COLORS[key]


def head_color(is_victim, agent_type, agent_path):
    return POLICY_TYPE_COLORS[agent_type]


GEOM_MAPPINGS = {
    '*': body_color,
    # Ant
    'torso_geom': head_color,
    # Humanoid
    'torso1': head_color,
    'uwaist': head_color,
    'lwaist': head_color,
    'head': head_color,
}


def set_geom_rgba(model, value):
    """Does what model.geom_rgba = ... should do, but doesn't because of a bug in mujoco-py."""
    val_ptr = np.array(value, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ctypes.memmove(model._wrapped.contents.geom_rgba, val_ptr,
                   model.ngeom * 4 * ctypes.sizeof(ctypes.c_float))


def set_geom_colors(model, patterns):
    names = [name.decode('utf-8') for name in model.geom_names]
    patterns = {re.compile(k): tuple(x / 255 for x in v) for k, v in patterns.items()}

    modified = np.array(model.geom_rgba)
    for row_idx, name in enumerate(names):
        for pattern, color in patterns.items():
            if pattern.match(name):
                modified[row_idx, :] = color

    set_geom_rgba(model, modified)


CAMERA_CONFIG = {
    # From behind
    'KickAndDefend-v0': {'azimuth': 0, 'distance': 10, 'elevation': -23},
    # From side, slightly behind (runner always goes forward, never back)
    'YouShallNotPassHumans-v0': {'azimuth': 110, 'distance': 9, 'elevation': -21},
    # Defaults fine for Sumo
    'SumoHumans-v0': {},
    'SumoAnts-v0': {},
}


class PrettyGymCompete(gym.Wrapper):
    def __init__(self, env, env_name, agent_a_type, agent_a_path, agent_b_type, agent_b_path,
                 font="times", font_size=24, spacing=0.02, color=(0, 0, 0, 255)):
        super(PrettyGymCompete, self).__init__(env)

        # Set agent colors
        self.env_name = env_name
        self.victim_index = VICTIM_INDEX[env_name]
        agent_mapping = {
            'agent0': (0 == self.victim_index, agent_a_type, agent_a_path),
            'agent1': (1 == self.victim_index, agent_b_type, agent_b_path)
        }
        color_patterns = {f'{agent_key}/{geom_key}': geom_fn(*agent_val)
                          for geom_key, geom_fn in GEOM_MAPPINGS.items()
                          for agent_key, agent_val in agent_mapping.items()}
        set_geom_colors(self.env.unwrapped.env_scene.model, color_patterns)

        # Text overlay
        self.font = ImageFont.truetype(f'{font}.ttf', font_size)
        self.font_bold = ImageFont.truetype(f'{font}bi.ttf', font_size)
        self.spacing = spacing
        self.color = color

        # Internal state
        self.result = collections.defaultdict(int)
        self.changed = collections.defaultdict(int)
        self.last_won = None

        self.env.unwrapped.env_scene._get_viewer()  # force viewer to start
        self.camera_setup()

    def camera_setup(self):
        canonical_env_name = env_name_to_canonical(self.env_name)
        camera_cfg = CAMERA_CONFIG[canonical_env_name]
        viewer = self.env.unwrapped.env_scene.viewer

        for k, v in camera_cfg.items():
            setattr(viewer.cam, k, v)

    def _reset(self):
        ob = super(PrettyGymCompete, self)._reset()

        if self.env.unwrapped.env_scene.viewer is not None:
            self.camera_setup()

        return ob

    def _step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done:
            # TODO: code duplication
            winner = game_outcome(info)
            if winner is None:
                k = 'ties'
            else:
                k = f'win{winner}'
            self.result[k] += 1
            self.last_won = k
        return obs, rew, done, info

    def _render(self, mode='human', close=False):
        res = self.env.render(mode, close)
        if mode == 'rgb_array':
            img = Image.fromarray(res)
            draw = ImageDraw.Draw(img)

            width, height = img.size
            ypos = height * 0.9
            texts = collections.OrderedDict([
                (f'win{1 - self.victim_index}', 'Opponent'),
                (f'win{self.victim_index}', 'Victim'),
                ('ties', 'Ties'),
            ])

            to_draw = []
            for k, label in texts.items():
                msg = f'{label} = {self.result[k]}'
                color = VICTIM_OPPONENT_COLORS[label]
                font = self.font_bold if k == self.last_won else self.font
                to_draw.append((msg, font, color))

            lengths = [font.getsize(msg)[0] + self.spacing * width
                       for (msg, font, color) in to_draw]
            total_length = sum(lengths)
            xpos = (width - total_length) / 2

            for (msg, font, color), length in zip(to_draw, lengths):
                draw.text((xpos, ypos), msg, font=font, fill=color)
                xpos += length

            res = np.array(img)

        return res
