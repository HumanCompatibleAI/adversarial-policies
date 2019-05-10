import collections
import ctypes
import math
import os.path as osp
import re

from PIL import Image, ImageDraw, ImageFont
import gym
import mujoco_py
import numpy as np

from modelfree.envs import VICTIM_INDEX
from modelfree.envs.gym_compete import env_name_to_canonical, game_outcome, is_symmetric
from modelfree.visualize.tb import read_sacred_config

VICTIM_OPPONENT_COLORS = {
    'Victim': (55, 126, 184, 255),
    'Opponent': (228, 26, 28, 255),
    'Ties': (0, 0, 0, 255),
}


def body_color(is_victim, agent_type, agent_path):
    key = 'Victim' if is_victim else 'Opponent'
    return VICTIM_OPPONENT_COLORS[key]


GEOM_MAPPINGS = {
    '*': body_color,
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
    'KickAndDefend-v0': {'azimuth': 0, 'distance': 10, 'elevation': -19},
    # From side, slightly behind (runner always goes forward, never back)
    'YouShallNotPassHumans-v0': {'azimuth': 110, 'distance': 9, 'elevation': -21},
    # From side, close up
    'SumoHumans-v0': {'azimuth': 90, 'distance': 8, 'elevation': -25},
    'SumoAnts-v0': {'azimuth': 90, 'distance': 10, 'elevation': -25},
}

PRETTY_POLICY_TYPES = {
    'ppo2': 'Adversary (Adv)',
    'zoo': 'Normal (Zoo)',
    'zero': 'Lifeless (Zero)',
    'random': 'Random (Rand)',
}


def pretty_policy_type(env_name, is_victim, policy_type, policy_path):
    if policy_type == 'zero':
        return 'Lifeless (Zero)'
    elif policy_type == 'random':
        return 'Random (Rand)'
    elif policy_type == 'ppo2':
        try:
            path_components = policy_path.split(osp.sep)
            experiment_root = osp.sep.join(path_components[:-4])
            cfg = read_sacred_config(experiment_root, 'train')
            victim_path = cfg['victim_path']
        except (IndexError, FileNotFoundError):
            victim_path = ''
        return f'Adversary (Adv{victim_path})'
    elif policy_type == 'zoo':
        if not is_symmetric(env_name):
            prefix = 'ZooV' if is_victim else 'ZooO'
        else:
            prefix = 'Zoo'
        return f'Normal ({prefix}{policy_path})'
    else:
        raise ValueError(f"Unrecognized policy type '{policy_type}'")


class AnnotatedGymCompete(gym.Wrapper):
    metadata = {
        'video.frames_per_second': 60,  # MuJoCo env default FPS is 67, round down to be standard
    }

    def __init__(self, env, env_name, agent_a_type, agent_a_path, agent_b_type, agent_b_path,
                 resolution, font, font_size, ypos=0.0, spacing=0.05, num_frames=120):
        super(AnnotatedGymCompete, self).__init__(env)

        # Set agent colors
        self.env_name = env_name
        self.victim_index = VICTIM_INDEX[env_name]
        self.agent_a_type = agent_a_type
        self.agent_a_path = agent_a_path
        self.agent_b_type = agent_b_type
        self.agent_b_path = agent_b_path
        self.agent_mapping = {
            0: (0 == self.victim_index, self.agent_a_type, self.agent_a_path),
            1: (1 == self.victim_index, self.agent_b_type, self.agent_b_path),
        }

        # Text overlay
        self.font = ImageFont.truetype(f'{font}.ttf', font_size)
        self.font_bold = ImageFont.truetype(f'{font}bd.ttf', font_size)
        self.ypos = ypos
        self.spacing = spacing

        # Internal state
        self.result = collections.defaultdict(int)
        self.changed = collections.defaultdict(int)
        self.last_won = None
        self.num_frames = num_frames

        env_scene = self.env.unwrapped.env_scene

        # Start the viewer ourself to control dimensions.
        # env_scene only sets this if None so will not be overwritten.
        width, height = resolution
        env_scene.viewer = mujoco_py.MjViewer(init_width=width, init_height=height)
        env_scene.viewer.start()
        env_scene.viewer.set_model(env_scene.model)
        env_scene.viewer_setup()
        self.camera_setup()

    def camera_setup(self):
        # Color mapping
        color_patterns = {f'agent{agent_key}/{geom_key}': geom_fn(*agent_val)
                          for geom_key, geom_fn in GEOM_MAPPINGS.items()
                          for agent_key, agent_val in self.agent_mapping.items()}
        set_geom_colors(self.env.unwrapped.env_scene.model, color_patterns)

        # Camera setup
        canonical_env_name = env_name_to_canonical(self.env_name)
        camera_cfg = CAMERA_CONFIG[canonical_env_name]
        viewer = self.env.unwrapped.env_scene.viewer

        for k, v in camera_cfg.items():
            setattr(viewer.cam, k, v)

    def _reset(self):
        ob = super(AnnotatedGymCompete, self)._reset()

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
            self.changed[k] = self.num_frames
            self.last_won = k
        self.changed[self.last_won] -= 1
        return obs, rew, done, info

    def _render(self, mode='human', close=False):
        res = self.env.render(mode, close)
        if mode == 'rgb_array':
            img = Image.fromarray(res)
            draw = ImageDraw.Draw(img)

            width, height = img.size
            ypos = height * self.ypos
            texts = collections.OrderedDict([
                (f'win{1 - self.victim_index}', 'Opponent'),
                ('ties', 'Ties'),
                (f'win{self.victim_index}', 'Victim'),
            ])

            to_draw = []
            for k, label in texts.items():
                cur_agent = None
                header = ""
                if label == 'Opponent':
                    cur_agent = self.agent_mapping[1 - self.victim_index]
                elif label == 'Victim':
                    cur_agent = self.agent_mapping[self.victim_index]
                if cur_agent is not None:
                    header = pretty_policy_type(self.env_name, *cur_agent)

                scores = f"{label} = {self.result[k]}"

                color = VICTIM_OPPONENT_COLORS[label]
                if self.changed[k] > 0:
                    weight = 0.7 * math.sqrt(self.changed[k] / self.num_frames)
                    color = tuple(int((255 * weight + (1 - weight) * x)) for x in color)
                font = self.font_bold if k == self.last_won else self.font
                to_draw.append(([scores, header], font, color))

            lengths = []
            for msgs, font, color in to_draw:
                length = max((font.getsize(msg)[0] for msg in msgs))
                lengths.append(length + self.spacing * width)
            total_length = sum(lengths)
            xpos = (width - total_length) / 2

            for (msgs, font, color), length in zip(to_draw, lengths):
                text = "\n".join(msgs)
                draw.multiline_text((xpos, ypos), text, align="center", font=font, fill=color)
                xpos += length

            res = np.array(img)

        return res
