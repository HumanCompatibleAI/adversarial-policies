import collections
import ctypes
import math
import os.path as osp
import re

from PIL import Image, ImageDraw, ImageFont
import gym
import mujoco_py_131
import numpy as np

from aprl.envs import VICTIM_INDEX
from aprl.envs.gym_compete import env_name_to_canonical, game_outcome, is_symmetric
from aprl.visualize import tb

VICTIM_OPPONENT_COLORS = {
    "Victim": (55, 126, 184, 255),
    "Opponent": (228, 26, 28, 255),
    "Ties": (0, 0, 0, 255),
}


def body_color(is_victim, is_masked, agent_type, agent_path):
    key = "Victim" if is_victim else "Opponent"
    return VICTIM_OPPONENT_COLORS[key]


GEOM_MAPPINGS = {
    "*": body_color,
}


def set_geom_rgba(model, value):
    """Does what model.geom_rgba = ... should do, but doesn't because of a bug in mujoco-py."""
    val_ptr = np.array(value, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ctypes.memmove(
        model._wrapped.contents.geom_rgba, val_ptr, model.ngeom * 4 * ctypes.sizeof(ctypes.c_float)
    )


def set_geom_colors(model, patterns):
    names = [name.decode("utf-8") for name in model.geom_names]
    patterns = {re.compile(k): tuple(x / 255 for x in v) for k, v in patterns.items()}

    modified = np.array(model.geom_rgba)
    for row_idx, name in enumerate(names):
        for pattern, color in patterns.items():
            if pattern.match(name):
                modified[row_idx, :] = color

    set_geom_rgba(model, modified)


CAMERA_CONFIGS = {
    # For website videos
    "default": {
        # From behind
        "KickAndDefend-v0": {"azimuth": 0, "distance": 10, "elevation": -16.5},
        # From side, slightly behind (runner always goes forward, never back)
        "YouShallNotPassHumans-v0": {
            "azimuth": 140,
            "distance": 9,
            "elevation": -21,
            "lookat": [-1.5, 0.5, 0.0],
            "trackbodyid": -1,
        },
        # From side, close up
        "SumoHumans-v0": {"azimuth": 90, "distance": 8, "elevation": -23},
        "SumoAnts-v0": {"azimuth": 90, "distance": 10, "elevation": -25},
    },
    # More closely cropped. May miss action, but easier to see on projector.
    "close": {
        "KickAndDefend-v0": {"azimuth": 0, "distance": 10, "elevation": -15},
        "YouShallNotPassHumans-v0": {
            "azimuth": 150,
            "distance": 9,
            "elevation": -23,
            "lookat": [-2.0, 1, 0.0],
            "trackbodyid": -1,
        },
        "SumoHumans-v0": {"azimuth": 90, "distance": 7.2, "elevation": -22.5},
        "SumoAnts-v0": {"azimuth": 90, "distance": 10, "elevation": -25},
    },
    # Camera tracks victim. Very tightly cropped. May miss what opponent is doing.
    "track": {
        "KickAndDefend-v0": {
            "azimuth": 0,
            "distance": 7,
            "elevation": -25,
            "trackbodyid": "agent0/torso",
        },
        "YouShallNotPassHumans-v0": {
            "azimuth": 140,
            "distance": 5,
            "elevation": -30,
            "trackbodyid": "agent1/torso",
        },
        "SumoHumans-v0": {"azimuth": 90, "distance": 7, "elevation": -30},
        "SumoAnts-v0": {"azimuth": 90, "distance": 10, "elevation": -25},
    },
}


def pretty_policy_type(env_name, short, is_victim, is_masked, policy_type, policy_path):
    if policy_type == "zero":
        friendly, code = "Lifeless", "Zero"
    elif policy_type == "random":
        friendly, code = "Random", "Rand"
    elif policy_type == "ppo2":
        try:
            path_components = policy_path.split(osp.sep)
            experiment_root = osp.sep.join(path_components[:-4])
            cfg = tb.read_sacred_config(experiment_root, "train")
            victim_path = cfg["victim_path"]
        except (IndexError, FileNotFoundError):
            victim_path = ""
        friendly, code = "Adversary", f"Adv{victim_path}"
    elif policy_type == "zoo":
        if not is_symmetric(env_name):
            prefix = "ZooV" if is_victim else "ZooO"
        else:
            prefix = "Zoo"
        friendly = "Normal"
        if is_masked:
            prefix += "M"
            friendly = "Masked"
        code = f"{prefix}{policy_path}"
    else:
        raise ValueError(f"Unrecognized policy type '{policy_type}'")

    if short:
        return friendly
    else:
        return f"{friendly} ({code})"


class AnnotatedGymCompete(gym.Wrapper):
    metadata = {
        "video.frames_per_second": 60,  # MuJoCo env default FPS is 67, round down to be standard
    }

    def __init__(
        self,
        env,
        env_name,
        agent_a_type,
        agent_a_path,
        agent_b_type,
        agent_b_path,
        mask_agent_index,
        resolution,
        font,
        font_size,
        short_labels,
        camera_config,
        ypos=0.0,
        spacing=0.05,
        num_frames=120,
        draw=True,
    ):
        super(AnnotatedGymCompete, self).__init__(env)

        # Set agent colors
        self.env_name = env_name
        self.victim_index = VICTIM_INDEX[env_name]
        self.mask_agent_index = mask_agent_index
        self.agent_a_type = agent_a_type
        self.agent_a_path = agent_a_path
        self.agent_b_type = agent_b_type
        self.agent_b_path = agent_b_path
        self.agent_mapping = {
            0: (
                0 == self.victim_index,
                0 == self.mask_agent_index,
                self.agent_a_type,
                self.agent_a_path,
            ),
            1: (
                1 == self.victim_index,
                1 == self.mask_agent_index,
                self.agent_b_type,
                self.agent_b_path,
            ),
        }

        # Text overlay
        self.font = ImageFont.truetype(f"{font}.ttf", font_size)
        self.font_bold = ImageFont.truetype(f"{font}bd.ttf", font_size)
        self.short_labels = short_labels
        self.ypos = ypos
        self.spacing = spacing

        # Camera settings
        self.camera_config = CAMERA_CONFIGS[camera_config]

        # Internal state
        self.result = collections.defaultdict(int)
        self.changed = collections.defaultdict(int)
        self.last_won = None
        self.num_frames = num_frames
        self.draw = draw

        env_scene = self.env.unwrapped.env_scene

        # Start the viewer ourself to control dimensions.
        # env_scene only sets this if None so will not be overwritten.
        width, height = resolution
        env_scene.viewer = mujoco_py_131.MjViewer(init_width=width, init_height=height)
        env_scene.viewer.start()
        env_scene.viewer.set_model(env_scene.model)
        env_scene.viewer_setup()
        self.camera_setup()

    def camera_setup(self):
        # Color mapping
        model = self.env.unwrapped.env_scene.model
        color_patterns = {
            f"agent{agent_key}/{geom_key}": geom_fn(*agent_val)
            for geom_key, geom_fn in GEOM_MAPPINGS.items()
            for agent_key, agent_val in self.agent_mapping.items()
        }
        set_geom_colors(model, color_patterns)

        # Camera setup
        canonical_env_name = env_name_to_canonical(self.env_name)
        camera_cfg = self.camera_config[canonical_env_name]

        if "trackbodyid" in camera_cfg:
            trackbodyid = camera_cfg["trackbodyid"]
            try:
                trackbodyid = int(trackbodyid)
            except ValueError:
                trackbodyid = str(trackbodyid).encode("utf-8")
                trackbodyid = model.body_names.index(trackbodyid)
            camera_cfg["trackbodyid"] = trackbodyid

        if "lookat" in camera_cfg:
            DoubleArray3 = ctypes.c_double * 3
            lookat = [float(x) for x in camera_cfg["lookat"]]
            assert len(lookat) == 3
            camera_cfg["lookat"] = DoubleArray3(*lookat)  # pytype:disable=not-callable

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
                k = "ties"
            else:
                k = f"win{winner}"
            self.result[k] += 1
            self.changed[k] = self.num_frames
            self.last_won = k
        self.changed[self.last_won] -= 1
        return obs, rew, done, info

    def _render(self, mode="human", close=False):
        res = self.env.render(mode, close)

        if mode == "rgb_array":
            if not self.draw:
                return res

            img = Image.fromarray(res)

            draw = ImageDraw.Draw(img)

            width, height = img.size
            ypos = height * self.ypos
            texts = collections.OrderedDict(
                [
                    (f"win{1 - self.victim_index}", "Opponent"),
                    ("ties", "Ties"),
                    (f"win{self.victim_index}", "Victim"),
                ]
            )

            to_draw = []
            for k, label in texts.items():
                cur_agent = None
                header = ""
                if label == "Opponent":
                    cur_agent = self.agent_mapping[1 - self.victim_index]
                elif label == "Victim":
                    cur_agent = self.agent_mapping[self.victim_index]
                if cur_agent is not None:
                    header = pretty_policy_type(self.env_name, self.short_labels, *cur_agent)

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
