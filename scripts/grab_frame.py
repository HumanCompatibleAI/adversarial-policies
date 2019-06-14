"""Extract a frame from the initial state of an environment for illustration purposes.

Lets user interactively move the camera, then takes a screenshot when ready."""

import argparse
import select
import sys
import time

import gym
import gym_compete  # side-effect: load multicomp/*
import imageio
import mujoco_py
import numpy as np

from modelfree.common.utils import make_env
from modelfree.envs.gym_compete import GymCompeteToOurs
from modelfree.visualize.annotated_gym_compete import AnnotatedGymCompete


def get_img(env_name, seed):
    pre_wrappers = [GymCompeteToOurs] if 'multicomp' in env_name else []

    env = make_env(env_name, int(seed), 0, None,
                   pre_wrappers=pre_wrappers)
    env = AnnotatedGymCompete(env, env_name, 'zoo', '1', 'zoo', '1', None,
                              resolution=(640, 480), font='times', font_size=24,
                              draw=False)
    env.reset()

    env_scene = env.unwrapped.env_scene
    env_scene.viewer = mujoco_py.MjViewer(init_width=1000, init_height=750)
    env_scene.viewer.start()
    env_scene.viewer.set_model(env_scene.model)
    env_scene.viewer_setup()

    print("Type save to save the image, step to take one timestep.")

    running = True
    while running:
        img = None
        while sys.stdin not in select.select([sys.stdin], [], [], 0)[0]:
            env.render()
            img = env.render(mode='rgb_array')

        input = sys.stdin.readline().strip()
        if input == 'save':
            running = False
        elif input == 'step':
            action = tuple(np.zeros(space.shape) for space in env.action_space.spaces)
            env.step(action)
        else:
            print(f"Unrecognized command '{input}'")

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help="environment name")
    parser.add_argument('--seed', type=int, default=time.time())
    parser.add_argument('--out', type=str, help="path to save figure")
    args = parser.parse_args()

    img = get_img(args.env, args.seed)
    imageio.imwrite(args.out, img)

if __name__ == '__main__':
    main()
