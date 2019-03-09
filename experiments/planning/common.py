import random
import time

import gym
from ilqr.controller import RecedingHorizonController
import numpy as np
import pandas as pd


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    info = "converged" if converged else ("accepted" if accepted else "failed")
    print("iteration", iteration_count, info, J_opt, xs[-1], us[-1])


def make_env(env_name, seed, horizon=None):
    env = gym.make(env_name)
    if horizon is None:
        horizon = env._max_episode_steps
    env = env.unwrapped
    env.frame_skip = 1
    env.seed(seed)
    env.reset()
    us_init = np.array([env.action_space.sample() for _ in range(horizon)])

    return env, us_init


def fit_ilqr(ilqrs, x0s, us_init, **kwargs):
    xs = {}
    us = {}
    print(ilqrs.keys())
    for k, ilqr in ilqrs.items():
        start = time.time()
        print('*** Fitting {} ***'.format(k))
        x0 = x0s[k]
        xs[k], us[k] = ilqr.fit(x0, us_init, on_iteration=on_iteration,
                                **kwargs)
        end = time.time()
        print('*** Fitted {} in {}s ***'.format(k, end - start))
    return xs, us


def receding(ilqr, x0, us_init, seed, step_size=1, horizon=None, **kwargs):
    if horizon is None:
        horizon = len(us_init)
    controller = RecedingHorizonController(x0, ilqr)
    controller.seed(seed)
    xs = np.zeros((horizon, ) + x0.shape)
    us = np.zeros((horizon, ) + us_init[0].shape)
    i = 0
    for x, u in controller.control(us_init, step_size=step_size, **kwargs):
        xs[i:i+step_size] = x[:-1]
        us[i:i+step_size] = u
        print('iteration {} x = {}, u = {}'.format(i, x, u))
        i += step_size
        if i == horizon:
            break
    return xs, us


def evaluate(env, dynamics, x0, us, render=False):
    dynamics.set_state(x0)
    if render:
        env.render()
    rew = []
    actual_xs = []
    for u in us:
        _obs, r, done, info = env.step(u)
        if done:
            print('warning: early termination! (assuming zero-reward from now)')
            break
        rew.append(r)
        actual_xs.append(dynamics.get_state())
        if render:
            env.render()
            time.sleep(0.01)
    return rew, actual_xs


def multi_evaluate(env, dynamics, x0s, us, **kwargs):
    rews = {}
    actual_xs = {}
    for k, solved_us in us.items():
        print(k)
        rews[k], actual_xs[k] = evaluate(env.unwrapped, dynamics[k], x0s[k],
                                         solved_us, **kwargs)
    rewards = {k: sum(r) for k, r in rews.items()}
    lengths = {k: len(r) for k, r in rews.items()}
    return pd.DataFrame({'rewards': rewards, 'lengths': lengths})
