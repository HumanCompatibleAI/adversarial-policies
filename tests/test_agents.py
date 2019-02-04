import functools
import os
import tempfile

import gym
from ilqr import iLQR
import numpy as np

from aprl.agents import MujocoFiniteDiffCost, MujocoFiniteDiffDynamics, MultiMonitor, PPOSelfPlay
from aprl.envs import DummyVecMultiEnv


def test_multi_monitor():
    """Smoke test for MultiMonitor."""
    env = gym.make('aprl/IMP-v0')
    with tempfile.TemporaryDirectory(prefix='test_multi_mon') as d:
        env = MultiMonitor(env, filename=os.path.join(d, 'test'))
        for eps in range(5):
            env.reset()
            done = False
            while not done:
                a = env.action_space.sample()
                _, _, done, info = env.step(a)
            epinfo = info['episode']
            assert set(epinfo.keys()) == {'r', 'r0', 'r1', 'l', 't'}


def test_ppo_self_play():
    """Smoke test for PPOSelfPlay."""
    with tempfile.TemporaryDirectory(prefix='test_ppo_self_play') as d:
        def make_env(i):
            env = gym.make('aprl/IMP-v0')
            fname = os.path.join(d, 'test{:d}'.format(i))
            env = MultiMonitor(env, filename=fname,
                               allow_early_resets=True)
            return env
        env_fns = [functools.partial(make_env, i) for i in range(4)]
        venv = DummyVecMultiEnv(env_fns)
        self_play = PPOSelfPlay(population_size=4,
                                training_type='best',
                                env=venv,
                                network='mlp')
        self_play.learn(total_timesteps=10000)


def test_lqr_mujoco():
    """Smoke test for MujcooFiniteDiff{Dynamics,Cost}.
    Jupyter notebook experiments/mujoco_control.ipynb has quantitative results
    attained; for efficiency, we only run for a few iterations here."""
    env = gym.make('Reacher-v2').unwrapped
    env.reset()
    dynamics = MujocoFiniteDiffDynamics(env)
    cost = MujocoFiniteDiffCost(env)
    N = 10
    ilqr = iLQR(dynamics, cost, N)
    x0 = dynamics.get_state()
    us_init = np.array([env.action_space.sample() for _ in range(N)])
    xs, us = ilqr.fit(x0, us_init, n_iterations=3)
    assert x0.shape == xs[0].shape
    assert xs.shape == (N + 1, 9)
    assert us.shape == (N, 2)
    assert env.action_space.contains(us[0])
