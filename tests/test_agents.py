import functools
import gym
import tempfile
import os

from aprl.agents import MultiMonitor, PPOSelfPlay
from aprl.envs import DummyVecMultiEnv


def test_multi_monitor():
    '''Smoke test for MultiMonitor.'''
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
    '''Smoke test for PPOSelfPlay.'''
    with tempfile.TemporaryDirectory(prefix='test_ppo_self_play') as d:
        def make_env(i):
            env = gym.make('aprl/IMP-v0')
            env = MultiMonitor(env, 'test{:d}'.format(i),
                               allow_early_resets=True)
            return env
        env_fns = [functools.partial(make_env, i) for i in range(4)]
        venv = DummyVecMultiEnv(env_fns)
        self_play = PPOSelfPlay(population_size=4,
                                num_competitors=2,
                                training_type='best',
                                env=venv,
                                network='mlp')
        self_play.learn(total_timesteps=10000)