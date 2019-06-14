import multiprocessing

import gym
from ilqr import iLQR
import numpy as np
import pytest

from aprl.agents.monte_carlo import (MonteCarloParallel, MonteCarloSingle, MujocoResettableWrapper,
                                     receding_horizon)
from aprl.agents.mujoco_lqr import (MujocoFiniteDiffCost, MujocoFiniteDiffDynamicsBasic,
                                    MujocoFiniteDiffDynamicsPerformance)

dynamics_list = [MujocoFiniteDiffDynamicsBasic, MujocoFiniteDiffDynamicsPerformance]
@pytest.mark.parametrize("dynamics_cls", dynamics_list)
def test_lqr_mujoco(dynamics_cls):
    """Smoke test for MujcooFiniteDiff{Dynamics,Cost}.
    Jupyter notebook experiments/mujoco_control.ipynb has quantitative results
    attained; for efficiency, we only run for a few iterations here."""
    env = gym.make('Reacher-v2').unwrapped
    env.seed(42)
    env.reset()
    dynamics = dynamics_cls(env)
    cost = MujocoFiniteDiffCost(env)
    N = 10
    ilqr = iLQR(dynamics, cost, N)
    x0 = dynamics.get_state()
    us_init = np.array([env.action_space.sample() for _ in range(N)])
    xs, us = ilqr.fit(x0, us_init, n_iterations=3)
    assert x0.shape == xs[0].shape
    assert xs.shape[0] == N + 1
    assert us.shape == (N, 2)
    assert env.action_space.contains(us[0])


def rollout(env, actions):
    obs, rews, dones, infos = [], [], [], []
    for a in actions:
        ob, rew, done, info = env.step(a)
        obs.append(ob)
        rews.append(rew)
        dones.append(done)
        infos.append(info)
    obs = np.array(obs)
    rews = np.array(rews)
    dones = np.array(dones)
    return obs, rews, dones, infos


def make_mujoco_env(env_name, seed):
    env = gym.make(env_name)
    env = MujocoResettableWrapper(env.unwrapped)
    env.seed(seed)
    env.reset()
    return env


MONTE_CARLO_ENVS = ['Reacher-v2', 'HalfCheetah-v2', 'Hopper-v2']


@pytest.mark.parametrize("env_name", MONTE_CARLO_ENVS)
def test_mujoco_reset_env(env_name, horizon=10, seed=42):
    env = make_mujoco_env(env_name, seed)
    state = env.get_state()
    actions = [env.action_space.sample() for _ in range(horizon)]

    first_obs, first_rews, first_dones, _first_infos = rollout(env, actions)
    env.set_state(state)
    second_obs, second_rews, second_dones, _second_infos = rollout(env, actions)

    np.testing.assert_almost_equal(second_obs, first_obs, decimal=5)
    np.testing.assert_almost_equal(second_rews, first_rews, decimal=5)
    assert (first_dones == second_dones).all()


def check_monte_carlo(kind, score_thresholds, total_horizon,
                      planning_horizon, trajectories, seed=42):
    def f(env_name):
        # Setup
        env = make_mujoco_env(env_name, seed)
        if kind == 'single':
            mc = MonteCarloSingle(env, planning_horizon, trajectories)
        elif kind == 'parallel':
            env_fns = [lambda: make_mujoco_env(env_name, seed)
                       for _ in range(multiprocessing.cpu_count())]
            mc = MonteCarloParallel(env_fns, planning_horizon, trajectories)
        else:
            raise ValueError("Unrecognized kind '{}'".format(kind))
        mc.seed(seed)

        # Check for side-effects
        state = env.get_state()
        _ = mc.best_action(state)
        assert (env.get_state() == state).all(), "Monte Carlo search has side effects"

        # One receding horizon rollout of Monte Carlo search
        total_rew = 0
        prev_done = False
        for i, (a, ob, rew, done, info) in enumerate(receding_horizon(mc, env)):
            assert not prev_done, "should terminate if env returns done"
            prev_done = done
            assert env.action_space.contains(a)
            assert env.observation_space.contains(ob)
            total_rew += rew

            if i >= total_horizon:
                break
        assert i == total_horizon or done

        # Check it does better than random sequences
        random_rews = []
        for i in range(10):
            env.action_space.np_random.seed(seed + i)
            action_seq = [env.action_space.sample() for _ in range(total_horizon)]
            env.set_state(state)
            _, rews, _, _ = rollout(env, action_seq)
            random_rew = sum(rews)
            random_rews.append(random_rew)
            assert total_rew >= random_rew, "random sequence {}".format(i)
        print(f'Random actions on {env_name} for {total_horizon} obtains '
              f'mean {np.mean(random_rews)} s.d. {np.std(random_rews)}')

        # Check against pre-defined score threshold
        assert total_rew >= score_thresholds[env_name]

        # Cleanup
        if kind == 'parallel':
            mc.close()
            with pytest.raises(BrokenPipeError):
                mc.best_action(state)
    return f


MC_SINGLE_THRESHOLDS = {
        'Reacher-v2': -11,  # tested -9.5, random -17.25 s.d. 1.5
        'HalfCheetah-v2': 19,  # tested 21.6, random -4.2 s.d. 3.7
        'Hopper-v2': 29,  # tested 31.1, random 15.2 s.d. 5.9
}
MC_PARALLEL_THRESHOLDS = {
        'Reacher-v2': -17,  # tested at -15.3; random -25.8 s.d. 1.8
        'HalfCheetah-v2': 33,  # tested at 35.5; random -6.0 s.d. 7.1
        'Hopper-v2': 52,  # tested at 54.7; random 21.1 s.d. 13.2
}
_test_mc_single = check_monte_carlo('single', MC_SINGLE_THRESHOLDS,
                                    total_horizon=20, planning_horizon=10, trajectories=100)
_test_mc_parallel = check_monte_carlo('parallel', MC_PARALLEL_THRESHOLDS,
                                      total_horizon=30, planning_horizon=15, trajectories=4096)
test_mc_single = pytest.mark.parametrize("env_name", MONTE_CARLO_ENVS)(_test_mc_single)
test_mc_parallel = pytest.mark.parametrize("env_name", MONTE_CARLO_ENVS)(_test_mc_parallel)
