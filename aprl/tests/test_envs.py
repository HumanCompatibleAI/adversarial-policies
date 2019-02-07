import functools

from baselines.common.vec_env.test_vec_env import assert_envs_equal
import gym
import numpy as np
import pytest

from aprl import envs

# Helper functions


def check_env(env):
    """Based on Gym smoke test in gym.envs.tests.test_envs."""
    ob_space = env.observation_space
    act_space = env.action_space
    ob = env.reset()
    assert ob_space.contains(ob), 'Reset observation: {!r} not in space'.format(ob)
    a = act_space.sample()
    ob, reward, done, _info = env.step(a)
    assert ob_space.contains(ob), 'Step observation: {!r} not in space'.format(ob)
    assert isinstance(done, bool), "Expected {} to be a boolean".format(done)

    if isinstance(env, envs.MultiAgentEnv):
        assert len(reward) == env.num_agents
        assert env.observation_space.shape[0] == env.num_agents
        assert env.action_space.shape[0] == env.num_agents
        assert env.observation_space.shape[1:] == env.agent_observation_space.shape
        assert env.action_space.shape[1:] == env.agent_action_space.shape
    else:
        assert np.isscalar(reward), "{} is not a scalar for {}".format(reward, env)

    for mode in env.metadata.get('render.modes', []):
        env.render(mode=mode)

    # Make sure we can render the environment after close.
    for mode in env.metadata.get('render.modes', []):
        env.render(mode=mode)

    env.close()


def check_random_rollout(env):
    """Based on Gym smoke test in gym.envs.tests.test_envs."""
    ob = env.reset()
    for _ in range(10):
        assert env.observation_space.contains(ob)
        a = env.action_space.sample()
        assert env.action_space.contains(a)
        ob, reward, done, info = env.step(a)
        if done:
            break
    env.close()


# Test aprl environments

spec_list = [spec
             for spec in sorted(gym.envs.registry.all(), key=lambda x: x.id)
             if spec.id.startswith('aprl/')]


def test_envs_exist():
    assert len(spec_list) > 0


def spec_to_env(fn):
    def helper(spec):
        env = spec.make()
        return fn(env)
    return helper


test_env = pytest.mark.parametrize("spec", spec_list)(spec_to_env(check_env))
test_random_rollout = pytest.mark.parametrize("spec", spec_list)(spec_to_env(check_random_rollout))

# Test VecMultiEnv classes


class SimpleMultiEnv(envs.MatrixGameEnv):
    def __init__(self, seed):
        num_actions = 0x100
        np.random.seed(seed)
        payoff = np.random.random((2, num_actions, num_actions))
        return super().__init__(num_actions=num_actions, payoff=payoff)


def test_vec_env():
    """Test that our {Dummy,Subproc}VecMultiEnv gives the same results as
       each other."""
    env_fns = [functools.partial(SimpleMultiEnv, i) for i in range(4)]
    venv1 = envs.DummyVecMultiEnv(env_fns)
    venv2 = envs.SubprocVecMultiEnv(env_fns)
    assert_envs_equal(venv1, venv2, 100)
