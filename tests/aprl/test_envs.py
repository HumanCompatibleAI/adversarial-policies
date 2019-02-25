import gym
from gym.spaces.tuple_space import Tuple
import numpy as np
import pytest

import aprl.envs as envs
from aprl.envs.multi_agent import tuple_transpose

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
        assert isinstance(env.observation_space, Tuple), 'Observations should be Tuples'
        assert isinstance(env.action_space, Tuple), 'Actions should be Tuples'
        assert len(env.observation_space.spaces) == env.num_agents
        assert len(env.action_space.spaces) == env.num_agents
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
    assert len(spec_list) > 0, "No aprl environments detected"


def spec_to_env(fn):
    def helper(spec):
        env = spec.make()
        return fn(env)
    return helper


test_env = pytest.mark.parametrize("spec", spec_list)(spec_to_env(check_env))
test_random_rollout = pytest.mark.parametrize("spec", spec_list)(spec_to_env(check_random_rollout))

# Test VecMultiEnv classes


def assert_envs_equal(env1, env2, num_steps):
    """
    Compare two environments over num_steps steps and make sure
    that the observations produced by each are the same when given
    the same actions.
    """
    assert env1.num_envs == env2.num_envs
    assert env1.observation_space == env2.observation_space
    assert env1.action_space == env2.action_space

    try:
        obs1, obs2 = env1.reset(), env2.reset()
        assert type(obs1) == type(obs2)
        # TODO: sample actions sensitive to num_envs.
        # (Maybe add a helper function to make this easy in VecEnv? Feels like a design flaw.)

        if isinstance(obs1, tuple):
            for x, y in zip(obs1, obs2):
                assert x.shape == y.shape
                assert np.allclose(x, y)
        else:
            assert np.array(obs1).shape == np.array(obs2).shape
            assert np.allclose(obs1, obs2)

        if isinstance(env1.action_space, Tuple):
            for space in env1.action_space.spaces:
                space.np_random.seed(1337)
        else:
            env1.action_space.np_random.seed(1337)

        for _ in range(num_steps):
            actions = tuple((env1.action_space.sample() for _ in range(env1.num_envs)))
            actions = tuple_transpose(actions)
            for env in [env1, env2]:
                env.step_async(actions)
            outs1 = env1.step_wait()
            outs2 = env2.step_wait()
            # Check ob, rew, done; ignore infos
            for out1, out2 in zip(outs1[:3], outs2[:3]):
                assert np.allclose(out1, out2)
            assert list(outs1[3]) == list(outs2[3])
    finally:
        env1.close()
        env2.close()


@pytest.mark.parametrize("spec", spec_list)
def test_vec_env(spec):
    """Test that our {Dummy,Subproc}VecMultiEnv gives the same results as
       each other."""
    def make_env(i):
        env = spec.make()
        env.seed(42 + i)
        return env
    env_fns = [lambda: make_env(i) for i in range(4)]
    venv1 = envs.make_dummy_vec_multi_env(env_fns)
    venv2 = envs.make_subproc_vec_multi_env(env_fns)
    assert_envs_equal(venv1, venv2, 100)
