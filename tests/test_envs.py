import gym
from gym.spaces import Tuple
import numpy as np
import pytest

from aprl.envs import multi_agent

spec_list = [
    spec
    for spec in sorted(gym.envs.registration.registry.all(), key=lambda x: x.id)
    if spec.id.startswith("aprl/") or spec.id.startswith("multicomp/")
]


def make_env(spec, i=0):
    env = spec.make()
    env.seed(42 + i)
    return env


@pytest.yield_fixture
def env_from_spec(spec):
    env = make_env(spec)
    yield env
    env.close()


def test_envs_exist():
    assert len(spec_list) > 0, "No aprl environments detected"


@pytest.mark.parametrize("spec", spec_list)
def test_random_rollout(env_from_spec):
    """Based on Gym smoke test in gym.envs.tests.test_envs."""
    ob = env_from_spec.reset()
    for _ in range(1000):
        assert env_from_spec.observation_space.contains(ob)
        a = env_from_spec.action_space.sample()
        assert env_from_spec.action_space.contains(a)
        ob, reward, done, info = env_from_spec.step(a)
        if done:
            break


@pytest.mark.parametrize("spec", spec_list)
def test_env(env_from_spec):
    """Based on Gym smoke test in gym.envs.tests.test_envs."""
    ob_space = env_from_spec.observation_space
    act_space = env_from_spec.action_space
    ob = env_from_spec.reset()
    assert ob_space.contains(ob), "Reset observation: {!r} not in space".format(ob)
    a = act_space.sample()
    ob, reward, done, _info = env_from_spec.step(a)
    assert ob_space.contains(ob), "Step observation: {!r} not in space".format(ob)
    assert isinstance(done, bool), "Expected {} to be a boolean".format(done)

    if hasattr(env_from_spec, "num_agents"):  # multi agent environment
        assert len(reward) == env_from_spec.num_agents
        assert isinstance(env_from_spec.observation_space, Tuple), "Observations should be Tuples"
        assert isinstance(env_from_spec.action_space, Tuple), "Actions should be Tuples"
        assert len(env_from_spec.observation_space.spaces) == env_from_spec.num_agents
        assert len(env_from_spec.action_space.spaces) == env_from_spec.num_agents
    else:
        assert np.isscalar(reward), "{} is not a scalar for {}".format(reward, env_from_spec)

    for mode in env_from_spec.metadata.get("render.modes", []):
        env_from_spec.render(mode=mode)

    # Make sure we can render the environment after close.
    for mode in env_from_spec.metadata.get("render.modes", []):
        env_from_spec.render(mode=mode)


# Test VecMultiEnv classes


def assert_envs_equal(env1, env2, num_steps, check_info: bool = True):
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
            actions = multi_agent.tuple_transpose(actions)
            for env in [env1, env2]:
                env.step_async(actions)
            outs1 = env1.step_wait()
            outs2 = env2.step_wait()
            # Check ob, rew, done; ignore infos
            for out1, out2 in zip(outs1[:3], outs2[:3]):
                assert np.allclose(out1, out2)
            if check_info:
                assert list(outs1[3]) == list(outs2[3])
    finally:
        env1.close()
        env2.close()


@pytest.mark.parametrize("spec", spec_list)
def test_vec_env(spec):
    """Test that our {Dummy,Subproc}VecMultiEnv gives the same results as
       each other."""
    env_fns = [lambda: make_env(spec, i) for i in range(4)]
    venv1 = multi_agent.make_dummy_vec_multi_env(env_fns)
    venv2 = multi_agent.make_subproc_vec_multi_env(env_fns)
    is_multicomp = spec.id.startswith("multicomp/")
    # Can't easily compare info dicts returned by multicomp/ environments, so just skip that check
    assert_envs_equal(venv1, venv2, 100, check_info=not is_multicomp)
