import contextlib
import math

import gym
from gym.wrappers import time_limit
import numpy as np
import pytest
from stable_baselines.common.policies import BasePolicy
from stable_baselines.common.vec_env import DummyVecEnv

from aprl.envs.multi_agent import (FakeSingleSpacesVec, FlattenSingletonVecEnv,
                                   make_dummy_vec_multi_env)
from aprl.envs.wrappers import make_env
from aprl.policies.base import ConstantPolicy, PolicyToModel
from aprl.policies.loader import load_policy
from aprl.policies.wrappers import MultiPolicyWrapper
from aprl.training.embedded_agents import CurryVecEnv


class ConstantStatefulPolicy(BasePolicy):
    """Policy that returns a constant action."""

    def __init__(self, env, constant, state_shape):
        assert env.action_space.contains(constant)
        super().__init__(sess=None,
                         ob_space=env.observation_space,
                         ac_space=env.action_space,
                         n_env=env.num_envs,
                         n_steps=1,
                         n_batch=1)
        self.state_shape = state_shape
        self.constant = constant

    def step(self, obs, state=None, mask=None, deterministic=False):
        if state is None:
            state = np.ones((obs.shape[0], ) + self.state_shape)
        assert state.shape[0] == obs.shape[0], "state.shape[0] does not match num_env"
        error_text = f"A state of shape {state.shape[1:]} passed in, requires {self.state_shape}"
        assert state.shape[1:] == self.state_shape, error_text

        actions = np.array([self.constant] * self.n_env)
        return actions, None, state, None

    def proba_step(self, obs, state=None, mask=None):
        return self.step(obs, state=state, mask=mask)


def _get_constant_policy(venv, constant_value, state_shape=None):
    if state_shape is None:
        policy = ConstantPolicy(venv, constant=constant_value)
    else:
        policy = ConstantStatefulPolicy(venv, constant=constant_value, state_shape=state_shape)
    return PolicyToModel(policy)


@contextlib.contextmanager
def create_simple_policy_wrapper(env_name, num_envs, state_shapes):
    vec_env = DummyVecEnv([lambda: gym.make(env_name) for _ in range(num_envs)])
    num_actions = vec_env.action_space.n  # for Discrete spaces

    policies = []
    for i, state_shape in enumerate(state_shapes):
        constant_value = np.full(shape=vec_env.action_space.shape, fill_value=i % num_actions)
        policy = _get_constant_policy(vec_env,
                                      constant_value=constant_value,
                                      state_shape=state_shape)
        policies.append(policy)
    policy_wrapper = MultiPolicyWrapper(policies=policies, num_envs=num_envs)

    yield vec_env, policy_wrapper
    policy_wrapper.close()


@contextlib.contextmanager
def create_multi_agent_curried_policy_wrapper(mon_dir, env_name, num_envs, embed_index, max_steps,
                                              state_shape=None, add_zoo=False, num_zoo=5):
    def episode_limit(env):
        return time_limit.TimeLimit(env, max_episode_steps=max_steps)

    def env_fn(i):
        return make_env(env_name, seed=42, i=i, out_dir=mon_dir,
                        pre_wrappers=[episode_limit])

    vec_env = make_dummy_vec_multi_env([lambda: env_fn(i) for i in range(num_envs)])

    zoo = load_policy(policy_path="1", policy_type="zoo", env=vec_env,
                      env_name=env_name, index=1 - embed_index, transparent_params=None)
    half_env = FakeSingleSpacesVec(vec_env, agent_id=embed_index)
    policies = [_get_constant_policy(half_env,
                                     constant_value=half_env.action_space.sample(),
                                     state_shape=state_shape)
                for _ in range(10)]
    if add_zoo:
        policies += [zoo] * num_zoo

    policy_wrapper = MultiPolicyWrapper(policies=policies, num_envs=num_envs)

    vec_env = CurryVecEnv(venv=vec_env,
                          policy=policy_wrapper,
                          agent_idx=embed_index,
                          deterministic=False)
    vec_env = FlattenSingletonVecEnv(vec_env)

    yield vec_env, policy_wrapper, zoo
    policy_wrapper.close()


def _check_switching(vec_env, num_steps, agent, policy_wrapper, extra_check=None):
    """Check policies switch at least once across episodes, and never switch within episode."""
    obs = vec_env.reset()
    dones = np.full(shape=vec_env.num_envs, fill_value=False)
    states = None
    current_policies = [ptm.policy_obj for ptm in policy_wrapper.current_env_policies]
    num_policies = len(current_policies)
    num_identical = 0
    num_switches = 0
    for step in range(num_steps):
        # PolicyWrapper is either `agent`, or is embedded in `vec_env`. So after we call
        # `agent.predict` and `vec_env.step`, `predict` has been called exactly once.
        actions, states = agent.predict(obs, state=states, mask=dones)
        obs, rew, new_dones, infos = vec_env.step(actions)

        new_current_policies = [ptm.policy_obj for ptm in policy_wrapper.current_env_policies]
        for ind, done in enumerate(dones):
            policies_match = new_current_policies[ind] == current_policies[ind]
            if done:
                num_switches += 1
                if policies_match:
                    num_identical += 1
            else:
                assert policies_match, f"Policies did not match on index {ind}, step {step}"

        if extra_check is not None:
            extra_check(locals())

        current_policies = new_current_policies
        dones = new_dones

    max_identical_ok = math.ceil(2 * num_switches / num_policies)
    dont_match_msg = (f"Same policy {num_identical} > {max_identical_ok} threshold over "
                      f"{num_switches} episodes")
    # Expected # of identical policies is num_switches / num_policies
    # Allow 2x margin of error to avoid flaky tests from randomness
    assert num_identical < max_identical_ok, dont_match_msg


def test_simple_multi_policy_switching():
    """Checks policy switching in simple environment."""
    num_steps = 5000

    def extra_check(vars):
        assert np.all(vars['actions'] == [p.constant for p in vars['new_current_policies']])

    with create_simple_policy_wrapper(env_name="CartPole-v1",
                                      num_envs=2,
                                      state_shapes=[None, (5,), (10,)],
                                      ) as (vec_env, policy_wrapper):
        _check_switching(vec_env=vec_env,
                         num_steps=num_steps,
                         agent=policy_wrapper,
                         policy_wrapper=policy_wrapper,
                         extra_check=extra_check)


MULTI_AGENT_CONFIGS = [dict(env_name="multicomp/YouShallNotPassHumans-v0",  # MLP Zoo
                            num_envs=2,
                            num_steps=200,
                            ep_len=10),
                       dict(env_name="multicomp/KickAndDefend-v0",  # LSTM Zoo
                            num_envs=2,
                            num_steps=200,
                            ep_len=10)]


@pytest.mark.parametrize("test_config", MULTI_AGENT_CONFIGS)
def test_multi_agent_policy_switching(test_config, tmpdir):
    """Checks policies switch when curried policy, for a mixture of Zoo and constant policies."""

    with create_multi_agent_curried_policy_wrapper(str(tmpdir),
                                                   env_name=test_config["env_name"],
                                                   num_envs=test_config["num_envs"],
                                                   embed_index=test_config.get("embed_index", 1),
                                                   max_steps=test_config["ep_len"],
                                                   state_shape=test_config.get("state_shape",
                                                                               None),
                                                   add_zoo=True,
                                                   num_zoo=test_config.get("num_zoo", 5),
                                                   ) as (vec_env, policy_wrapper, zoo_agent):
        _check_switching(vec_env=vec_env,
                         num_steps=test_config["num_steps"],
                         agent=zoo_agent,
                         policy_wrapper=policy_wrapper)
