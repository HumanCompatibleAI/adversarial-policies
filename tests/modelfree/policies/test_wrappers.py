import pdb

import gym
import numpy as np
import pytest
from stable_baselines.common.policies import BasePolicy
from stable_baselines.common.vec_env import DummyVecEnv

from aprl.envs.multi_agent import FakeSingleSpacesVec
from modelfree.policies.base import ConstantPolicy, PolicyToModel
from modelfree.policies.loader import load_policy
from modelfree.policies.wrappers import MultiPolicyWrapper
from modelfree.train import build_env
from modelfree.training.victim_envs import EmbedVictimWrapper


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
        if state is not None:
            error_text = f"A state of shape {state.shape} passed in, requires {self.state_shape}"
            assert state.shape == self.state_shape, error_text
        actions = np.array([self.constant] * self.n_env)
        return actions, state, None, None

    def proba_step(self, obs, state=None, mask=None):
        return self.step(obs, state=state, mask=mask)


def _get_constant_policy(venv, constant_value, state_shape=None):
    if state_shape is None:
        policy = ConstantPolicy(venv, constant=constant_value)
    else:
        policy = ConstantStatefulPolicy(venv, constant=constant_value, state_shape=state_shape)
    return PolicyToModel(policy)


def _get_constant_action(constant, action_space):
    sample_action = action_space.sample()
    try:
        shape = sample_action.shape
    except AttributeError:
        shape = 1
    constant_action = np.full(shape=shape, fill_value=constant)
    if shape == 1:
        constant_action = constant_action[0]
    return constant_action


def env_func(gym_name):
    def f():
        return gym.make(gym_name)
    return f


def create_simple_policy_wrapper(env_name, num_envs, state_shape=None):
    vec_env = DummyVecEnv([env_func(env_name) for _ in range(num_envs)])
    policies = [
        _get_constant_policy(vec_env,
                             constant_value=_get_constant_action(i, vec_env.action_space),
                             state_shape=state_shape)
        for i in range(vec_env.action_space.n)
    ]
    policy_wrapper = MultiPolicyWrapper(policies=policies, num_envs=num_envs)
    return vec_env, policy_wrapper


def create_multi_agent_curried_policy_wrapper(env_name, num_envs, victim_index,
                                              state_shape=None, add_zoo=False, num_zoo=5):
    vec_env, my_idx = build_env(".", _seed=43, env_name=env_name,
                                num_env=num_envs, victim_type="zoo", victim_index=victim_index,
                                mask_victim=False, mask_victim_kwargs=dict(),
                                lookback_params={'lb_num': 0}, debug=False)

    half_env = FakeSingleSpacesVec(vec_env, agent_id=victim_index)
    zoo = load_policy(policy_path="1", policy_type="zoo", env=vec_env,
                      env_name=env_name, index=1 - victim_index, transparent_params=None)
    policies = [_get_constant_policy(half_env,
                                     constant_value=half_env.action_space.sample(),
                                     state_shape=state_shape) for _ in range(10)]
    if add_zoo:
        policies += [zoo] * num_zoo

    policy_wrapper = MultiPolicyWrapper(policies=policies, num_envs=num_envs)

    vec_env = EmbedVictimWrapper(multi_env=vec_env,
                                 victim=policy_wrapper,
                                 victim_index=victim_index, transparent=False,
                                 deterministic=False)
    return vec_env, policy_wrapper, zoo


SIMPLE_CONFIGS = [dict(env_name="CartPole-v1",
                       num_envs=8,
                       num_steps=1000),
                  dict(env_name="CartPole-v1",
                       num_envs=8,
                       num_steps=1000,
                       state_shape=(10,))]


@pytest.mark.parametrize("test_config", SIMPLE_CONFIGS)
def test_simple_multi_policy_wrapper(test_config):
    env_name, num_envs, num_steps = (test_config["env_name"],
                                     test_config["num_envs"],
                                     test_config["num_steps"])
    state_shape = test_config.get("state_shape", None)

    vec_env, policy_wrapper = create_simple_policy_wrapper(env_name, num_envs, state_shape)
    obs = vec_env.reset()
    dones = np.full(shape=num_envs, fill_value=False)
    current_policies = [ptm.policy.constant for ptm in policy_wrapper.current_env_policies]
    num_identical = 0
    num_switches = 0
    for i in range(num_steps):
        actions, states = policy_wrapper.predict(obs, mask=dones)
        new_current_policies = [ptm.policy.constant for ptm in
                                policy_wrapper.current_env_policies]
        for ind, done in enumerate(dones):
            policies_match = new_current_policies[ind] == current_policies[ind]
            if not done:
                assert policies_match
            if done:
                num_switches += 1
                if policies_match:
                    num_identical += 1

        current_policies = new_current_policies
        obs, rew, dones, infos = vec_env.step(actions)

    if num_identical / num_switches == 1:
        raise AssertionError("Your policies never empirically change at episode boundaries")


MULTI_AGENT_ZOO_CONFIGS = [dict(env_name="multicomp/YouShallNotPassHumans-v0",
                                num_envs=8,
                                num_steps=1000)]


@pytest.mark.parametrize("test_config", MULTI_AGENT_ZOO_CONFIGS)
def test_constant_and_zoo_multi_agent_policy_wrapper(test_config):
    """Doesn't check policy switching, just ensures nothing breaks when you are wrapping and
    switching between both constant and zoo policies as your curried policy"""

    env_name, num_envs, num_steps = (test_config["env_name"],
                                     test_config["num_envs"],
                                     test_config["num_steps"])
    state_shape, victim_index, num_zoo = (test_config.get("state_shape", None),
                                          test_config.get("victim_index", 1),
                                          test_config.get("num_zoo", 5))

    vec_env, policy_wrapper, zoo_agent = create_multi_agent_curried_policy_wrapper(env_name,
                                                                                   num_envs,
                                                                                   victim_index,
                                                                                   state_shape,
                                                                                   add_zoo=True,
                                                                                   num_zoo=num_zoo)
    obs = vec_env.reset()
    dones = np.full(shape=num_envs, fill_value=False)
    for step in range(num_steps):
        actions, states = zoo_agent.predict(obs[0], mask=dones)
        actions_list = [actions]
        obs, rew, new_dones, infos = vec_env.step(actions_list)


MULTI_AGENT_CONFIGS = [dict(env_name="multicomp/YouShallNotPassHumans-v0",
                            num_envs=8,
                            num_steps=500),
                       dict(env_name="multicomp/KickAndDefend-v0",
                            num_envs=8,
                            num_steps=500)]


@pytest.mark.parametrize("test_config", MULTI_AGENT_CONFIGS)
def test_constant_multi_agent_multi_policy_wrapper(test_config):
    """Tests whether, when multi policy wrapper is acting as the curried policy within a
    multi-policy setting, the policies switch on episode boundaries as expected """

    env_name, num_envs, num_steps = (test_config["env_name"],
                                     test_config["num_envs"],
                                     test_config["num_steps"])
    state_shape, victim_index = (test_config.get("state_shape", None),
                                 test_config.get("victim_index", 1))

    vec_env, policy_wrapper, zoo_agent = create_multi_agent_curried_policy_wrapper(env_name,
                                                                                   num_envs,
                                                                                   victim_index,
                                                                                   state_shape)
    obs = vec_env.reset()
    state = None
    dones = np.full(shape=num_envs, fill_value=False)
    current_policies = [ptm.policy for ptm in policy_wrapper.current_env_policies]
    num_identical = 0
    num_switches = 0
    for step in range(num_steps):
        actions, state = zoo_agent.predict(obs[0], mask=dones, state=state)
        actions_list = [actions]
        obs, rew, new_dones, infos = vec_env.step(actions_list)

        new_current_policies = [ptm.policy for ptm in
                                policy_wrapper.current_env_policies]
        for ind, done in enumerate(dones):
            policies_match = new_current_policies[ind] == current_policies[ind]
            if not done:
                try:
                    assert policies_match, f"Policies did not match on index {ind}, step {step}"
                except AssertionError:
                    pdb.set_trace()
            if done:
                num_switches += 1
                if policies_match:
                    num_identical += 1
        dones = new_dones
        current_policies = new_current_policies

    if num_identical / num_switches == 1:
        raise AssertionError("Your policies never empirically change at episode boundaries")
