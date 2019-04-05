"""Test environments from gym_compete and our wrappers around them."""

import gym
import pytest
from tests.aprl.test_envs import check_env, check_random_rollout

from modelfree.envs.gym_compete import GymCompeteToOurs

spec_list = [spec
             for spec in sorted(gym.envs.registry.all(), key=lambda x: x.id)
             if spec.id.startswith('multicomp/')]


def test_envs_exist():
    assert len(spec_list) > 0, "No multi-agent competition environments detected"


def spec_to_env(fn):
    def helper(spec):
        their_env = spec.make()
        our_env = GymCompeteToOurs(their_env)
        return fn(our_env)
    return helper


test_env = pytest.mark.parametrize("spec", spec_list)(spec_to_env(check_env))
test_random_rollout = pytest.mark.parametrize("spec", spec_list)(spec_to_env(check_random_rollout))
