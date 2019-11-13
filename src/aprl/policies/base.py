"""RL policies, models and adaptor classes."""

from typing import Optional, Type

import gym
import numpy as np
from stable_baselines.common import BaseRLModel
from stable_baselines.common.policies import BasePolicy
import tensorflow as tf


class PredictOnlyModel(BaseRLModel):
    """Abstract class for policies pretending to be RL algorithms (models).

    Provides stub implementations that raise NotImplementedError.
    The predict method is left as abstract and must be implemented in base class."""

    def __init__(self,
                 policy: Type[BasePolicy],
                 sess: Optional[tf.Session],
                 observation_space: gym.Space,
                 action_space: gym.Space):
        """Constructs a DummyModel with given policy and session.
        :param policy: (BasePolicy) a loaded policy.
        :param sess: (tf.Session or None) a TensorFlow session.
        :return an instance of BaseRLModel.
        """
        super().__init__(policy=policy, env=None, requires_vec_env=True, policy_base='Dummy')
        self.sess = sess
        self.observation_space = observation_space
        self.action_space = action_space

    def setup_model(self):
        raise NotImplementedError()

    def learn(self):
        raise NotImplementedError()

    def action_probability(self, observation, state=None, mask=None, actions=None):
        raise NotImplementedError()

    def save(self, save_path):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def _get_pretrain_placeholders(self):
        raise NotImplementedError()

    def get_parameter_list(self):
        raise NotImplementedError()


class ModelWrapper(PredictOnlyModel):
    """Base class for wrapping RL algorithms (models)."""

    def __init__(self, model: BaseRLModel):
        super().__init__(policy=model.policy,
                         sess=model.sess,
                         observation_space=model.observation_space,
                         action_space=model.action_space)
        self.model = model


class PolicyToModel(PredictOnlyModel):
    """Converts BasePolicy to a BaseRLModel with only predict implemented."""

    def __init__(self, policy_obj: BasePolicy):
        """Constructs a BaseRLModel using policy for predictions.
        :param policy: a loaded policy.
        :return an instance of BaseRLModel.
        """
        super().__init__(policy=type(policy_obj),
                         sess=policy_obj.sess,
                         observation_space=policy_obj.ob_space,
                         action_space=policy_obj.ac_space)
        self.policy_obj = policy_obj

    def _get_policy_out(self, observation, state, mask, transparent, deterministic=False):
        if state is None:
            state = self.policy_obj.initial_state
        if mask is None:
            mask = [False for _ in range(self.policy_obj.n_env)]

        step_fn = self.policy_obj.step_transparent if transparent else self.policy_obj.step
        return step_fn(observation, state, mask, deterministic=deterministic)

    def predict(self, observation, state=None, mask=None, deterministic=False):
        policy_out = self._get_policy_out(observation, state, mask, transparent=False,
                                          deterministic=deterministic)
        actions, _val, states, _neglogp = policy_out
        return actions, states

    def predict_transparent(self, observation, state=None, mask=None, deterministic=False):
        """Returns same values as predict, as well as a dictionary with transparent data."""
        policy_out = self._get_policy_out(observation, state, mask, transparent=True,
                                          deterministic=deterministic)
        actions, _val, states, _neglogp, data = policy_out
        return actions, states, data


class OpenAIToStablePolicy(BasePolicy):
    """Converts an OpenAI Baselines Policy to a Stable Baselines policy."""

    def __init__(self, old_policy, ob_space: gym.Space, ac_space: gym.Space):
        super().__init__(sess=old_policy.sess, ob_space=ob_space, ac_space=ac_space,
                         n_env=1, n_steps=1, n_batch=1)
        self.old = old_policy

    @property
    def initial_state(self):
        return self.old.initial_state

    def step(self, obs, state=None, mask=None, deterministic=False):
        stochastic = not deterministic
        return self.old.step(obs, S=state, M=mask, stochastic=stochastic)

    def proba_step(self, obs, state=None, mask=None):
        raise NotImplementedError()


class ConstantPolicy(BasePolicy):
    """Policy that returns a constant action."""

    def __init__(self, env, constant):
        assert env.action_space.contains(constant)
        super().__init__(sess=None,
                         ob_space=env.observation_space,
                         ac_space=env.action_space,
                         n_env=env.num_envs,
                         n_steps=1,
                         n_batch=1)
        self.constant = constant

    def step(self, obs, state=None, mask=None, deterministic=False):
        actions = np.array([self.constant] * self.n_env)
        return actions, None, None, None

    def proba_step(self, obs, state=None, mask=None):
        return self.step(obs, state=state, mask=mask)


class ZeroPolicy(ConstantPolicy):
    """Policy that returns a zero action."""

    def __init__(self, env):
        super().__init__(env, np.zeros(env.action_space.shape))


class RandomPolicy(BasePolicy):
    def __init__(self, env):
        super().__init__(sess=None,
                         ob_space=env.observation_space,
                         ac_space=env.action_space,
                         n_env=env.num_envs,
                         n_steps=1,
                         n_batch=1)

    def step(self, obs, state=None, mask=None, deterministic=False):
        actions = np.array([self.ac_space.sample() for _ in range(self.n_env)])
        return actions, None, None, None

    def proba_step(self, obs, state=None, mask=None):
        raise NotImplementedError()
