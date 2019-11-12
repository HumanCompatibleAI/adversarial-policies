"""RL policies, models and adaptor classes."""

import numpy as np
from stable_baselines.common import BaseRLModel
from stable_baselines.common.policies import BasePolicy


class DummyModel(BaseRLModel):
    """Abstract class for policies pretending to be RL algorithms (models).

    Provides stub implementations that raise NotImplementedError.
    The predict method is left as abstract and must be implemented in base class."""

    def __init__(self, policy, sess):
        """Constructs a DummyModel with given policy and session.
        :param policy: (BasePolicy) a loaded policy.
        :param sess: (tf.Session or None) a TensorFlow session.
        :return an instance of BaseRLModel.
        """
        super().__init__(policy=policy, env=None, requires_vec_env=True, policy_base='Dummy')
        self.sess = sess

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


class PolicyToModel(DummyModel):
    """Converts BasePolicy to a BaseRLModel with only predict implemented."""

    def __init__(self, policy):
        """Constructs a BaseRLModel using policy for predictions.
        :param policy: (BasePolicy) a loaded policy.
        :return an instance of BaseRLModel.
        """
        super().__init__(policy=policy, sess=policy.sess)

    def _get_policy_out(self, observation, state, mask, transparent, deterministic=False):
        if state is None:
            state = self.policy.initial_state
        if mask is None:
            mask = [False for _ in range(self.policy.n_env)]

        step_fn = self.policy.step_transparent if transparent else self.policy.step
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

    def __init__(self, old_policy):
        self.old = old_policy
        self.sess = old_policy.sess

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
