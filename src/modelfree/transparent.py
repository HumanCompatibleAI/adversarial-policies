from abc import ABC, abstractmethod

import gym
from gym_compete.policy import LSTMPolicy, MlpPolicyValue
import numpy as np
from stable_baselines.common.policies import FeedForwardPolicy, nature_cnn
import tensorflow as tf
import pickle
from aprl.envs.multi_agent import CurryVecEnv, _tuple_pop, _tuple_space_augment

TRANSPARENCY_KEYS = ('obs', 'ff', 'hid')


class TransparentPolicy(ABC):
    def __init__(self, transparent_params, policy=None):
        self.transparent_params = transparent_params
        self.policy = policy

    @abstractmethod
    def get_obs_aug_amount(self):
        raise NotImplementedError()


class TransparentFeedForwardPolicy(TransparentPolicy, FeedForwardPolicy):
    """FeedForwardPolicy which is also transparent."""
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, transparent_params,
                 reuse=False, layers=None, net_arch=None, act_fun=tf.tanh,
                 cnn_extractor=nature_cnn, feature_extraction="cnn", policy=None, **kwargs):
        """
        :param transparent_params: dict with potential keys 'obs', 'ff', 'hid'.
        If key is not present, then we don't provide this data as part of the data dict in step.
        If key is present, value (bool) corresponds to whether we augment the observation space
        with it. This is because TransparentCurryVecEnv needs this information to modify its
        observation space, and we want all of the transparency-related parameters in one dict.
        """
        FeedForwardPolicy.__init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                   layers, net_arch, act_fun, cnn_extractor, feature_extraction,
                                   **kwargs)
        TransparentPolicy.__init__(self, transparent_params, policy=policy)

    def get_obs_aug_amount(self):
        obs_aug_amount = 0
        obs_sizes = (self.ob_space.shape[0], sum(self.layers), None)
        for key, val in list(zip(TRANSPARENCY_KEYS, obs_sizes)):
            if self.transparent_params.get(key):
                obs_aug_amount += val
        return obs_aug_amount

    def step(self, obs, state=None, mask=None, deterministic=False):
        actor = self.policy if self.policy is not None else self
        action_op = self.deterministic_action if deterministic else self.action
        action, value, neglogp, ff = self.sess.run([action_op, self._value, self.neglogp,
                                                    self.ff_out], {self.obs_ph: obs})
        transparent_objs = (obs, ff, None)
        transparency_dict = {k: v for k, v in list(zip(TRANSPARENCY_KEYS, transparent_objs))
                             if k in self.transparent_params}
        return action, value, actor.initial_state, neglogp, transparency_dict


class TransparentMlpPolicyWrapper(TransparentFeedForwardPolicy):
    def __init__(self, policy, transparent_params):
        self.policy = policy
        self.policy.transparent_params = transparent_params
        self.transparent_params = transparent_params

    def get_obs_aug_amount(self):
        return 0

    def step(self, obs, state=None, mask=None, deterministic=False):
        return TransparentFeedForwardPolicy.step(self.policy, obs, state, mask, deterministic)

    def proba_step(self, obs, state=None, mask=None):
        return TransparentFeedForwardPolicy.proba_step(self.policy, obs, state, mask)

    def value(self, obs, state=None, mask=None):
        return TransparentFeedForwardPolicy.value(self.policy, obs, state, mask)


class TransparentMlpPolicy(TransparentFeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, transparent_params,
                 reuse=False, **_kwargs):
        super(TransparentMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                   n_batch, transparent_params, reuse,
                                                   feature_extraction="mlp", **_kwargs)


class TransparentLSTMPolicy(TransparentPolicy, LSTMPolicy):
    """LSTMPolicy which also gives information about itself as outputs."""
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, transparent_params,
                 hiddens=None, scope="input", reuse=False, normalize=False):
        """
        :param transparent_params: dict with potential keys 'obs', 'ff', 'hid'.
        If key is not present, then we don't provide this data as part of the data dict in step.
        If key is present, value (bool) corresponds to whether we augment the observation space
        with it. This is because TransparentCurryVecEnv needs this information to modify its
        observation space, and we want all of the transparency-related parameters in one dict.
        """
        LSTMPolicy.__init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, hiddens,
                            scope, reuse, normalize)
        TransparentPolicy.__init__(self, transparent_params)

    def get_obs_aug_amount(self):
        obs_aug_amount = 0
        obs_sizes = (self.ob_space.shape[0], self.hiddens[-2], self.hiddens[-1])
        for key, val in list(zip(TRANSPARENCY_KEYS, obs_sizes)):
            if self.transparent_params.get(key):
                obs_aug_amount += val
        return obs_aug_amount

    def step(self, obs, state=None, mask=None, deterministic=False):
        action = self.deterministic_action if deterministic else self.action
        outputs = [action, self._value, self.state_out, self.neglogp, self.ff_out]
        feed_dict = self._make_feed_dict(obs, state, mask)
        a, v, s, neglogp, ff = self.sess.run(outputs, feed_dict)
        state = []
        for x in s:
            state.append(x.c)
            state.append(x.h)
        state = np.array(state)
        state = np.transpose(state, (1, 0, 2))

        transparent_objs = (obs, ff, state)
        transparency_dict = {k: v for k, v in list(zip(TRANSPARENCY_KEYS, transparent_objs))
                             if k in self.transparent_params}
        return a, v, state, neglogp, transparency_dict


class TransparentMlpPolicyValue(TransparentPolicy, MlpPolicyValue):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, transparent_params,
                 hiddens=None, scope="input", reuse=False, normalize=False):
        """
        :param transparent_params: dict with potential keys 'obs', 'ff', 'hid'.
        If key is not present, then we don't provide this data as part of the data dict in step.
        If key is present, value (bool) corresponds to whether we augment the observation space
        with it. This is because TransparentCurryVecEnv needs this information to modify its
        observation space, and we want all of the transparency-related parameters in one dict.
        """
        MlpPolicyValue.__init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, hiddens,
                            scope, reuse, normalize)
        TransparentPolicy.__init__(self, transparent_params)

    def get_obs_aug_amount(self):
        obs_aug_amount = 0
        obs_sizes = (self.ob_space.shape[0], sum(self.hiddens), None)
        for key, val in list(zip(TRANSPARENCY_KEYS, obs_sizes)):
            if self.transparent_params.get(key):
                obs_aug_amount += val
        return obs_aug_amount

    def step(self, obs, state=None, mask=None, deterministic=False):
        action = self.deterministic_action if deterministic else self.action
        outputs = [action, self._value, self.neglogp, self.ff_out]
        a, v, neglogp, ff = self.sess.run(outputs, {self.obs_ph: obs})
        return a, v, self.initial_state, neglogp, ff


class TransparentCurryVecEnv(CurryVecEnv):
    """CurryVecEnv that gives out much more info about its policy."""
    def __init__(self, venv, policy, agent_idx=0):
        super().__init__(venv, policy, agent_idx)

        # this is super annoying. Need to make general solution for stable_baselines/gym_compete.
        self.underlying_policy = policy.policy
        if not isinstance(self.underlying_policy, TransparentPolicy):
            raise TypeError("Error: policy must be transparent")
        self._action = None
        self.debug_file = None
        self.debug_dict = {}
        self.t = 0

        obs_aug_amount = self.underlying_policy.get_obs_aug_amount()
        if obs_aug_amount > 0:
            obs_aug_space = gym.spaces.Box(-np.inf, np.inf, obs_aug_amount)
            self.observation_space = _tuple_space_augment(self.observation_space, agent_idx,
                                                          augment_space=obs_aug_space)

    def step_async(self, actions):
        self._action, self._state, self._data = self._policy.predict(self._obs, state=self._state,
                                                                     mask=self._dones)
        actions.insert(self._agent_to_fix, self._action)
        all_state_data = self.venv.unwrapped.env_method('get_full_state')[0]
        if self.debug_file is not None:
            self.debug_dict.update({'actions': actions, 'env': 'curry', 't': self.t, 'data': self._data})
        self.venv.step_async(actions)

    def step_wait(self):
        observations, rewards, self._dones, infos = self.venv.step_wait()
        observations, self._obs = _tuple_pop(observations, self._agent_to_fix)
        if self.debug_file is not None:
            self.debug_dict.update({'obs': observations, 'rewards': rewards})
            pickle.dump(self.debug_dict, self.debug_file)
            self.t += 1
        for env_idx in range(self.num_envs):
            infos[env_idx][self._agent_to_fix].update(self._data)
        return observations, rewards, self._dones, infos

    def reset(self):
        observations = self.venv.reset()
        observations, self._obs = _tuple_pop(observations, self._agent_to_fix)
        return observations
