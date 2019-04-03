from abc import ABC, abstractmethod

from gym_compete.policy import LSTMPolicy
import numpy as np
from stable_baselines.common.policies import FeedForwardPolicy, nature_cnn
import tensorflow as tf

TRANSPARENCY_KEYS = ('obs', 'ff', 'hid')


class TransparentPolicy(ABC):
    @abstractmethod
    def get_obs_aug_amount(self):
        raise NotImplementedError()


class TransparentFeedForwardPolicy(TransparentPolicy, FeedForwardPolicy):
    """FeedForwardPolicy which is also transparent."""
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, transparent_params,
                 reuse=False, layers=None, net_arch=None, act_fun=tf.tanh,
                 cnn_extractor=nature_cnn, feature_extraction="cnn", **kwargs):
        """
        :param transparent_params: dict with potential keys 'obs', 'ff', 'hid'.
        If key is not present, then we don't provide this data as part of the data dict in step.
        If key is present, value (bool) corresponds to whether we augment the observation space with it.
        This is because TransparentCurryVecEnv needs this information to modify its observation space,
        and we would like to keep all of the transparency-related parameters in one dictionary.
        """
        FeedForwardPolicy.__init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                   layers, net_arch, act_fun, cnn_extractor, feature_extraction,
                                   **kwargs)
        self.transparent_params = transparent_params

    def get_obs_aug_amount(self):
        obs_aug_amount = 0
        obs_sizes = (self.ob_space.shape[0], sum(self.layers), None)
        for key, val in list(zip(TRANSPARENCY_KEYS, obs_sizes)):
            if self.transparent_params.get(key):
                obs_aug_amount += val
        return obs_aug_amount

    def step(self, obs, state=None, mask=None, deterministic=False):
        action_op = self.deterministic_action if deterministic else self.action
        action, value, neglogp, ff = self.sess.run([action_op, self._value, self.neglogp, self.pi_latent],
                                                   {self.obs_ph: obs})
        transparent_objs = (obs, ff, None)
        transparency_dict = {k: v for k, v in list(zip(TRANSPARENCY_KEYS, transparent_objs))
                             if k in self.transparent_params}
        return action, value, self.initial_state, neglogp, transparency_dict


class TransparentMlpPolicy(TransparentFeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, transparent_params,
                 reuse=False, **_kwargs):
        super(TransparentMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                   n_batch, transparent_params, reuse,
                                                   feature_extraction="mlp", **_kwargs)


class TransparentLSTMPolicy(TransparentPolicy, LSTMPolicy):
    """LSTMPolicy which also gives information about itself as outputs."""
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, hiddens,
                 transparent_params, scope="input", reuse=False, normalize=False):
        """
        :param transparent_params: dict with potential keys 'obs', 'ff', 'hid'.
        If key is not present, then we don't provide this data as part of the data dict in step.
        If key is present, value (bool) corresponds to whether we augment the observation space with it.
        This is because TransparentCurryVecEnv needs this information to modify its observation space,
        and we would like to keep all of the transparency-related parameters in one dictionary.
        """
        LSTMPolicy.__init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, hiddens,
                            scope, reuse, normalize)
        self.hiddens = hiddens
        self.transparent_params = transparent_params

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
