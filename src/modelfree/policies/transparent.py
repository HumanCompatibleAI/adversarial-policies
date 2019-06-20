"""Policies """

from abc import ABC

import numpy as np
from stable_baselines.common.policies import FeedForwardPolicy, nature_cnn
import tensorflow as tf

from modelfree.envs.wrappers import _filter_dict

TRANSPARENCY_KEYS = set(['obs', 'ff_policy', 'ff_value', 'hid'])


class TransparentPolicy(ABC):
    """Policy which returns its observations and/or activations in its call to self.predict

    :param transparent_params: (set) a subset of TRANSPARENCY_KEYS.
           If key is present, that data will be included in the transparency_dict
           returned in step_transparent.
    """
    def __init__(self, transparent_params):
        if transparent_params is None:
            transparent_params = set()
        unexpected_keys = set(transparent_params).difference(TRANSPARENCY_KEYS)
        if unexpected_keys:
            raise KeyError(f"Unrecognized transparency keys: {unexpected_keys}")
        self.transparent_params = transparent_params

    def _get_default_transparency_dict(self, obs, ff, hid):
        """This structure is typical for subclasses of TransparentPolicy

        :param obs: ([float]) array of observations
        :param ff: (dict<str>, [float]) dictionary of lists of feedforward activations.
        :param hid: ([float] or None) LSTM hidden state.
        """
        # TODO: Do not consolidate -- have this happen later down the pipeline.
        def consolidate(acts):
            """Turn a list of activations into one array with shape (num_env,) + action_space"""
            return np.squeeze(np.concatenate(acts))

        transparency_dict = {'obs': obs, 'hid': hid,
                             'ff_policy': consolidate(ff['policy']),
                             'ff_value': consolidate(ff['value'])}
        transparency_dict = _filter_dict(transparency_dict, self.transparent_params)
        return transparency_dict


class TransparentFeedForwardPolicy(TransparentPolicy, FeedForwardPolicy):
    """stable_baselines FeedForwardPolicy which is also transparent."""
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, transparent_params,
                 reuse=False, layers=None, net_arch=None, act_fun=tf.tanh,
                 cnn_extractor=nature_cnn, feature_extraction="cnn", **kwargs):
        FeedForwardPolicy.__init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                   layers, net_arch, act_fun, cnn_extractor, feature_extraction,
                                   **kwargs)
        TransparentPolicy.__init__(self, transparent_params)

    def step_transparent(self, obs, state=None, mask=None, deterministic=False):
        action_op = self.deterministic_action if deterministic else self.action
        outputs = [action_op, self.value_flat, self.neglogp, self.ff_out]

        action, value, neglogp, ff = self.sess.run(outputs, {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp, ff


class TransparentMlpPolicy(TransparentFeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, transparent_params,
                 reuse=False, **_kwargs):
        super(TransparentMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                   n_batch, transparent_params, reuse,
                                                   feature_extraction="mlp", **_kwargs)
