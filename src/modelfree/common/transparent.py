from abc import ABC

import gym
import numpy as np
from stable_baselines.common.policies import FeedForwardPolicy, nature_cnn
import tensorflow as tf

from aprl.envs.multi_agent import CurryVecEnv, _tuple_pop
from modelfree.common.utils import _filter_dict

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


class TransparentCurryVecEnv(CurryVecEnv):
    """CurryVecEnv that provides transparency data about its policy by updating infos dicts."""
    def __init__(self, venv, policy, agent_idx=0):
        """
        :param venv (VecMultiEnv): the environments
        :param policy (BaseRLModel): model which wraps a BasePolicy object
        :param agent_idx (int): the index of the agent that should be fixed.
        :return: a new VecMultiEnv with num_agents decremented. It behaves like env but
                 with all actions at index agent_idx set to those returned by agent."""
        super().__init__(venv, policy, agent_idx)
        if not hasattr(self._policy.policy, 'step_transparent'):
            raise TypeError("Error: policy must be transparent")
        self._action = None

    def step_async(self, actions):
        policy_out = self._policy.predict_transparent(self._obs, state=self._state,
                                                      mask=self._dones)
        self._action, self._state, self._data = policy_out
        actions.insert(self._agent_to_fix, self._action)
        self.venv.step_async(actions)

    def step_wait(self):
        observations, rewards, self._dones, infos = self.venv.step_wait()
        observations, self._obs = _tuple_pop(observations, self._agent_to_fix)
        for env_idx in range(self.num_envs):
            env_data = {k: v[env_idx] for k, v in self._data.items()}
            infos[env_idx][self._agent_to_fix].update(env_data)
        return observations, rewards, self._dones, infos
