from abc import ABC, abstractmethod

import gym
from gym_compete.policy import LSTMPolicy, MlpPolicyValue
import numpy as np
from stable_baselines.common.policies import FeedForwardPolicy, nature_cnn
import tensorflow as tf

from aprl.envs.multi_agent import CurryVecEnv, _tuple_pop, _tuple_space_augment

TRANSPARENCY_KEYS = ('obs', 'ff_policy', 'ff_value', 'hid')


class TransparentPolicy(ABC):
    """Policy which returns its observations and/or activations in its call to self.predict

    :param transparent_params: dict with potential keys in TRANSPARENCY_KEYS
    - If key is not present, then we don't provide this data as part of the data dict in step.
    - If key is present, value (bool) corresponds to whether we augment the observation space
    with it. This is because TransparentCurryVecEnv needs this information to modify its
    observation space, and we want all of the transparency-related parameters in one dict.
    """
    def __init__(self, transparent_params):
        if transparent_params is None:
            raise ValueError("TransparentPolicy requires transparent_params.")
        for key in transparent_params:
            if key not in TRANSPARENCY_KEYS:
                raise KeyError(f"Unrecognized transparency key: {key}")
        self.transparent_params = transparent_params

    @abstractmethod
    def get_obs_sizes(self):
        """List of dimensionalities of different data that could be appended to an observation.
        Specifically (observations, (concatenated) policy acts., value acts., and hidden state)

        This is used in get_obs_aug_amount, which has the common logic for all subclasses.
        If class does not support a key, its value should be None.

        :return: (list<int,None>) sizes of data from TRANSPARENCY_KEYS for this policy.
        """
        raise NotImplementedError()

    def get_obs_aug_amount(self):
        """Execute loop for calculating amount by which to augment observations"""
        obs_aug_amount = 0
        obs_sizes = self.get_obs_sizes()
        for key, val in list(zip(TRANSPARENCY_KEYS, obs_sizes)):
            if self.transparent_params.get(key):
                obs_aug_amount += val
        return obs_aug_amount

    def get_transparency_dict(self, transparent_objs):
        """Construct dictionary of transparent data

        :param transparent_objs: (tuple<np.ndarray,None>) values for keys in TRANSPARENCY_KEYS
        :return: (dict[str, np.ndarray]) dict with all exposed data
        """
        transparency_dict = {k: np.squeeze(v) for k, v in zip(TRANSPARENCY_KEYS, transparent_objs)
                             if k in self.transparent_params}
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

    def get_obs_sizes(self):
        return self.ob_space.shape[0], sum(self.layers), sum(self.layers), None

    def step(self, obs, state=None, mask=None, deterministic=False):
        action_op = self.deterministic_action if deterministic else self.action
        action, value, neglogp, ff = self.sess.run([action_op, self.value_flat, self.neglogp,
                                                    self.ff_out], {self.obs_ph: obs})

        transparent_objs = (obs, np.concatenate(ff['policy']), np.concatenate(ff['value']), None)
        transparency_dict = self.get_transparency_dict(transparent_objs)
        return action, value, self.initial_state, neglogp, transparency_dict


class TransparentMlpPolicy(TransparentFeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, transparent_params,
                 reuse=False, **_kwargs):
        super(TransparentMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                   n_batch, transparent_params, reuse,
                                                   feature_extraction="mlp", **_kwargs)


class TransparentLSTMPolicy(TransparentPolicy, LSTMPolicy):
    """gym_compete LSTMPolicy which also transparent."""
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, transparent_params,
                 hiddens=None, scope="input", reuse=False, normalize=False):
        LSTMPolicy.__init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, hiddens,
                            scope, reuse, normalize)
        TransparentPolicy.__init__(self, transparent_params)

    def get_obs_sizes(self):
        ff_size = sum(self.hiddens[:-1])
        return self.ob_space.shape[0], ff_size, ff_size, self.hiddens[-1]

    def step(self, obs, state=None, mask=None, deterministic=False):
        action = self.deterministic_action if deterministic else self.action
        outputs = [action, self.value_flat, self.state_out, self.neglogp, self.ff_out]
        feed_dict = self._make_feed_dict(obs, state, mask)
        a, v, s, neglogp, ff = self.sess.run(outputs, feed_dict)
        state = []
        for x in s:
            state.append(x.c)
            state.append(x.h)
        state = np.array(state)
        state = np.transpose(state, (1, 0, 2))

        # right now 'hid' only supports hidden state of policy
        # which is the last of the four state vectors
        transparent_objs = (obs, np.concatenate(ff['policy']),
                            np.concatenate(ff['value']), state[:, -1, :])
        transparency_dict = self.get_transparency_dict(transparent_objs)
        return a, v, state, neglogp, transparency_dict


class TransparentMlpPolicyValue(TransparentPolicy, MlpPolicyValue):
    """gym_compete MlpPolicyValue which is also transparent."""
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, transparent_params,
                 hiddens=None, scope="input", reuse=False, normalize=False):
        MlpPolicyValue.__init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                hiddens=hiddens, scope=scope, reuse=reuse, normalize=normalize)
        TransparentPolicy.__init__(self, transparent_params)

    def get_obs_sizes(self):
        return self.ob_space.shape[0], sum(self.hiddens), sum(self.hiddens), None

    def step(self, obs, state=None, mask=None, deterministic=False):
        action = self.deterministic_action if deterministic else self.action
        outputs = [action, self.value_flat, self.neglogp, self.ff_out]
        a, v, neglogp, ff = self.sess.run(outputs, {self.obs_ph: obs})

        transparent_objs = (obs, np.concatenate(ff['policy']), np.concatenate(ff['value']), None)
        transparency_dict = self.get_transparency_dict(transparent_objs)
        return a, v, self.initial_state, neglogp, transparency_dict


class TransparentCurryVecEnv(CurryVecEnv):
    """CurryVecEnv that provides transparency data about its policy in its infos dicts."""
    def __init__(self, venv, policy, agent_idx=0):
        super().__init__(venv, policy, agent_idx)
        self.underlying_policy = policy.policy
        if not isinstance(self.underlying_policy, TransparentPolicy):
            raise TypeError("Error: policy must be transparent")
        self._action = None

        obs_aug_amount = self.underlying_policy.get_obs_aug_amount()
        if obs_aug_amount > 0:
            obs_aug_space = gym.spaces.Box(-np.inf, np.inf, (obs_aug_amount,))
            self.observation_space = _tuple_space_augment(self.observation_space, agent_idx,
                                                          augment_space=obs_aug_space)

    def step_async(self, actions):
        actions.insert(self._agent_to_fix, self._action)
        self.venv.step_async(actions)

    def step_wait(self):
        observations, rewards, self._dones, infos = self.venv.step_wait()
        observations = self._get_updated_obs(observations)
        for env_idx in range(self.num_envs):
            env_data = {k: v[env_idx] for k, v in self._data.items()}
            infos[env_idx][self._agent_to_fix].update(env_data)
        return observations, rewards, self._dones, infos

    def reset(self):
        observations = self._get_updated_obs(self.venv.reset())
        return observations

    def _get_updated_obs(self, observations):
        observations, self._obs = _tuple_pop(observations, self._agent_to_fix)
        self._action, self._state, self._data = self._policy.predict(self._obs, state=self._state,
                                                                     mask=self._dones)
        # we assume that there is only one other agent in the MultiEnv.
        assert len(observations) == 1
        obs_copy = observations[0]
        for k in TRANSPARENCY_KEYS:
            if k in self._data and self.underlying_policy.transparent_params.get(k):
                obs_copy = np.concatenate([obs_copy, self._data[k]], -1)
        return (obs_copy,)
