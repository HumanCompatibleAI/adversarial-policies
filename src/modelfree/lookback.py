from collections import defaultdict

import numpy as np

from stable_baselines.common.vec_env import VecEnvWrapper


class LookbackRewardVecWrapper(VecEnvWrapper):
    """Retains information about prior episodes and rollouts to be used in k-lookback whitebox attacks"""
    def __init__(self, venv, policy, base_venvs, lookback_space=1):
        super().__init__(venv)
        # assume that base_venvs have get_state and set_state
        self._policy = policy
        self.base_venvs = base_venvs
        self.lookback_num = len(self.base_venvs)
        self.lookback_space = lookback_space

        self._action = None
        self._obs = None
        self._state = None
        self._pseudo_base_state = None
        self._dones = [False] * self.num_envs
        self.ep_lens = np.zeros(self.num_envs).astype(int)

        state_dict = {'state': None, 'action': None, 'reward': 0, 'info': defaultdict(dict)}
        self.base_data = [state_dict.copy() for _ in range(self.lookback_num)]

    def step_async(self, actions):
        current_states = self.venv.unwrapped.env_method('get_state')

        # cycle the base_venvs and step all but the first. Then reset the first one with self.venv.
        self.base_venvs = [self.base_venvs[-1]] + self.base_venvs[:-1]
        self.base_data = [self.base_data[-1]] + self.base_data[:-1]
        for base_venv, base_dict in list(zip(self.base_venvs, self.base_data))[1:]:
            # base_action is calculated from reset() or the most recent step_wait()
            base_venv.step_async(base_dict['action'])

        new_baseline_venv = self.base_venvs[0]
        for env_idx in range(self.num_envs):
            new_baseline_venv.unwrapped.env_method('set_state', env_idx, current_states[env_idx])

        # the baseline policy's state is what it would have been if it had observed all of
        # the same things as our policy. self._pseudo_base_state comes from seeing only self._obs
        base_action, self._pseudo_base_state = self._policy.predict(self._obs, state=self._pseudo_base_state,
                                                                    mask=self._dones)
        self.base_data[0]['state'] = self._pseudo_base_state
        new_baseline_venv.step_async(base_action)
        self.venv.step_async(actions)
        self.ep_lens += 1

    def step_wait(self):
        observations, rewards, self._dones, infos = self.venv.step_wait()
        self._process_own_obs(observations)
        base_data = [venv.step_wait() for venv in self.base_venvs]
        self._process_base_data(base_data)

        self.ep_lens *= ~np.array(self._dones)
        for env_idx in range(self.num_envs):
            if self._dones[env_idx]:
                # align this env with self.venv since self.venv was reset
                self._reset_state_data(observations, env_idx)
            valid_baseline_dicts = self.base_data[:self.ep_lens[env_idx]]
            for base_dict in valid_baseline_dicts:
                diff_reward = rewards[env_idx] - base_dict['reward'][env_idx]
                rewards[env_idx] += diff_reward
        return observations, rewards, self._dones, infos

    def reset(self):
        observations = self.venv.reset()
        self._process_own_obs(observations)
        self._reset_state_data(observations)
        return observations

    def _process_own_obs(self, observations):
        """Record action, state and observations of our policy"""
        self._obs = observations
        self._action, self._state = self._policy.predict(self._obs, state=self._state,
                                                         mask=self._dones)

    def _process_base_data(self, base_data):
        """Record action and state of baseline policy
        :param base_data: list of (observations, rewards, dones, infos), one for each base_venv
        """
        for idx, (base_obs, base_reward, _, base_info) in enumerate(base_data):
            base_action, base_state = self._policy.predict(base_obs, state=self.base_data[idx]['state'],
                                                           mask=self._dones)
            self.base_data[idx]['action'] = base_action
            self.base_data[idx]['state'] = base_state
            self.base_data[idx]['reward'] = base_reward
            for env_idx in range(self.num_envs):
                self.base_data[idx]['info'][env_idx].update(base_info[env_idx])

    def _reset_state_data(self, initial_observations, env_idx=None):
        """Reset base_venv states when self.venv resets. Also reset data for baseline policy."""
        action, state = self._policy.predict(initial_observations, state=None, mask=None)
        initial_env_states = self.venv.unwrapped.env_method('get_state', env_idx)
        for base_dict, base_venv in list(zip(self.base_data, self.base_venvs)):
            if env_idx is None:
                # this gets called only in self.reset()
                base_venv.reset()
                base_dict['action'] = action
                base_dict['state'] = state
                base_dict['reward'] *= 0
                base_dict['info'] = defaultdict(dict)
            else:
                # this gets called when an episode ends in one of the environments
                base_dict['action'][env_idx] = action[env_idx]
                base_dict['state'][:, env_idx, :] = state[:, env_idx, :]
                base_dict['reward'][env_idx] = 0
                base_dict['info'][env_idx] = {}
            envs_iter = range(self.num_envs) if env_idx is None else (0,)
            for env_to_set in envs_iter:
                base_venv.unwrapped.env_method('set_state', env_to_set, initial_env_states[env_to_set])
