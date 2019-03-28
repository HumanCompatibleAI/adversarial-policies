from collections import deque

import numpy as np

from stable_baselines.common.vec_env import VecEnvWrapper


class LookbackRewardVecWrapper(VecEnvWrapper):
    """Retains information about prior episodes and rollouts to be used in k-lookback whitebox attacks"""
    def __init__(self, venv, policy, past_venvs, lookback_space=1):
        super().__init__(venv)
        # assume that past_venvs have get_state and set_state
        self._policy = policy
        self.past_venvs = past_venvs
        self.lookback_num = len(self.past_venvs)
        self.lookback_space = lookback_space

        self._action = None
        self._obs = None
        self._obs_queue = deque([], maxlen=self.lookback_num)
        self._state = None
        self._pseudo_base_state = None
        self._dones = [False] * self.num_envs
        self.ep_lens = np.zeros(self.num_envs)

        state_dict = {k: None for k in ('base_state', 'base_obs', 'base_action')}
        self.past_data = [state_dict.copy() for _ in range(self.lookback_num)]

    def step_async(self, actions):
        self.ep_lens += 1
        current_states = self.venv.env_method('get_state')

        # cycle the past_venvs and step all but the first. Then reset the first one with self.venv.
        self.past_venvs = [self.past_venvs[-1]] + self.past_venvs[:-1]
        self.past_data = [self.past_data[-1]] + self.past_data[:-1]
        for past_venv, past_dict in zip(self.past_venvs, self.past_data)[1:]:
            # base_action is calculated from the most recent step_wait()
            past_venv.step_async(past_dict['base_action'])

        for env_idx in range(self.num_envs):
            self.past_venvs[0].env_method('set_state', indices=env_idx, *[current_states[env_idx]])

        # the baseline policy's state is what it would have been if it had observed all of
        # the same things as our policy. self._pseudo_base_state comes from seeing only self._obs
        base_action, self._pseudo_base_state = self._policy.predict(self._obs, state=self._pseudo_base_state,
                                                                    mask=self._dones)
        self.past_data[0]['base_state'] = self._pseudo_base_state
        self.past_venvs[0].step_async(base_action)

    def step_wait(self):
        observations, rewards, self._dones, infos = self.venv.step_wait()
        self._process_own_obs(observations)
        past_data = [venv.step_wait() for venv in self.past_venvs]
        self._process_base_obs([data[0] for data in past_data])  # observations are first

        self.ep_lens *= ~np.array(self._dones)
        for env_idx in range(self.num_envs):
            if self._dones[env_idx]:
                # align this env with self.venv since self.venv was reset
                self._reset_state_data(observations, env_idx)
            # TODO: process rewards based on past_data and self.ep_lens
            valid_baseline_times = self.past_data[:self.ep_lens[env_idx]]
            for data in valid_baseline_times:
                # rew[env_idx] += f(data)
                pass

        return observations, rewards, self._dones, infos

    def reset(self):
        observations = self.venv.reset()
        self._process_own_obs(observations)
        for env_idx in range(self.num_envs):
            # align all of the past_venvs with self.venv
            self._reset_state_data(observations, env_idx)
        return observations

    def _process_own_obs(self, observations):
        """Record action, state and observation of our policy"""
        self._obs = observations
        self._obs_queue.append(observations)
        self._action, self._state = self._policy.predict(self._obs, state=self._state,
                                                         mask=self._dones)

    def _process_base_obs(self, past_obs_list):
        """Record action, state and observation of baseline policy"""
        for idx, base_obs in enumerate(past_obs_list):
            base_action, base_state = self._policy.predict(base_obs, state=self.past_data[idx]['base_state'],
                                                           mask=self._dones)
            self.past_data[idx]['base_action'] = base_action
            self.past_data[idx]['base_state'] = base_state
            self.past_data[idx]['base_obs'] = base_obs

    def _reset_state_data(self, observations, env_idx):
        """Reset past_venv states when self.venv resets. Also reset data for baseline policy."""
        initial_state = self.venv.env_method('get_state', indices=env_idx)[0]
        initial_base_obs = observations[env_idx]
        for past_step, past_venv in enumerate(self.past_venvs):
            self.past_data[past_step] = {'base_obs': initial_base_obs,
                                         'base_state': None,
                                         'base_action': None}
            past_venv.env_method('set_state', indices=env_idx, *[initial_state])
