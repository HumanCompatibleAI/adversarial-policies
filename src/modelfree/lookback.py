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
        self._state = None
        self._pseudo_base_state = None
        self._dones = [False] * self.num_envs
        self.ep_lens = np.zeros(self.num_envs).astype(int)

        state_dict = {'state': None, 'action': None, 'reward': 0, 'info': [{}]}
        self.past_data = [state_dict.copy() for _ in range(self.lookback_num)]

    def step_async(self, actions):
        current_states = self.venv.unwrapped.env_method('get_state')

        # cycle the past_venvs and step all but the first. Then reset the first one with self.venv.
        self.past_venvs = [self.past_venvs[-1]] + self.past_venvs[:-1]
        self.past_data = [self.past_data[-1]] + self.past_data[:-1]
        for past_venv, past_dict in list(zip(self.past_venvs, self.past_data))[1:]:
            # base_action is calculated from reset() or the most recent step_wait()
            past_venv.step_async(past_dict['action'])

        new_baseline_venv = self.past_venvs[0]
        for env_idx in range(self.num_envs):
            new_baseline_venv.unwrapped.env_method('set_state', env_idx, current_states[env_idx])

        # the baseline policy's state is what it would have been if it had observed all of
        # the same things as our policy. self._pseudo_base_state comes from seeing only self._obs
        base_action, self._pseudo_base_state = self._policy.predict(self._obs, state=self._pseudo_base_state,
                                                                    mask=self._dones)
        self.past_data[0]['state'] = self._pseudo_base_state
        new_baseline_venv.step_async(base_action)
        self.venv.step_async(actions)
        self.ep_lens += 1

    def step_wait(self):
        observations, rewards, self._dones, infos = self.venv.step_wait()
        self._process_own_obs(observations)
        past_data = [venv.step_wait() for venv in self.past_venvs]
        self._process_base_data(past_data)

        self.ep_lens *= ~np.array(self._dones)
        for env_idx in range(self.num_envs):
            if self._dones[env_idx]:
                # align this env with self.venv since self.venv was reset
                self._reset_state_data(observations, env_idx)
            # TODO: process rewards based on past_data and self.ep_lens
            valid_baseline_times = self.past_data[:self.ep_lens[env_idx]]
            for data in valid_baseline_times:
                base_reward = data['reward']
                # rew[env_idx] += f(data)
                pass

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

    def _process_base_data(self, past_data):
        """Record action and state of baseline policy
        :param past_data: list of (observations, rewards, dones, infos), one for each past_venv
        """
        for idx, data in enumerate(past_data):
            base_obs = data[0]
            base_action, base_state = self._policy.predict(base_obs, state=self.past_data[idx]['state'],
                                                           mask=self._dones)
            self.past_data[idx]['action'] = base_action
            self.past_data[idx]['state'] = base_state
            self.past_data[idx]['reward'] += data[1]
            for env_idx in range(self.num_envs):
                self.past_data[idx]['info'][env_idx].update(data[2][env_idx])

    def _reset_state_data(self, observations, env_idx=None):
        """Reset past_venv states when self.venv resets. Also reset data for baseline policy."""
        action, state = self._policy.predict(observations, state=None, mask=None)
        initial_env_states = self.venv.unwrapped.env_method('get_state', env_idx)
        for past_dict, past_venv in list(zip(self.past_data, self.past_venvs)):
            if env_idx is None:
                # this gets called only in self.reset()
                past_venv.reset()
                past_dict['action'] = action
                past_dict['state'] = state
                past_dict['reward'] *= 0
                past_dict['info'] = [{} for _ in range(self.num_envs)]
            else:
                # this gets called when an episode ends in one of the environments
                past_dict['action'][env_idx] = action[env_idx]
                past_dict['state'][:, env_idx, :] = state[:, env_idx, :]
                past_dict['reward'][env_idx] = 0
                past_dict['info'][env_idx] = {}
            envs_iter = range(self.num_envs) if env_idx is None else (0,)
            for env_to_set in envs_iter:
                past_venv.unwrapped.env_method('set_state', env_to_set, initial_env_states[env_to_set])
