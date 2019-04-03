from collections import defaultdict

import numpy as np
from stable_baselines.common.vec_env import VecEnvWrapper
from stable_baselines.common.base_class import ActorCriticRLModel
from stable_baselines.common.vec_env.vec_normalize import VecNormalize

from aprl.envs.multi_agent import FlattenSingletonVecEnv, make_dummy_vec_multi_env, make_subproc_vec_multi_env
from modelfree.gym_compete_conversion import GymCompeteToOurs
from modelfree.policy_loader import load_policy
from modelfree.utils import make_env


class LookbackRewardVecWrapper(VecEnvWrapper):
    """Retains information about prior episodes and rollouts to be used in k-lookback whitebox attacks"""
    def __init__(self, venv, lookback_num, env_name, use_dummy, victim_idx, victim_path, victim_type, lookback_space=1):
        super().__init__(venv)
        # assume that base_venvs have get_state and set_state
        self.lookback_num = lookback_num
        self.lookback_space = lookback_space

        self.base_currys = None
        self.base_venvs = None
        self._policy = self.venv.venv.get_policy()
        self._create_base_venvs(env_name, use_dummy, victim_idx, victim_path, victim_type)
        self._action = None
        self._obs = None
        self._state = None
        self._pseudo_base_state = None
        self._dones = [False] * self.num_envs
        self.ep_lens = np.zeros(self.num_envs).astype(int)

        state_dict = {'state': None, 'action': None, 'reward': np.zeros(self.num_envs), 'info': defaultdict(dict)}
        self.base_data = [state_dict.copy() for _ in range(self.lookback_num)]

    def _create_base_venvs(self, env_name, use_dummy, victim_idx, victim_path, victim_type):
        from modelfree.train import EmbedVictimWrapper

        def env_fn(i):
            return make_env(env_name, 0, i, out_dir='data/extraneous/', pre_wrapper=GymCompeteToOurs, resettable=True)

        self.base_venvs = []
        self.base_currys = []
        for _ in range(self.lookback_num):
            make_vec_env = make_dummy_vec_multi_env if use_dummy else make_subproc_vec_multi_env
            multi_venv = make_vec_env([lambda: env_fn(i) for i in range(self.num_envs)])

            victim = load_policy(policy_path=victim_path, policy_type=victim_type, env=multi_venv,
                                 env_name=env_name, index=victim_idx)
            multi_venv = EmbedVictimWrapper(multi_env=multi_venv, victim=victim, victim_index=victim_idx)
            self.base_currys.append(multi_venv.venv)

            single_venv = FlattenSingletonVecEnv(multi_venv)
            self.base_venvs.append(single_venv)

    def step_async(self, actions):
        current_states = self.venv.unwrapped.env_method('get_state')

        # cycle the base_venvs and step all but the first. Then reset the first one with self.venv.
        self.base_venvs = [self.base_venvs[-1]] + self.base_venvs[:-1]
        self.base_data = [self.base_data[-1]] + self.base_data[:-1]
        self.base_currys = [self.base_currys[-1]] + self.base_currys[:-1]

        new_baseline_venv = self.base_venvs[0]
        curry_obs = self.venv.venv.venv.get_obs()
        self.base_currys[0].set_obs(curry_obs)
        for env_idx in range(self.num_envs):
            new_baseline_venv.unwrapped.env_method('set_state', env_idx, current_states[env_idx])

        for base_venv, base_dict in list(zip(self.base_venvs, self.base_data))[1:]:
            # base_action is calculated from reset() or the most recent step_wait()
            base_venv.step_async(base_dict['action'])

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
            env_diff_reward = 0
            for i, base_dict in enumerate(valid_baseline_dicts):
                diff_reward = rewards[env_idx] - base_dict['reward'][env_idx]
                print('diff:', "{:8.6f}".format(diff_reward),
                      'base:', "{:8.6f}".format(base_dict['reward'][env_idx]),
                      'rew:', "{:8.6f}".format(rewards[env_idx]), 'idx', i)
                env_diff_reward += diff_reward
            rewards[env_idx] += env_diff_reward
        return observations, rewards, self._dones, infos

    def reset(self):
        observations = self.venv.reset()
        self._process_own_obs(observations)
        self._reset_state_data(observations)
        return observations

    def _process_own_obs(self, observations):
        """Record action, state and observations of our policy"""
        self._obs = self._get_truncated_obs(observations)
        self._action, self._state = self._policy.predict(self._obs, state=self._state,
                                                         mask=self._dones)

    def _process_base_data(self, base_data):
        """Record action and state of baseline policy
        :param base_data: list of (observations, rewards, dones, infos), one for each base_venv
        """
        for idx, (base_obs, base_reward, _, base_info) in enumerate(base_data):
            base_obs = self._get_truncated_obs(base_obs)
            base_action, base_state = self._policy.predict(base_obs, state=self.base_data[idx]['state'],
                                                           mask=self._dones)
            self.base_data[idx]['action'] = base_action
            self.base_data[idx]['state'] = base_state
            self.base_data[idx]['reward'] = base_reward
            for env_idx in range(self.num_envs):
                self.base_data[idx]['info'][env_idx].update(base_info[env_idx])

    def _reset_state_data(self, initial_observations, env_idx=None):
        """Reset base_venv states when self.venv resets. Also reset data for baseline policy."""
        truncated_obs = self._get_truncated_obs(initial_observations)
        curry_obs = self.venv.venv.venv.get_obs()
        action, state = self._policy.predict(truncated_obs, state=None, mask=None)
        initial_env_states = self.venv.unwrapped.env_method('get_state', env_idx)
        for base_dict, base_venv, base_curry in list(zip(self.base_data, self.base_venvs, self.base_currys)):
            if env_idx is None:
                # this gets called only in self.reset()
                base_venv.reset()
                base_dict['action'] = action
                base_dict['state'] = state
                base_dict['reward'] *= 0
                base_dict['info'] = defaultdict(dict)
                base_curry.set_obs(curry_obs)
            else:
                # this gets called when an episode ends in one of the environments
                base_dict['action'][env_idx] = action[env_idx]
                if state is None:
                    base_dict['state'] = None
                else:
                    base_dict['state'][:, env_idx, :] = state[:, env_idx, :]
                base_dict['reward'][env_idx] = 0
                base_dict['info'][env_idx] = {}
                base_curry.set_obs(curry_obs, env_idx)
            envs_iter = range(self.num_envs) if env_idx is None else (0,)
            for env_to_set in envs_iter:
                base_venv.unwrapped.env_method('set_state', env_to_set, initial_env_states[env_to_set])

    def _get_truncated_obs(self, obs):
        """Truncate the observation given to self._policy if we are using adversarial noise ball"""
        if isinstance(self._policy.policy, ActorCriticRLModel):
            # stable_baselines policy
            return obs[:, :self._policy.policy.observation_space.shape[0]]
        else:
            # gym_compete policy
            return obs[:, :self._policy.policy.ob_space.shape[0]]
