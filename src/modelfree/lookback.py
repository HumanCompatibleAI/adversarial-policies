from collections import defaultdict, namedtuple

import numpy as np
from stable_baselines.common.vec_env import VecEnvWrapper
from stable_baselines.common.base_class import ActorCriticRLModel
from stable_baselines.common.vec_env.vec_normalize import VecNormalize

from aprl.envs.multi_agent import FlattenSingletonVecEnv, make_dummy_vec_multi_env, make_subproc_vec_multi_env
from modelfree.gym_compete_conversion import GymCompeteToOurs
from modelfree.policy_loader import load_policy
from modelfree.utils import make_env

LookbackDict = namedtuple('LookbackDict', ['curry', 'venv', 'data'])

class LookbackRewardVecWrapper(VecEnvWrapper):
    """Retains information about prior episodes and rollouts to be used in k-lookback whitebox attacks"""
    def __init__(self, venv, lookback_num, env_name, use_dummy,
                 victim_idx, victim_path, victim_type, lookback_space=1):
        super().__init__(venv)
        self.lookback_num = lookback_num
        self.lookback_space = lookback_space

        self.lb_dicts = self._create_lb_dicts(env_name, use_dummy, victim_idx, victim_path, victim_type)
        self._policy = self.venv.venv.get_policy()
        self._action = None
        self._obs = None
        self._state = None
        self._new_lb_state = None
        self._dones = [False] * self.num_envs
        self.ep_lens = np.zeros(self.num_envs).astype(int)

        state_dict = {'state': None, 'action': None, 'reward': np.zeros(self.num_envs),
                      'info': defaultdict(dict)}
        self.lb_data = [state_dict.copy() for _ in range(self.lookback_num)]

    def _create_lb_dicts(self, env_name, use_dummy, victim_idx, victim_path, victim_type):
        from modelfree.train import EmbedVictimWrapper

        def env_fn(i):
            return make_env(env_name, 0, i, out_dir='data/extraneous/',
                            pre_wrapper=GymCompeteToOurs, resettable=True)
        lb_dicts = []
        for _ in range(self.lookback_num):
            make_vec_env = make_dummy_vec_multi_env if use_dummy else make_subproc_vec_multi_env
            multi_venv = make_vec_env([lambda: env_fn(i) for i in range(self.num_envs)])

            victim = load_policy(policy_path=victim_path, policy_type=victim_type, env=multi_venv,
                                 env_name=env_name, index=victim_idx)
            multi_venv = EmbedVictimWrapper(multi_env=multi_venv, victim=victim,
                                            victim_index=victim_idx)

            single_venv = FlattenSingletonVecEnv(multi_venv)
            data_dict = {'state': None, 'action': None, 'reward': np.zeros(self.num_envs),
                         'info': defaultdict(dict)}
            lb_dicts.append(LookbackDict(curry=multi_venv.venv, venv=single_venv, data=data_dict))
        return lb_dicts

    def step_async(self, actions):
        current_states = self.venv.unwrapped.env_method('get_state')

        # cycle the lb_venvs and step all but the first. Then reset the first one with self.venv.
        self.lb_dicts = [self.lb_dicts[-1]] + self.lb_dicts[:-1]

        new_baseline_venv = self.lb_dicts[0].venv
        curry_obs = self.venv.venv.venv.get_obs()
        self.lb_dicts[0].curry.set_obs(curry_obs)
        for env_idx in range(self.num_envs):
            new_baseline_venv.unwrapped.env_method('set_state', env_idx, current_states[env_idx])

        for lb_dict in self.lb_dicts[1:]:
            # lb_action is calculated from reset() or the most recent step_wait()
            lb_dict.venv.step_async(lb_dict.data['action'])

        # the baseline policy's state is what it would have been if it had observed all of
        # the same things as our policy. self._pseudo_lb_state comes from seeing only self._obs
        lb_action, self._new_lb_state = self._policy.predict(self._obs, state=self._new_lb_state,
                                                             mask=self._dones)
        self.lb_dicts[0].data['state'] = self._new_lb_state
        new_baseline_venv.step_async(lb_action)
        self.venv.step_async(actions)
        self.ep_lens += 1

    def step_wait(self):
        observations, rewards, self._dones, infos = self.venv.step_wait()
        self._process_own_obs(observations)
        lb_data = [lb_dict.venv.step_wait() for lb_dict in self.lb_dicts]
        self._process_lb_data(lb_data)

        self.ep_lens *= ~np.array(self._dones)
        for env_idx in range(self.num_envs):
            if self._dones[env_idx]:
                # align this env with self.venv since self.venv was reset
                self._reset_state_data(observations, env_idx)
            valid_lb_dicts = self.lb_dicts[:self.ep_lens[env_idx]]
            env_diff_reward = 0
            for i, lb_dict in enumerate(valid_lb_dicts):
                diff_reward = rewards[env_idx] - lb_dict.data['reward'][env_idx]
                print('diff:', "{:8.6f}".format(diff_reward),
                      'base:', "{:8.6f}".format(lb_dict.data['reward'][env_idx]),
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

    def _process_lb_data(self, lb_data):
        """Record action and state of baseline policy
        :param lb_data: list of (observations, rewards, dones, infos), one for each lb_venv
        """
        for idx, (lb_obs, lb_reward, _, lb_info) in enumerate(lb_data):
            lb_obs = self._get_truncated_obs(lb_obs)
            lb_action, lb_state = self._policy.predict(lb_obs,
                                                       state=self.lb_dicts[idx].data['state'],
                                                       mask=self._dones)
            self.lb_dicts[idx].data['action'] = lb_action
            self.lb_dicts[idx].data['state'] = lb_state
            self.lb_dicts[idx].data['reward'] = lb_reward
            for env_idx in range(self.num_envs):
                self.lb_dicts[idx].data['info'][env_idx].update(lb_info[env_idx])

    def _reset_state_data(self, initial_observations, env_idx=None):
        """Reset lb_venv states when self.venv resets. Also reset data for baseline policy."""
        truncated_obs = self._get_truncated_obs(initial_observations)
        curry_obs = self.venv.venv.venv.get_obs()
        action, state = self._policy.predict(truncated_obs, state=None, mask=None)
        initial_env_states = self.venv.unwrapped.env_method('get_state', env_idx)
        for lb_dict in self.lb_dicts:
            if env_idx is None:
                # this gets called only in self.reset()
                lb_dict.venv.reset()
                lb_dict.data['action'] = action
                lb_dict.data['state'] = state
                lb_dict.data['reward'] *= 0
                lb_dict.data['info'] = defaultdict(dict)
                lb_dict.curry.set_obs(curry_obs)
            else:
                # this gets called when an episode ends in one of the environments
                lb_dict.data['action'][env_idx] = action[env_idx]
                if state is None:
                    lb_dict.data['state'] = None
                else:
                    lb_dict.data['state'][:, env_idx, :] = state[:, env_idx, :]
                lb_dict.data['reward'][env_idx] = 0
                lb_dict.data['info'][env_idx] = {}
                lb_dict.curry.set_obs(curry_obs, env_idx)
            envs_iter = range(self.num_envs) if env_idx is None else (0,)
            for env_to_set in envs_iter:
                lb_dict.venv.unwrapped.env_method('set_state', env_to_set, initial_env_states[env_to_set])

    def _get_truncated_obs(self, obs):
        """Truncate the observation given to self._policy if we are using adversarial noise ball"""
        if isinstance(self._policy.policy, ActorCriticRLModel):
            return obs[:, :self._policy.policy.observation_space.shape[0]]  # stable_baselines
        else:
            return obs[:, :self._policy.policy.ob_space.shape[0]]           # gym_compete
