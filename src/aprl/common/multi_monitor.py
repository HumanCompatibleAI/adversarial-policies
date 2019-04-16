import time

import numpy as np
from stable_baselines.bench import Monitor

from aprl.agents.monte_carlo import ResettableEnv
from aprl.utils import getattr_unwrapped


class MultiMonitor(Monitor):
    def __init__(self, env, filename, our_idx=None, allow_early_resets=False,
                 reset_keywords=(), info_keywords=()):
        num_agents = getattr_unwrapped(env, 'num_agents')
        extra_rks = tuple("r{:d}".format(i) for i in range(num_agents))
        super().__init__(env, filename, allow_early_resets=allow_early_resets,
                         reset_keywords=reset_keywords,
                         info_keywords=extra_rks + info_keywords)
        self.our_idx = our_idx
        self.info_keywords = info_keywords

    def step(self, action):
        """
        Step the environment with the given action

        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            self.needs_reset = True
            eplen = len(self.rewards)
            ep_rew = np.asarray(self.rewards).sum(axis=0).round(6)
            our_rew = float('nan') if self.our_idx is None else ep_rew[self.our_idx]
            ep_info = {"r": our_rew,
                       "l": eplen,
                       "t": round(time.time() - self.t_start, 6)}
            for i, rew in enumerate(ep_rew):
                ep_info["r{:d}".format(i)] = rew
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_rewards.append(ep_rew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.logger:
                self.logger.writerow(ep_info)
                self.file_handler.flush()
            info['episode'] = ep_info
        self.total_steps += 1
        return observation, reward, done, info

    def get_state(self):
        """Pass through to a ResettableEnv"""
        assert isinstance(self.env, ResettableEnv)
        return self.env.get_state()

    def get_full_state(self):
        return self.env.get_full_state()

    def set_state(self, x, sim_data=None, forward=True):
        """Pass through to a ResettableEnv"""
        assert isinstance(self.env, ResettableEnv)
        return self.env.set_state(x, sim_data=sim_data, forward=forward)

    def get_radius(self):
        """Pass through to a ResettableEnv"""
        assert isinstance(self.env, ResettableEnv)
        return self.env.get_radius()

    def set_radius(self, r):
        """Pass through to a ResettableEnv"""
        assert isinstance(self.env, ResettableEnv)
        return self.env.set_radius(r)

    def set_arbitrary_state(self, sim_data):
        return self.env.set_arbitrary_state(sim_data)
