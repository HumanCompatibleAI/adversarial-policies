import time

from baselines.bench import Monitor
import numpy as np

from aprl.utils import getattr_unwrapped


class MultiMonitor(Monitor):
    def __init__(self, env, filename, allow_early_resets=False,
                 reset_keywords=(), info_keywords=()):
        num_agents = getattr_unwrapped(env, 'num_agents')
        extra_rks = tuple("r{:d}".format(i) for i in range(num_agents))
        super().__init__(env, filename, allow_early_resets=allow_early_resets,
                         reset_keywords=reset_keywords,
                         info_keywords=extra_rks + info_keywords)
        self.info_keywords = info_keywords

    def update(self, ob, rew, done, info):
        # Same as Monitor.update, except handle rewards being vector-valued
        self.rewards.append(rew)
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eprew = list(map(lambda x: round(x, 6), eprew))
            joint_eprew = np.mean(eprew)
            eplen = len(self.rewards)
            epinfo = {"r": joint_eprew,
                      "l": eplen,
                      "t": round(time.time() - self.tstart, 6)}
            for i, rew in enumerate(eprew):
                epinfo["r{:d}".format(i)] = rew
            for k in self.info_keywords:
                epinfo[k] = info[k]
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            epinfo.update(self.current_reset_info)
            self.results_writer.write_row(epinfo)

            if isinstance(info, dict):
                info['episode'] = epinfo

        self.total_steps += 1
