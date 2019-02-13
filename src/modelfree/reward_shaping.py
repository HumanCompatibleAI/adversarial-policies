from abc import ABC, abstractmethod

from baselines.common.vec_env import VecEnvWrapper


class RewardShapingEnv(VecEnvWrapper):
    """A more direct interface for shaping the reward of the attacking agent."""

    default_shaping_params = {
        # 'center_reward' : 1,
        # 'ctrl_cost'     : -1,
        # 'contact_cost'  : -1,
        # 'survive'       : 1,

        # sparse reward field as per gym_compete/new_envs/multi_agent_env:151
        'reward_remaining': 1,
        # dense reward field as per gym_compete/new_envs/agents/humanoid_fighter:45
        'reward_move': 1
    }

    def __init__(self, env, shaping_params=default_shaping_params,
                 reward_annealer=None):
        super().__init__(env)
        self.env = env
        self.shaping_params = shaping_params
        self.reward_annealer = reward_annealer

    def reset(self):
        return self.env.reset()

    def step_wait(self):
        obs, rew, done, infos = self.env.step_wait()
        num_envs = len(obs)
        # replace rew with differently shaped rew
        # victim is agent 0, attacker is agent 1
        for env_num in range(num_envs):
            shaped_reward = 0
            for rew_type, rew_value in infos[env_num][1].items():
                if rew_type not in self.shaping_params:
                    continue
                weighted_reward = self.shaping_params[rew_type] * rew_value

                if self.reward_annealer is not None:
                    c = self.get_annealed_exploration_reward()
                    if rew_type == 'reward_remaining':
                        weighted_reward *= (1 - c)
                    elif rew_type == 'reward_move':
                        weighted_reward *= c

                shaped_reward += weighted_reward
            rew[env_num] = shaped_reward
        return obs, rew, done, infos

    def get_annealed_exploration_reward(self):
        """
        Returns c (which will be annealed to zero) s.t. the total reward equals
        c * reward_move + (1 - c) * reward_remaining
        """
        c = self.reward_annealer()
        assert 0 <= c <= 1
        return c


# TODO: should this inherit from something?
class Scheduler(object):
    """Keep track of timefrac_remainings and return time-dependent values"""
    schedule_num = 0

    def __init__(self, lr_func, rew_shape_func):
        # print('made a scheduler')
        # print(Scheduler.schedule_num)
        self._lr_func = lr_func
        self._rew_shape_func = rew_shape_func
        self.frac_remaining = 1  # frac_remaining goes from 1 to 0
        self.same_thing = False
        Scheduler.schedule_num += 1

    def _update_frac_remaining(self, frac_remaining=None):
        if frac_remaining is not None:
            self.same_thing = True
            self.frac_remaining = frac_remaining
            # print("new frac_remaining", frac_remaining)
            # print("reward shaping", self.get_rew_shape_val())

    def get_lr(self, frac_remaining=None):
        # Note that baselines expects this function to be [0, 1] -> R+,
        # where 1 is the beginning of training and 0 is the end of training.
        self._update_frac_remaining(frac_remaining)
        return self._lr_func(self.frac_remaining)

    def get_rew_shape_val(self, frac_remaining=None):
        self._update_frac_remaining(frac_remaining)
        val = self._rew_shape_func(self.frac_remaining)
        # if self.same_thing:
        #    print('same thing')
        if not self.same_thing:
            print('not anymore')
        return val


class Annealer(ABC):
    """Abstract class for implementing Annealers."""
    def __init__(self, start_val, end_val):
        self.start_val = start_val
        self.end_val = end_val

    @abstractmethod
    def get_value(self, step):
        raise NotImplementedError()


class ConstantAnnealer(Annealer):
    """Returns a constant value"""
    def __init__(self, const_val):
        self.const_val = const_val
        super().__init__(const_val, None)

    def get_value(self, frac_remaining):
        return self.const_val


class LinearAnnealer(Annealer):
    """Linearly anneals from start_val to end_val over end_frac fraction of training"""
    def __init__(self, start_val, end_val, end_frac):
        super().__init__(start_val, end_val)
        self.end_frac = end_frac

    def get_value(self, frac_remaining):
        anneal_progress = min(1.0, (1 - frac_remaining) / self.end_frac)
        return (1 - anneal_progress) * self.start_val + anneal_progress * self.end_val


annealer_collection = {
    # Schedule used in the multiagent competition paper for reward shaping.
    'default_reward': LinearAnnealer(1, 0, 0.5),
    # Default baselines.ppo2 learning rate
    'default_lr': ConstantAnnealer(3e-4)
}
