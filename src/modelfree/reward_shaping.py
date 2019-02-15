from abc import ABC, abstractmethod
import numpy as np
import logging
from baselines.common.vec_env import VecEnvWrapper
from gym.envs.mujoco import HumanoidEnv


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


class NoisyAgentWrapper(object):
    def __init__(self, agent, noise_annealer, noise_type='gaussian'):
        """
        - agent: ResettableAgent (most likely) - noise will be added to the actions of
        this agent in order to build a curriculum of weaker to stronger victims
        - noise_annealer: Annealer.get_value - presumably the noise should be decreased over time
        in order to get the adversarial policy to perform well on a normal victim.
        This function should be tied to a Scheduler to keep it stateless.
        - noise_type: str - the type of noise parametrized by noise_annealer's value.
        Current options are [gaussian]
        """
        self.agent = agent
        self.noise_annealer = noise_annealer
        self.noise_generator = self._get_noise_generator(noise_type)

    def _get_noise_generator(self, noise_type):
        noise_generators = {
            'gaussian': lambda x, size: np.random.normal(scale=x, size=size)
        }
        return noise_generators[noise_type]

    def get_action(self, observation):
        noise_param = self.noise_annealer()
        original_action = self.agent.get_action(observation)
        action_size = original_action.shape

        noise = self.noise_generator(noise_param, action_size)
        noisy_action = original_action + noise
        return noisy_action

    def reset(self):
        return self.agent.reset()


class HumanoidEnvWrapper(VecEnvWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(self):
        return self.env.reset()

    def step_wait(self):
        obs, rew, done, infos = self.env.step_wait()
        num_envs = len(obs)
        for env_num in range(num_envs):
            rew[env_num] -= infos[env_num]['reward_linvel']

        return obs, rew, done, infos


# TODO: should this inherit from something?
class Scheduler(object):
    """Keep track of frac_remaining and return time-dependent values"""
    schedule_num = 0

    def __init__(self, lr_func=None, rew_shape_func=None, noise_func=None):
        # print('made a scheduler')
        # print(Scheduler.schedule_num)
        self._lr_func = lr_func
        self._rew_shape_func = rew_shape_func
        self._noise_func = noise_func

        self.frac_remaining = 1  # frac_remaining goes from 1 to 0
        self.same_thing = False
        Scheduler.schedule_num += 1

    def _update_frac_remaining(self, frac_remaining=None):
        if frac_remaining is not None:
            self.same_thing = True
            self.frac_remaining = frac_remaining

    def set_lr_func(self, lr_func):
        self._lr_func = lr_func

    def get_lr_val(self, frac_remaining=None):
        # Note that baselines expects this function to be [0, 1] -> R+,
        # where 1 is the beginning of training and 0 is the end of training.
        self._update_frac_remaining(frac_remaining)
        assert callable(self._lr_func)
        return self._lr_func(self.frac_remaining)

    def set_rew_shape_func(self, rew_shape_func):
        self._rew_shape_func = rew_shape_func

    def get_rew_shape_val(self, frac_remaining=None):
        self._update_frac_remaining(frac_remaining)
        assert callable(self._rew_shape_func)
        val = self._rew_shape_func(self.frac_remaining)
        # if self.same_thing:
        #    print('same thing')
        if not self.same_thing:
            logging.warning('not anymore')
        return val

    def set_noise_func(self, noise_func):
        self._noise_func = noise_func

    def get_noise_val(self, frac_remaining=None):
        self._update_frac_remaining(frac_remaining)
        assert callable(self._noise_func)
        return self._noise_func(self.frac_remaining)


# Annealers

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
