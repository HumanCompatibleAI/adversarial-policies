from collections import defaultdict
import numpy as np
from baselines import logger
from baselines.common.vec_env import VecEnvWrapper
from modelfree.simulation_utils import ResettableAgent


class RewardShapingEnv(VecEnvWrapper):
    """A more direct interface for shaping the reward of the attacking agent."""

    default_shaping_params = {
        'reward_center': 1,
        'reward_ctrl': -1,
        'reward_contact': -1,
        'reward_survive': 1,

        # sparse reward field as per gym_compete/new_envs/multi_agent_env:151
        'reward_remaining': 1,
        # dense reward field as per gym_compete/new_envs/agents/humanoid_fighter:45
        # 'reward_move': 10,
    }

    def __init__(self, env, shaping_params=default_shaping_params,
                 reward_annealer=None):
        super().__init__(env)
        self.env = env
        self.shaping_params = shaping_params
        self.reward_annealer = reward_annealer
        self.counter = 0
        self.num_envs = len(self.env.ps)

        self.ep_rew_dict = defaultdict(list)
        self.step_rew_dict = defaultdict(lambda: [[] for _ in range(self.num_envs)])

    def _log_sparse_dense_rewards(self):
        if self.counter == 2048 * self.num_envs:
            num_episodes = len(self.ep_rew_dict['reward_remaining'])
            dense_terms = ['reward_center', 'reward_ctrl', 'reward_contact', 'reward_survive']
            for term in dense_terms:
                assert len(self.ep_rew_dict[term]) == num_episodes

            ep_dense_mean = sum([sum(self.ep_rew_dict[t]) for t in dense_terms]) / num_episodes
            logger.logkv('epdensemean', ep_dense_mean)

            ep_sparse_mean = sum(self.ep_rew_dict['reward_remaining']) / num_episodes
            logger.logkv('epsparsemean', ep_sparse_mean)

            c = self.reward_annealer()
            ep_rew_mean = c * ep_sparse_mean + (1 - c) * ep_dense_mean
            logger.logkv('eprewmean_true', ep_rew_mean)

            for rew_type in self.ep_rew_dict:
                self.ep_rew_dict[rew_type] = []
            self.counter = 0

    def reset(self):
        return self.env.reset()

    def step_wait(self):
        obs, rew, done, infos = self.env.step_wait()

        # replace rew with differently shaped rew
        # victim is agent 0, attacker is agent 1
        for env_num in range(self.num_envs):
            shaped_reward = 0
            for rew_type, rew_value in infos[env_num][1].items():
                if rew_type not in self.shaping_params:
                    continue

                weighted_reward = self.shaping_params[rew_type] * rew_value
                self.step_rew_dict[rew_type][env_num].append(weighted_reward)
                if self.reward_annealer is not None:
                    c = self.get_annealed_exploration_reward()
                    if rew_type == 'reward_remaining':
                        weighted_reward *= (1 - c)
                    else:
                        weighted_reward *= c
                shaped_reward += weighted_reward

            if done[env_num]:
                for rew_type in self.step_rew_dict:
                    self.ep_rew_dict[rew_type].append(sum(self.step_rew_dict[rew_type][env_num]))
                    self.step_rew_dict[rew_type][env_num] = []
            self.counter += 1
            self._log_sparse_dense_rewards()
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


class NoisyAgentWrapper(ResettableAgent):
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

