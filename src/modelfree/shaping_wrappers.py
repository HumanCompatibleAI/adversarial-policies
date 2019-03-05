from collections import defaultdict, deque
import json
import os

import numpy as np
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import VecEnvWrapper

from modelfree.scheduling import ConditionalAnnealer, LinearAnnealer


class RewardShapingVecWrapper(VecEnvWrapper):
    """
    A more direct interface for shaping the reward of the attacking agent.
    - shaping_params schema: {'sparse': {k: v}, 'dense': {k: v}, **kwargs}
    """
    def __init__(self, venv, agent_idx, logger, shaping_params, batch_size, reward_annealer=None):
        super().__init__(venv)
        self.shaping_params = shaping_params
        self.reward_annealer = reward_annealer
        self.batch_size = batch_size
        self.agent_idx = agent_idx
        self.logger = logger
        self.log_buffers = defaultdict(lambda: deque([], maxlen=1000))
        self.num_episodes = 0

        self.ep_rew_dict = defaultdict(list)
        self.step_rew_dict = defaultdict(lambda: [[] for _ in range(self.num_envs)])

    def log_callback(self):
        """Logs various metrics. This is given as a callback to PPO2.learn()"""
        all_keys = list(self.shaping_params['dense'].keys()) + \
            list(self.shaping_params['sparse'].keys())
        num_episodes = len(self.ep_rew_dict[all_keys[0]])
        if num_episodes == 0:
            return

        for k in all_keys:
            assert len(self.ep_rew_dict[k]) == num_episodes

        ep_dense_mean = self.get_aggregate_data(buffer=self.ep_rew_dict, episode_avg=True,
                                                terms=self.shaping_params['dense'].keys())
        self.logger.logkv('shaping/epdensemean', ep_dense_mean)

        ep_sparse_mean = self.get_aggregate_data(buffer=self.ep_rew_dict, episode_avg=True,
                                                 terms=self.shaping_params['sparse'].keys())
        self.logger.logkv('shaping/epsparsemean', ep_sparse_mean)

        if self.reward_annealer is not None:
            c = self.reward_annealer()
            self.logger.logkv('shaping/rew_anneal', c)
            ep_rew_mean = c * ep_dense_mean + (1 - c) * ep_sparse_mean
            self.logger.logkv('shaping/eprewmean_true', ep_rew_mean)

        for rew_type in self.ep_rew_dict:
            self.ep_rew_dict[rew_type] = []

    def get_log_buffer_data(self):
        self.log_buffers['epdensereward'] = self.get_aggregate_data(
            buffer=self.log_buffers, episode_avg=False,
            terms=self.shaping_params['dense'].keys())
        self.log_buffers['epsparsereward'] = self.get_aggregate_data(
            buffer=self.log_buffers, episode_avg=False,
            terms=self.shaping_params['sparse'].keys())
        self.log_buffers['num_episodes'] = self.num_episodes
        if self.num_episodes == 0:
            return None
        return self.log_buffers

    @staticmethod
    def get_aggregate_data(buffer, episode_avg, terms):
        term_buffers = [buffer[t] for t in terms]
        aggregated = [sum(x) for x in zip(*term_buffers)]

        if episode_avg is True:
            num_episodes = len(buffer[list(terms)[0]])
            return sum(aggregated) / num_episodes

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rew, done, infos = self.venv.step_wait()
        for env_num in range(self.num_envs):
            shaped_reward = 0
            for rew_term, rew_value in infos[env_num][self.agent_idx].items():
                # our shaping_params dictionary is hierarchically separated the rew_terms
                # into two rew_types, 'dense' and 'sparse'
                if rew_term in self.shaping_params['sparse']:
                    rew_type = 'sparse'
                elif rew_term in self.shaping_params['dense']:
                    rew_type = 'dense'
                else:
                    continue

                # weighted_reward := our weight for that term * value of that term
                weighted_reward = self.shaping_params[rew_type][rew_term] * rew_value
                self.step_rew_dict[rew_term][env_num].append(weighted_reward)

                # perform annealing if necessary and then accumulate total shaped_reward
                if self.reward_annealer is not None:
                    c = self.get_annealed_exploration_reward()
                    if rew_type == 'sparse':
                        weighted_reward *= (1 - c)
                    else:
                        weighted_reward *= c
                shaped_reward += weighted_reward

            # log the results of an episode into buffers and then pass on the shaped reward
            if done[env_num]:
                for rew_term in self.step_rew_dict:
                    rew_term_total = sum(self.step_rew_dict[rew_term][env_num])
                    self.ep_rew_dict[rew_term].append(rew_term_total)
                    self.log_buffers[rew_term].appendleft(rew_term_total)
                    self.step_rew_dict[rew_term][env_num] = []
                    self.num_episodes += 1
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


class NoisyAgentWrapper(BaseRLModel):
    def __init__(self, agent, logger, noise_annealer, noise_type='gaussian'):
        """
        Wrap an agent and add noise to its actions
        :param agent: BaseRLModel the agent to wrap
        :param noise_annealer: Annealer.get_value - presumably the noise should be decreased
        over time in order to get the adversarial policy to perform well on a normal victim.
        :param noise_type: str - the type of noise parametrized by noise_annealer's value.
        Current options are [gaussian]
        """
        self.agent = agent
        self.sess = agent.sess
        self.logger = logger
        self.noise_annealer = noise_annealer
        self.noise_generator = self._get_noise_generator(noise_type)

    @staticmethod
    def _get_noise_generator(noise_type):
        noise_generators = {
            'gaussian': lambda x, size: np.random.normal(scale=x, size=size)
        }
        return noise_generators[noise_type]

    def log_callback(self):
        current_noise_param = self.noise_annealer()
        self.logger.logkv('shaping/victim_noise', current_noise_param)

    def predict(self, observation, state=None, mask=None, deterministic=False):
        original_actions, states = self.agent.predict(observation, state, mask, deterministic)
        action_shape = original_actions.shape
        noise_param = self.noise_annealer()

        noise = self.noise_generator(noise_param, action_shape)
        noisy_actions = original_actions * (1 + noise)
        return noisy_actions, states

    def reset(self):
        return self.agent.reset()

    def setup_model(self):
        pass

    def learn(self):
        raise NotImplementedError()

    def action_probability(self, observation, state=None, mask=None, actions=None):
        raise NotImplementedError()

    def save(self, save_path):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()


def load_wrapper_params(params_path, env_name, rew_shaping=False, noisy_victim=False):
    msg = "Exactly one of rew_shaping and noisy_victim must be True"
    assert bool(rew_shaping) != bool(noisy_victim), msg

    config_dir = 'rew_configs' if rew_shaping else 'noise_configs'
    path_stem = os.path.join('experiments', config_dir)
    default_configs = {
        'Humanoid-v1': 'default_hwalk.json',
        'multicomp/SumoHumans-v0': 'default_hsumo.json',
        'multicomp/SumoHumansAutoContact-v0': 'default_hsumo.json'
    }
    fname = os.path.join(path_stem, default_configs[env_name])
    with open(fname) as default_config_file:
        final_params = json.load(default_config_file)
    if params_path != 'default':
        with open(params_path) as config_file:
            config = json.load(config_file)
        if rew_shaping:
            final_params['sparse'].update(config.get('sparse', {}))
            final_params['dense'].update(config.get('dense', {}))
            final_params['rew_shape_anneal_frac'] = config.get('rew_shape_anneal_frac', 0)
        elif noisy_victim:
            final_params.update(config)
    return final_params


def apply_env_wrapper(single_env, rew_shape_params, env_name, agent_idx,
                      logger, batch_size, scheduler):
    shaping_params = load_wrapper_params(rew_shape_params, env_name, rew_shaping=True)
    if 'metric' in shaping_params:
        rew_shape_annealer = ConditionalAnnealer.from_dict(shaping_params, shaping_env=None)
        scheduler.set_conditional('rew_shape')
    else:
        rew_shape_anneal_frac = shaping_params.get('rew_shape_anneal_frac', 0)
        if rew_shape_anneal_frac > 0:
            rew_shape_annealer = LinearAnnealer(1, 0, rew_shape_anneal_frac)
        else:
            # In this case, the different reward types are weighted differently
            # but reward is not annealed over time.
            rew_shape_annealer = None

    scheduler.set_annealer_and_func('rew_shape', rew_shape_annealer)
    return RewardShapingVecWrapper(single_env, agent_idx=agent_idx, logger=logger,
                                   shaping_params=shaping_params, batch_size=batch_size,
                                   reward_annealer=scheduler.get_func('rew_shape'))


def apply_victim_wrapper(victim, victim_noise_params, env_name, scheduler, logger):
    noise_params = load_wrapper_params(victim_noise_params, env_name, noisy_victim=True)
    if 'metric' in noise_params:
        noise_annealer = ConditionalAnnealer.from_dict(noise_params, shaping_env=None)
        scheduler.set_conditional('noise')
    else:
        victim_noise_anneal_frac = noise_params.get('victim_noise_anneal_frac', 0)
        victim_noise_param = noise_params.get('victim_noise_param', 0)

        if victim_noise_anneal_frac <= 0:
            msg = "victim_noise_anneal_frac must be greater than 0 if using a NoisyAgentWrapper."
            raise ValueError(msg)
        noise_annealer = LinearAnnealer(victim_noise_param, 0, victim_noise_anneal_frac)
    scheduler.set_annealer_and_func('noise', noise_annealer)
    return NoisyAgentWrapper(victim, logger=logger, noise_annealer=scheduler.get_func('noise'))
