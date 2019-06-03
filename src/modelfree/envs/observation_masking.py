import functools

from gym.envs import registry
from gym_compete.new_envs.agents.agent import Agent
from gym_compete.new_envs.multi_agent_env import MultiAgentEnv
import numpy as np


def make_mask_from_class(cls):
    if not issubclass(cls, Agent):
        raise TypeError("You have passed in '{cls}', expected subclass of 'Agent'")

    class AdversaryMaskedGymCompeteAgent(cls):
        def __init__(self, agent_to_mask, agents_to_hide=None, masking_type='initialization',
                     noise_magnitude=None):
            if not isinstance(agent_to_mask, cls):
                raise TypeError(f"You have passed in '{type(agent_to_mask)}', "
                                f"requires instance of '{cls}'")

            self.agent_to_mask = agent_to_mask
            self.agents_to_hide = agents_to_hide
            self.noise_magnitude = noise_magnitude
            self.masking_type = masking_type
            if self.masking_type == 'additive_noise' and self.noise_magnitude is None:
                raise ValueError("To create a noisy observation masker, you must specify magnitude"
                                 "of desired Gaussian noise")

            other_agent_qpos = super(AdversaryMaskedGymCompeteAgent, self).get_other_agent_qpos()
            self.initial_values = {}
            for other_agent_id in other_agent_qpos:
                self.initial_values[other_agent_id] = other_agent_qpos[other_agent_id]
            self.initial_other_qpos = super(AdversaryMaskedGymCompeteAgent, self).get_other_qpos()

        def _get_masking_given_initial(self, initial_position_value, true_current_position):
            if self.masking_type == 'zeros':
                return np.zeros_like(initial_position_value)
            elif self.masking_type == 'debug':
                return np.full_like(initial_position_value, fill_value=-4.2)
            elif self.masking_type == 'initialization':
                return initial_position_value
            elif self.masking_type == 'additive_noise':
                noise = np.random.normal(scale=self.noise_magnitude,
                                         size=initial_position_value.shape)
                return true_current_position + noise
            else:
                raise ValueError(f"Unsupported masking type '{self.masking_type}'")

        def get_other_agent_qpos(self):
            outp = {}
            for other_agent_id in self.initial_values:
                if self.agents_to_hide is None or other_agent_id in self.agents_to_hide:
                    true_current_pos = self.agent_to_mask.get_other_agent_qpos()[other_agent_id]
                    outp[other_agent_id] = self._get_masking_given_initial(
                        initial_position_value=self.initial_values[other_agent_id],
                        true_current_position=true_current_pos)
            return outp

        def get_other_qpos(self):
            true_current_pos = self.agent_to_mask.get_other_qpos()
            return self._get_masking_given_initial(initial_position_value=self.initial_other_qpos,
                                                   true_current_position=true_current_pos)

        def __getattr__(self, item):
            return getattr(self.agent_to_mask, item)

    return AdversaryMaskedGymCompeteAgent


def make_mask_for_env(env_name, agent_index):
    spec = registry.spec(env_name)
    agent_names = spec._kwargs['agent_names']
    agent_name = agent_names[agent_index]
    agent_cls = MultiAgentEnv.AGENT_MAP[agent_name][1]
    return make_mask_from_class(agent_cls)


def make_mask_agent_wrappers(env_name, agent_index, **kwargs):
    masker = make_mask_for_env(env_name, agent_index)
    masker = functools.partial(masker, **kwargs)
    return {agent_index: masker}
