from gym.envs import registry
from gym_compete.new_envs.agents.agent import Agent
from gym_compete.new_envs.multi_agent_env import MultiAgentEnv
import numpy as np


def make_mask_from_class(cls):
    if not issubclass(cls, Agent):
        raise TypeError("You have passed in '{cls}', expected subclass of 'Agent'")

    class AdversaryMaskedGymCompeteAgent(cls):
        def __init__(self, agent_to_mask, agents_to_hide=None, masking_type='initialization'):
            if not isinstance(agent_to_mask, cls):
                raise TypeError(f"You have passed in '{type(agent_to_mask)}', "
                                f"requires instance of '{cls}'")

            self.agent_to_mask = agent_to_mask
            self.agents_to_hide = agents_to_hide
            self.masking_type = masking_type

            other_agent_qpos = super(AdversaryMaskedGymCompeteAgent, self).get_other_agent_qpos()
            self.other_agent_shapes = {}
            self.initial_values = {}
            for other_agent_id in other_agent_qpos:
                self.other_agent_shapes[other_agent_id] = other_agent_qpos[other_agent_id].shape
                self.initial_values[other_agent_id] = other_agent_qpos[other_agent_id]

            self.initial_other_qpos = super(AdversaryMaskedGymCompeteAgent, self).get_other_qpos()

        def get_other_agent_qpos(self):
            outp = {}
            for other_agent in self.other_agent_shapes:
                if self.agents_to_hide is None or other_agent in self.agents_to_hide:
                    if self.masking_type == 'zeros':
                        outp[other_agent] = np.zeros(self.other_agent_shapes[other_agent])
                    elif self.masking_type == 'debug':
                        outp[other_agent] = np.full(self.other_agent_shapes[other_agent],
                                                    fill_value=-4.2)
                    elif self.masking_type == 'initialization':
                        outp[other_agent] = self.initial_values[other_agent]
                    else:
                        raise ValueError(f"Unsupported masking type '{self.masking_type}'")
            return outp

        def get_other_qpos(self):
            if self.masking_type == 'zeros':
                return np.zeros_like(self.initial_other_qpos)
            elif self.masking_type == 'debug':
                return np.full(self.initial_other_qpos.shape, fill_value=-4.2)
            elif self.masking_type == 'initialization':
                return self.initial_other_qpos
            else:
                raise ValueError(f"Unsupported masking type '{self.masking_type}'")

        def __getattr__(self, item):
            return getattr(self.agent_to_mask, item)

    return AdversaryMaskedGymCompeteAgent


def make_mask_for_env(env_name, agent_index):
    spec = registry.spec(env_name)
    agent_names = spec._kwargs['agent_names']
    agent_name = agent_names[agent_index]
    agent_cls = MultiAgentEnv.AGENT_MAP[agent_name][1]
    return make_mask_from_class(agent_cls)
