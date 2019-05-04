from gym_compete.new_envs.agents.agent import Agent
import numpy as np


def make_masker(cls):
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
            other_qpos = super(AdversaryMaskedGymCompeteAgent, self).get_other_agent_qpos()
            self.other_agent_shapes = {}
            self.initial_values = {}
            for other_agent_id in other_qpos:
                self.other_agent_shapes[other_agent_id] = other_qpos[other_agent_id].shape
                self.initial_values[other_agent_id] = other_qpos[other_agent_id]

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
                        raise NotImplementedError(
                            "Oops, haven't implemented Gaussian noise yet"
                            " since I haven't found the right variance")

            return outp

        def __getattr__(self, item):
            return getattr(self.agent_to_mask, item)

    return AdversaryMaskedGymCompeteAgent
