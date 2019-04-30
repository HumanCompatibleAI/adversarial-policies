
from gym_compete.new_envs.agents.agent import Agent
from gym_compete.new_envs.agents.ant_fighter import AntFighter
from gym_compete.new_envs.agents.humanoid import Humanoid
from gym_compete.new_envs.agents.humanoid_blocker import HumanoidBlocker
from gym_compete.new_envs.agents.humanoid_kicker import HumanoidKicker
import numpy as np


class AdversaryMaskedGymCompeteAgent(Agent):

    def __init__(self, agent_to_mask, agents_to_hide=None, masking_type='initialization'):
        if isinstance(agent_to_mask, Agent):
            self.__dict__ = agent_to_mask.__dict__.copy()
        else:
            raise ValueError("Must pass in an agent to create a masked agent")

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


class AdversaryMaskedGymCompeteAntFighter(AdversaryMaskedGymCompeteAgent, AntFighter):
    def __init__(self, agent_to_mask, agents_to_hide=None, masking_type='initialization'):
        assert isinstance(
            agent_to_mask, AntFighter), \
            "You have passed in {}, requires instance of AntFighter".format(
            type(agent_to_mask))
        AdversaryMaskedGymCompeteAgent.__init__(self,
                                                agent_to_mask=agent_to_mask,
                                                agents_to_hide=agents_to_hide,
                                                masking_type=masking_type)


class AdversaryMaskedGymCompeteHumanoidKicker(AdversaryMaskedGymCompeteAgent, HumanoidKicker):
    def __init__(self, agent_to_mask, agents_to_hide=None, masking_type='initialization'):
        assert isinstance(agent_to_mask,
                          HumanoidKicker), \
            "You have passed in {}, requires instance of HumanoidKicker".format(
            type(agent_to_mask))
        AdversaryMaskedGymCompeteAgent.__init__(self,
                                                agent_to_mask=agent_to_mask,
                                                agents_to_hide=agents_to_hide,
                                                masking_type=masking_type)


class AdversaryMaskedGymCompeteHumanoid(AdversaryMaskedGymCompeteAgent, Humanoid):
    def __init__(self, agent_to_mask, agents_to_hide=None, masking_type='initialization'):
        assert isinstance(agent_to_mask,
                          Humanoid), \
            "You have passed in {}, requires instance of Humanoid".format(
            type(agent_to_mask))
        AdversaryMaskedGymCompeteAgent.__init__(self,
                                                agent_to_mask=agent_to_mask,
                                                agents_to_hide=agents_to_hide,
                                                masking_type=masking_type)


class AdversaryMaskedGymCompeteHumanoidBlocker(AdversaryMaskedGymCompeteAgent, HumanoidBlocker):
    def __init__(self, agent_to_mask, agents_to_hide=None, masking_type='initialization'):
        assert isinstance(agent_to_mask,
                          HumanoidBlocker),\
            "You have passed in {}, requires instance of HumanoidBlocker".format(
            type(agent_to_mask))
        AdversaryMaskedGymCompeteAgent.__init__(self,
                                                agent_to_mask=agent_to_mask,
                                                agents_to_hide=agents_to_hide,
                                                masking_type=masking_type)
