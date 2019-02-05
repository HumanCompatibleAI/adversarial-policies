

#TODO Make an Agent_Wrapper class that takes a function from OxM -> AxM and makes an agent.

def simulate(env, agents, render=False):
    """
    Run Environment env with the agents in agents
    :param env: any enviroment following the openai-gym spec
    :param agents: agents that have get-action functions
    :return: streams information about the simulation
    """
    observations = env.reset()
    dones = [False]

    while not dones[0]:
        if render:
            env.render()
        actions = []
        for agent, observation in zip(agents, observations):
            actions.append(agent.get_action(observation))

        observations, rewards, dones, infos = env.step(actions)

        yield observations, rewards, dones, infos


class Agent(object):

    def __init__(self, action_selector, reseter, values=None, sess = None):
        """
        Takes policies from their format to mine
        :param actable: a policy in the format used by mult-agent-compeitition
        """
        self._action_selector = action_selector
        self._reseter = reseter
        self._values = values
        self._sess = sess

    def get_action(self, observation):
        action = self._action_selector(observation)
        return action

    def reset(self):
        return self._reseter()