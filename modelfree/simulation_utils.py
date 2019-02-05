

def simulate(env, agents, render=False):
    """
    Run Environment env with the agents in agents
    :param env: any enviroment following the openai-gym spec
    :param agents: agents that have get-action functions
    :param render: true if the run should be rendered to the screen
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


class ResettableAgent(object):

    def __init__(self, get_action_in, reset_in, values=None, sess=None):
        """
        Takes a get_action and reset function and makes a resettable agent
        :param get_action_in: a function to get an action
        :param reset_in: a function to reset the agent
        """
        self._get_action = get_action_in
        self._reset = reset_in
        self._values = values
        self._sess = sess

    def get_action(self, observation):
        return self._get_action(observation)

    def reset(self):
        return self._reset()
