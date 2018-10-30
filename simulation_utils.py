from typing import TypeVar, Generator, Callable, Generic, Any


def simulate(env, agents) -> Generator[Any, None, None]:
    """
    Run Environment env with the agents in agents
    :param env: any enviroment following the openai-gym spec
    :param agents: agents that have get-action functions
    :return: streams information about the simulation
    """
    observations = env.reset()
    dones = [False]

    while not dones[0]:
        env.render()
        actions = []
        for agent, observation in zip(agents, observations):
            actions.append(agent.get_action(observation))

        observations, rewards, dones, infos = env.step(actions)

        yield observations, rewards, dones, infos


def anounce_winner(sim_stream):
    for _, _, dones, infos in sim_stream:
        if dones[0]:
            draw = True
            for i in range(len(infos)):
                if 'winner' in infos[i]:
                    draw = False
                    print("Winner: Agent {}, Scores: {}, Total Episodes: {}".format(i, 1,1))
            if draw:
                print("Game Tied: Agent {}, Scores: {}, Total Episodes: {}".format(i, 1,1))


def utility(sim_stream):
    """
    Calculates the utility acheived from a simulation stream
    :param sim_stream: a single agent simulation stream
    :return: the utility achieved by the agent
    """
    util = 0
    for _, reward, _, _ in sim_stream:
        util += reward

    return util


class FiniteHorizonEnv(object):
    def __init__(self, env, horizon):
        """
        Converts a multi-agent environment with one agent(actions as lists) to
        a single agent environment(actions without lists)
        :param env: a multi agent environment with one agent
        :return: a single agent environment
        """

        self._env = env
        self._steps = 0
        self._horizon = horizon

    def step(self, action):
        observations, rewards, done, infos = self._env([action])

        if self._steps >= self._horizon:
            done = True

        return observations, rewards, done, infos

    def reset(self):
        return self._env.reset()


class MultiToSingle(object):
    def __init__(self, env):
        """
        Converts a multi-agent environment with one agent(actions as lists) to
        a single agent environment(actions without lists)
        :param env: a multi agent environment with one agent
        :return: a single agent environment
        """

        self._env = env

    def step(self, action):
        observations, rewards, dones, infos = self._env([action])

        return observations[0], rewards[0], dones[0], infos[0]

    def reset(self):
        return self._env.reset()[0]


class CurryEnv(object):

    def __init__(self, env, agent, agent_to_fix=0):
        """
        Take a multi agent environment and fix one of the agents
        :param env: The multi-agent environment
        :param agent: The agent to be fixed
        :param agent_to_fix: The index of the agent that should be fixed
        :return: a new environment which behaves like "env" with the agent at position "agent_to_fix" fixed as "agent"
        """
        self._env = env
        self._agent_to_fix = agent_to_fix
        self._agent = agent
        self._last_obs = None
        self._last_reward = None
        self._last_infos = None


    #TODO Check if dones are handeled correctly (if you ever have an env in which it matters)
    def step(self, actions):
        action = self._agent.get_action(self._last_obs)
        actions = actions.insert(self._agent_to_fix, action)
        observations, rewards, done, infos = self._env(actions)

        self._last_obs = observations.pop(self._agent_to_fix)
        self._last_reward = rewards.pop(self._agent_to_fix)
        self._last_infos = infos.pop(self._agent_to_fix)

        return observations, rewards, done, infos

    def reset(self):
        observations = self._env.reset()

        self._last_obs = observations.pop(self._agent_to_fix)

        return observations


def randomSearch(env, policy_sampler, samples=10000):
    """
    Performs random search for policies from "policy_sampler" which maximize reward on "env"
    :param env: The environment to be optimized for
    :param policy_sampler: The policy sampler to optimize from
    :param samples: how many samples to take
    :return: the best policy that has been found
    """
    first_itteration = True
    best_util = 0
    best_policy = None
    for _ in range(samples):
        policy = policy_sampler()
        util = utility(simulate(env, policy))

        if best_util < util or first_itteration:
            best_util = util
            best_policy = policy

    return best_policy
