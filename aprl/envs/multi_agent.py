import gym

class MultiAgentEnv(gym.Env):
    '''Abstract class for multi-agent environments.
       This isn't really a gym.Env, since it returns a vector of
       rewards and dones. However, it's very convenient to have it interoperate
       with the rest of the Gym infrastructure, so we'll abuse this.
       Sadly there is still no standard for multi-agent environments in Gym,
       issue #934 is working on it.
       '''

    # Set these in ALL subclasses
    num_agents = None
    # Spaces must have num_agents as the first dimension.
    action_space = None
    observation_space = None

    def step(self, action_n):
        '''Run one timestep of the environment's dynamics.
           Accepts an action_n of self.num_agents long, each containing
           an action from self.action_space.

           Args:
                action_n (list<object>): actions per agent.
            Returns:
                obs_n (list<object>): observations per agent.
                reward_n (list<float>): reward per agent.
                done (list<boolean>): done per agent.
                info (dict): auxiliary diagnostic info.
        '''
        raise NotImplementedError

    def reset(self):
        '''Resets state of environment.

        Returns: observation (list<object>): per agent.'''
        raise NotImplementedError