import numpy as np

from aprl.envs.multi_agent import VecMultiWrapper, _tuple_pop, _tuple_space_filter


class EmbedVictimWrapper(VecMultiWrapper):
    """Embeds victim in a (Transparent)CurryVecEnv. Also takes care of closing victim's session"""
    def __init__(self, multi_env, victims, victim_index, transparent, deterministic):
        multi_victim = len(victims) > 0
        self.victims = victims

        if multi_victim and not transparent:
            cls = MultiCurryVecEnv
            victim_arg = self.victims
        elif multi_victim and transparent:
            raise NotImplementedError("VecEnvs cannot be transparent with multiple victims")
        elif transparent:
            victim_arg = self.victims[0]
            cls = TransparentCurryVecEnv
        else:
            victim_arg = self.victims[0]
            cls = CurryVecEnv

        curried_env = cls(multi_env, victim_arg, agent_idx=victim_index,
                          deterministic=deterministic)
        super().__init__(curried_env)

    def get_policy(self):
        return self.venv.get_policy()

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()

    def close(self):
        for individual_victim in self.victims:
            individual_victim.sess.close()

        super().close()


class CurryVecEnv(VecMultiWrapper):
    """Substitutes in a fixed agent for one of the players in a VecMultiEnv."""
    def __init__(self, venv, policy, agent_idx=0, deterministic=False):
        """Fixes one of the players in a VecMultiEnv.
        :param venv(VecMultiEnv): the environments.
        :param policy(Policy): the policy to use for the agent at agent_idx.
        :param agent_idx(int): the index of the agent that should be fixed.
        :return: a new VecMultiEnv with num_agents decremented. It behaves like env but
                 with all actions at index agent_idx set to those returned by agent."""
        super().__init__(venv)

        assert venv.num_agents >= 1  # allow currying the last agent
        self.num_agents = venv.num_agents - 1
        self.observation_space = _tuple_space_filter(self.observation_space, agent_idx)
        self.action_space = _tuple_space_filter(self.action_space, agent_idx)

        self._agent_to_fix = agent_idx
        self._policy = policy
        self._state = None
        self._obs = None
        self._dones = [False] * venv.num_envs
        self.deterministic = deterministic

    def step_async(self, actions):
        action, self._state = self._policy.predict(self._obs, state=self._state, mask=self._dones,
                                                   deterministic=self.deterministic)
        actions.insert(self._agent_to_fix, action)
        self.venv.step_async(actions)

    def step_wait(self):
        observations, rewards, self._dones, infos = self.venv.step_wait()
        observations, self._obs = _tuple_pop(observations, self._agent_to_fix)
        rewards, _ = _tuple_pop(rewards, self._agent_to_fix)
        return observations, rewards, self._dones, infos

    def reset(self):
        observations = self.venv.reset()
        observations, self._obs = _tuple_pop(observations, self._agent_to_fix)
        return observations

    def get_policy(self):
        return self._policy

    def get_curry_venv(self):
        """Helper method to locate self in a stack of nested VecEnvWrappers"""
        return self

    def set_curry_obs(self, obs, env_idx=None):
        """Setter for observation of embedded agent

        :param obs ([float]) a vectorized observation from either one or all environments
        :param env_idx (int,None) indices of observations to set. None means all.
        """
        if env_idx is None:
            self._obs = obs
        else:
            self._obs[env_idx] = obs

    def get_curry_obs(self, env_idx=None):
        """Getter for observation of embedded agent

        :param env_idx (int,None) indices of observations to get. None means all.
        :return: ([float]) observations from specified environments
        """
        if env_idx is None:
            return self._obs
        else:
            return self._obs[env_idx]


class MultiCurryVecEnv(CurryVecEnv):
    def __init__(self, venv, policies, agent_idx=0, deterministic=False):
        """Fixes one of the players in a VecMultiEnv, but alternates between policies
        used to perform response action.
        :param venv(VecMultiEnv): the environments.
        :param policies(iterable of Policy): the policies to use for the agent at agent_idx.
        :param agent_idx(int): the index of the agent that should be fixed.
        :return: a new VecMultiEnv with num_agents decremented. It behaves like env but
                 with all actions at index agent_idx set to those sampled from one policy
                 within policies"""
        super().__init__(venv, policies, agent_idx, deterministic)

        self.policies = policies
        self.num_envs = venv.num_envs
        self.state_array = [None]*self.num_envs
        self.current_policy_idx = [None]*self.num_envs
        self.current_policies = [None]*self.num_envs
        self.switch_policy()

    def step_async(self, actions):
        policy_actions = []
        new_state_array = []
        for i in range(len(self._obs)):
            policy = self.current_policies[i]
            observations = np.array([self._obs[i]]*self.num_envs)
            state = np.array([self.state_array[i]]*self.num_envs)
            mask = np.array([self._dones[i]]*self.num_envs)
            action, new_state = policy.predict(observations,
                                               state=state,
                                               mask=mask,
                                               deterministic=self.deterministic)
            policy_actions.append(action[0])
            new_state_array.append(new_state)
        policy_actions_array = np.array(policy_actions)
        self.state_array = np.array(new_state_array)
        actions.insert(self._agent_to_fix, policy_actions_array)
        self.venv.step_async(actions)

    def get_policy(self):
        return self.current_policies[0]

    def step_wait(self):
        obs, rew, dones, info = super().step_wait()
        done_environments = np.where(dones)[0]
        if len(done_environments) > 0:
            self.switch_policy(done_environments)
        return obs, rew, dones, info

    def switch_policy(self, indicies_to_change=None):
        if indicies_to_change is None:
            indicies_to_change = range(self.num_envs)

        for i in indicies_to_change:
            policy_ind = np.random.choice(range(len(self.policies)))
            self.current_policy_idx[i] = policy_ind
            self.current_policies[i] = self.policies[self.current_policy_idx[i]]


class TransparentCurryVecEnv(CurryVecEnv):
    """CurryVecEnv that provides transparency data about its policy by updating infos dicts."""
    def __init__(self, venv, policy, agent_idx=0, deterministic=False):
        """
        :param venv (VecMultiEnv): the environments
        :param policy (BaseRLModel): model which wraps a BasePolicy object
        :param agent_idx (int): the index of the agent that should be fixed.
        :return: a new VecMultiEnv with num_agents decremented. It behaves like env but
                 with all actions at index agent_idx set to those returned by agent."""
        super().__init__(venv, policy, agent_idx, deterministic)
        if not hasattr(self._policy.policy, 'step_transparent'):
            raise TypeError("Error: policy must be transparent")
        self._action = None

    def step_async(self, actions):
        policy_out = self._policy.predict_transparent(self._obs, state=self._state,
                                                      mask=self._dones,
                                                      deterministic=self.deterministic)
        self._action, self._state, self._data = policy_out
        actions.insert(self._agent_to_fix, self._action)
        self.venv.step_async(actions)

    def step_wait(self):
        observations, rewards, self._dones, infos = self.venv.step_wait()
        observations, self._obs = _tuple_pop(observations, self._agent_to_fix)
        for env_idx in range(self.num_envs):
            env_data = {k: v[env_idx] for k, v in self._data.items()}
            infos[env_idx][self._agent_to_fix].update(env_data)
        return observations, rewards, self._dones, infos
