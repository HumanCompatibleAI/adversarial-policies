import numpy as np

from aprl.envs.multi_agent import VecMultiWrapper, _tuple_pop, _tuple_space_filter


class EmbedVictimWrapper(VecMultiWrapper):
    """Embeds victim in a (Transparent)CurryVecEnv. Also takes care of closing victim's session"""
    def __init__(self, multi_env, victim, victim_index, transparent, deterministic,
                 multi_victim):
        self.multi_victim = multi_victim
        self.victim = victim
        if multi_victim:
            # currently don't have transparent and multi-victim combo,
            # so multi-victim is prioritized
            cls = MultiCurryVecEnv
        elif transparent:
            cls = TransparentCurryVecEnv
            self.victim = self.victim[0]
        else:
            cls = CurryVecEnv
            self.victim = self.victim[0]

        curried_env = cls(multi_env, self.victim, agent_idx=victim_index,
                          deterministic=deterministic)
        super().__init__(curried_env)

    def get_policy(self):
        return self.venv.get_policy()

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()

    def close(self):
        if self.multi_victim:
            for individual_victim in self.victim:
                individual_victim.sess.close()
        else:
            self.victim.sess.close()
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
        :param policy_selector(func): A function that takes in
        :return: a new VecMultiEnv with num_agents decremented. It behaves like env but
                 with all actions at index agent_idx set to those sampled from one policy
                 within policies"""
        super().__init__(venv, policies, agent_idx, deterministic)

        self.policies = policies
        self.state_array = [None]*len(policies)
        self.current_policy_idx = np.random.choice(range(len(self.policies)))
        self._policy = self.policies[self.current_policy_idx]
        self.step_count = 0
        print(f"Initialized: now sampling from policy of type {type(self._policy)}")

    def step_async(self, actions):
        action, new_state = self._policy.predict(self._obs,
                                                 state=self.state_array[self.current_policy_idx],
                                                 mask=self._dones,
                                                 deterministic=self.deterministic)
        self.state_array[self.current_policy_idx] = new_state
        actions.insert(self._agent_to_fix, action)
        self.step_count += 1
        if self.step_count > 175:
            self.current_policy_idx = np.random.choice(range(len(self.policies)))
            self._policy = self.policies[self.current_policy_idx]
            print(f"Hit step count: now sampling from policy of type {type(self._policy)}")
            self.step_count = 0
        self.venv.step_async(actions)

    def reset(self):
        self.current_policy_idx = np.random.choice(range(len(self.policies)))
        self._policy = self.policies[self.current_policy_idx]
        print(f"Resetting: now sampling from policy of type {type(self._policy)}")
        return super().reset()


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
