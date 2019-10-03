import numpy as np

from modelfree.policies.base import DummyModel


class NoisyAgentWrapper(DummyModel):
    def __init__(self, agent, noise_annealer, noise_type='gaussian'):
        """
        Wrap an agent and add noise to its actions
        :param agent: (BaseRLModel) the agent to wrap
        :param noise_annealer: Annealer.get_value - presumably the noise should be decreased
        over time in order to get the adversarial policy to perform well on a normal victim.
        :param noise_type: str - the type of noise parametrized by noise_annealer's value.
        Current options are [gaussian]
        """
        super().__init__(policy=agent, sess=agent.sess)
        self.noise_annealer = noise_annealer
        self.noise_generator = self._get_noise_generator(noise_type)

    @staticmethod
    def _get_noise_generator(noise_type):
        noise_generators = {
            'gaussian': lambda x, size: np.random.normal(scale=x, size=size)
        }
        return noise_generators[noise_type]

    def log_callback(self, logger):
        current_noise_param = self.noise_annealer()
        logger.logkv('shaping/victim_noise', current_noise_param)

    def predict(self, observation, state=None, mask=None, deterministic=False):
        original_actions, states = self.policy.predict(observation, state, mask, deterministic)
        action_shape = original_actions.shape
        noise_param = self.noise_annealer()

        noise = self.noise_generator(noise_param, action_shape)
        noisy_actions = original_actions * (1 + noise)
        return noisy_actions, states


class MultiPolicyWrapper(DummyModel):
    def __init__(self, policies, num_envs):
        # TODO how do we do this properly, since DummyModel requires a single policy and sess?
        super().__init__(policies, policies[0].sess)
        self.policies = policies
        # I ended up keeping num_envs as a parameter because you need it to construct
        # self.current_env_policies which makes sense to do the first time at initialization
        self.num_envs = num_envs
        self.action_space_shape = self.policies[0].policy.action_space.shape
        self.obs_space_shape = self.policies[0].policy.observation_space.shape
        for p in self.policies:
            err_txt = "All policies must have the same {} space"
            assert p.policy.action_space.shape == self.action_space_shape, err_txt.format("action")

            assert p.policy.observation_space.shape == self.obs_space_shape, err_txt.format("obs")

        self.current_env_policies = np.random.choice(self.policies, size=self.num_envs)

    def predict(self, observation, state=None, mask=None, deterministic=False):
        self.reset_current_policies(mask)
        policy_actions = None
        new_state_array = None

        for policy in self.policies:
            env_mask = [el == policy for el in self.current_env_policies]
            predicted_actions, new_states = policy.predict(observation,
                                                           state=state,
                                                           mask=mask,
                                                           deterministic=deterministic)
            if policy_actions is None:
                policy_actions = np.empty(shape=predicted_actions.shape)

            policy_actions[env_mask] = predicted_actions[env_mask]
            # can we get the state shape from the policy?
            if new_states is not None:
                if new_state_array is None:
                    new_state_array = np.empty(shape=new_states.shape)
                new_state_array[env_mask] = new_states[env_mask]
        return policy_actions, new_state_array

    def reset_current_policies(self, mask):
        for ind, done in enumerate(mask):
            if done:
                self.current_env_policies[ind] = np.random.choice(self.policies)

    def close(self):
        for policy in self.policies:
            policy.sess.close()
