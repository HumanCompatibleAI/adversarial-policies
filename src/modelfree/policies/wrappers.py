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


def _array_mask_assign(arr, mask, vals):
    """
    A helper method for basically doing Numpy-style mask assignment on a Python array.
    The `mask` variable contains boolean True values at all locations within `vals` that we
    want to copy over to `arr`. If `vals` is not the same first-dimension as mask,
    it will be broadcast to all locations
    locations that `mask` specifies.
    """
    arr_copy = arr.copy()
    # Check the first-dimension length of vals, checking for numpy array, python arr, and None
    if vals is None:
        length = None
    else:
        try:
            length = vals.shape[0]
        except AttributeError:
            length = len(vals)
    # If the first dimension is not the same as the mask, assume we want this value
    # tiled for every True-valued location in env_mask
    tile_val = length != len(mask)
    for i in range(len(arr_copy)):
        if mask[i]:
            if tile_val:
                arr_copy[i] = vals
            else:
                arr_copy[i] = vals[i]
    return arr_copy


def _standardize_state(state_arr, env_mask, inferred_state_shape):
    """
    Solves the problem of different policies taking in different types of state vector:
    either different shapes of array, or None in the case of a MLP policy. Takes all the
    entries of state_arr that are true in env_mask
    """
    env_mask = env_mask.copy()
    if inferred_state_shape is None:
        filler_value = None
    else:
        filler_value = np.zeros(shape=inferred_state_shape)
        for i in range(len(state_arr)):
            if env_mask[i] and state_arr[i] is None:
                # This is to account for the edge case where we are sending this state to a
                # stateful policy, but at the last step this env had a stateless policy,
                # and thus had its state value set to None. Stateful policies can take in
                # None as a state value, but it can't accommodate some of its envs having
                # arrays and others being None
                env_mask[i] = False

    # Fill in `filler_value` for all indices of state_arr where env_mask is False
    filled_state = _array_mask_assign(state_arr,
                                      [not el for el in env_mask],
                                      filler_value)
    return filled_state


class MultiPolicyWrapper(DummyModel):
    def __init__(self, policies, num_envs):
        super().__init__(policies[0], policies[0].sess)
        self.policies = policies
        # Num_envs is kept as a parameter because you need it to construct
        # self.current_env_policies which makes sense to do the first time at initialization
        self.num_envs = num_envs
        self.action_space_shape = self.policies[0].policy.action_space.shape
        self.obs_space_shape = self.policies[0].policy.observation_space.shape
        for p in self.policies:
            err_txt = "All policies must have the same {} space"
            assert p.policy.action_space.shape == self.action_space_shape, err_txt.format("action")

            assert p.policy.observation_space.shape == self.obs_space_shape, err_txt.format("obs")

        self.current_env_policies = np.random.choice(self.policies, size=self.num_envs)
        self.inferred_state_shapes = [None]*len(policies)

    def predict(self, observation, state=None, mask=None, deterministic=False):
        self._reset_current_policies(mask)
        policy_actions = [None]*self.num_envs
        new_state_array = [None]*self.num_envs

        for i, policy in enumerate(self.policies):
            env_mask = [el == policy for el in self.current_env_policies]
            if sum(env_mask) == 0:
                # If this policy isn't active for any environments, don't run predict on it
                continue
            if state is None:
                # If it's the first training step, and the global state is None, just pass that
                # through to policies without standardizing, because stateful policies can accept
                # a single None as input, but not an array with None values
                standardized_state = None
            else:
                # Otherwise, create filler values for places where env_mask is False
                # with the shape that the policy expects. Inferred state shapes will be set
                # for stateful policies as soon as they return a state vector, i.e. after the
                # first prediction step
                standardized_state = self._standardize_state(state, env_mask,
                                                             self.inferred_state_shapes[i])

            predicted_actions, new_states = policy.predict(observation,
                                                           state=standardized_state,
                                                           mask=mask,
                                                           deterministic=deterministic)
            if new_states is not None and self.inferred_state_shapes[i] is None:
                # If this is a policy that returns state, and its current inferred state
                # is None, update the inferred state value to this shape
                self.inferred_state_shapes[i] = new_states.shape

            # This is basically numpy mask value assignment, but created to work with python
            # arrays to accommodate the fact that different policies won't have state
            # values of the same shape/type
            policy_actions = self._array_mask_assign(policy_actions, env_mask, predicted_actions)
            new_state_array = self._array_mask_assign(new_state_array, env_mask, new_states)

        return policy_actions, new_state_array

    def _reset_current_policies(self, mask):
        for ind, done in enumerate(mask):
            if done:
                new_policy = np.random.choice(self.policies)
                self.current_env_policies[ind] = new_policy

    def close(self):
        for policy in self.policies:
            policy.sess.close()
