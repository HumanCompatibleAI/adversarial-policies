from typing import List, Optional, Sequence, Tuple, TypeVar

import numpy as np
from stable_baselines.common.base_class import BaseRLModel

from modelfree.policies.base import ModelWrapper


class NoisyAgentWrapper(ModelWrapper):
    def __init__(self, model: BaseRLModel, noise_annealer, noise_type: str = 'gaussian'):
        """
        Wrap an agent and add noise to its actions
        :param model: the agent to wrap
        :param noise_annealer: Annealer.get_value - presumably the noise should be decreased
        over time in order to get the adversarial policy to perform well on a normal victim.
        :param noise_type: the type of noise parametrized by noise_annealer's value.
            Current options are [gaussian]
        """
        super().__init__(model=model)
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
        original_actions, states = self.model.predict(observation, state, mask, deterministic)
        action_shape = original_actions.shape
        noise_param = self.noise_annealer()

        noise = self.noise_generator(noise_param, action_shape)
        noisy_actions = original_actions * (1 + noise)
        return noisy_actions, states


T = TypeVar("T")


def _array_mask_assign(arr: List[T], mask: Sequence[bool], vals: Optional[List[T]]) -> List[T]:
    """Assign val to indices of `arr` that are True in `mask`.

    :param arr: a Python list.
    :param mask: a list of boolean values of the same length as `arr`.
    :param vals: value to assign.
    :return A copy of `arr` with masked values updated to `val`.
    """
    if vals is None:
        vals = [None] * sum(mask)

    arr = list(arr)
    inds = np.arange(len(arr))[mask]
    for i, v in zip(inds, vals):
        arr[i] = v
    return arr


def _standardize_state(state_arr: Sequence[np.ndarray],
                       mask: Sequence[bool],
                       filler_shape: Tuple[int, ...]) -> np.ndarray:
    """Replaces values in state_arr[env_mask] with a filler value.

    The returned value should have entries of a consistent type, suitable to pass to a policy.
    The input `state_arr` may contain entries produced by different policies, which may include
    `None` values and NumPy arrays of various shapes.

    :param state_arr: The state from the previous timestep.
    :param mask: Mask of indices to replace with filler values. These should be environments
        the policy does not control -- so it does not matter what output it produces.
    :param filler_shape: The shape of the value to fill in.
    :return `None` if `filler_shape` is None, otherwise `state_arr` with appropriate entries
        masked by the filler value.
    """
    if filler_shape is None:
        # If the policy is stateless, it should take a `None` state entry
        return None

    # The policy is stateful, and expects entries of shape inferred_state_shape.
    num_env = len(state_arr)
    standardized_arr = np.zeros(shape=(num_env, ) + filler_shape)

    if np.any(mask):
        # Copy over values from state_arr in mask. The others are OK to leave as zero:
        # we'll ignore actions predicted in those indices anyway.
        to_copy = np.array(state_arr)[mask]  # extract subset
        to_copy = np.stack(to_copy)  # ensure it is a 2D array
        standardized_arr[mask] = to_copy

    return standardized_arr


class MultiPolicyWrapper(ModelWrapper):
    """Combines multiple policies into a single policy.

    Each policy executes for the entirety of an episode, and then a new policy is randomly
    selected from the list of policies.

    WARNING: Only suitable for inference, not for training!"""

    def __init__(self, policies: Sequence[BaseRLModel], num_envs: int):
        """Creates MultiPolicyWrapper.

        :param policies: The underlying policies to execute.
        :param num_envs: The number of environments to execute in parallel.
        """
        super().__init__(policies[0])
        self.policies = policies

        self.action_space = self.policies[0].action_space
        self.obs_space = self.policies[0].observation_space
        for p in self.policies:
            err_txt = "All policies must have the same {} space"
            assert p.action_space == self.action_space, err_txt.format("action")
            assert p.observation_space == self.obs_space, err_txt.format("obs")

        # Strictly we do not need `num_envs`, but it is convenient to have it so we can
        # construct an appropriate sized `self.current_env_policies` at initialization.
        self.num_envs = num_envs
        self.current_env_policies = np.random.choice(self.policies, size=self.num_envs)
        self.inferred_state_shapes = [None] * len(policies)

    def predict(self, observation, state=None, mask=None, deterministic=False):
        self._reset_current_policies(mask)
        policy_actions = np.zeros((self.num_envs, ) + self.action_space.shape,
                                  dtype=self.action_space.dtype)
        new_state_array = [None] * self.num_envs

        for i, policy in enumerate(self.policies):
            env_mask = np.array([el == policy for el in self.current_env_policies])
            if not np.any(env_mask):
                # If this policy isn't active for any environments, don't run predict on it
                continue

            if state is None:
                # If it's the first training step, and the global state is None, just pass that
                # through to policies without standardizing, because stateful policies can accept
                # a single None as input, but not an array with None values
                standardized_state = None
            else:
                # Otherwise, fill in values for places where env_mask is False, i.e. that belong
                # to other policies. Also fill in values if the environment has just been reset
                # (mask is True), as the state may have originated from a different policy.
                #
                # Note initially we do not what shape stateful policies expect, so we default to
                # `None`, which is always OK at the first time step. Inferred state shapes will be
                # set for stateful policies as soon as they return a state vector.
                retain = env_mask & ~np.array(mask)
                standardized_state = _standardize_state(state, mask=retain,
                                                        filler_shape=self.inferred_state_shapes[i])

            predicted_actions, new_states = policy.predict(observation,
                                                           state=standardized_state,
                                                           mask=mask,
                                                           deterministic=deterministic)
            if new_states is not None and self.inferred_state_shapes[i] is None:
                # If this is a policy that returns state, and its current inferred state
                # is None, update the inferred state value to this shape
                self.inferred_state_shapes[i] = new_states.shape[1:]
            assert ((new_states is None and self.inferred_state_shapes[i] is None) or
                    new_states.shape[1:] == self.inferred_state_shapes[i])

            policy_actions[env_mask] = predicted_actions[env_mask]
            new_state_array = _array_mask_assign(new_state_array, env_mask, new_states)

        return policy_actions, new_state_array

    def _reset_current_policies(self, mask):
        num_done = sum(mask)
        self.current_env_policies[mask] = np.random.choice(self.policies, size=num_done)

    def close(self):
        for policy in self.policies:
            if policy.sess is not None:
                policy.sess.close()
