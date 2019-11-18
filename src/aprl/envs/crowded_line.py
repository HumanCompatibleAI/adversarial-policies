"""Agents want to be close to 'food' but not be too crowded on a 1D line."""

from gym.spaces import Box, Tuple
import numpy as np

from aprl.envs.multi_agent import MultiAgentEnv


class CrowdedLineEnv(MultiAgentEnv):
    dt = 1e-1

    """Agents live on a line in [-1,1]. States consist of a position and velocity
    for each agent, with actions consisting of acceleration."""

    def __init__(self, num_agents):
        agent_action_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        agent_observation_space = Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = Tuple(tuple(agent_action_space for _ in range(num_agents)))
        self.observation_space = Tuple(tuple(agent_observation_space for _ in range(num_agents)))
        super().__init__(num_agents=num_agents)
        self.np_random = np.random.RandomState()

    def _get_obs(self):
        return tuple((np.array(row) for row in self.state))

    def reset(self):
        self.state = self.np_random.rand(self.num_agents, 2) * 2 - 1
        return self._get_obs()

    def step(self, action_n):
        # Dynamics
        positions = self.state[:, 0]
        velocities = self.state[:, 1]
        positions += velocities * self.dt
        velocities += np.array(action_n).flatten()
        self.state = np.clip(self.state, -1, 1)

        # Reward: zero-sum game, agents want to be close to food items that other
        # agents are not close to. They should end up spreading out to cover the line.
        # One food item per agent, equally spaced:
        # at [-1, -1 + 2/(N-1), ..., 0, 1 - 2/(N-1), 1]
        # Each agent induces a quasi-Gaussian around its current position,
        # and gets a weighted average of the value of each of the food items.
        # The value of the food item is inversely proportional to the weights
        # induced by the agents.
        foods = np.arange(self.num_agents) * 2 / (self.num_agents - 1) - 1
        positions = positions.reshape(self.num_agents, 1)
        foods = foods.reshape(1, self.num_agents)
        # (num_agents, num_agents) matrix where rows are agents and columns food
        distance = positions - foods
        weights = np.exp(-np.square(distance))
        food_values = 1 / weights.sum(axis=0)
        rewards = tuple(weights.dot(food_values) - 1)

        obs = self._get_obs()
        done = False
        info = {}
        return obs, rewards, done, info

    def seed(self, seed):
        self.np_random.seed(seed)

    def render(self, mode="human"):
        return ", ".join(["{:3f} @ {:3f}".format(pos, vel) for pos, vel in self.state])
