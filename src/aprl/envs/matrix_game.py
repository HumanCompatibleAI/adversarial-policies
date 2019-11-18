"""Two-player, normal-form games with symmetric action spaces."""

from gym.spaces import Discrete, Tuple
import numpy as np

from aprl.envs.multi_agent import MultiAgentEnv


class MatrixGameEnv(MultiAgentEnv):
    """Models two-player, normal-form games with symmetrically sized action space."""

    metadata = {"render.modes": ["human"]}
    ACTION_TO_SYM = None

    def __init__(self, num_actions, payoff):
        """payoff_matrices must be a pair of num_actions*num_actions payoff matrices."""
        agent_space = Discrete(num_actions)
        overall_space = Tuple((agent_space, agent_space))
        self.action_space = overall_space
        self.observation_space = overall_space
        super().__init__(num_agents=2)

        payoff = np.array(payoff)
        assert payoff.shape == (2, num_actions, num_actions)
        self.payoff = payoff

    def step(self, action_n):
        assert len(action_n) == 2
        i, j = action_n
        # observation is the other players move
        self.obs_n = (j, i)
        rew_n = self.payoff[:, i, j]
        done = False
        return self.obs_n, rew_n, done, dict()

    def reset(self):
        # State is previous players action, so this doesn't make much sense;
        # just assume (0, 0) is start.
        self.obs_n = (0, 0)
        return self.obs_n

    def seed(self, seed=None):
        # No-op, there is no randomness in this environment.
        return

    def render(self, mode="human"):
        # note observations are flipped -- observe other agents actions
        p2, p1 = self.obs_n
        if self.ACTION_TO_SYM is not None:
            p1, p2 = tuple(map(self.ACTION_TO_SYM.get, (p1, p2)))
        return f"P1: {p1}, P2: {p2}"


class IteratedMatchingPenniesEnv(MatrixGameEnv):
    ACTION_TO_SYM = {0: "H", 1: "T"}

    def __init__(self):
        p1_payoff = np.array([[1, -1], [-1, 1]])
        payoff = [p1_payoff, -p1_payoff]
        return super().__init__(num_actions=2, payoff=payoff)


class RockPaperScissorsEnv(MatrixGameEnv):
    ACTION_TO_SYM = {0: "R", 1: "P", 2: "S"}

    def __init__(self):
        p1_payoff = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        payoff = [p1_payoff, -p1_payoff]
        return super().__init__(num_actions=3, payoff=payoff)
