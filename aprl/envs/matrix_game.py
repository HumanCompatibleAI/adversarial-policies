from gym.spaces.discrete import Discrete
import numpy as np

from aprl.envs import MultiAgentEnv

class MatrixGame(MultiAgentEnv):
    '''Models two-player, normal-form games with symetrically sized action space.'''
    ACTION_TO_SYM = None

    def __init__(self, num_actions, payoff):
        '''payoff_matrices must be a pair of num_actions*num_actions payoff matrices.'''
        agent_space = Discrete(num_actions)
        super().__init__(num_agents=2,
                         agent_action_space=agent_space,
                         agent_observation_space=agent_space)

        payoff = np.array(payoff)
        assert (payoff.shape == (2, num_actions, num_actions))
        self.payoff = payoff

    def step(self, action_n):
        assert(len(action_n) == 2)
        i, j = action_n
        # observation is the other players move
        self.obs_n = [j, i]
        rew_n = self.payoff[:, i, j]
        done = False
        return self.obs_n, rew_n, done, dict()

    def reset(self):
        # State is previous players action, so this doesn't make much sense;
        # just give a random result.
        self.obs_n = self.observation_space.sample()
        return self.obs_n

    def seed(self, seed=None):
        # No-op, there is no randomness in this environment.
        return

    def render(self, mode='human'):
        if self.ACTION_TO_SYM is None:
            raise NotImplementedError
        # note observations are flipped -- observe other agents actions
        p2, p1 = list(map(self.ACTION_TO_SYM.get, self.obs_n))
        return f'P1: {p1}, P2: {p2}'


class IteratedMatchingPennies(MatrixGame):
    ACTION_TO_SYM = {0: 'H', 1: 'T'}

    def __init__(self):
        p1_payoff = np.array([[1, -1], [-1, 1]])
        payoff = [p1_payoff, -p1_payoff]
        return super().__init__(num_actions=2, payoff=payoff)


class RockPaperScissors(MatrixGame):
    ACTION_TO_SYM = {0: 'R', 1: 'P', 2: 'S'}

    def __init__(self):
        p1_payoff = np.array([
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0]
        ])
        payoff = [p1_payoff, -p1_payoff]
        return super().__init__(num_actions=3, payoff=payoff)