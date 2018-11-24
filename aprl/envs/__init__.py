from gym.envs.registration import register

from aprl.envs.multi_agent import MultiAgentEnv, MultiToSingleObs, MultiToSingleObsVec, DummyVecMultiEnv, SubprocVecMultiEnv
from aprl.envs.matrix_game import MatrixGame, IteratedMatchingPennies, RockPaperScissors

register(
    id='aprl/IMP-v0',
    entry_point='aprl.envs.matrix_game:IteratedMatchingPennies',
    max_episode_steps=200,
    reward_threshold=100,
)

register(
    id='aprl/RPS-v0',
    entry_point='aprl.envs.matrix_game:RockPaperScissors',
    max_episode_steps=200,
    reward_threshold=100,
)