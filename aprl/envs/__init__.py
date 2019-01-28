from gym.envs.registration import register

from aprl.envs.multi_agent import MultiAgentEnv, MultiToSingleObs, MultiToSingleObsVec, DummyVecMultiEnv, SubprocVecMultiEnv
from aprl.envs.matrix_game import MatrixGameEnv, IteratedMatchingPenniesEnv, RockPaperScissorsEnv

register(
    id='aprl/IMP-v0',
    entry_point='aprl.envs.matrix_game:IteratedMatchingPenniesEnv',
    max_episode_steps=200,
    reward_threshold=100,
)

register(
    id='aprl/RPS-v0',
    entry_point='aprl.envs.matrix_game:RockPaperScissorsEnv',
    max_episode_steps=200,
    reward_threshold=100,
)