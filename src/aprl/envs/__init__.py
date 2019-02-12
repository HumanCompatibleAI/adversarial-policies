# flake8: noqa: F401

from gym.envs.registration import register

from aprl.envs.multi_agent import (FakeSingleSpaces, FakeSingleSpacesVec, MultiAgentEnv,
                                   make_dummy_vec_multi_env, make_subproc_vec_multi_env)
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