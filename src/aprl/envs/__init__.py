# flake8: noqa: F401

from gym.envs.registration import register

from aprl.envs.multi_agent import (FakeSingleSpaces, FakeSingleSpacesVec, MultiAgentEnv,
                                   make_dummy_vec_multi_env, make_subproc_vec_multi_env)
from aprl.envs.crowded_line import CrowdedLineEnv
from aprl.envs.matrix_game import MatrixGameEnv, IteratedMatchingPenniesEnv, RockPaperScissorsEnv

register(
    id='aprl/CrowdedLine-v0',
    entry_point='aprl.envs.crowded_line:CrowdedLineEnv',
    max_episode_steps=200,
    reward_threshold=0,
    kwargs={'num_agents': 3},
)

register(
    id='aprl/IteratedMatchingPennies-v0',
    entry_point='aprl.envs.matrix_game:IteratedMatchingPenniesEnv',
    max_episode_steps=200,
    reward_threshold=100,
)

register(
    id='aprl/RockPaperScissors-v0',
    entry_point='aprl.envs.matrix_game:RockPaperScissorsEnv',
    max_episode_steps=200,
    reward_threshold=100,
)