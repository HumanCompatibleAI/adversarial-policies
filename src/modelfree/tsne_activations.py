import argparse
import itertools
import pickle
import os
import sacred

from get_best_adversary import get_best_adversary_path
from modelfree.score_agent import score_ex

BASE_DIR = "/Users/cody/Data/adversarial_policies/"



## TODO
## Find a T-SNE library that works to learn whatever the projection matrix is
## Figure out some sensible tags to attach to timesteps
# (probably just "index in trajectory", "did you win", "what policy were you playing against" would be good)


def score_and_store(episodes, skip_scoring):
    base_config = dict(transparent_params={'ff_policy': True, 'ff_value': True},
                       record_traj_params={'agent_indices': 0},
                       env_name='multicomp/KickAndDefend-v0',
                       num_env=1, record_traj=True, episodes=episodes, render=False)

    best_adversary_path_agent_1 = get_best_adversary_path(environment=base_config['env_name'],
                                                          zoo_id=1,
                                                          base_path="/Users/cody/Data/adversarial_policies/")

    tests_to_run = [
        {'dir': 'kad-adv', 'path': best_adversary_path_agent_1, 'type': 'ppo2'},
        {'dir': 'kad-rand', 'path': None, 'type': 'random'},
        {'dir': 'kad-zoo', 'path': '1', 'type': 'zoo'},
    ]

    if not skip_scoring:
        for experiment in tests_to_run:
            config_copy = base_config.copy()
            config_copy['record_traj_params']['save_dir'] = f'{BASE_DIR}/{experiment["dir"]}'
            config_copy.update({'agent_b_type': experiment["type"], 'agent_b_path': experiment["path"]})
            run = score_ex.run(config_updates=config_copy)
            assert run.status == 'COMPLETED'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--skip-scoring', action='store_true')
    args = parser.parse_args()
    score_and_store(args.episodes, args.skip_scoring)