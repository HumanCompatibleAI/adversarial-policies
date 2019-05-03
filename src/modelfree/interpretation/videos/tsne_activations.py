import argparse
import os

from sacred.observers import FileStorageObserver

from modelfree.interpretation.videos.get_best_adversary import get_best_adversary_path
from modelfree.score_agent import score_ex

# TODO
# Find a T-SNE library that works to learn whatever the projection matrix is
# Figure out some sensible tags to attach to timesteps
# (probably just "index in trajectory", "did you win",
# "what policy were you playing against" would be good)

BASE_DIR = "/Users/cody/Data/adversarial_policies/"


def score_and_store(episodes, skip_scoring, sacred_dir):

    base_config = dict(transparent_params={'ff_policy': True, 'ff_value': True},
                       record_traj_params={'agent_indices': 0},
                       env_name='multicomp/SumoAnts-v0',
                       num_env=1, record_traj=True,
                       videos=True,
                       video_dir=None,
                       episodes=episodes, render=False)

    best_adversary_path_agent_1 = get_best_adversary_path(
        environment=base_config['env_name'], zoo_id=1, base_path=BASE_DIR)

    tests_to_run = [
        {'dir': 'kad-adv', 'path': best_adversary_path_agent_1,
         'policy_type': 'ppo2', 'opponent_type': 'adversary'},

        {'dir': 'kad-rand', 'path': None, 'policy_type': 'random',
         'opponent_type': 'random'},
        {'dir': 'kad-zoo', 'path': '1', 'policy_type': 'zoo',
         'opponent_type': 'zoo'},
    ]

    if not skip_scoring:
        for experiment in tests_to_run:
            observer = FileStorageObserver.create(os.path.join(sacred_dir,
                                                               experiment['opponent_type']))
            score_ex.observers.append(observer)

            config_copy = base_config.copy()
            config_copy['record_traj_params']['save_dir'] = f'{BASE_DIR}/{experiment["dir"]}'
            config_copy.update({'agent_b_type': experiment["policy_type"],
                                'agent_b_path': experiment["path"]})
            run = score_ex.run(config_updates=config_copy)
            assert run.status == 'COMPLETED'
            score_ex.observers.pop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--skip-scoring', action='store_true')
    parser.add_argument('--sacred_dir', type=str, default="tsne_save_activations")
    args = parser.parse_args()
    score_and_store(args.episodes, args.skip_scoring,
                    sacred_dir="{}/{}/".format(BASE_DIR, args.sacred_dir))
