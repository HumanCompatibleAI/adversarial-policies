import logging
import os
import os.path as osp
import tempfile

import sacred
from sacred.observers import FileStorageObserver

from modelfree.interpretation.fit_tsne import fit_tsne_ex
from modelfree.interpretation.get_best_adversary import get_best_adversary_path
from modelfree.interpretation.visualize_tsne import vis_tsne_ex
from modelfree.score_agent import score_ex

tsne_ex = sacred.Experiment('tsne')
logger = logging.getLogger('modelfree.interpretation.tsne_pipeline')


@tsne_ex.config
def activation_storing_config():
    base_path = "data"
    perplexity = 250,
    score_agent_configs = dict(
        transparent_params={'ff_policy': True, 'ff_value': True},
        record_traj_params={'agent_indices': 0},
        env_name='multicomp/YouShallNotPassHumans-v0',
        num_env=1,
        record_traj=True,
        videos=True,
        video_dir=None,
        episodes=2,
        render=False
    )

    fit_tsne_configs = dict(
        relative_dirs=['adversary', 'random', 'zoo'],
        data_type='ff_policy',
        num_components=2,
        sacred_dir_ids=None,
        np_file_name="victim_activations.npz",
        num_observations=None,
        env_name='multicomp/YouShallNotPassHumans-v0',
        seed=0,
        perplexity=perplexity[0],
    )

    visualize_configs = dict(
        base_path=None,
        subsample_rate=0.15,
        perplexity=perplexity[0],
        video_path="data/video_frames",
        chart_type="seaborn",
        opacity=0.75,
        dot_size=0.25,
        palette_name="cube_bright",
        save_type="pdf",
        hue_order=["adversary", "zoo", "random"],
    )

    skip_scoring = False
    _ = locals()
    del _


@tsne_ex.capture
def store_activations(sacred_base_dir, base_path, score_agent_configs):
    # TODO: make adversary_path a config parameter
    best_adversary_path_agent_1 = get_best_adversary_path(
                    best_path=osp.join(base_path, "score_agents", "best_adversaries.json"),
                    environment=score_agent_configs['env_name'],
                    zoo_id=1, base_path=base_path)

    tests_to_run = [
        {'dir': 'kad-adv', 'path': best_adversary_path_agent_1,
         'policy_type': 'ppo2', 'opponent_type': 'adversary'},

        {'dir': 'kad-rand', 'path': None, 'policy_type': 'random',
         'opponent_type': 'random'},
        {'dir': 'kad-zoo', 'path': '1', 'policy_type': 'zoo',
         'opponent_type': 'zoo'},
    ]
    observer_dirs = []

    for test_config in tests_to_run:
        observer = FileStorageObserver.create(os.path.join(sacred_base_dir,
                                                           test_config['opponent_type']))
        score_ex.observers.append(observer)
        observer_dirs.append(observer.dir)
        config_copy = score_agent_configs.copy()
        with tempfile.TemporaryDirectory() as td:  # noqa: F841
            # config_copy['record_traj_params']['save_dir'] = td
            # TODO: use temporary directory but copy it somewhere?
            config_copy['record_traj_params']['save_dir'] = 'data/tsne_save_activations'
            config_copy.update({'agent_b_type': test_config["policy_type"],
                                'agent_b_path': test_config["path"]})
            run = score_ex.run(config_updates=config_copy)
        assert run.status == 'COMPLETED'
        score_ex.observers.pop()


@tsne_ex.main
def pipeline(fit_tsne_configs, visualize_configs):
    activation_path = os.path.join(tsne_ex.observers[0].dir, "store_activations")
    activation_directories = store_activations(sacred_base_dir=activation_path)
    logger.info("Activations saved")

    fitted_tsne_path = os.path.join(tsne_ex.observers[0].dir, "fit")
    fitting_observer = FileStorageObserver.create(fitted_tsne_path)
    fit_tsne_ex.observers.append(fitting_observer)
    fit_tsne_configs['sacred_dir_ids'] = activation_directories
    fit_tsne_ex.run(config_updates=fit_tsne_configs)
    logger.info("Fitting complete")

    visualize_tsne_path = os.path.join(tsne_ex.observers[0].dir, "visualize")
    visualize_observer = FileStorageObserver.create(visualize_tsne_path)
    vis_tsne_ex.observers.append(visualize_observer)
    visualize_configs["sacred_id"] = fitting_observer.dir
    vis_tsne_ex.run(config_updates=visualize_configs)
    logger.info("Visualization complete")


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'tsne'))
    tsne_ex.observers.append(observer)
    tsne_ex.run_commandline()


if __name__ == '__main__':
    main()
