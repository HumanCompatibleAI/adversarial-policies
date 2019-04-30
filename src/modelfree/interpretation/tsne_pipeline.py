import argparse
import os

from get_best_adversary import get_best_adversary_path
from sacred.observers import FileStorageObserver

from modelfree.score_agent import score_ex
import sacred
from fit_tsne import tsne_experiment as tsne_fitting_experiment
from visualize_tsne import tsne_vis_ex
import pdb
import tempfile


global_save_dir = "/Users/cody/Data/adversarial_policies/master_runs/"
master_exp = sacred.Experiment('master_tsne_pipeline')
master_observer = FileStorageObserver.create(global_save_dir)
master_exp.observers.append(master_observer)

@master_exp.config
def activation_storing_config():
    adversary_path = "/Users/cody/Data/adversarial_policies/"
    perplexity = 250,
    score_agent_configs = dict(
        transparent_params={'ff_policy': True, 'ff_value': True},
        record_traj_params={'agent_indices': 0},
        env_name='multicomp/SumoAnts-v0',
        # multicomp/YouShallNotPassHumans-v0 // multicomp/KickAndDefend-v0
        num_env=1,
        record_traj=True,
        videos=True,
        video_dir=None,
        episodes=2,
        render=False
    ),

    fit_tsne_configs = dict(
        relative_dirs=['adversary', 'random', 'zoo'],
        data_type = 'ff_policy',
        num_components = 2,
        sacred_dir_ids = None,
        np_file_name = "victim_activations.npz",
        num_observations = None,
        env_name = "multicomp/YouShallNotPassHumans-v0",
        seed = 0,
        perplexity = perplexity[0],
    )

    visualize_configs = dict(
        base_path=None,
        subsample_rate = 0.15,
        perplexity = perplexity[0],
        video_path = "/Users/cody/Data/adversarial_policies/video_frames",
        chart_type = "seaborn",
        opacity = 0.75,
        dot_size = 0.25,
        palette_name = "cube_bright",
        save_type = "pdf",
        hue_order = ["adversary", "zoo", "random"],
    )

    skip_scoring = False
    _ = locals()
    del _


@master_exp.capture
def store_activations(sacred_base_dir, adversary_path, score_agent_configs):


    best_adversary_path_agent_1 = get_best_adversary_path(
                    environment=score_agent_configs[0]['env_name'],
                    zoo_id=1, base_path=adversary_path)

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
        config_copy = score_agent_configs[0].copy()
        with tempfile.TemporaryDirectory() as td:
            config_copy['record_traj_params']['save_dir'] = td
            config_copy.update({'agent_b_type': test_config["policy_type"],
                                'agent_b_path': test_config["path"]})
            run = score_ex.run(config_updates=config_copy)
        assert run.status == 'COMPLETED'
        score_ex.observers.pop()

@master_exp.automain
def pipeline(fit_tsne_configs, visualize_configs):
    activation_path = os.path.join(master_exp.observers[0].dir, "store_activations")
    activation_directories = store_activations(sacred_base_dir=activation_path)

    print("Activations saved")

    fitted_tsne_path = os.path.join(master_exp.observers[0].dir, "fit")
    fitting_observer = FileStorageObserver.create(fitted_tsne_path)
    tsne_fitting_experiment.observers.append(fitting_observer)
    fit_tsne_configs['sacred_dir_ids'] = activation_directories
    tsne_fitting_experiment.run(config_updates=fit_tsne_configs)
    print("Fitting complete")

    visualize_tsne_path = os.path.join(master_exp.observers[0].dir, "visualize")
    visualize_observer = FileStorageObserver.create(visualize_tsne_path)
    tsne_vis_ex.observers.append(visualize_observer)
    visualize_configs["sacred_id"] = fitting_observer.dir
    tsne_vis_ex.run(config_updates=visualize_configs)
    print("Visualization complete")


