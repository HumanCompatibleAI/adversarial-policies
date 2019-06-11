import json
import math
import os
import os.path as osp

from matplotlib import pyplot as plt
import pandas as pd
from sacred import Experiment
from sacred.observers import FileStorageObserver
import seaborn as sns

from modelfree.envs.gym_compete import NUM_ZOO_POLICIES
from modelfree.visualize.util import PRETTY_ENV

plot_noisy_obs_exp = Experiment('plot_noisy_observations')


@plot_noisy_obs_exp.config
def base_config():
    root_dir = "data/aws/score_agents/victim_masked_noise/"
    out_dir = "data/aws/score_agents/masked_obs_visualization/"
    env_lookup = {
        'SumoHumans': 'multicomp/SumoHumansAutoContact-v0',
        'SumoAnts': 'multicomp/SumoAntsAutoContact-v0',
        'KickAndDefend': 'multicomp/KickAndDefend-v0'
    }
    available_zoos = {
        'SumoHumans': 3,
        'SumoAnts': 4,
        'KickAndDefend': 3
    }
    input_run = "ep_500_5-22_all_zoo"
    # Runs known to work: ["ep_500_5-22_single_zoo", "ep_100_5-21", "ep_500_5-22_all_zoo"]
    _ = locals()  # quieten flake8 unused variable warning
    del _


def transform(df, transform_list):
    new_df = df.copy()
    for trans_dict in transform_list:
        new_df[trans_dict['new_col']] = new_df[trans_dict['old_col']].apply(trans_dict['func'])
    return new_df


def subset(df, spec):
    ret = df.copy()
    for constraint, constraint_value in spec.items():
        ret = ret[ret[constraint] == constraint_value]
    return ret


def process_element_into_flat_dict(el, key_order):
    outp = {}
    for i, k in enumerate(key_order):
        outp[k] = el['k'][i]
    outp['agent0_wins'] = el['v']['win0']
    outp['agent1_wins'] = el['v']['win1']
    outp['ties'] = el['v']['ties']
    return outp


def noisy_adversary_opponent_subset_plot(original_df, subset_specs, transform_specs,
                                         logistic=True, plot_line=True, savefile=None):
    subset_df = subset(original_df, subset_specs)
    if len(subset_df) == 0:
        return
    transformed_df = transform(subset_df, transform_specs)
    plt.figure(figsize=(10, 7))
    if plot_line:
        sns.lmplot(data=transformed_df, x='log_noise', y='agent0_win_perc',
                   logistic=logistic)
    else:
        sns.scatterplot(data=transformed_df, x='log_noise', y='agent0_win_perc')
    plt.title("{}: Noisy Zoo{} Observations vs Adversary".format(subset_specs['env'],
                                                                 subset_specs['agent0_path']))
    if savefile is not None:
        plt.savefig(savefile)
    else:
        plt.show()
    plt.close()


def noisy_multiple_opponent_subset_plot(original_df, subset_specs, transform_specs,
                                        logistic=True, savefile=None):
    subset_df = subset(original_df, subset_specs)
    if len(subset_df) == 0:
        return
    transformed_df = transform(subset_df, transform_specs)
    plt.figure(figsize=(10, 7))
    sns.lmplot(data=transformed_df, x='log_noise', y='agent0_win_perc', hue='agent1_path',
               logistic=logistic)
    plt.title("{}: Noisy Zoo{} Observations vs Normal Zoos".format(subset_specs['env'],
                                                                   subset_specs['agent0_path']))
    if savefile is not None:
        plt.savefig(savefile)
    else:
        plt.show()
    plt.close()


@plot_noisy_obs_exp.main
def generate_plots(input_run, root_dir, out_dir, env_lookup, available_zoos):
    num_episodes = int(input_run.split("_")[1])
    baseline_transformations = [
        {'new_col': 'log_noise', 'old_col': 'noise_magnitude',
         'func': lambda x: math.log(x)},
        {'new_col': 'agent0_win_perc', 'old_col': 'agent0_wins',
         'func': lambda x: x / num_episodes},
        {'new_col': 'agent1_win_perc', 'old_col': 'agent1_wins',
         'func': lambda x: x / num_episodes}
    ]

    zoo_path = os.path.join(root_dir, input_run, "noisy_zoo_observations.json")
    adversary_path = os.path.join(root_dir, input_run, "noisy_adversary_observations.json")
    with open(adversary_path, "r") as fp:
        noisy_obs_against_adv = json.load(fp)

    DATAFRAME_KEYS = ['env', 'agent0_type', 'agent0_path', 'agent1_type',
                      'agent1_path', 'masking_param', 'noise_magnitude']

    with open(zoo_path, "r") as fp:
        noisy_obs_against_zoo = json.load(fp)
    noisy_zoo_obs_df = pd.DataFrame(
        [process_element_into_flat_dict(el, key_order=DATAFRAME_KEYS)
         for el in noisy_obs_against_zoo])
    noisy_adv_obs_df = pd.DataFrame(
        [process_element_into_flat_dict(el, key_order=DATAFRAME_KEYS)
         for el in noisy_obs_against_adv])
    experiment_out_dir = os.path.join(out_dir, input_run)

    if not os.path.exists(experiment_out_dir):
        os.mkdir(experiment_out_dir)

    for env_name, pretty_env in PRETTY_ENV.items():
        short_env = pretty_env.replace(' ', '')
        if env_name == 'multicomp/YouShallNotPassHumans-v0':
            # skip for now as has different victim index, need to fix plotting code
            continue

        for zoo_id in range(1, NUM_ZOO_POLICIES[short_env] + 1):
            subset_params = {
                'agent0_path': str(zoo_id),
                'env': env_name
            }

            zoo_plot_path = os.path.join(experiment_out_dir,
                                         f"{input_run}_ZooBaseline_"
                                         f"{short_env}_AgainstZoo{zoo_id}")

            adversary_plot_path = os.path.join(experiment_out_dir,
                                               f"{input_run}_AdversaryTrained_"
                                               f"{short_env}_AgainstZoo{zoo_id}")
            noisy_multiple_opponent_subset_plot(noisy_zoo_obs_df, subset_specs=subset_params,
                                                transform_specs=baseline_transformations,
                                                savefile=zoo_plot_path)
            noisy_adversary_opponent_subset_plot(noisy_adv_obs_df, subset_specs=subset_params,
                                                 transform_specs=baseline_transformations,
                                                 savefile=os.path.join(adversary_plot_path))


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'plot_noisy_observations'))
    plot_noisy_obs_exp.observers.append(observer)
    plot_noisy_obs_exp.run_commandline()


if __name__ == '__main__':
    main()
