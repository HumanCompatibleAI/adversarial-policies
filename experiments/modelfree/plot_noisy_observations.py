import json
import math
import os

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

RESULT_DIR = "/Users/cody/Data/adversarial_policies/ray_results/noisy_victim_obs/"
OUT_DIR = "/Users/cody/Data/adversarial_policies/noisy_obs_plots/results_may_22"
ENV_LOOKUP = {
    'SumoHumans': 'multicomp/SumoHumansAutoContact-v0',
    'SumoAnts': 'multicomp/SumoAntsAutoContact-v0',
    'KickAndDefend': 'multicomp/KickAndDefend-v0',
    # 'YouShallNotPass': 'multicomp/YouShallNotPassHumans-v0' (Removed for now because I don't have
    # time at the moment to fix the plotting logic to deal with victim in different location)
}
AVAILABLE_ZOOS = {
    'SumoHumans': 3,
    'SumoAnts': 4,
    'KickAndDefend': 3
}


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


def generate_plots(experiment):
    num_episodes = int(experiment.split("_")[1])

    baseline_transformations = [
        {'new_col': 'log_noise', 'old_col': 'noise_magnitude',
         'func': lambda x: math.log(x)},
        {'new_col': 'agent0_win_perc', 'old_col': 'agent0_wins',
         'func': lambda x: x / num_episodes},
        {'new_col': 'agent1_win_perc', 'old_col': 'agent1_wins',
         'func': lambda x: x / num_episodes}
    ]

    zoo_path = os.path.join(RESULT_DIR, experiment, "noisy_zoo_observations.json")
    adversary_path = os.path.join(RESULT_DIR, experiment, "noisy_adversary_observations.json")
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
    experiment_out_dir = os.path.join(OUT_DIR, experiment)

    if not os.path.exists(experiment_out_dir):
        os.mkdir(experiment_out_dir)
    for short_env in ENV_LOOKUP:
        for zoo_id in range(1, AVAILABLE_ZOOS[short_env] + 1):
            subset_params = {
                'agent0_path': str(zoo_id),
                'env': ENV_LOOKUP[short_env]}

            zoo_plot_path = os.path.join(experiment_out_dir,
                                         f"{experiment}_ZooBaseline_"
                                         f"{short_env}_AgainstZoo{zoo_id}")

            adversary_plot_path = os.path.join(experiment_out_dir,
                                               f"{experiment}_AdversaryTrained_"
                                               f"{short_env}_AgainstZoo{zoo_id}")
            noisy_multiple_opponent_subset_plot(noisy_zoo_obs_df, subset_specs=subset_params,
                                                transform_specs=baseline_transformations,
                                                savefile=zoo_plot_path)
            noisy_adversary_opponent_subset_plot(noisy_adv_obs_df, subset_specs=subset_params,
                                                 transform_specs=baseline_transformations,
                                                 savefile=os.path.join(adversary_plot_path))


if __name__ == "__main__":
    for experiment_run in ["ep_500_5-22_single_zoo", "ep_100_5-21", "ep_500_5-22_all_zoo"]:
        generate_plots(experiment_run)
