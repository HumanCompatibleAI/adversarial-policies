import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def opponent_convert(x):
    if 'zoo' in x:
        return 'zoo'
    else:
        return x

# This was run on top of the density directories with timestamp, and fitted/activations inside


def get_train_test_merged_df(env, victim_id, exp_id="20190517_114008"):
    env_dir = f"{env}-v0_victim_zoo_{victim_id}"
    train_df = pd.read_csv(os.path.join(exp_id, 'fitted', env_dir, "train_metadata.csv"))
    train_df['is_train'] = True
    test_df = pd.read_csv(os.path.join(exp_id, 'fitted', env_dir, "test_metadata.csv"))
    test_df['is_train'] = False
    merged = pd.concat([train_df, test_df])
    merged['broad_opponent'] = merged['opponent_id'].apply(lambda x: opponent_convert(x))
    merged['raw_proba'] = np.exp(merged['log_proba'])
    return merged


def comparative_densities(group_var, env_name, victim, shade=False, cutoff_point=None):
    df = get_train_test_merged_df(env_name, victim)
    plt.figure(figsize=(10, 7))
    if cutoff_point is not None:
        subset = df[df['log_proba'] > cutoff_point]
    else:
        subset = df.copy()
    grped = subset.groupby(group_var)
    for name, grp in grped:
        # clean up random_none to just random
        name = name.replace('_none', '')
        avg_log_proba = round(np.mean(grp['log_proba']), 2)
        sns.kdeplot(grp['log_proba'], label=f"{name}: {avg_log_proba}", shade=shade)
    plt.suptitle(f"{env_name} Densities, Victim Zoo {victim}: Trained on Zoo 1", y=0.95)
    plt.title("Avg Log Proba* in Legend")

    # Note that this was for running in a notebook with matplotlib inline; probably would want
    # to dynamically construct filename and do savefig, but I'll leave it this way for
    # efficiency's sake
    plt.show()


if __name__ == "__main__":
    # Example plot
    comparative_densities(group_var='opponent_id', env_name='KickAndDefend', victim='1',
                          cutoff_point=-1000)
