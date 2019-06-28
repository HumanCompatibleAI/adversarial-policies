import copy
import datetime
import os

import tensorflow as tf


def make_session(graph=None):
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=tf_config)
    return sess


def make_timestamp():
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    return datetime.datetime.now().strftime(ISO_TIMESTAMP)


def add_artifacts(run, dirname, ingredient=None):
    """Convenience function for Sacred to add artifacts inside directory dirname to current run.

    :param run: (sacred.Run) object representing current experiment. Can be captured as `_run`.
    :param dirname: (str) root of directory to save.
    :param ingredient: (sacred.Ingredient or None) optional, ingredient that generated the
                       artifacts. Will be used to tag saved files. This is ignored if ingredient
                       is equal to the currently running experiment.
    :return None"""
    prefix = ""
    if ingredient is not None:
        exp_name = run.experiment_info['name']
        ingredient_name = ingredient.path
        if exp_name != ingredient_name:
            prefix = ingredient_name + "_"

    for root, dirs, files in os.walk(dirname):
        for file in files:
            path = os.path.join(root, file)
            relroot = os.path.relpath(path, dirname)
            name = prefix + relroot.replace('/', '_') + '_' + file
            run.add_artifact(path, name=name)


def dict_deep_copy(d):
    """Perform a deep copy on nested dictionaries.

    If `d` is an instance-of dict, copies `d` to a dict where the values are
    recursively copied using `dict_deep_copy`. If `d` is not a dict, copies it
    using `copy.deepcopy`. Note this intentioanlly loses subclasses.
    This is useful if e.g. `d` is a Sacred read-only dict. However, it can be
    undesirable if e.g. `d` is an OrderedDict.

    :param d: (object) if dict, copy recursively; otherwise, use `copy.deepcopy`.
    :return A deep copy of d."""
    if isinstance(d, dict):
        return {k: dict_deep_copy(v) for k, v in d.items()}
    else:
        return copy.deepcopy(d)
