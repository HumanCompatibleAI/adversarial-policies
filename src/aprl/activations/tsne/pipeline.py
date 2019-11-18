import logging
import os
import os.path as osp

import sacred
from sacred.observers import FileStorageObserver

from aprl.activations import generate_activations
from aprl.activations.tsne import visualize
from aprl.activations.tsne.fit_model import fit_model, fit_model_ex
from aprl.common import utils

tsne_ex = sacred.Experiment(
    "tsne",
    ingredients=[
        generate_activations.generate_activations_ex,
        fit_model_ex,
        visualize.visualize_ex,
    ],
)
logger = logging.getLogger("aprl.activations.tsne.pipeline")


@tsne_ex.config
def activation_storing_config():
    output_root = "data/tsne"  # where to produce output
    exp_name = "default"  # experiment name

    _ = locals()  # quieten flake8 unused variable warning
    del _


@tsne_ex.named_config
def debug_config(generate_activations, tsne_visualize):
    generate_activations = dict(generate_activations)
    generate_activations["score_configs"] = [("debug_two_agents",)]
    generate_activations["score_update"] = {"score": {"timesteps": 100}}

    tsne_visualize = dict(tsne_visualize)
    tsne_visualize["ordering"] = ["Zoo", "Rand"]

    exp_name = "debug"

    _ = locals()  # quieten flake8 unused variable warning
    del _


@tsne_ex.main
def pipeline(_run, output_root, exp_name):
    out_dir = osp.join(output_root, exp_name, utils.make_timestamp())
    os.makedirs(out_dir)

    activation_dst_dir = osp.join(out_dir, "activations")
    generate_activations.generate_activations(out_dir=activation_dst_dir)

    model_dir = osp.join(out_dir, "fitted")
    fit_model(activation_dir=activation_dst_dir, output_root=model_dir)

    figure_dst_dir = osp.join(out_dir, "figures")
    visualize.visualize(model_glob=osp.join(model_dir, "*"), output_root=figure_dst_dir)

    return out_dir


def main():
    observer = FileStorageObserver(osp.join("data", "sacred", "tsne"))
    tsne_ex.observers.append(observer)
    tsne_ex.run_commandline()


if __name__ == "__main__":
    main()
