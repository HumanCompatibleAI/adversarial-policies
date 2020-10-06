import json
import logging
import os
import os.path as osp

import sacred
from sacred.observers import FileStorageObserver

from aprl.common import utils
from aprl.multi.score import extract_data, run_external

generate_activations_ex = sacred.Experiment("generate_activations")
logger = logging.getLogger("aprl.activations.generate_activations")


@generate_activations_ex.config
def activation_storing_config():
    adversary_path = osp.join(
        "data",
        "aws",
        "score_agents",
        "normal",
        "2019-05-05T18:12:24+00:00",
        "best_adversaries.json",
    )
    ray_upload_dir = "data"  # where Ray will upload multi.score outputs. 'data' works on local
    out_dir = None

    # Configs for the multi-score experiments
    score_configs = [(x,) for x in ["zoo_baseline", "random_baseline", "adversary_trained"]]
    score_update = {}

    _ = locals()  # quieten flake8 unused variable warning
    del _


def _activations_path_generator(
    trial_root,
    cfg,
    env_sanitized,
    victim_index,
    victim_type,
    victim_path,
    opponent_type,
    opponent_path,
):
    del cfg
    src_path = osp.join(trial_root, "data", "trajectories", f"agent_{victim_index}.npz")

    if opponent_path.startswith("/"):  # is path name
        opponent_root = osp.sep.join(opponent_path.split(osp.sep)[:-3])
        opponent_sacred = osp.join(opponent_root, "sacred", "train", "1", "config.json")

        with open(opponent_sacred, "r") as f:
            opponent_cfg = json.load(f)

        if "embed_path" in opponent_cfg:
            opponent_path = opponent_cfg["embed_path"]
        elif "victim_path" in opponent_cfg:
            # TODO(adam): remove backwards compatibility when all policies retrained
            opponent_path = opponent_cfg["victim_path"]
        else:
            raise KeyError("'embed_path' and 'victim_path' not present in 'opponent_cfg'")

    new_name = (
        f"{env_sanitized}_victim_{victim_type}_{victim_path}"
        f"_opponent_{opponent_type}_{opponent_path}"
    )
    return src_path, new_name, "npz"


@generate_activations_ex.main
def generate_activations(
    _run, out_dir, score_configs, score_update, adversary_path, ray_upload_dir
):
    """Uses multi.score to generate activations, then extracts them into a convenient
    directory structure."""
    logger.info("Generating activations")
    activation_dirs = run_external(
        score_configs,
        post_named_configs=["save_activations"],
        config_updates=score_update,
        adversary_path=adversary_path,
    )

    os.makedirs(out_dir)
    extract_data(_activations_path_generator, out_dir, activation_dirs, ray_upload_dir)
    logger.info("Activations saved")

    utils.add_artifacts(_run, out_dir)


def main():
    observer = FileStorageObserver(osp.join("data", "sacred", "generate_activations"))
    generate_activations_ex.observers.append(observer)
    generate_activations_ex.run_commandline()


if __name__ == "__main__":
    main()
