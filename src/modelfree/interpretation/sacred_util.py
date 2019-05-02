import json
import logging
import os

logger = logging.getLogger('modelfree.interpretation.sacred_util')


def get_latest_sacred_dir_with_params(base_path, param_dict):
    sacred_dirs = os.listdir(base_path)
    max_int_dir = -1
    for sd in sacred_dirs:
        if param_dict is not None:
            try:
                with open(os.path.join(base_path, sd, "config.json")) as fp:
                    config_params = json.load(fp)
            except (NotADirectoryError, FileNotFoundError):
                logger.info("No config json found at {}".format(sd))
                continue
            all_match = True
            for param in param_dict:
                if param not in config_params or config_params[param] != param_dict[param]:
                    all_match = False
                    break
            if not all_match:
                continue
        try:
            int_dir = int(sd)
            if int_dir > max_int_dir:
                max_int_dir = int_dir
        except ValueError:
            continue
    if max_int_dir < 0:
        format_str = "No sacred directory found for base path {}, param dict {}"
        raise ValueError(format_str.format(base_path, param_dict))
    return str(max_int_dir)
