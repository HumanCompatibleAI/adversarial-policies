import json
import logging


def get_best_adversary_path(best_path, environment, zoo_id, base_path, victim_idx=None):
    with open(best_path, 'rb') as fp:
        best_adversaries = json.load(fp)
    best_adv_policy_paths = best_adversaries['policies']
    best_adv_for_env = best_adv_policy_paths[environment]
    victim_idxs = best_adv_for_env.keys()
    if victim_idx is None:
        logging.debug("No victim_idx specified, getting policy for victim_idx in 0th position")
        victim_idx = list(victim_idxs)[0]

    best_adv_for_victim_idx = best_adv_for_env[victim_idx]
    best_adv_for_zoo_agent_path = best_adv_for_victim_idx[str(zoo_id)]
    return best_adv_for_zoo_agent_path