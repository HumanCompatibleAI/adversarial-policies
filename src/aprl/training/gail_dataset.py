import numpy as np
from stable_baselines.gail.dataset.dataset import ExpertDataset


class ExpertDatasetFromOurFormat(ExpertDataset):
    """GAIL Expert Dataset. Loads in our format, rather than the GAIL default.

    In particular, GAIL expects a dict of flattened arrays, with episodes concatenated together.
    The episode start is delineated by an `episode_starts` array. See `ExpertDataset` base class
    for more information.

    By contrast, our format consists of a list of NumPy arrays, one for each episode."""

    def __init__(self, expert_path, **kwargs):
        traj_data = np.load(expert_path, allow_pickle=True)

        # Add in episode starts
        episode_starts = []
        for reward_dict in traj_data["rewards"]:
            ep_len = len(reward_dict)
            # used to index episodes since they are flattened in GAIL format.
            ep_starts = [True] + [False] * (ep_len - 1)
            episode_starts.append(np.array(ep_starts))

        # Flatten arrays
        traj_data = {k: np.concatenate(v) for k, v in traj_data.items()}
        traj_data["episode_starts"] = np.concatenate(episode_starts)

        # Rename observations->obs
        traj_data["obs"] = traj_data["observations"]
        del traj_data["observations"]

        super().__init__(traj_data=traj_data, **kwargs)
