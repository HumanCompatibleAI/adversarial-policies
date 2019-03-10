"""Named configs for modelfree.hyperparams."""

import numpy as np
from ray import tune


def make_configs(hyper_ex):
    @hyper_ex.named_config
    def basic_spec(train):
        train = dict(train)
        train['total_timesteps'] = 3000000
        spec = {
            'config': {
                'env_name': tune.grid_search(
                    ['multicomp/KickAndDefend-v0', 'multicomp/SumoHumans-v0']
                ),
                'seed': tune.sample_from(
                    lambda spec: np.random.randint(1000)
                ),
                # Dec 2018 experiments used 2^11 = 2048 batch size.
                # Aurick Zhou used 2^14 = 16384; Bansal et al use 409600 ~= 2^19.
                'batch_size': tune.sample_from(
                    lambda spec: 2 ** np.random.randint(11, 16)
                ),
                'rl_args': {
                    # PPO2 default is 0.01. run_humanoid.py uses 0.00.
                    'ent_coef': tune.sample_from(
                        lambda spec: np.random.uniform(low=0.00, high=0.02)
                    ),
                    # nminibatches must be a factor of batch size; OK provided power of two
                    # PPO2 default is 2^2 = 4; run_humanoid.py is 2^5 = 32
                    'nminibatches': tune.sample_from(
                        lambda spec: 2 ** (np.random.randint(0, 7))
                    ),
                    # PPO2 default is 4; run_humanoid.py is 10
                    'noptepochs': tune.sample_from(
                        lambda spec: np.random.randint(1, 11),
                    ),
                },
                # PPO2 default is 3e-4; run_humanoid uses 1e-4;
                # Bansal et al use 1e-2 (but with huge batch size).
                # Sample log-uniform between 1e-2 and 1e-5.
                'learning_rate': tune.sample_from(
                    lambda spec: 10 ** (-2 + -3 * np.random.random())
                ),
            },
            'num_samples': 100,
        }
        exp_name = 'basic'
        _ = locals()  # quieten flake8 unused variable warning
        del _
