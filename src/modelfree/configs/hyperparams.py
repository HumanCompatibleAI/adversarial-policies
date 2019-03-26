"""Named configs for modelfree.hyperparams."""

import numpy as np
from ray import tune

TARGET_VICTIMS = {
    'multicomp/KickAndDefend-v0': 2,
}


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
                'victim_path': tune.sample_from(
                    lambda spec: TARGET_VICTIMS.get(spec.config.env_name, 1)
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

    @hyper_ex.named_config
    def gail_spec(train):
        train = dict(train)
        train['total_timesteps'] = int(1e7)
        spec = {
            'config': {
                'rl_algo': 'gail',
                'env_name': tune.grid_search(
                    ['multicomp/KickAndDefend-v0', 'multicomp/SumoHumans-v0']
                ),
                'victim_index': tune.grid_search(
                    [0, 1]
                ),
                'victim_path': tune.sample_from(
                    lambda spec: 1 if spec.config.env_name == 'multicomp/SumoHumans-v0'
                    else spec.config.victim_index + 1
                ),
                'seed': tune.sample_from(
                    lambda spec: np.random.randint(1000)
                ),
                # default is 1024 (2^10)
                'batch_size': tune.sample_from(
                    lambda spec: 2 ** np.random.randint(9, 15)
                ),
                'num_env': 1,
                'expert_dataset_path': 'default',
                'rl_args': {
                    # default is 100
                    'hidden_size_adversary': tune.sample_from(
                        lambda spec: 50 * np.random.randint(1, 6)
                    ),
                    # default is 1e-3
                    # log-uniform between 1e-2 and 1e-4
                    'adversary_entcoeff': tune.sample_from(
                        lambda spec: 10 ** (-2 + -2 * np.random.random())
                    ),
                    # default is 3
                    'g_step': tune.sample_from(
                        lambda spec: np.random.randint(2, 5)
                    ),
                    # default is 3e-4
                    # log-uniform between 1e-2.5 and 1e-5
                    'd_stepsize': tune.sample_from(
                        lambda spec: 10 ** (-2.5 + -2.5 * np.random.random())
                    ),
                    # default is 10,
                    'cg_iters': tune.sample_from(
                        lambda spec: 3 * np.random.randint(2, 8)
                    ),
                    # default is 1e-2
                    # log-uniform between 1e-1.5 and 1e-3
                    'cg_damping': tune.sample_from(
                        lambda spec: 10 ** (-1.5 + -1.5 * np.random.random())
                    ),
                    # default is 3e-4
                    # log-uniform between 1e-2.5 and 1e-5
                    'vf_stepsize': tune.sample_from(
                        lambda spec: 10 ** (-2.5 + -2.5 * np.random.random())
                    ),
                    # default is 3
                    'vf_iters': tune.sample_from(
                        lambda spec: np.random.randint(2, 7)
                    ),
                },

            },
            'num_samples': 50,
        }
        exp_name = 'gail'
        _ = locals()  # quieten flake8 unused variable warning
        del _
