"""Named configs for modelfree.hyperparams."""

import numpy as np
from ray import tune

TARGET_VICTIMS = {
    'multicomp/KickAndDefend-v0': 2,
}


def _sparse_reward(train):
    train['rew_shape'] = True
    train['rew_shape_params'] = {'anneal_frac': 0}


def make_configs(multi_train_ex):
    @multi_train_ex.named_config
    def hyper(train):
        """A random search to find good hyperparameters in Bansal et al's environments."""
        train = dict(train)
        _sparse_reward(train)
        # Checkpoints take up a lot of disk space, only save every ~500k steps
        train['checkpoint_interval'] = 2 ** 19
        train['total_timesteps'] = int(3e6)
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
        exp_name = 'hyper'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def single_agent_baseline(train):
        """Baseline applying our method to standard single-agent Gym MuJoCo environments.

        Should perform similarly to the results given in PPO paper."""
        train = dict(train)
        _sparse_reward(train)
        train['victim_type'] = 'none'
        train['total_timesteps'] = int(5e6)
        train['batch_size'] = 2048
        spec = {
            'env_name': tune.grid_search(['Reacher-v1', 'Hopper-v1', 'Ant-v1', 'Humanoid-v1']),
            'seed': tune.grid_search([0, 1, 2]),
        }
        exp_name = 'single_agent_baseline'
        _ = locals()   # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def vec_normalize(train):
        """Does using VecNormalize make a difference in performance?
        (Answer: not much after we rescaled reward; before the reward clipping had big effect.)"""
        train = dict(train)
        _sparse_reward(train)
        train['total_timesteps'] = int(5e6)
        train['learning_rate'] = 2.5e-4
        train['batch_size'] = 2048
        train['rl_args'] = {'ent_coef': 0.00}
        spec = {
            'env_name': tune.grid_search(
                ['multicomp/KickAndDefend-v0', 'multicomp/SumoAnts-v0'],
            ),
            'seed': tune.grid_search([0, 1, 2]),
            'victim_path': tune.grid_search(['1', '2', '3']),
            'normalize': tune.grid_search([True, False]),
        }
        exp_name = 'vec_normalize'
        _ = locals()   # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def dec2018rep(train):
        """Reproduce results from December 2018 draft paper with new codebase."""
        train = dict(train)
        train['rew_shape'] = True
        train['total_timesteps'] = int(5e6)
        train['batch_size'] = 2048
        train['learning_rate'] = 2.5e-4
        train['rl_args'] = {'ent_coef': 0.0}
        spec = {
            'env_name': tune.grid_search(
                ['multicomp/KickAndDefend-v0', 'multicomp/SumoAnts-v0'],
            ),
            'victim_path': tune.grid_search(['1', '2', '3']),
            'seed': tune.grid_search([0, 1, 2]),
            'rew_shape_params': {
                'anneal_frac': tune.grid_search([0.0, 0.1]),
            },
        }
        exp_name = 'dec2018rep'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def finetune_nolearn(train):
        """Sanity check finetuning: with a learning rate of 0.0, do we get performance the
           same as that given by `score_agent`? Tests the training-specific loading pipeline."""
        train = dict(train)
        train['total_timesteps'] = int(1e6)
        train['learning_rate'] = 0.0
        train['load_policy'] = {
            'path': '1',
            'type': 'zoo',
        }
        train['normalize'] = False
        spec = {
            'config': {
                'env_name': tune.grid_search([
                    'multicomp/SumoHumans-v0',  # LSTM
                    'multicomp/RunToGoalHumans-v0'  # MLP
                ]),
            },
        }
        exp_name = 'finetune_nolearn'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def gym_compete_from_scratch(train):
        """Use the policy architecture in gym_compete, but train from random initialization
           (i.e. not loading one of their zoo agents). There's no reason to actually do this
           (there are nicer implementations of these architectures in Stable Baselines),
           but it confirms that training works, and together with `finetune_nolearn` gives
           confidence that finetuning is operating correctly."""
        train = dict(train)
        _sparse_reward(train)
        train['total_timesteps'] = int(5e6)
        train['batch_size'] = 2048
        train['learning_rate'] = 2.5e-4
        train['rl_args'] = {'ent_coef': 0.0}
        spec = {
            'config': {
                'env_name': tune.grid_search([
                    'multicomp/KickAndDefend-v0',  # should be able to win in this
                    'multicomp/SumoHumans-v0',  # should be able to get to 50% in this
                ]),
                'victim_path': tune.sample_from(
                    lambda spec: TARGET_VICTIMS.get(spec.config.env_name, 1)
                ),
                'policy': tune.grid_search(['BansalMlpPolicy', 'BansalLstmPolicy']),
                'seed': tune.grid_search([0, 1, 2]),
            },
        }
        exp_name = 'gym_compete_from_scratch'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def finetune_hyper_search(train):
        """Hyperparameter search for finetuning gym_compete policies."""
        train = dict(train)
        _sparse_reward(train)
        # Checkpoints take up a lot of disk space, only save every ~500k steps
        train['checkpoint_interval'] = 2 ** 19
        train['total_timesteps'] = int(3e6)
        spec = {
            'config': {
                'env_name': tune.grid_search([
                    'multicomp/KickAndDefend-v0',
                    'multicomp/SumoHumans-v0'
                ]),
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
        exp_name = 'finetune_hyper'
        _ = locals()  # quieten flake8 unused variable warning
        del _
