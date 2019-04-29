"""Named configs for modelfree.multi.train."""

import collections
import itertools

import numpy as np
from ray import tune

from modelfree.configs.multi.common import BANSAL_ENVS, BANSAL_GOOD_ENVS, VICTIM_INDEX
from modelfree.envs import gym_compete

MLP_ENVS = [env for env in BANSAL_ENVS if not gym_compete.is_stateful(env)]
LSTM_ENVS = [env for env in BANSAL_ENVS if gym_compete.is_stateful(env)]

TARGET_VICTIM = collections.defaultdict(lambda: 1)
TARGET_VICTIM['multicomp/KickAndDefend-v0'] = 2


def _env_victim(envs=None):
    if envs is None:
        envs = BANSAL_GOOD_ENVS
    env_and_victims = [[(env, i + 1) for i in range(gym_compete.num_zoo_policies(env))]
                       for env in envs]
    return list(itertools.chain(*env_and_victims))


def _sparse_reward(train):
    train['rew_shape'] = True
    train['rew_shape_params'] = {'anneal_frac': 0}


def _best_guess_train(train):
    train['total_timesteps'] = int(10e6)
    train['batch_size'] = 16384
    train['learning_rate'] = 3e-4
    train['rl_args'] = {
        'ent_coef': 0.0,
        'nminibatches': 4,
        'noptepochs': 4,
    }


def _best_guess_spec(envs=None):
    spec = {
        'config': {
            'env_name:victim_path': tune.grid_search(_env_victim(envs)),
            'victim_index': tune.sample_from(
                lambda spec: VICTIM_INDEX[spec.config['env_name:victim_path'][0]]
            ),
            'seed': tune.grid_search([0, 1, 2]),
        },
    }
    return spec


def _finetune_train(train):
    train['load_policy'] = {
        'path': '1',
        'type': 'zoo',
    }
    train['normalize'] = False


def _finetune_spec(envs=None):
    spec = {
        'config': {
            'env_name:victim_path': tune.grid_search(_env_victim(envs)),
            'seed': tune.grid_search([0, 1, 2]),
            'load_policy': {
                'path': tune.sample_from(lambda spec: spec.config['env_name:victim_path'][1]),
            },
            'victim_index': tune.sample_from(
                lambda spec: VICTIM_INDEX[spec.config['env_name:victim_path'][0]]
            ),
        },
    }
    return spec


def make_configs(multi_train_ex):
    # From-scratch sparse reward training
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
                    lambda spec: TARGET_VICTIM[spec.config.env_name]
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
    def paper(train):
        """Final experiments for paper. Like best_guess but more seeds & timesteps."""
        train = dict(train)
        _sparse_reward(train)
        _best_guess_train(train)
        train['total_timesteps'] = int(20e6)
        spec = _best_guess_spec()
        spec['config']['seed'] = tune.grid_search([0, 1, 2, 3, 4])
        exp_name = 'paper'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def best_guess(train):
        """Train with promising hyperparameters for 10 million timesteps."""
        train = dict(train)
        _sparse_reward(train)
        _best_guess_train(train)
        spec = _best_guess_spec()
        exp_name = 'best_guess'
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
        train['rew_shape'] = False
        spec = {
            'config': {
                'env_name': tune.grid_search(['Reacher-v1', 'Hopper-v1', 'Ant-v1', 'Humanoid-v1']),
                'seed': tune.grid_search([0, 1, 2]),
            },
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
            'config': {
                'env_name': tune.grid_search(
                    ['multicomp/KickAndDefend-v0', 'multicomp/SumoAnts-v0'],
                ),
                'seed': tune.grid_search([0, 1, 2]),
                'victim_path': tune.grid_search(['1', '2', '3']),
                'normalize': tune.grid_search([True, False]),
            },
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
            'config': {
                'env_name': tune.grid_search(
                    ['multicomp/KickAndDefend-v0', 'multicomp/SumoAnts-v0'],
                ),
                'victim_path': tune.grid_search(['1', '2', '3']),
                'seed': tune.grid_search([0, 1, 2]),
                'rew_shape_params': {
                    'anneal_frac': tune.grid_search([0.0, 0.1]),
                },
            },
        }
        exp_name = 'dec2018rep'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    # From-scratch dense reward
    @multi_train_ex.named_config
    def dense_env_reward(train):
        """Train with the dense reward defined by the environment."""
        train = dict(train)
        _best_guess_train(train)
        train['total_timesteps'] = int(40e6)
        train['rew_shape'] = True
        train['rew_shape_params'] = {'anneal_frac': 0.25}
        spec = _best_guess_spec(envs=['multicomp/SumoHumansAutoContact-v0',
                                      'multicomp/YouShallNotPassHumans-v0'])
        exp_name = 'dense_env_reward'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def dense_env_reward_anneal_search(train):
        """Search for the best annealing fraction in SumoHumans."""
        train = dict(train)
        _best_guess_train(train)
        train['total_timesteps'] = int(40e6)
        train['rew_shape'] = True
        train['env_name'] = 'multicomp/SumoHumansAutoContact-v0'
        train['victim_path'] = 3  # median difficulty victim (1 is easy, 2 is hard)
        spec = {
            'config': {
                'rew_shape_params': {
                    'anneal_frac': tune.sample_from(
                        lambda spec: np.random.rand()
                    ),
                },
                'seed': tune.sample_from(
                    lambda spec: np.random.randint(1000)
                ),
            },
            'num_samples': 10,
        }
        exp_name = 'dense_env_reward_anneal_search'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def kick_and_defend_sparse_vs_dense(train):
        """Does dense reward help KickAndDefend, even though the final policy never stands up?"""
        train = dict(train)
        _best_guess_train(train)
        train['rew_shape'] = True
        train['env_name'] = 'multicomp/KickAndDefend-v0'
        spec = {
            'config': {
                'rew_shape_params': {
                    'anneal_frac': tune.grid_search([0.0, 0.25]),
                },
                'victim_path': tune.grid_search(['1', '2', '3']),
                'seed': tune.grid_search([10, 20, 30, 40, 50]),
            },
        }
        exp_name = 'kick_and_defend_sparse_vs_dense'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    # Finetuning
    @multi_train_ex.named_config
    def finetune_nolearn(train):
        """Sanity check finetuning: with a learning rate of 0.0, do we get performance the
           same as that given by `score_agent`? Tests the training-specific loading pipeline."""
        train = dict(train)
        _finetune_train(train)
        train['total_timesteps'] = int(1e6)
        train['learning_rate'] = 0.0
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
        """Use the policy architecture in gym_compete, but training from random initialization
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
                    lambda spec: TARGET_VICTIM[spec.config.env_name]
                ),
                'policy': tune.grid_search(['BansalMlpPolicy', 'BansalLstmPolicy']),
                'seed': tune.grid_search([0, 1, 2]),
            },
        }
        exp_name = 'gym_compete_from_scratch'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def finetune_best_guess(train):
        """Finetuning gym_compete policies with current best guess of hyperparameters. Policy
        initialization is the same path as the victim path. (In symmetric environments, they'll be
        the same policy.)"""
        train = dict(train)
        _sparse_reward(train)
        _finetune_train(train)
        _best_guess_train(train)
        spec = _finetune_spec()
        exp_name = 'finetune_best_guess'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def finetune_gentle_mlp(train):
        """Finetuning gym_compete MLP policies with lower learning rate / larger batch size.
        This more closely emulates the hyperparameters they were originally trained with."""
        train = dict(train)
        _sparse_reward(train)
        _finetune_train(train)
        _best_guess_train(train)
        train['batch_size'] = 32768
        train['learning_rate'] = 1e-4
        spec = _finetune_spec(envs=MLP_ENVS)
        exp_name = 'finetune_gentle_mlp'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def finetune_gentle_youshallnotpass(train):
        """Finetune gym_compete policy on YouShallNotPassHumans.
        This is the only environment that I saw some improvement on.
        Same hyperparams as finetune_gentle_mlp, but greater total number of timesteps."""
        train = dict(train)
        _sparse_reward(train)
        _finetune_train(train)
        _best_guess_train(train)
        train['batch_size'] = 32768
        train['learning_rate'] = 1e-4
        train['total_timesteps'] = int(40e6)
        spec = _finetune_spec(envs=['multicomp/YouShallNotPassHumans-v0'])
        exp_name = 'finetune_gentle_youshallnotpass'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def finetune_gentle_lstm(train):
        """Finetuning gym_compete LSTM policies with lower learning rate / larger batch size.
        This more closely emulates the hyperparameters they were originally trained with."""
        train = dict(train)
        _sparse_reward(train)
        _finetune_train(train)
        _best_guess_train(train)
        train['num_env'] = 16
        train['batch_size'] = train['num_env'] * 128  # Note Bansal used n_steps=10
        train['learning_rate'] = 1e-4
        spec = _finetune_spec(envs=['multicomp/SumoHumansAutoContact-v0'])
        exp_name = 'finetune_gentle_lstm'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def lstm_policies(train):
        """Do LSTM policies work? This is likely to require some hyperparameter tuning;
           just my best guess."""
        train = dict(train)
        _sparse_reward(train)
        train['total_timesteps'] = int(10e6)
        train['learning_rate'] = 1e-4
        train['num_env'] = 16
        train['batch_size'] = train['num_env'] * 128
        train['rl_args'] = {
            'ent_coef': 0.0,
            'nminibatches': 4,
            'noptepochs': 4,
        }
        spec = {
            'config': {
                'env_name': tune.grid_search(LSTM_ENVS),
                'policy': tune.grid_search(['MlpLstmPolicy', 'BansalLstmPolicy']),
                'seed': tune.grid_search([0, 1, 2]),
            },
        }
        exp_name = 'lstm_policies'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    # Adversarial noise ball
    @multi_train_ex.named_config
    def noise_ball_search(train):
        """Random search of size of allowed noise ball."""
        train = dict(train)
        _sparse_reward(train)
        _best_guess_train(train)
        train['adv_noise_params'] = {
            'base_type': 'zoo',
            'base_path': '1',
        }
        spec = {
            'config': {
                'env_name': tune.grid_search(
                    ['multicomp/SumoHumansAutoContact-v0', 'multicomp/KickAndDefend-v0'],
                ),
                'victim_path': tune.sample_from(
                    lambda spec: TARGET_VICTIM[spec.config.env_name]
                ),
                'victim_index': tune.sample_from(
                    lambda spec: VICTIM_INDEX[spec.config.env_name]
                ),
                'adv_noise_params': {
                    'noise_val': tune.sample_from(
                        lambda spec: 10 ** (np.random.rand() * -2)
                    ),
                }
            },
            'num_samples': 10,
        }
        exp_name = 'noise_ball_search'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def noise_ball(train):
        """Test adversarial noise ball policy in a range of environments."""
        train = dict(train)
        _sparse_reward(train)
        _best_guess_train(train)
        train['adv_noise_params'] = {
            'base_type': 'zoo',
            'noise_val': 1.0,
        }
        spec = {
            'config': {
                'env_name:victim_path': tune.grid_search(_env_victim(
                    ['multicomp/SumoHumansAutoContact-v0', 'multicomp/RunToGoalHumans-v0'],
                )),
                'adv_noise_params': {
                    'base_path': tune.sample_from(
                        lambda spec: spec.config['env_name:victim_path'][1],
                    )
                },
                'seed': tune.grid_search([0, 1, 2]),
            },
        }
        exp_name = 'noise_ball'
        _ = locals()  # quieten flake8 unused variable warning
        del _
