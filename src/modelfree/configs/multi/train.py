"""Named configs for modelfree.multi.train."""

import collections
import itertools
import json
import os
import os.path as osp

import numpy as np
from ray import tune

from modelfree.configs.multi.common import BANSAL_ENVS, BANSAL_GOOD_ENVS, _get_adversary_paths
from modelfree.envs import VICTIM_INDEX, gym_compete

MLP_ENVS = [env for env in BANSAL_ENVS if not gym_compete.is_stateful(env)]
LSTM_ENVS = [env for env in BANSAL_ENVS if gym_compete.is_stateful(env)]

TARGET_VICTIM = collections.defaultdict(lambda: 1)
TARGET_VICTIM['multicomp/KickAndDefend-v0'] = 2

HYPERPARAM_SEARCH_VALUES = {
    'seed': tune.sample_from(
        lambda spec: np.random.randint(1000)),

    # Dec 2018 experiments used 2^11 = 2048 batch size.
    # Aurick Zhou used 2^14 = 16384; Bansal et al use 409600 ~= 2^19.
    'batch_size': tune.sample_from(
        lambda spec: 2 ** np.random.randint(11, 16)),

    'rl_args': {
        # PPO2 default is 0.01. run_humanoid.py uses 0.00.
        'ent_coef': tune.sample_from(
            lambda spec: np.random.uniform(low=0.00, high=0.02)),

        # PPO2 default is 4; run_humanoid.py is 10
        'noptepochs': tune.sample_from(
            lambda spec: np.random.randint(1, 11)),
                },
    # PPO2 default is 3e-4; run_humanoid uses 1e-4;
    # Bansal et al use 1e-2 (but with huge batch size).
    # Sample log-uniform between 1e-2 and 1e-5.
    'learning_rate': tune.sample_from(
        lambda spec: 10 ** (-2 + -3 * np.random.random())),
}

MULTI_TRAIN_LOCATION = osp.join(os.environ.get('DATA_LOC', 'data'), "multi_train")


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
    train['normalize_observations'] = False


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


def _get_path_from_exp_name(exp_name, json_file_path=None):
    # Takes in an experiment name and auto-constructs the JSON path containing its best policies
    if json_file_path is None:
        json_file_path = "highest_win_policies_and_rates.json"
    full_json_path = os.path.join(MULTI_TRAIN_LOCATION, exp_name, json_file_path)
    try:
        with open(full_json_path, 'r') as f:
            return json.load(f)['policies']
    except FileNotFoundError:
        raise FileNotFoundError(f"Please run highest_win_rate.py for experiment {exp_name} before"
                                " trying to use it ")


# ### CONFIGS FOR FINETUNING AGAINST ADVERSARY OR ADVERSARY + DUAL ### #

def _finetune_configs(envs=None, dual_defense=False):
    """
    Generates configs for finetuning a zoo model either against an adversary, or jointly against
    an adversary and zoo agent. An odd thing about this setup is that it requires an adversary
    path to exist in `victim_path` &  `victim_type` since the adversary is the one being held
    constant for the zoo agent to train on. Another thing to note is that for dual training (zoo &
    adversary jointly), we pass lists for `victim_paths` and `victim_types` (plural)
    :param envs: A list of envs; if set to None, uses all BANSAL_GOOD_ENVS
    :param dual_defense: A boolean set to true if we're generating configs for an
    adversary + zoo joint training finetuning run
    :return:
    """
    if envs is None:
        envs = BANSAL_GOOD_ENVS
    configs = []
    adversary_paths = _get_adversary_paths()
    for env in envs:
        original_victim_index = VICTIM_INDEX[env]
        num_zoo = gym_compete.num_zoo_policies(env)
        for original_victim in range(1, num_zoo+1):
            original_victim = str(original_victim)
            load_policy = {'type': 'zoo', 'path': original_victim}

            adversary = adversary_paths.get(env,
                                            {}).get(str(original_victim_index),
                                                    {}).get(original_victim)
            # If adversary paths are not absolute paths, assume they're relative to
            # MULTI_TRAIN_LOCATION, which is derived from the env variable DATA_LOC
            if not os.path.isabs(adversary):
                adversary = os.path.join(MULTI_TRAIN_LOCATION, adversary)

            if dual_defense:
                # If we're training both best adversary and zoo, experiment with different
                # zoo agents to slot into this role
                for finetuning_zoo in range(1, num_zoo+1):
                    finetuning_zoo = str(finetuning_zoo)
                    victim_paths = [adversary, finetuning_zoo]
                    victim_types = ["ppo2", "zoo"]
                    configs.append((env, victim_paths, victim_types,
                                    1-original_victim_index, load_policy))
            else:
                configs.append((env, adversary, "ppo2", 1-original_victim_index, load_policy))
    return configs


FINETUNE_PATHS_TYPES = "env_name:victim_path:victim_type:victim_index:load_policy"
FINETUNE_PATHS_TYPES_DUAL = "env_name:victim_paths:victim_types:victim_index:load_policy"


def _finetuning_defense(train, dual_defense=False, envs=None):
    _sparse_reward(train)
    # A hack to make it so  you can in theory fine tune LSTMs
    train['num_env'] = 16
    train['normalize_observations'] = False
    ray_config = {}
    paths_and_types = tune.grid_search(_finetune_configs(envs=envs, dual_defense=dual_defense))
    if dual_defense:
        ray_config[FINETUNE_PATHS_TYPES_DUAL] = paths_and_types
    else:
        ray_config[FINETUNE_PATHS_TYPES] = paths_and_types

    return ray_config


def _hyper_finetune_defense(train, dual_defense=False, envs=None, num_samples=20):
    """
       Creates a spec for conducting a hyperparameter search for
       finetuning a zoo agent against an adversary
       """
    ray_config = _finetuning_defense(train, dual_defense, envs)
    train['total_timesteps'] = int(10e6)
    ray_config.update(HYPERPARAM_SEARCH_VALUES)
    spec = {
        'config': ray_config,
        'num_samples': num_samples,
    }
    return spec


def _finetune_defense_long(train, dual_defense=False):
    """
    Creates a spec for conducting a multi-seed long finetuning run against
    either an adversary or an (adversary, zoo) combined setup across environments
    """
    ray_config = _finetuning_defense(train, dual_defense)
    train['total_timesteps'] = int(20e6)
    # "Victim" here is the adversary
    ray_config['seed'] = tune.grid_search(range(5))
    spec = {
        "config": ray_config
    }
    return spec


# ### RETRAINING ADVERSARY AGAINST ADVERSARIALLY-FINETUNED VICTIM ### #

def _train_against_finetuned_configs(finetune_run, envs=None, from_scratch=True):
    """
    Generates configs for training an adversary against an adversarially-finetuned zoo agent.

    :param finetune_run: An experiment name (or <experiment_name/experiment_timestamp>)
    representing to finetuned zoo agent you'd like to train against. This code assumes that
    highest_win_rate.py has been run, and takes the best-performing finetuned agent for each
    (env, zoo_id) combination.
    :param envs: A list of envs; if set to None, uses all BANSAL_GOOD_ENVS
    :param from_scratch: If True, trains an adversary from random initialization; if False,
    finetunes an adversary starting with the already-existing adversary for that (env, zoo_id)
    :return:
    """

    if envs is None:
        envs = BANSAL_GOOD_ENVS
    configs = []
    finetuned_paths = _get_path_from_exp_name(finetune_run)
    adversary_paths = _get_adversary_paths()
    for env in envs:
        victim_index = VICTIM_INDEX[env]
        finetuned_victim_index = 1 - victim_index
        num_zoo = gym_compete.num_zoo_policies(env)
        for original_victim in range(1, num_zoo + 1):
            original_victim = str(original_victim)
            finetuned_victim = finetuned_paths.get(env,
                                                   {}).get(str(finetuned_victim_index),
                                                           {}).get(original_victim, {})
            if from_scratch:
                load_policy = {'type': 'ppo2', 'path': None}

            else:
                adversary = adversary_paths.get(env,
                                                {}).get(str(victim_index),
                                                        {}).get(original_victim, {})
                load_policy = {'type': 'ppo2', 'path': adversary}

            configs.append((env, finetuned_victim, victim_index, load_policy))

    return configs


TRAIN_AGAINST_FINETUNED_PATHS = "env_name:victim_path:victim_index:load_policy"


def _train_against_finetuned(train, finetune_run, from_scratch=True):
    _sparse_reward(train)
    ray_config = {TRAIN_AGAINST_FINETUNED_PATHS:
                  tune.grid_search(
                                _train_against_finetuned_configs(finetune_run=finetune_run,
                                                                 from_scratch=from_scratch))}
    # All victims are new-style policies because we finetuned them
    train['victim_type'] = "ppo2"
    return ray_config


def _hyper_train_adversary_against_finetuned(train, finetune_run, from_scratch=True):
    """
    Creates a spec for doing a hyperparameter search across environments of retraining an adversary
    against a policy that has been defensively finetuned
    """
    ray_config = _train_against_finetuned(train, finetune_run, from_scratch)
    train['total_timesteps'] = int(10e6)
    # Victim is back to being the finetuned victim again
    ray_config.update(HYPERPARAM_SEARCH_VALUES)
    spec = {
        'config': ray_config,
        'num_samples': 2,
    }
    return spec


def _train_adversary_against_finetuned_long(train, finetune_run, from_scratch=True):
    """
        Creates a spec for doing a long run across seeds of retraining an adversary against a
        finetuned zoo agent
        """
    ray_config = _train_against_finetuned(train, finetune_run, from_scratch)
    train['total_timesteps'] = int(20e6)
    ray_config['seed'] = tune.grid_search(range(5))
    spec = {
        'config': ray_config,
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
            },
            'num_samples': 100,
        }
        spec['config'].update(HYPERPARAM_SEARCH_VALUES)
        # This isn't present in default HYPERPARAM_SEARCH_VALUES because trying to vary it for LSTM
        # models causes problems
        spec['config']['rl_args']['minibatches'] = tune.sample_from(
            lambda spec: 2 ** (np.random.randint(0, 7)))

        exp_name = 'hyper'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    ###############################################################################################
    # BEGIN CONFIGS FOR DEFENSE EXPERIMENTS
    ###############################################################################################

    # HYPERPARAMETER TUNING: FINETUNE ZOO

    @multi_train_ex.named_config
    def hyper_finetune_defense_mlp(train):
        """Hyperparameter search for finetuning defense against only the adversary
        MLP (YSNP) only"""
        train = dict(train)
        spec = _hyper_finetune_defense(train, dual_defense=False,
                                       envs=['multicomp/YouShallNotPassHumans-v0'],
                                       num_samples=100)
        exp_name = 'hyper_finetune_defense_mlp'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def hyper_finetune_defense(train):
        """Hyperparameter search for finetuning defense against only the adversary"""
        train = dict(train)
        spec = _hyper_finetune_defense(train, dual_defense=False)
        exp_name = 'hyper_finetune_defense'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def hyper_finetune_dual_defense_mlp(train):
        """Hyperparameter search for finetuning defense against the adversary and a zoo agent
        MLP (YSNP) only """
        train = dict(train)
        spec = _hyper_finetune_defense(train, dual_defense=True,
                                       envs=['multicomp/YouShallNotPassHumans-v0'],
                                       num_samples=100
                                       )
        exp_name = 'hyper_finetune_dual_defense_mlp'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def hyper_finetune_dual_defense(train):
        """Hyperparameter search for finetuning defense against the adversary and a zoo agent"""
        train = dict(train)
        spec = _hyper_finetune_defense(train, dual_defense=True)
        exp_name = 'hyper_finetune_dual_defense'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    # BEST-HYPERPARAM LONG RUN: FINETUNE

    @multi_train_ex.named_config
    def finetune_adv_defense_long_run(train):
        """ Longer run of finetuning across seeds for training against just an adversary
            with hyperparameters generated by hyperparameter search
        """
        train = dict(train)
        spec = _finetune_defense_long(train, dual_defense=False)
        train['learning_rate'] = .00005
        train['batch_size'] = 2048
        exp_name = "finetune_adv_defense_long_run"
        _ = locals()
        del _

    @multi_train_ex.named_config
    def finetune_dual_defense_long_run(train):
        """ Longer run of finetuning across seeds for training against both an adversary and
        zoo agent with hyperparameters generated by hyperparameter search
        """
        train = dict(train)
        spec = _finetune_defense_long(train, dual_defense=True)
        train['learning_rate'] = .000025
        train['batch_size'] = 4096
        exp_name = "finetune_dual_defense_long_run"
        _ = locals()
        del _

    # HYPERPARAMETER TUNING: RETRAIN ADVERSARY

    @multi_train_ex.named_config
    def hyper_against_adv_finetuned_from_scratch(train):
        """Hyperparameter search for training adversary from scratch against best policy from
        hyper_finetune_defense"""
        train = dict(train)
        spec = _hyper_train_adversary_against_finetuned(train,
                                                        finetune_run="hyper_finetune_defense",
                                                        from_scratch=True)
        exp_name = 'hyper_against_adv_finetuned_from_scratch'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def hyper_against_dual_finetuned_from_scratch(train):
        """Hyperparameter search for training adversary from scratch against best policy from
        hyper_finetune_dual_defense """
        train = dict(train)
        spec = _hyper_train_adversary_against_finetuned(train,
                                                        finetune_run="hyper_finetune_dual_defense",
                                                        from_scratch=True)
        exp_name = 'hyper_against_dual_finetuned_from_scratch'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def hyper_against_adv_finetuned_from_existing(train):
        """Hyperparameter search for finetuning adversary from existing adversary against
        best policy from hyper_finetune_defense"""
        train = dict(train)
        spec = _hyper_train_adversary_against_finetuned(train,
                                                        finetune_run="hyper_finetune_defense",
                                                        from_scratch=False)
        exp_name = 'hyper_against_adv_finetuned_from_existing'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def hyper_against_dual_finetuned_from_existing(train):
        """Hyperparameter search for finetuning adversary from existing adversary against
                best policy from hyper_finetune_dual_defense"""
        train = dict(train)
        spec = _hyper_train_adversary_against_finetuned(train,
                                                        finetune_run="hyper_finetune_dual_defense",
                                                        from_scratch=False)
        exp_name = 'hyper_against_dual_finetuned_from_existing'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    # BEST-HYPERPARAM LONG RUN: RETRAIN ADVERSARY
    @multi_train_ex.named_config
    def train_against_finetuned_adv(train):
        """Longer run of an adversary retraining run against the current best finetuned
        zoo agent from the long run of finetuning against just an adversary
        """
        train = dict(train)
        train['learning_rate'] = 8e-4
        train['batch_size'] = 2048
        spec = _train_adversary_against_finetuned_long(train,
                                                       finetune_run="finetune_adv_"
                                                                    "defense_long_run",
                                                       from_scratch=True)
        exp_name = 'train_against_finetuned_adv'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def train_against_finetuned_dual(train):
        """Longer run of an adversary retraining run against the current best finetuned
        zoo agent from the long run of finetuning against an adversary and zoo agent jointly
        """
        train = dict(train)
        train['learning_rate'] = 2.2e-4
        train['batch_size'] = 2048
        spec = _train_adversary_against_finetuned_long(train,
                                                       finetune_run="finetune_dual_"
                                                                    "defense_long_run",
                                                       from_scratch=True)
        exp_name = "train_against_finetuned_adv"
        _ = locals()  # quieten flake8 unused variable warning
        del _

    ###############################################################################################
    # END CONFIGS FOR DEFENSE EXPERIMENTS
    ###############################################################################################

    @multi_train_ex.named_config
    def paper(train):
        """Final experiments for paper. Like best_guess but more seeds & timesteps."""
        train = dict(train)
        _sparse_reward(train)
        _best_guess_train(train)
        train['total_timesteps'] = int(20e6)
        spec = _best_guess_spec()
        spec['config']['seed'] = tune.grid_search(range(5))
        exp_name = 'paper'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def paper_sumohumans_only(train):
        """Final experiments for paper. Like best_guess but more seeds & timesteps."""
        train = dict(train)
        _sparse_reward(train)
        _best_guess_train(train)
        train['total_timesteps'] = int(20e6)
        spec = _best_guess_spec(envs=['multicomp/SumoHumans-v0'])
        spec['config']['seed'] = tune.grid_search(range(5))
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
