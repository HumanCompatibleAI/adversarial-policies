"""Named configs for modelfree.multi.train."""

import collections
import itertools
import json
import os
import os.path as osp

import numpy as np
from ray import tune

from modelfree.configs.multi.common import BANSAL_ENVS, BANSAL_GOOD_ENVS, get_adversary_paths
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
            'env_name:embed_path': tune.grid_search(_env_victim(envs)),
            'embed_index': tune.sample_from(
                lambda spec: VICTIM_INDEX[spec.config['env_name:embed_path'][0]]
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
            'env_name:embed_path': tune.grid_search(_env_victim(envs)),
            'seed': tune.grid_search([0, 1, 2]),
            'load_policy': {
                'path': tune.sample_from(lambda spec: spec.config['env_name:embed_path'][1]),
            },
            'embed_index': tune.sample_from(
                lambda spec: VICTIM_INDEX[spec.config['env_name:embed_path'][0]]
            ),
        },
    }
    return spec


def _get_path_from_exp_name(exp_name,
                            json_file_path="highest_win_policies_and_rates.json"):
    """Takes in an experiment name and constructs the JSON path containing its best policies."""
    full_json_path = os.path.join(MULTI_TRAIN_LOCATION, exp_name, json_file_path)
    try:
        with open(full_json_path, 'r') as f:
            return json.load(f)['policies']
    except FileNotFoundError:
        raise FileNotFoundError(f"Please run highest_win_rate.py for experiment {exp_name} before"
                                " trying to use it ")


# ### CONFIGS FOR FINETUNING AGAINST ADVERSARY (SINGLE) OR ADVERSARY + ZOO (DUAL) ### #

def _finetune_configs(envs=None, dual_defense=False):
    """Generates configs for finetuning a Zoo model.

    Note in this setup, the adversary is the embedded agent, whereas usually the victim is.
    :param envs: A list of envs; if set to None, uses all BANSAL_GOOD_ENVS
    :param dual_defense: If True, fine-tune against both an adversary and Zoo agent (randomly
        selected per episode); if False, fine-tune against just the adversary.
    """
    if envs is None:
        envs = BANSAL_GOOD_ENVS
    configs = []
    adversary_paths = get_adversary_paths()
    for env in envs:
        original_embed_index = VICTIM_INDEX[env]
        num_zoo = gym_compete.num_zoo_policies(env)
        for original_victim in range(1, num_zoo+1):
            original_victim = str(original_victim)
            load_policy = {'type': 'zoo', 'path': original_victim}

            adversary = (adversary_paths.get(env, {})
                                        .get(str(original_embed_index), {})
                                        .get(original_victim))
            adversary = os.path.join(MULTI_TRAIN_LOCATION, adversary)

            if dual_defense:
                # If training both best adversary and Zoo, try each possible Zoo agent
                for finetuning_zoo in range(1, num_zoo+1):
                    finetuning_zoo = str(finetuning_zoo)
                    embed_paths = [adversary, finetuning_zoo]
                    embed_types = ["ppo2", "zoo"]
                    configs.append((env, embed_paths, embed_types,
                                    1-original_embed_index, load_policy))
            else:
                configs.append((env, [adversary], ["ppo2"], 1-original_embed_index, load_policy))
    return configs


FINETUNE_PATHS_TYPES = "env_name:embed_paths:embed_types:embed_index:load_policy"


def _generic_finetune_defense(train, dual_defense=False, envs=None, exp_suffix=''):
    """Finetuning victim against adversary.

    This is the most generic helper method, used as a base for `_hyper_finetune_defense`
    and `_finetune_defense`.
    """
    _sparse_reward(train)
    train['num_env'] = 16  # TODO(adam): cleaner way of allowing finetuning LSTMs
    train['normalize_observations'] = False
    ray_config = {
        FINETUNE_PATHS_TYPES: tune.grid_search(
            _finetune_configs(envs=envs, dual_defense=dual_defense)
        ),
    }
    dual_name = 'dual' if dual_defense else 'single'
    exp_name = f'finetune_defense_{dual_name}_{exp_suffix}'

    return ray_config, exp_name


def _hyper_finetune_defense(train, num_samples=20, **kwargs):
    """Hyperparameter search for finetuning Zoo agent against adversary."""
    ray_config, exp_name = _generic_finetune_defense(train, **kwargs)
    train['total_timesteps'] = int(10e6)
    ray_config.update(HYPERPARAM_SEARCH_VALUES)
    spec = {
        'config': ray_config,
        'run_kwargs': {
            'num_samples': num_samples,
        }
    }
    exp_name = f'hyper_{exp_name}'
    return spec, exp_name


def _finetune_defense(train, **kwargs):
    """Multi-seed, long (20e6) timestep finetuning against adversary."""
    ray_config, exp_name = _generic_finetune_defense(train, **kwargs)
    ray_config['seed'] = tune.grid_search(list(range(5)))
    spec = {
        'config': ray_config,
    }
    return spec, exp_name


# ### RETRAINING ADVERSARY AGAINST ADVERSARIALLY-FINETUNED VICTIM ### #

def _train_against_finetuned_configs(finetune_run, envs=None, from_scratch=True):
    """Train an adversary against an adversarially-finetuned Zoo agent.

    :param finetune_run: An experiment name (or <experiment_name/experiment_timestamp>)
    representing the finetuned Zoo agent you'd like to train against. This method assumes that
    highest_win_rate.py has been run, and takes the best-performing finetuned agent for each
    (env, zoo_id) combination.
    :param envs: A list of envs; if set to None, uses all BANSAL_GOOD_ENVS
    :param from_scratch: If True, trains an adversary from random initialization; if False,
        finetunes an adversary starting with the already-existing adversary.
    :return:
    """

    if envs is None:
        envs = BANSAL_GOOD_ENVS
    configs = []
    finetuned_paths = _get_path_from_exp_name(finetune_run)
    adversary_paths = get_adversary_paths()
    for env in envs:
        embed_index = VICTIM_INDEX[env]
        finetuned_embed_index = 1 - embed_index
        num_zoo = gym_compete.num_zoo_policies(env)
        for original_victim in range(1, num_zoo + 1):
            original_victim = str(original_victim)
            finetuned_victim = (finetuned_paths.get(env, {})
                                               .get(str(finetuned_embed_index), {})
                                               .get(original_victim, {}))

            if from_scratch:
                load_policy = {'type': 'ppo2', 'path': None}
            else:
                adversary = (adversary_paths.get(env, {})
                                            .get(str(embed_index), {})
                                            .get(original_victim, {}))
                load_policy = {'type': 'ppo2', 'path': adversary}

            configs.append((env, finetuned_victim, embed_index, load_policy))

    return configs


TRAIN_AGAINST_FINETUNED_PATHS = "env_name:embed_path:embed_index:load_policy"


def _generic_train_adv_against_finetuned(train, finetune_run, from_scratch=True):
    """Retrain adversary against defensively finetuned victim.

    This is the most generic helper method, that is used by `_hyper_train_adv_against_finetuned`
    and `_train_adv_against_finetuned`."""
    _sparse_reward(train)
    train['embed_type'] = "ppo2"  # all victims are new-style policies because we finetuned them
    ray_config = {
        TRAIN_AGAINST_FINETUNED_PATHS: tune.grid_search(
            _train_against_finetuned_configs(finetune_run=finetune_run,
                                             from_scratch=from_scratch)
        ),
    }
    from_scratch_name = 'from_scratch' if from_scratch else 'finetune'
    exp_name = f'adv_{from_scratch_name}_against_{finetune_run}'
    return ray_config, exp_name


def _hyper_train_adv_against_finetuned(train, finetune_run, from_scratch=True):
    """Hyperparameter search for retraining an adversary against defensively finetuned victim."""
    ray_config, exp_name = _generic_train_adv_against_finetuned(train, finetune_run, from_scratch)
    train['total_timesteps'] = int(10e6)
    ray_config.update(HYPERPARAM_SEARCH_VALUES)
    spec = {
        'config': ray_config,
        'run_kwargs': {'num_samples': 2}
    }
    exp_name = f'hyper_{exp_name}'
    return spec, exp_name


def _train_adv_against_finetuned(train, finetune_run, from_scratch=True):
    """Multi-seed, long (20e6) retraining of adversary against finetuned Zoo agent."""
    ray_config = _generic_train_adv_against_finetuned(train, finetune_run, from_scratch)
    ray_config['seed'] = tune.grid_search(list(range(5)))
    spec = {
        'config': ray_config,
    }
    return spec


def make_configs(multi_train_ex):

    # ### STANDARD ADVERSARIAL EXPERIMENTS ### #
    # ### Train adversary from scratch against a fixed victim. ### #

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
                'embed_path': tune.sample_from(
                    lambda spec: TARGET_VICTIM[spec.config.env_name]
                ),
            },
            'run_kwargs': {'num_samples': 100}
        }
        spec['config'].update(HYPERPARAM_SEARCH_VALUES)
        # This isn't present in default HYPERPARAM_SEARCH_VALUES because trying to vary it for LSTM
        # models causes problems
        spec['config']['rl_args']['minibatches'] = tune.sample_from(
            lambda spec: 2 ** (np.random.randint(0, 7)))

        exp_name = 'hyper'
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
    def paper(train):
        """Final experiments for paper. Like best_guess but more seeds & timesteps."""
        train = dict(train)
        _sparse_reward(train)
        _best_guess_train(train)
        train['total_timesteps'] = int(20e6)
        spec = _best_guess_spec()
        spec['config']['seed'] = tune.grid_search(list(range(5)))
        exp_name = 'paper'
        _ = locals()  # quieten flake8 unused variable warning
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
                'embed_path': tune.grid_search(['1', '2', '3']),
                'seed': tune.grid_search([0, 1, 2]),
                'rew_shape_params': {
                    'anneal_frac': tune.grid_search([0.0, 0.1]),
                },
            },
        }
        exp_name = 'dec2018rep'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    # ### SPARSE VS DENSE REWARDS ### #

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
        train['embed_path'] = 3  # median difficulty victim (1 is easy, 2 is hard)
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
            'run_kwargs': {'num_samples': 10}
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
                'embed_path': tune.grid_search(['1', '2', '3']),
                'seed': tune.grid_search([10, 20, 30, 40, 50]),
            },
        }
        exp_name = 'kick_and_defend_sparse_vs_dense'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    # ### DIAGNOSTIC EXPERIMENTS ### #

    @multi_train_ex.named_config
    def single_agent_baseline(train):
        """Baseline applying our method to standard single-agent Gym MuJoCo environments.

        Should perform similarly to the results given in PPO paper."""
        train = dict(train)
        _sparse_reward(train)
        train['embed_type'] = 'none'
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
                'embed_path': tune.grid_search(['1', '2', '3']),
                'normalize': tune.grid_search([True, False]),
            },
        }
        exp_name = 'vec_normalize'
        _ = locals()   # quieten flake8 unused variable warning
        del _

    # ### DEFENSE EXPERIMENTS ### #

    # MODIFIERS: Used with all experiments

    @multi_train_ex.named_config
    def defense_dual():
        defense_kwargs = {'dual_defense': True}  # noqa: F841
        hyper_defense_kwargs = {}  # noqa: F841

    @multi_train_ex.named_config
    def defense_single():
        defense_kwargs = {'dual_defense': False}  # noqa: F841
        hyper_defense_kwargs = {}  # noqa: F841

    @multi_train_ex.named_config
    def defense_only_mlp(defense_kwargs, hyper_defense_kwargs):
        defense_kwargs['envs'] = ['multicomp/YouShallNotPassHumans-v0']
        defense_kwargs['exp_suffix'] = 'mlp'
        hyper_defense_kwargs['num_samples'] = 100

    @multi_train_ex.named_config
    def adv_from_scratch():
        adv_retrain_kwargs = {'from_scratch': True}  # noqa: F841

    @multi_train_ex.named_config
    def adv_finetune():
        adv_retrain_kwargs = {'from_scratch': False}  # noqa: F841

    # HYPERPARAMETER TUNING

    @multi_train_ex.named_config
    def hyper_finetune_defense(train, defense_kwargs, hyper_defense_kwargs):
        """Hyperparameter search for finetuning defense.

        You must use this with one of the modifiers `defense_dual` or `defense_single`,
        specified before this named config. You may optionally use `defense_only_mlp`.
        """
        train = dict(train)
        spec, exp_name = _hyper_finetune_defense(train,
                                                 **defense_kwargs,
                                                 **hyper_defense_kwargs)
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.capture
    def squash_sacred_warning(defense_kwargs, hyper_defense_kwargs):
        """Sacred thinks we don't use these arguments, but we do in the above named_config.

        This is because Sacred is only looking at capture functions, not at named_config's.
        """

    @multi_train_ex.named_config
    def hyper_adv_against_hardened(train, defense_kwargs, adv_retrain_kwargs):
        """Hyperparameter search for training adversary against best hardened victim.

        You must specify the same modifiers as used with `hyper_finetune_defense` to locate
        the correct victim.

        You must also specify one of the modifiers `adv_from_scratch` or `adv_finetune`.
        """
        train = dict(train)
        _, finetune_run = _hyper_finetune_defense({}, **defense_kwargs)
        spec, exp_name = _hyper_train_adv_against_finetuned(train,
                                                            **adv_retrain_kwargs,
                                                            finetune_run=finetune_run)
        _ = locals()  # quieten flake8 unused variable warning
        del _

    # BEST-HYPERPARAM LONG RUNS

    @multi_train_ex.named_config
    def finetune_defense(train, defense_kwargs):
        """Finetune victim against adversary and optionally a normal opponent."""
        train = dict(train)
        # Hyperparameter search found similar hyperparameters for dual finetuning as for
        # training an adversary, so reuse _best_guess_train for consistency.
        # TODO(adam): do we want different hyperparameters for single defense?
        _best_guess_train(train)
        train['total_timesteps'] = int(20e6)
        spec, exp_name = _finetune_defense(train, **defense_kwargs)
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_train_ex.named_config
    def adv_against_hardened(train, defense_kwargs, adv_retrain_kwargs):
        """Retrain adversary against the current best finetuned Zoo agent, from finetune_defense.

        You must specify the same modifiers as used with `hyper_finetune_defense` to locate
        the correct victim.

        You must also specify one of the modifiers `adv_from_scratch` or `adv_finetune`.
        """
        train = dict(train)
        _best_guess_train(train)
        train['total_timesteps'] = int(20e6)
        _, finetune_run = _finetune_defense({}, **defense_kwargs)
        spec, exp_name = _train_adv_against_finetuned(train,
                                                      **adv_retrain_kwargs,
                                                      finetune_run=finetune_run)
        _ = locals()  # quieten flake8 unused variable warning
        del _

    # ### FINETUNING (Not as a defense) ### #

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
                'embed_path': tune.sample_from(
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

    # ### ADVERSARIAL NOISE BALL ### #

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
                'embed_path': tune.sample_from(
                    lambda spec: TARGET_VICTIM[spec.config.env_name]
                ),
                'embed_index': tune.sample_from(
                    lambda spec: VICTIM_INDEX[spec.config.env_name]
                ),
                'adv_noise_params': {
                    'noise_val': tune.sample_from(
                        lambda spec: 10 ** (np.random.rand() * -2)
                    ),
                }
            },
            'run_kwargs': {'num_samples': 10}
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
                'env_name:embed_path': tune.grid_search(_env_victim(
                    ['multicomp/SumoHumansAutoContact-v0', 'multicomp/RunToGoalHumans-v0'],
                )),
                'adv_noise_params': {
                    'base_path': tune.sample_from(
                        lambda spec: spec.config['env_name:embed_path'][1],
                    )
                },
                'seed': tune.grid_search([0, 1, 2]),
            },
        }
        exp_name = 'noise_ball'
        _ = locals()  # quieten flake8 unused variable warning
        del _
