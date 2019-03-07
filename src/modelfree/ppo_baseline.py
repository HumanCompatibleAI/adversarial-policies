"""Uses PPO to train an attack policy against a fixed victim policy."""

import json
import os
import os.path as osp

from sacred import Experiment
from sacred.observers import FileStorageObserver
from stable_baselines import PPO2
from stable_baselines.common.vec_env.vec_normalize import VecNormalize

from aprl.envs.multi_agent import (CurryVecEnv, FlattenSingletonVecEnv, VecMultiWrapper,
                                   make_dummy_vec_multi_env, make_subproc_vec_multi_env)
from modelfree.gym_compete_conversion import GameOutcomeMonitor, GymCompeteToOurs
from modelfree.logger import setup_logger
from modelfree.policy_loader import load_policy
from modelfree.scheduling import ConstantAnnealer, Scheduler
from modelfree.shaping_wrappers import apply_env_wrapper, apply_victim_wrapper
from modelfree.utils import make_env

ppo_baseline_ex = Experiment("ppo_baseline")
ppo_baseline_ex.observers.append(FileStorageObserver.create("data/sacred"))


class EmbedVictimWrapper(VecMultiWrapper):
    def __init__(self, multi_env, victim, victim_index):
        self.victim = victim
        curried_env = CurryVecEnv(multi_env, self.victim, agent_idx=victim_index)

        super().__init__(curried_env)

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()

    def close(self):
        self.victim.sess.close()
        super().close()


@ppo_baseline_ex.capture
def train(_seed, env, out_dir, total_timesteps, num_env, policy,
          batch_size, load_path, learning_rate, rl_args, debug,
          log_callbacks=None, save_callbacks=None):
    kwargs = dict(env=env,
                  n_steps=batch_size // num_env,
                  verbose=1 if not debug else 2,
                  learning_rate=learning_rate,
                  **rl_args)
    if load_path is not None:
        # SOMEDAY: Counterintuitively this inherits any extra arguments saved in the policy
        model = PPO2.load(load_path, **kwargs)
    else:
        model = PPO2(policy=policy, **kwargs)

    def save(root_dir):
        os.makedirs(root_dir, exist_ok=True)
        model_path = osp.join(root_dir, 'model.pkl')
        model.save(model_path)
        if save_callbacks is not None:
            for f in save_callbacks:
                f(root_dir)

    def callback(locals, globals):
        update = locals['update']
        checkpoint_dir = osp.join(out_dir, 'checkpoint', f'{update:05}')
        save(checkpoint_dir)

        if log_callbacks is not None:
            for f in log_callbacks:
                f(locals, globals)

    model.learn(total_timesteps=total_timesteps, log_interval=1,
                seed=_seed, callback=callback)
    final_path = osp.join(out_dir, 'final_model')
    save(final_path)
    model.sess.close()
    return final_path


@ppo_baseline_ex.named_config
def human_default():
    env = "multicomp/SumoHumans-v0"
    total_timesteps = int(1e8)
    num_env = 32
    batch_size = 8192
    _ = locals()
    del _


@ppo_baseline_ex.config
def default_ppo_config():
    env_name = "multicomp/SumoAnts-v0"   # Gym environment ID
    victim_type = "zoo"             # type supported by policy_loader.py
    victim_path = "1"               # path or other unique identifier
    victim_index = 0                # which agent the victim is (we default to other agent)
    num_env = 8                     # number of environments to run in parallel
    root_dir = "data/baselines"     # root of directory to store baselines log
    exp_name = "Dummy Exp Name"     # name of experiment
    total_timesteps = 4096          # total number of timesteps to train for
    policy = "MlpPolicy"            # policy network type
    batch_size = 2048               # batch size
    learning_rate = 3e-4            # learning rate
    normalize = True                # normalize environment observations and reward
    rl_args = {}                    # extra RL algorithm arguments
    load_path = None                # path to load initial policy from
    debug = False                   # debug mode; may run more slowly
    seed = 0
    _ = locals()  # quieten flake8 unused variable warning
    del _


DEFAULT_CONFIGS = {
    'multicomp/SumoHumans-v0': 'SumoHumans.json',
    'multicomp/SumoHumansAutoContact-v0': 'SumoHumans.json'
}


def load_default(env_name, config_dir):
    path_stem = os.path.join('experiments', config_dir)
    default_config = DEFAULT_CONFIGS.get(env_name, 'default.json')
    fname = os.path.join(path_stem, default_config)
    with open(fname) as f:
        return json.load(f)


@ppo_baseline_ex.config
def rew_shaping(env_name):
    rew_shape = False  # enable reward shaping
    victim_noise = False  # enable adding noise to victim
    rew_shape_params = load_default(env_name, 'rew_configs')  # parameters for reward shaping
    victim_noise_params = load_default(env_name, 'noise_configs')  # parameters for victim noise
    _ = locals()  # quieten flake8 unused variable warning
    del _


@ppo_baseline_ex.automain
def ppo_baseline(_run, env_name, victim_path, victim_type, victim_index, root_dir, exp_name,
                 learning_rate, normalize, num_env, seed, rew_shape, rew_shape_params,
                 victim_noise, victim_noise_params, debug):
    out_dir, logger = setup_logger(root_dir, exp_name)
    scheduler = Scheduler(func_dict={'lr': ConstantAnnealer(learning_rate).get_value})
    log_callbacks, save_callbacks = [], []
    pre_wrapper = GymCompeteToOurs if env_name.startswith('multicomp/') else None

    def env_fn(i):
        return make_env(env_name, seed, i, out_dir, pre_wrapper=pre_wrapper)

    make_vec_env = make_subproc_vec_multi_env if not debug else make_dummy_vec_multi_env
    multi_env = make_vec_env([lambda: env_fn(i) for i in range(num_env)])

    if env_name.startswith('multicomp/'):
        game_outcome = GameOutcomeMonitor(multi_env, logger)
        # Need game_outcome as separate variable as Python closures bind late
        log_callbacks.append(lambda locals, globals: game_outcome.log_callback())
        multi_env = game_outcome

    if victim_type == 'none':
        if multi_env.num_agents > 1:
            raise ValueError("Victim needed for multi-agent environments")
        agent_idx = 0
    else:
        assert multi_env.num_agents == 2
        agent_idx = 1 - victim_index

        # Load the victim and then wrap it if appropriate.
        victim = load_policy(policy_path=victim_path, policy_type=victim_type, env=multi_env,
                             env_name=env_name, index=victim_index)
        if victim_noise:
            victim = apply_victim_wrapper(victim=victim, noise_params=victim_noise_params,
                                          scheduler=scheduler)

        # Curry the victim
        multi_env = EmbedVictimWrapper(multi_env=multi_env, victim=victim,
                                       victim_index=victim_index)
    single_env = FlattenSingletonVecEnv(multi_env)

    if rew_shape:
        rew_shape_env = apply_env_wrapper(single_env=single_env, shaping_params=rew_shape_params,
                                          agent_idx=agent_idx, logger=logger, scheduler=scheduler)
        log_callbacks.append(lambda locals, globals: rew_shape_env.log_callback())
        single_env = rew_shape_env

    if normalize:
        vec_normalize = VecNormalize(single_env)
        save_callbacks.append(lambda root_dir: vec_normalize.save_running_average(root_dir))
        single_env = vec_normalize

    res = train(env=single_env, out_dir=out_dir, learning_rate=scheduler.get_func('lr'),
                log_callbacks=log_callbacks, save_callbacks=save_callbacks)
    single_env.close()

    return res
