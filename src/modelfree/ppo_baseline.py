"""Uses PPO to train an attack policy against a fixed victim policy."""

import json
import os
import os.path as osp

from gym.spaces import Box
from sacred import Experiment
from sacred.observers import FileStorageObserver
from stable_baselines import PPO1, PPO2, SAC, logger
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from aprl.envs.multi_agent import (CurryVecEnv, FlattenSingletonVecEnv, MergeAgentVecEnv,
                                   VecMultiWrapper, make_dummy_vec_multi_env,
                                   make_subproc_vec_multi_env)
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
          batch_size, load_path, learning_rate, rl_args, model_type, debug,
          log_callbacks=None, save_callbacks=None):
    model_kwargs = dict(
        env=env,
        verbose=1 if not debug else 2,
        tensorboard_log=out_dir
    )
    ppo2_kwargs = dict(
        n_steps=batch_size // num_env,
        learning_rate=learning_rate
    )

    # these should be run with mpirun -np 16, batch_size=2048
    ppo1_kwargs = dict(
        timesteps_per_actorbatch=batch_size // num_env,
        clip_param=0.1,
        entcoeff=0.0,
        optim_epochs=10,
        optim_stepsize=1e-4,
        optim_batchsize=64,
        schedule='constant'
    )
    sac_kwargs = dict(
        buffer_size=100000,
        batch_size=batch_size // num_env,  # should be 256
        learning_rate=learning_rate
    )

    model_dict = {
        'ppo1': [PPO1, ppo1_kwargs, 'iters_so_far', 1],
        'ppo2': [PPO2, ppo2_kwargs, 'update', 1],
        'sac': [SAC, sac_kwargs, 'n_updates', 500]
    }
    selected_model, selected_model_kwargs, update_str, log_interval = model_dict[model_type]
    model_kwargs.update(selected_model_kwargs)

    if load_path is not None:
        # SOMEDAY: Counterintuitively this will inherit any extra arguments saved in the policy
        model = selected_model.load(load_path, **model_kwargs)
    else:
        model = selected_model(policy=policy, **model_kwargs)

    def save(root_dir):
        os.makedirs(root_dir, exist_ok=True)
        model_path = osp.join(root_dir, 'model.pkl')
        model.save(model_path)
        if save_callbacks is not None:
            for f in save_callbacks:
                f(root_dir)

    def callback(locals, globals):
        update = locals[update_str]
        if update % (log_interval * 500) == 0:
            checkpoint_dir = osp.join(out_dir, 'checkpoint', f'{update:05}')
            save(checkpoint_dir)

        if log_callbacks is not None and update % log_interval == 0:
            for f in log_callbacks:
                f(locals, globals)

    model.learn(total_timesteps=total_timesteps, log_interval=log_interval,
                seed=_seed, callback=callback)
    final_path = osp.join(out_dir, 'final_model.pkl')
    save(final_path)
    model.sess.close()
    return final_path


@ppo_baseline_ex.named_config
def human_default():
    env_name = "multicomp/SumoHumansAutoContact-v0"
    total_timesteps = int(1e9)
    model_type = 'ppo1'
    num_env = 1
    _ = locals()
    del _


@ppo_baseline_ex.named_config
def humanoid():
    env_name = 'Humanoid-v1'
    victim_index = 1
    victim_type = None
    rew_shape_params = 'default'
    root_dir = 'data/hwalk'
    total_timesteps = int(1e9)
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
    adv_noise_agent_val = None      # epsilon-ball noise policy added to existing zoo policy
    model_type = 'ppo2'             # pick which RL algorithm to use between ppo1, ppo2, sac
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
                 victim_noise, victim_noise_params, adv_noise_agent_val, debug):
    out_dir, logger = setup_logger(root_dir, exp_name)
    scheduler = Scheduler(annealer_dict={'lr': ConstantAnnealer(learning_rate)})
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

        # If we are actually training an epsilon-ball noise agent on top of a zoo agent
        if adv_noise_agent_val is not None:
            base_policy = load_policy(policy_path=victim_path, policy_type=victim_type,
                                      env=multi_env, env_name=env_name, index=agent_idx)
            act_space_shape = multi_env.action_space.spaces[0].shape
            adv_noise_action_space = Box(-adv_noise_agent_val, adv_noise_agent_val, act_space_shape)
            multi_env = MergeAgentVecEnv(venv=multi_env, policy=base_policy,
                                         replace_action_space=adv_noise_action_space,
                                         merge_agent_idx=1-victim_index)

        # Load the victim and then wrap it if appropriate.
        victim = load_policy(policy_path=victim_path, policy_type=victim_type, env=multi_env,
                             env_name=env_name, index=victim_index)
        if victim_noise:
            victim = apply_victim_wrapper(victim=victim, noise_params=victim_noise_params,
                                          scheduler=scheduler, logger=logger)
            log_callbacks.append(lambda locals, globals: victim.log_callback())

        # Curry the victim
        multi_env = EmbedVictimWrapper(multi_env=multi_env, victim=victim,
                                       victim_index=victim_index)
    single_env = FlattenSingletonVecEnv(multi_env)

    if rew_shape:
        rew_shape_env = apply_env_wrapper(single_env=single_env, shaping_params=rew_shape_params,
                                          agent_idx=agent_idx, logger=logger, scheduler=scheduler)
        log_callbacks.append(lambda locals, globals: rew_shape_env.log_callback())
        single_env = rew_shape_env

    for anneal_type in ['noise', 'rew_shape']:
        if scheduler.is_conditional(anneal_type):
            scheduler.set_annealer_shaping_env(anneal_type, single_env)

    if normalize:
        vec_normalize = VecNormalize(single_env)
        save_callbacks.append(lambda root_dir: vec_normalize.save_running_average(root_dir))
        single_env = vec_normalize

    res = train(env=single_env, out_dir=out_dir, learning_rate=scheduler.get_func('lr'),
                log_callbacks=log_callbacks, save_callbacks=save_callbacks)
    single_env.close()

    return res
