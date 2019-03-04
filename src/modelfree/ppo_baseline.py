"""Uses PPO to train an attack policy against a fixed victim policy."""

import datetime
import os
import os.path as osp

from sacred import Experiment
from sacred.observers import FileStorageObserver
from stable_baselines import PPO1, PPO2, SAC, logger
from stable_baselines.common.vec_env.vec_normalize import VecEnvWrapper

from aprl.envs.multi_agent import CurryVecEnv, FlattenSingletonVecEnv, make_subproc_vec_multi_env
from modelfree.gym_compete_conversion import GameOutcomeMonitor, GymCompeteToOurs
from modelfree.policy_loader import load_policy
from modelfree.scheduling import DEFAULT_ANNEALERS, Scheduler
from modelfree.shaping_wrappers import apply_env_wrapper, apply_victim_wrapper
from modelfree.utils import make_env

ppo_baseline_ex = Experiment("ppo_baseline")
ppo_baseline_ex.observers.append(FileStorageObserver.create("data/sacred"))


class EmbedVictimWrapper(VecEnvWrapper):
    def __init__(self, multi_env, victim, victim_index):
        self.victim = victim
        curried_env = CurryVecEnv(multi_env, self.victim, agent_idx=victim_index)
        single_env = FlattenSingletonVecEnv(curried_env)

        super().__init__(single_env)

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()

    def close(self):
        self.victim.sess.close()
        super().close()


@ppo_baseline_ex.capture
def train(_seed, env, out_dir, total_timesteps, num_env, policy,
          batch_size, load_path, learning_rate, model_type, callbacks=None):
    ppo2_kwargs = dict(
        env=env,
        n_steps=batch_size // num_env,
        verbose=1,
        tensorboard_log=out_dir,
        learning_rate=learning_rate)

    ppo1_kwargs = dict(
        env=env,
        timesteps_per_actorbatch=batch_size // num_env,
        verbose=1,
        tensorboard_log=out_dir,
        entcoeff=0.0,
        optim_epochs=10,
        optim_stepsize=3e-4,
        optim_batchsize=64,
    )
    sac_kwargs = dict(
        env=env,
        buffer_size=100000,
        batch_size=batch_size // num_env,  # should be 256
        verbose=1,
        tensorboard_log=out_dir,
        learning_rate=learning_rate
    )

    model_dict = {
        'ppo1': [PPO1, ppo1_kwargs, 'iters_so_far', 1],
        'ppo2': [PPO2, ppo2_kwargs, 'update', 1],
        'sac': [SAC, sac_kwargs, 'n_updates', 20]
    }
    selected_model, model_kwargs, update_str, log_interval = model_dict[model_type]
    if load_path is not None:
        # SOMEDAY: Counterintuitively this will inherit any extra arguments saved in the policy
        model = selected_model.load(load_path, **model_kwargs)
    else:
        model = selected_model(policy=policy, **model_kwargs)

    def checkpoint(locals, globals):
        update = locals[update_str]
        if update % (log_interval * 500) == 0:
            checkpoint_dir = osp.join(out_dir, 'checkpoint')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = osp.join(checkpoint_dir, f'{update:05}')
            model.save(checkpoint_path)

        if callbacks is not None and update % log_interval == 0:
            for f in callbacks:
                f(locals, globals)
    model.learn(total_timesteps=total_timesteps, log_interval=log_interval,
                seed=_seed, callback=checkpoint)

    model_path = osp.join(out_dir, 'final_model.pkl')
    model.save(model_path)
    model.sess.close()
    return model_path


ISO_TIMESTAMP = "%Y%m%d_%H%M%S"


def setup_logger(out_dir="results", exp_name="test"):
    timestamp = datetime.datetime.now().strftime(ISO_TIMESTAMP)
    out_dir = osp.join(out_dir, '{}-{}'.format(timestamp, exp_name))
    os.makedirs(out_dir, exist_ok=True)
    logger.configure(folder=osp.join(out_dir, 'mon'),
                     format_strs=['tensorboard', 'stdout'])
    return out_dir


@ppo_baseline_ex.named_config
def human_default():
    env = "multicomp/SumoHumans-v0"
    total_timesteps = int(1e8)
    num_env = 32
    batch_size = 8192
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
    seed = 0
    load_path = None                # path to load initial policy from
    rew_shape_params = None         # path to file. 'default' uses default settings for env_name
    victim_noise_params = None      # path to file. 'default' uses default settings for env_name
    model_type = 'ppo2'
    # then default settings for that environment will be used.
    _ = locals()  # quieten flake8 unused variable warning
    del _


@ppo_baseline_ex.automain
def ppo_baseline(_run, env_name, victim_path, victim_type, victim_index, root_dir, exp_name,
                 num_env, seed, rew_shape_params, victim_noise_params, batch_size):
    out_dir = setup_logger(root_dir, exp_name)
    scheduler = Scheduler(func_dict={'lr': DEFAULT_ANNEALERS['default_lr'].get_value})
    callbacks = []
    pre_wrapper = GymCompeteToOurs if 'multicomp' in env_name else None

    def env_fn(i):
        return make_env(env_name, seed, i, root_dir, pre_wrapper=pre_wrapper)

    multi_env = make_subproc_vec_multi_env([lambda: env_fn(i) for i in range(num_env)])
    multi_env = GameOutcomeMonitor(multi_env, logger)
    assert bool(victim_type is None) == bool(multi_env.num_agents == 1)
    callbacks.append(lambda locals, globals: multi_env.log_callback())

    # Get the correct victim and then wrap it accordingly.
    if multi_env.num_agents > 1:
        victim = load_policy(policy_path=victim_path, policy_type=victim_type, env=multi_env,
                             env_name=env_name, index=victim_index)
        if victim_noise_params is not None:
            victim = apply_victim_wrapper(victim=victim, victim_noise_params=victim_noise_params,
                                          env_name=env_name, scheduler=scheduler, logger=logger)
            callbacks.append(lambda locals, globals: victim.log_callback())

        # Get the correct environment and then wrap it accordingly.
        single_env = EmbedVictimWrapper(multi_env=multi_env, victim=victim,
                                        victim_index=victim_index)
    else:
        single_env = FlattenSingletonVecEnv(multi_env)

    if rew_shape_params is not None:
        single_env = apply_env_wrapper(single_env=single_env, rew_shape_params=rew_shape_params,
                                       env_name=env_name, agent_idx=1 - victim_index,
                                       logger=logger, batch_size=batch_size, scheduler=scheduler)
        callbacks.append(lambda locals, globals: single_env.log_callback())

    res = train(env=single_env, out_dir=out_dir, learning_rate=scheduler.get_func('lr'),
                callbacks=callbacks)
    single_env.close()

    return res
