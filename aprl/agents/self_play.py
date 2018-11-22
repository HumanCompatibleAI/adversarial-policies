from collections import deque
import functools
import time
import os
import os.path as osp

import numpy as np
from baselines import logger
from baselines.bench import Monitor
from baselines.common.runners import AbstractEnvRunner
from baselines.common import explained_variance, set_global_seeds, tf_util
from baselines.common.policies import build_policy
import tensorflow as tf
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from aprl.envs import MultiToSingleObsVec
from aprl.utils import getattr_unwrapped

class MultiMonitor(Monitor):
    def __init__(self, env, filename, allow_early_resets=False,
                 reset_keywords=(), info_keywords=()):
        num_agents = getattr_unwrapped(env, 'num_agents')
        extra_rks = tuple("r{:d}".format(i) for i in range(num_agents))
        super().__init__(env, filename, allow_early_resets=allow_early_resets,
                         reset_keywords=reset_keywords,
                         info_keywords=extra_rks + info_keywords)
        self.info_keywords = info_keywords

    def update(self, ob, rew, done, info):
        # Same as Monitor.update, except handle rewards being vector-valued
        self.rewards.append(rew)
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eprew = list(map(lambda x: round(x, 6), eprew))
            joint_eprew = np.mean(eprew)
            eplen = len(self.rewards)
            epinfo = {"r": joint_eprew,
                      "l": eplen,
                      "t": round(time.time() - self.tstart, 6)}
            for i, rew in enumerate(eprew):
                epinfo["r{:d}".format(i)] = rew
            for k in self.info_keywords:
                epinfo[k] = info[k]
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            epinfo.update(self.current_reset_info)
            self.results_writer.write_row(epinfo)

            if isinstance(info, dict):
                info['episode'] = epinfo

        self.total_steps += 1

class AbstractMultiEnvRunner(AbstractEnvRunner):
    def __init__(self, *, env, models, nsteps):
        super().__init__(env=env, model=models[0], nsteps=nsteps)
        self.nmodels = len(models)
        self.models = models
        self.states = [model.initial_state for model in models]
        self.dones = np.zeros(self.nenv, dtype=np.bool)

class PPOMultiRunner(AbstractMultiEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, models, nsteps, gamma, lam):
        super().__init__(env=env, models=models, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

    def run(self):
        # Memory buffer values mb_* are indexed [model_id, timestep, env_id].
        # self.obs, self.dones and actions are indexed [env_id, model_id].

        # Here, we init the lists that will contain the experience buffer
        def e():
            return [[] for _ in self.models]
        mb_obs, mb_rewards, mb_actions = e(), e(), e()
        mb_values, mb_dones, mb_neglogpacs = e(), e(), e()
        epinfos = []

        # Gather environment experience for n in range number of steps
        actions = np.zeros((self.nenv,) + self.env.action_space.shape,
                           dtype=self.env.action_space.dtype.name)
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass
            # runs self.obs[:] = env.reset() on init.
            for i, model in enumerate(self.models):
                # TODO: evaluate models in parallel?
                # self.obs, self.dones are [env_id, model_id]
                a, v, self.states[i], nlogp = model.step(self.obs[:, i],
                                                         S=self.states[i],
                                                         M=self.dones)
                mb_obs[i].append(self.obs[:, i].copy())
                actions[:, i] = a
                mb_actions[i].append(a)
                mb_values[i].append(v)
                mb_neglogpacs[i].append(nlogp)
                mb_dones[i].append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones[:], infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            for i in range(self.nmodels):
                mb_rewards[i].append(rewards[:, i])

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=actions.dtype)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        #TODO: check the shape of last_values
        last_values = np.zeros((self.nenv,) + self.env.observation_space.shape,
                               dtype=np.float32)
        for i, model in enumerate(self.models):
            last_values[:, i] = self.model.value(self.obs[:, i],
                                                 S=self.states[i],
                                                 M=self.dones)
        last_values = last_values.swapaxes(0, 1)

        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[:, t+1]
                nextvalues = mb_values[:, t+1]
            delta = mb_rewards[:, t] + self.gamma * nextvalues * nextnonterminal - mb_values[:, t]
            mb_advs[:, t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        res = []
        for i in range(self.nmodels):
            x = (*map(sf01, (mb_obs[i], mb_returns[i], mb_dones[i],
                            mb_actions[i], mb_values[i], mb_neglogpacs[i])),
                 self.states[i])
            res.append(x)
        return res, epinfos

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

#TODO: make less dependent on PPO2 implementation choices
class SelfPlay(object):
    TRAINING_TYPES = ['best']

    def __init__(self, runner_class, population_size, training_type, env):
        self.population_size = int(population_size)
        if training_type not in self.TRAINING_TYPES:
            raise NotImplementedError
        self.training_type = training_type
        self.env = env
        self.nenv = env.num_envs
        self.models = [None for _ in range(population_size)]
        self.runner_class = runner_class

    def rollout(self, nsteps):
        # Select two models to play each other.
        pa, pb = np.random.choice(self.population_size, size=2, replace=False)
        if self.training_type == 'best':
            modela, modelb = self.models[pa], self.models[pb]
        else:
            raise NotImplementedError

        # Generate a rollout
        #TODO: support persistent replay buffers?
        runner = self.runner_class(env=self.env,
                                   models=[modela, modelb],
                                   nsteps=nsteps)
        (traja, trajb), epinfos = runner.run()

        return (pa, modela, traja), (pb, modelb, trajb), epinfos

    def learn(self):
        raise NotImplementedError

def constfn(val):
    def f(_):
        return val
    return f

class PPOSelfPlay(SelfPlay):
    def __init__(self, population_size, training_type, env, network,
                 make_sess=tf_util.make_session, seed=None, nsteps=2048,
                 gamma=0.99, lam=0.95, ent_coef=0.0,  vf_coef=0.5,
                 max_grad_norm=0.5, nminibatches=4, load_paths=None,
                 model_fn=None, **network_kwargs):
        runner = functools.partial(PPOMultiRunner, gamma=gamma, lam=lam)
        super().__init__(runner, population_size, training_type, env)

        set_global_seeds(seed)

        # Get state_space and action_space
        # TODO: these are for both agents
        ob_space = env.observation_space
        ac_space = env.action_space

        # Calculate the batch_size
        self.nsteps = nsteps
        self.nminibatches = nminibatches
        self.nbatch = self.nenv * self.nsteps
        self.nbatch_train = self.nbatch // nminibatches

        self.graphs = [tf.Graph() for _ in range(population_size)]
        self.sess = [make_sess(graph=graph) for graph in self.graphs]
        fake_env = MultiToSingleObsVec(env)
        for i in range(population_size):
            policy = build_policy(fake_env, network, **network_kwargs)

            # Instantiate the model object (that creates act_model and train_model)
            if model_fn is None:
                from baselines.ppo2.model import Model
                model_fn = Model

            #TODO: construct all in one graph & session?
            #This might give performance improvements, e.g. evaluate actions
            #for each agent in one pass. (Although since they're independent,
            #possibly not -- depends how clever TF is at optimizing.)
            #However, it breaks PPO's Model, which uses
            #tf.trainable_variables('ppo2_model') and so does not support
            #multiple variables scopes.
            with self.sess[i].as_default():
                with self.graphs[i].as_default():
                    # Both of these are needed -- making a session default
                    # does not change the default graph.
                    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space,
                                     nbatch_act=self.nenv, nbatch_train=self.nbatch_train,
                                     nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                                     max_grad_norm=max_grad_norm)

            if load_paths is not None:
                model.load(load_paths[i])

            self.models[i] = model

        self.epinfobufs = [deque(maxlen=1000) for _ in range(population_size)]
        #TODO: eval_env?
        # if eval_env is not None:
        #     eval_epinfobuf = deque(maxlen=100)

    def learn(self, total_timesteps, eval_env = None, lr=3e-4, cliprange=0.2,
              log_interval=10, noptepochs=4,
              save_interval=0):
        if isinstance(lr, float):
            lr = constfn(lr)
        else:
            assert callable(lr)
        if isinstance(cliprange, float):
            cliprange = constfn(cliprange)
        else:
            assert callable(cliprange)

        total_timesteps = int(total_timesteps)

        nupdates = total_timesteps // self.nbatch
        # Start total timer
        tfirststart = time.time()
        for update in range(1, nupdates + 1):
            assert self.nbatch % self.nminibatches == 0
            # Start timer
            tstart = time.time()
            frac = 1.0 - (update - 1.0) / nupdates
            # Calculate the learning rate
            lrnow = lr(frac)
            # Calculate the cliprange
            cliprangenow = cliprange(frac)

            # Get minibatch
            traja, trajb, epinfos = self.rollout(self.nsteps)
            for i, (pi, model, traj) in enumerate([traja, trajb]):
                epinfobuf = self.epinfobufs[pi]
                #TODO: parallelize
                obs, returns, masks, actions, values, neglogpacs, states = traj
                # Extract relevant subset of epinfo
                our_epinfos = [{
                        'r': epinfo['r{:d}'.format(i)],
                        'l': epinfo['l'],
                        't': epinfo['t'],
                    }
                    for epinfo in epinfos
                ]
                epinfobuf.extend(our_epinfos)

                # Here what we're going to do is for each minibatch calculate the loss and append it.
                mblossvals = []
                if states is None:  # nonrecurrent version
                    # Index of each element of batch_size
                    # Create the indices array
                    inds = np.arange(self.nbatch)
                    for _ in range(noptepochs):
                        # Randomize the indexes
                        np.random.shuffle(inds)
                        # 0 to batch_size with batch_train_size step
                        for start in range(0, self.nbatch, self.nbatch_train):
                            end = start + self.nbatch_train
                            mbinds = inds[start:end]
                            slices = (arr[mbinds] for arr in (
                            obs, returns, masks, actions, values, neglogpacs))
                            mblossvals.append(model.train(lrnow, cliprangenow, *slices))
                else:  # recurrent version
                    assert self.nenv % self.nminibatches == 0
                    envinds = np.arange(self.nenv)
                    flatinds = np.arange(self.nenv * self.nsteps).reshape(self.nenv, self.nsteps)
                    envsperbatch = self.nbatch_train // self.nsteps
                    for _ in range(noptepochs):
                        np.random.shuffle(envinds)
                        for start in range(0, self.nenv, envsperbatch):
                            end = start + envsperbatch
                            mbenvinds = envinds[start:end]
                            mbflatinds = flatinds[mbenvinds].ravel()
                            slices = (arr[mbflatinds] for arr in (
                            obs, returns, masks, actions, values, neglogpacs))
                            mbstates = states[mbenvinds]
                            mblossvals.append(
                                model.train(lrnow, cliprangenow, *slices, mbstates))

                # Feedforward --> get losses --> update
                lossvals = np.mean(mblossvals, axis=0)
                # End timer
                tnow = time.time()
                # Calculate the fps (frame per second)
                fps = int(self.nbatch / (tnow - tstart))
                if update % log_interval == 0 or update == 1:
                    # Calculates if value function is a good predicator of the returns (ev > 1)
                    # or if it's just worse than predicting nothing (ev =< 0)
                    ev = explained_variance(values, returns)
                    # SOMEDAY: better metrics?
                    # Mean reward is not very meaningful in self-play.
                    logger.logkv("player", pi)
                    logger.logkv("serial_timesteps", update * self.nsteps)
                    logger.logkv("nupdates", update)
                    logger.logkv("total_timesteps", update * self.nbatch)
                    logger.logkv("fps", fps)
                    logger.logkv("explained_variance", float(ev))
                    logger.logkv('eprewmean_rollout',
                                 safemean([epinfo['r'] for epinfo in our_epinfos]))
                    logger.logkv('eplenmean_rollout',
                                 safemean([epinfo['l'] for epinfo in our_epinfos]))
                    logger.logkv('eprewmean_all',
                                 safemean([epinfo['r'] for epinfo in epinfobuf]))
                    logger.logkv('eplenmean_all',
                                 safemean([epinfo['l'] for epinfo in epinfobuf]))
                    logger.logkv('time_elapsed', tnow - tfirststart)
                    for (lossval, lossname) in zip(lossvals, model.loss_names):
                        logger.logkv(lossname, lossval)
                    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
                        logger.dumpkvs()
                if save_interval and (
                        update % save_interval == 0 or update == 1) and logger.get_dir() and (
                        MPI is None or MPI.COMM_WORLD.Get_rank() == 0):
                    checkdir = osp.join(logger.get_dir(), 'checkpoints', '.%2i' % pi)
                    os.makedirs(checkdir, exist_ok=True)
                    savepath = osp.join(checkdir, '%.5i' % update)
                    print('Saving to', savepath)
                    model.save(savepath)
        return self.models
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)