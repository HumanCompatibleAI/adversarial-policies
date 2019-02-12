# flake8: noqa
# TODO: fix PEP8 violations inherited from Baselines

from collections.__init__ import deque
import functools
import os
from os import path as osp
import time

from baselines import logger
from baselines.common import explained_variance, set_global_seeds, tf_util
from baselines.common.policies import build_policy
import numpy as np
import tensorflow as tf

from aprl.agents.self_play import AbstractMultiEnvRunner, SelfPlay
from aprl.envs import FakeSingleSpacesVec

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def _sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


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
                # SOMEDAY: evaluate models in parallel?
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
            x = (*map(_sf01, (mb_obs[i], mb_returns[i], mb_dones[i],
                              mb_actions[i], mb_values[i], mb_neglogpacs[i])),
                 self.states[i])
            res.append(x)
        return res, epinfos


def _constfn(val):
    def f(_):
        return val
    return f


def _safemean(xs):
    """Avoid division error when calculate the mean
       (in our case if epinfo is empty returns np.nan, not return an error)"""
    return np.nan if len(xs) == 0 else np.mean(xs)


class PPOSelfPlay(SelfPlay):
    def __init__(self, population_size, training_type, env,
                 network, make_sess=tf_util.make_session, seed=None,
                 nsteps=2048, gamma=0.99, lam=0.95, ent_coef=0.0,  vf_coef=0.5,
                 max_grad_norm=0.5, nminibatches=4, load_paths=None,
                 model_fn=None, **network_kwargs):
        runner = functools.partial(PPOMultiRunner, gamma=gamma, lam=lam)
        super().__init__(population_size, training_type, runner, env)

        set_global_seeds(seed)

        # Get state_space and action_space
        ob_space = env.observation_space
        ac_space = env.action_space

        # Calculate the batch_size
        self.nsteps = nsteps
        self.nminibatches = nminibatches
        self.nbatch = self.nenv * self.nsteps
        self.nbatch_train = self.nbatch // nminibatches

        self.graphs = [tf.Graph() for _ in range(population_size)]
        self.sess = [make_sess(graph=graph) for graph in self.graphs]
        fake_env = FakeSingleSpacesVec(env)
        for i in range(population_size):
            policy = build_policy(fake_env, network, **network_kwargs)

            # Instantiate the model object (that creates act_model and train_model)
            if model_fn is None:
                from baselines.ppo2.model import Model
                model_fn = Model

            # SOMEDAY: construct everything in one graph & session?
            # This might give performance improvements, e.g. evaluate actions
            # for each agent in one pass. (Although since they're independent,
            # possibly not -- depends how clever TF is at optimizing.)
            # However, it breaks PPO's Model, which uses
            # tf.trainable_variables('ppo2_model') and so does not support
            # multiple variables scopes.
            with self.sess[i].as_default():
                with self.graphs[i].as_default():
                    # Both of these are needed -- making a session default
                    # does not change the default graph.
                    model = model_fn(policy=policy, ob_space=ob_space,
                                     ac_space=ac_space, nbatch_act=self.nenv,
                                     nbatch_train=self.nbatch_train,
                                     nsteps=nsteps, ent_coef=ent_coef,
                                     vf_coef=vf_coef,
                                     max_grad_norm=max_grad_norm)

            if load_paths is not None:
                model.load(load_paths[i])

            self.models[i] = model

        self.epinfobufs = [deque(maxlen=1000) for _ in range(population_size)]

    def learn(self, total_timesteps, lr=3e-4, cliprange=0.2,
              log_interval=10, noptepochs=4, save_interval=0):
        if isinstance(lr, float):
            lr = _constfn(lr)
        else:
            assert callable(lr)
        if isinstance(cliprange, float):
            cliprange = _constfn(cliprange)
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
            rollout = self.rollout(self.nsteps)
            # SOMEDAY: parallelize model training
            for pi, model, traj, epinfos in rollout:
                obs, returns, masks, actions, values, neglogpacs, states = traj
                epinfobuf = self.epinfobufs[pi]
                epinfobuf.extend(epinfos)

                # Here what we're going to do is for each minibatch calculate
                # the loss and append it.
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
                            loss = model.train(lrnow, cliprangenow, *slices)
                            mblossvals.append(loss)
                else:  # recurrent version
                    assert self.nenv % self.nminibatches == 0
                    envinds = np.arange(self.nenv)
                    flatinds = np.arange(self.nenv * self.nsteps)
                    flatinds = flatinds.reshape(self.nenv, self.nsteps)
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
                            loss = model.train(lrnow, cliprangenow,
                                               *slices, mbstates)
                            mblossvals.append(loss)

                # Feedforward --> get losses --> update
                lossvals = np.mean(mblossvals, axis=0)
                # End timer
                tnow = time.time()
                # Calculate the fps (frame per second)
                fps = int(self.nbatch / (tnow - tstart))
                if update % log_interval == 0 or update == 1:
                    # Calculates if value function is a good predicator of the
                    # returns (ev > 1) or worse than nothing (ev =< 0)
                    ev = explained_variance(values, returns)
                    # SOMEDAY: better metrics?
                    # Mean reward is not very meaningful in self-play.
                    logger.logkv("player", pi)
                    logger.logkv("serial_timesteps", update * self.nsteps)
                    logger.logkv("nupdates", update)
                    logger.logkv("total_timesteps", update * self.nbatch)
                    logger.logkv("fps", fps)
                    logger.logkv("explained_variance", float(ev))
                    logger.logkv('eprewmean_rollout', _safemean(
                                    [epinfo['r'] for epinfo in epinfos]))
                    logger.logkv('eplenmean_rollout', _safemean(
                                    [epinfo['l'] for epinfo in epinfos]))
                    logger.logkv('eprewmean_all', _safemean(
                                    [epinfo['r'] for epinfo in epinfobuf]))
                    logger.logkv('eplenmean_all', _safemean(
                                    [epinfo['l'] for epinfo in epinfobuf]))
                    logger.logkv('time_elapsed', tnow - tfirststart)
                    for (lossval, lossname) in zip(lossvals, model.loss_names):
                        logger.logkv(lossname, lossval)
                    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
                        logger.dumpkvs()
                if (save_interval and
                      (update % save_interval == 0 or update == 1) and
                      logger.get_dir() and
                      (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)):
                    checkdir = osp.join(logger.get_dir(), 'checkpoints', '.%2i' % pi)
                    os.makedirs(checkdir, exist_ok=True)
                    savepath = osp.join(checkdir, '%.5i' % update)
                    print('Saving to', savepath)
                    model.save(savepath)
        return self.models
