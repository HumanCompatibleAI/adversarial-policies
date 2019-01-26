"""LQR control for Gym MujocoEnv classes using Anassinator's iLQR library, see
https://github.com/anassinator/ilqr. Specifically, implements
finite-difference approximations for dynamics and cost."""

from collections import namedtuple
from functools import reduce

from ilqr.cost import FiniteDiffCost
from ilqr.dynamics import FiniteDiffDynamics
from mujoco_py import MjSimState
import numpy as np

from aprl.utils import getattr_unwrapped

class MujocoFiniteDiff(object):
    def __init__(self, env):
        if hasattr(env, '_max_episode_steps'):
            # We step multiple times, then reset to a previous state.
            # Timestep limit doesn't make much sense at this level.
            # (Instead, apply it outside of the controller.)
            raise TypeError("Environment must not have a timestep limit.")
        self.env = env
        self.sim = getattr_unwrapped(env, 'sim')
        state = self.get_state()
        self._state_size = len(state)
        self._action_size = reduce(lambda x, y : x * y, env.action_space.shape)

    def get_state(self):
        return self.sim.get_state().flatten()

    def set_state(self, x):
        state = MjSimState.from_flattened(x, self.sim)
        self.sim.set_state(state)
        self.sim.forward()  # put mjData in consistent state


class MujocoFiniteDiffDynamics(MujocoFiniteDiff, FiniteDiffDynamics):
    """Finite difference approximation to dynamics of a MujocoEnv.
    WARNING: assumes environment has no termination condition -- will ignore
    step returning done. (It's not clear what the correct thing to do here is.)
    """

    def __init__(self, env, x_eps=1e-6, u_eps=1e-6):
        MujocoFiniteDiff.__init__(self, env)
        FiniteDiffDynamics.__init__(self, self._mujoco_f,
                                    self.state_size, self.action_size,
                                    x_eps=x_eps, u_eps=u_eps)

    def _mujoco_f(self, x, u, i):
        self.set_state(x)
        self.env.step(u)  # evaluated for side-effects on state
        return self.get_state()


class MujocoFiniteDiffCost(MujocoFiniteDiff, FiniteDiffCost):
    def __init__(self, env, x_eps=None, u_eps=None):
        MujocoFiniteDiff.__init__(self, env)
        FiniteDiffCost.__init__(self, self._mujoco_l, self._mujoco_l_terminal,
                                self._state_size, self._action_size,
                                x_eps=x_eps, u_eps=u_eps)

    def _mujoco_l(self, x, u, i):
        self.set_state(x)
        _, r, _, _ = self.env.step(u)
        return -r  # we want cost, not reward!

    def _mujoco_l_terminal(self, x, i):
        """Return a zero terminal cost: i.e. no reward at end of planning
        horizon. In a fixed-horizon environment where our planning horizon
        matches that of the underlying environment, this is the right thing
        to do. Otherwise, it would be better for this to be some "reward to go"
        estimate, but there's no universally sensible way to do this. May want
        to override this on a case-by-case basis."""
        return 0

# TODO: This is fairly inefficient, and ugly to boot.
# Once you know what fields you actually care about, get rid of the dynamic logic.
class MujocoRelevantState(namedtuple('MujocoStateBase',
        'qpos qvel qacc qacc_warmstart qfrc_applied xfrc_applied')):
    '''Based off MjSimState'''
    def flatten(self):
        """ Flattens a state into a numpy array of numbers."""
        res = []
        for f in self._fields:
            v = getattr(self, f)
            if v is not None:
                res.append(v.flatten())
        return np.concatenate(res)

    @staticmethod
    def from_flattened(a, sim):
        len_a = len(a)
        idx_qpos = 0
        idx_qvel = idx_qpos + sim.model.nq
        idx_qacc = idx_qvel + sim.model.nv
        idx_qacc_warmstart = idx_qacc + sim.model.nv
        idx_qfrc_applied = idx_qacc_warmstart + sim.model.nv
        idx_xfrc_applied = idx_qfrc_applied + sim.model.nv

        qpos = a[idx_qpos:idx_qpos + sim.model.nq]

        qvel = None
        if idx_qvel < len_a:
            qvel = a[idx_qvel:idx_qvel + sim.model.nv]

        qacc = None
        if idx_qacc < len_a:
            qacc = a[idx_qacc:idx_qacc + sim.model.nv]

        qacc_warmstart = None
        if idx_qacc_warmstart < len_a:
            qacc_warmstart = a[idx_qacc_warmstart:idx_qacc_warmstart + sim.model.nv]

        qfrc_applied = None
        if idx_qfrc_applied < len_a:
            qfrc_applied = a[idx_qfrc_applied:idx_qfrc_applied + sim.model.nv]

        xfrc_applied = None
        if idx_xfrc_applied < len_a:
            xfrc_applied = a[idx_xfrc_applied:idx_xfrc_applied + sim.model.nbody * 6]
            xfrc_applied = xfrc_applied.reshape(sim.model.nbody, 6)

        return MujocoRelevantState(qpos, qvel,
                                   qacc, qacc_warmstart,
                                   qfrc_applied, xfrc_applied)

    @classmethod
    def from_mjdata(cls, data, fields=None):
        if fields is None:
            fields = cls._fields
        kwargs = {f: getattr(data, f) for f in fields}
        return MujocoRelevantState(**kwargs)

    def set_mjdata(self, data):
        for k in self._fields:
            assert hasattr(data, k)
            v = getattr(self, k)
            if v is not None:
                getattr(data, k)[:] = v
MujocoRelevantState.__new__.__defaults__ = (None,) * 6


class MujocoFiniteDiffDynamicsLowLevel(MujocoFiniteDiff, FiniteDiffDynamics):
    def __init__(self, env, x_eps=1e-6, u_eps=1e-6, kind='recommended'):
        if kind == 'all':
            self.fields = None
        elif kind == 'basic':
            self.fields = ['qpos', 'qvel']
        elif kind == 'basic_plus':
            self.fields = ['qpos', 'qvel', 'qacc']
        elif kind == 'recommended':
            self.fields = ['qpos', 'qvel', 'qacc', 'qacc_warmstart']
        else:
            raise ValueError("Unrecognised kind: '{}'".format(kind))

        MujocoFiniteDiff.__init__(self, env)
        FiniteDiffDynamics.__init__(self, self._mujoco_f,
                                    self.state_size, self.action_size,
                                    x_eps=x_eps, u_eps=u_eps)

    def get_state(self):
        return MujocoRelevantState.from_mjdata(self.sim.data, self.fields).flatten()

    def set_state(self, x):
        state = MujocoRelevantState.from_flattened(x, self.sim)
        state.set_mjdata(self.sim.data)
        self.sim.forward()  # put mjData in consistent state

    def _mujoco_f(self, x, u, i):
        #TODO: something more efficient, also figure out if step vs forward breaks things
        self.set_state(x)
        self.env.step(u)  # evaluated for side-effects on state
        return self.get_state()

# TODO:
# + What attributes are actually needed to be saved? Check derivative.cpp again.
# + What to do about step v.s. forward?
# + Is the state actually being reset properly? (I don't fully trust MuJoCo-Py.)

# derivative.cpp copies from dmain:
# qpos; qvel; qacc; qacc_warmstart; qfrc_applied; xfrc_applied; ctrl
# Does one forward step, and nwarmup forwardSkip(mjSTAGE_VEL).
# For each differencing step:
# + Copies warmstart; does forwardSkip with appropriate stage.
# + Everything else is stage-specific.
# + For perturrbations in qfrc_applied, only needs to restore qacc. Since forwardSkip does not integrate, it doesn't change the rest of the state.
# + For perturbations in qvel, have to restore qvel; otherwise same.
# + For perturbations in position, have to restore qpos. They also do different perturbations depending on joint type (to get consistent eps, I think).
# I think the parts important for correctness are:
# + Copying all the relevant bits from dmain. Done, I think (except ctrl, but you set that.)
# + Warmstarting.
# + Angle perturbation. I'm not directly changing ctrl, but rather action in env.step, so hard to normalize in general. I'd be a bit surprised if this really broke things, though.
# Rest is for efficiency, including using forward rather than skip.
# So probably go for correctness first. Then figure out how to make it efficient.
# (May want to just use their C++ code, if you can factor MuJoCo v.s. policy FD.)

# MjSimState: time qpos qvel act udd_state
# It's ok to omit qfrc_applied and xfrc_applied since Gym never sets these,
# just using the ctrl field.

# Hmm, using MujocoRelevantState rather than MjSimState made no difference.
# Literally no difference: same up to 8 d.p. at least!
# This is actually quite surprising -- we'll be perturbing some new values!

# Gym environments use frame-skip (2 for Reacher, 5 for HalfCheetah), so
# cannot easily swap out for forward. I'm a bit confused why they do
# frame skip rather than changing dt. I guess smaller dt makes better physics sim?

# Why does MjSimState contain act but derivative.cpp does not?