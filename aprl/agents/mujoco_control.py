"""LQR control for Gym MujocoEnv classes using Anassinator's iLQR library, see
https://github.com/anassinator/ilqr. Specifically, implements
finite-difference approximations for dynamics and cost."""

from collections import namedtuple
from enum import Enum
from functools import reduce

from ilqr.cost import FiniteDiffCost
from ilqr.dynamics import Dynamics, FiniteDiffDynamics
from mujoco_py import functions as mjfunc
import numpy as np

from aprl.utils import getattr_unwrapped

#TODO: Cythonize
class MujocoState(namedtuple('MujocoStateBase', 'qpos qvel')):
    """Represents state from the MuJoCo simulator needed for planning,
       namely position and velocity."""

    @staticmethod
    def from_mjdata(data):
        return MujocoState(data.qpos, data.qvel)

    @staticmethod
    def from_flattened(flattened, sim):
        qpos = flattened[0:sim.model.nq]
        qvel = flattened[sim.model.nq:sim.model.nq + sim.model.nv]
        return MujocoState(qpos, qvel)

    def set_mjdata(self, data):
        data.qpos[:] = self.qpos
        data.qvel[:] = self.qvel

    def flatten(self):
        return np.concatenate((self. qpos, self.qvel))


class MujocoFiniteDiff(object):
    def __init__(self, env):
        if hasattr(env, '_max_episode_steps'):
            # We step multiple times, then reset to a previous state.
            # Timestep limit doesn't make much sense at this level.
            # (Instead, apply it outside of the controller.)
            raise TypeError("Environment must not have a timestep limit.")
        self.env = env
        self.sim = getattr_unwrapped(env, 'sim')
        state = MujocoState.from_mjdata(self.sim.data).flatten()
        self._state_size = len(state)
        self._action_size = reduce(lambda x, y : x * y, env.action_space.shape)

    def get_state(self):
        return MujocoState.from_mjdata(self.sim.data).flatten()

    def set_state(self, x):
        state = MujocoState.from_flattened(x, self.sim)
        state.set_mjdata(self.sim.data)
        self.sim.forward()  # put mjData in consistent state


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


class MujocoFiniteDiffDynamicsBasic(MujocoFiniteDiff, FiniteDiffDynamics):
    """Computes finite difference approximation to dynamics of a MujocoEnv.

    Uses Gym step() directly to advance the simulation and exert control.
    However, does directly access MuJoCo qpos and qvel to use as an state,
    instead of Gym observations.

    WARNING: assumes environment has no termination condition -- will ignore
    step returning done. (It's not clear what the correct thing to do here is.)
    """
    def __init__(self, env, x_eps=1e-6, u_eps=1e-6):
        """Inits MujocoFiniteDiffDynamicsBase with a MujocoEnv.

        :param env (MujocoEnv): a Gym MuJoCo environment. Must not be wrapped.
        :param x_eps (float): increment to use for finite difference on state.
        :param u_eps (float): increment to use for finite difference on control.
        """
        MujocoFiniteDiff.__init__(self, env)
        FiniteDiffDynamics.__init__(self, self._mujoco_f,
                                    self.state_size, self.action_size,
                                    x_eps=x_eps, u_eps=u_eps)

    def _mujoco_f(self, x, u, i):
        self.set_state(x)
        self.env.step(u)  # evaluated for side-effects on state
        return self.get_state()


class SkipStages(Enum):
    NONE = 0
    POS = 1
    VEL = 2
    ACC = 3


def _finite_diff(sim, xptr, qacc_warmstart, skip, eps):
    #TODO: do positions in quaternion space
    orig = np.copy(xptr)
    center = np.copy(sim.data.qacc)
    nin = len(orig)
    nout = len(center)
    jacobian = np.zeros((nout, nin), dtype=np.float64)
    for j in range(nin):
        xptr[j] += eps
        sim.data.qacc_warmstart[:] = qacc_warmstart
        mjfunc.mj_forwardSkip(sim.model, sim.data, skip.value, 1)
        jacobian[:, j] = (sim.data.qacc - center) / eps
        xptr[j] = orig[j]
    return jacobian


class MujocoFiniteDiffDynamicsPerformance(MujocoFiniteDiff, Dynamics):
    """Finite difference dynamics for a MuJoCo environment."""
    NWARMUP = 3
    #TODO: assumes frame skip is 1. Can we generalize? Do we want to?
    def __init__(self, env, x_eps=1e-6, u_eps=1e-6):
        """Inits MujocoFiniteDiffDynamicsPerformance with a MujocoEnv.

        :param env (MujocoEnv): a Gym MuJoCo environment. Must not be wrapped.
               CAUTION: assumes env.step(u) sets MuJoCo ctrl to u. (This is true
               for all the built-in Gym MuJoCo environments as of 2019-01.)
        :param x_eps (float): increment to use for finite difference on state.
        :param u_eps (float): increment to use for finite difference on control.
        """
        MujocoFiniteDiff.__init__(self, env)
        Dynamics.__init__(self)
        self._x_eps = x_eps
        self._u_eps = u_eps
        self.warmstart = np.zeros_like(self.sim.data.qacc_warmstart)
        self.warmstart_for = None

    @property
    def has_hessians(self):
        """See base class Dynamics."""
        return False

    @property
    def state_size(self):
        """State size."""
        return self._state_size

    @property
    def action_size(self):
        """Action size."""
        return self._action_size

    def _warmstart(self):
        """Calculate qacc_warmstart. Initializing qacc_warmstart in the
           simulator will result in a more rapid computation of qacc when
           other fields (e.g. qpos, qvel, ctrl) are only slightly changed."""
        # TODO: This is largely undocumented in MuJoCo; I'm following the lead
        # of derivative.cpp. Does this have any implications for correctness,
        # or just performance?

        curr_state = np.concatenate((self.get_state(), self.sim.data.ctrl))
        if np.array_equal(self.warmstart_for, curr_state):
            # self.qacc_warmstart has already been computed for this x.
            return

        self.sim.forward()  # run full computation
        for i in range(self.NWARMUP):
            # iterate several times to get a good qacc_warmstart
            mjfunc.mj_forwardSkip(self.env.sim.model, self.env.sim.data,
                                  SkipStages.VEL.value, 1)
        self.warmstart[:] = self.sim.data.qacc_warmstart
        self.warmstart_for = curr_state

    def f(self, x, u, i):
        """Dynamics model. See FiniteDiffDynamics."""
        self._set_state_action(x, u)
        self._warmstart()
        self.sim.step()
        return self.get_state()

    def f_x(self, x, u, i):
        """Partial derivative of dynamics model w.r.t. to x.
           See base class Dynamics.

           Assumes that the control input u is mapped into raw control ctrl
           in the simulator by a function that does not depend on x."""
        self._set_state_action(x, u)
        self._warmstart()

        # Finite difference over velocity: skip = POS
        jacob_vel = _finite_diff(self.sim, self.sim.data.qvel, self.warmstart,
                                 SkipStages.POS, self._x_eps)
        jacob_pos = _finite_diff(self.sim, self.sim.data.qpos, self.warmstart,
                                 SkipStages.NONE, self._x_eps)
        # This Jacobian is qacc differentiated w.r.t. (qpos, qvel).
        acc_jacobian = np.concatenate((jacob_pos, jacob_vel), axis=1)
        # But we want (qpos, qvel) differentiated w.r.t. (qpos, qvel).
        # x_{t+1} = x_t + dt * \dot{x}_t
        # \dot{x}_{t+1} = \dot{x}_t + \ddot{x}_t
        # \nabla_{(x_t, \dot{x}_t)} \ddot{x}_t = acc_jacobian
        # So \nabla_{(x_t, \dot{x}_t)} \dot{x}_t = (
        #      dt * \nabla_{x_t} \ddot{x}_t,
        #      I + dt * \nabla_{\dot{x}_t} \ddot{x}_t)
        # And \nabla_{(x_t, \dot{x}_t)} x_t = (I, dt * I)
        # TODO: Could use an alternative linearization?
        dt = self.sim.model.opt.timestep
        A = np.eye(self.sim.model.nq)
        B = dt * np.eye(self.sim.model.nv)
        C = dt * acc_jacobian[:, :self.sim.model.nq]
        D = np.eye(self.sim.model.nv) + dt * acc_jacobian[:, self.sim.model.nq:]
        state_jacobian = np.vstack([
            np.hstack([A, B]),
            np.hstack([C, D]),
        ])

        return state_jacobian

    def f_u(self, x, u, i):
        """Partial derivative of dynamics model w.r.t. to u.
           See base class Dynamics.

           Assumes that the control input u is mapped into raw control ctrl
           in the simulator by a function that does not depend on x."""
        self._set_state_action(x, u)
        self._warmstart()

        # Finite difference over control: skip = VEL
        # This Jacobian is qacc differentiated w.r.t. ctrl.
        jacobian_acc = _finite_diff(self.sim, self.sim.data.ctrl, self.warmstart,
                                    SkipStages.VEL, self._u_eps)
        # But we want (qpos, qvel) differentiated w.r.t. ctrl.
        # \nabla_{u_t} \ddot{x}_t = acc_jacobian
        # \nabla_{u_t} \dot{x}_t = dt * \nabla_{u_t} \ddot{x}_t
        # \nabla_{u_t} x_t = 0
        # TODO: Could use an alternative linearization?
        dt = self.sim.model.opt.timestep
        state_jacobian = np.vstack([
            np.zeros((self.sim.model.nq, self.sim.model.nu)),
            dt * jacobian_acc
        ])
        return state_jacobian

    def f_xx(self, x, u, i):
        """See base class Dynamics."""
        raise NotImplementedError()

    def f_ux(self, x, u, i):
        """See base class Dynamics."""
        raise NotImplementedError()

    def f_uu(self, x, u, i):
        """See base class Dynamics."""
        raise NotImplementedError()

    def _set_state_action(self, x, u):
        self.sim.data.ctrl[:] = u
        self.set_state(x)