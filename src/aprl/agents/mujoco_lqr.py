"""LQR control for Gym MujocoEnv classes using Anassinator's iLQR library, see
https://github.com/anassinator/ilqr. Specifically, implements
finite-difference approximations for dynamics and cost."""

from contextlib import contextmanager
from enum import Enum
from functools import reduce

from ilqr.cost import FiniteDiffCost
from ilqr.dynamics import Dynamics, FiniteDiffDynamics
from mujoco_py import functions as mjfunc
import numpy as np

from aprl.common.mujoco import MujocoState
from aprl.common.utils import getattr_unwrapped


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
        self._action_size = reduce(lambda x, y: x * y, env.action_space.shape)

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


class JointTypes(Enum):
    FREE = 0
    BALL = 1
    SLIDE = 2
    HINGE = 3


class SkipStages(Enum):
    NONE = 0
    POS = 1
    VEL = 2
    ACC = 3


class VaryValue(Enum):
    POS = 0
    VEL = 1
    CTRL = 2


def _finite_diff(sim, qacc_warmstart, vary, eps):
    # Select the variable to perform finite differences over
    variables = {
        VaryValue.POS: sim.data.qpos,
        VaryValue.VEL: sim.data.qvel,
        VaryValue.CTRL: sim.data.ctrl,
    }
    variable = variables[vary]

    # Select what parts of the forward dynamics computation can be skipped.
    # Everything depends on position, so perform full computation in that case.
    # When varying velocity, can skip position-dependent computations.
    # When varying control, can skip acceleration/force-dependent computations.
    skip_stages = {
        VaryValue.POS: SkipStages.NONE,
        VaryValue.VEL: SkipStages.POS,
        VaryValue.CTRL: SkipStages.VEL,
    }
    skip_stage = skip_stages[vary]

    # Make a copy of variable, to restore after each perturbation
    original = np.copy(variable)
    # Run forward simulation and copy the qacc output. We will take finite
    # differences w.r.t. this value later.
    sim.data.qacc_warmstart[:] = qacc_warmstart
    sim.forward()  # TODO: this computation is probably often redundant
    center = np.copy(sim.data.qacc)

    if vary in [VaryValue.VEL, VaryValue.CTRL]:
        nin = len(variable)
        nout = len(center)
        jacobian = np.zeros((nout, nin), dtype=np.float64)
        for j in range(nin):
            # perturb
            variable[j] += eps
            # compute finite-difference
            sim.data.qacc_warmstart[:] = qacc_warmstart
            mjfunc.mj_forwardSkip(sim.model, sim.data, skip_stage.value, 1)
            jacobian[:, j] = (sim.data.qacc - center) / eps
            # unperturb
            variable[j] = original[j]
    elif vary == VaryValue.POS:
        # When varying qpos, positions corresponding to ball and free joints
        # are specified in quaternions. The derivative w.r.t. a quaternion
        # is only defined in the tangent space to the configuration manifold.
        # Accordingly, we do not vary the quaternion directly, but instead
        # use mju_quatIntegrate to perturb with a certain angular velocity.
        # Note that the Jacobian is always square with size sim.model.nv,
        # which in the presence of quaternions is smaller than sim.model.nq.
        nv = sim.model.nv
        jacobian = np.zeros((nv, nv), dtype=np.float64)

        # If qvel[j] belongs to a quaternion, then quatadr[j] gives the address
        # of that quaternion and  dofpos[j] gives the corresponding degree of
        # freedom within the quaternion.
        #
        # If qvel[j] is not part of a quaternion, then quatadr[j] is -1
        # and ids[j] gives the index in qpos to that joint. If there are no
        # quaternions, ids will just be a sequence of consecutive numbers.
        # But in the presence of a quaternion, it will skip a number.

        # Common variables
        joint_ids = sim.model.dof_jntid
        jnt_types = sim.model.jnt_type[joint_ids]
        jnt_qposadr = sim.model.jnt_qposadr[joint_ids]
        dofadrs = sim.model.jnt_dofadr[joint_ids]

        # Compute ids
        js = np.arange(nv)
        ids = jnt_qposadr + js - dofadrs

        # Compute quatadr and dofpos
        quatadr = np.tile(-1, nv)
        dofpos = np.zeros(nv)

        # A ball joint is always a quaternion.
        ball_joints = (jnt_types == JointTypes.BALL.value)
        quatadr[ball_joints] = jnt_qposadr[ball_joints]
        dofpos[ball_joints] = js[ball_joints] - dofadrs[ball_joints]

        # A free joint consists of a Cartesian (x,y,z) and a quaternion.
        free_joints = (jnt_types == JointTypes.FREE.value) & (js > dofadrs + 3)
        quatadr[free_joints] = jnt_qposadr[free_joints] + 3
        dofpos[free_joints] = js[free_joints] - dofadrs[ball_joints] - 3

        for j in js:
            # perturb
            qa = quatadr[j]
            if qa >= 0:  # quaternion perturbation
                angvel = np.array([0, 0, 0])
                angvel[dofpos[j]] = eps
                # side-effect: perturbs variable[qa:qa+4]
                mjfunc.mju_quatIntegrate(variable[qa], angvel, 1)
            else:  # simple perturbation
                variable[ids[j]] += eps

            # compute finite-difference
            sim.data.qacc_warmstart[:] = qacc_warmstart
            mjfunc.mj_forwardSkip(sim.model, sim.data, skip_stage.value, 1)
            jacobian[:, j] = (sim.data.qacc - center) / eps

            # unperturb
            if qa >= 0:  # quaternion perturbation
                variable[qa:qa+4] = original[qa:qa+4]
            else:
                variable[ids[j]] = original[ids[j]]
    else:
        assert False

    return jacobian


@contextmanager
def _consistent_solver(sim, niter=30):
    saved_iterations = sim.model.opt.iterations
    saved_tolerance = sim.model.opt.tolerance
    sim.model.opt.iterations = niter
    sim.model.opt.tolerance = 0
    yield sim
    sim.model.opt.iterations = saved_iterations
    sim.model.opt.tolerance = saved_tolerance


class MujocoFiniteDiffDynamicsPerformance(MujocoFiniteDiff, Dynamics):
    """Finite difference dynamics for a MuJoCo environment."""
    # TODO: assumes frame skip is 1. Can we generalize? Do we want to?
    NWARMUP = 3

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
        self.sim.step()
        return self.get_state()

    def f_x(self, x, u, i):
        """Partial derivative of dynamics model w.r.t. to x.
           See base class Dynamics.

           Assumes that the control input u is mapped into raw control ctrl
           in the simulator by a function that does not depend on x."""
        with _consistent_solver(self.sim) as sim:
            self._set_state_action(x, u)
            self._warmstart()
            jacob_vel = _finite_diff(sim, self.warmstart,
                                     VaryValue.VEL, self._x_eps)
            jacob_pos = _finite_diff(sim, self.warmstart,
                                     VaryValue.POS, self._x_eps)

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
            dt = sim.model.opt.timestep
            A = np.eye(sim.model.nq)
            B = dt * np.eye(sim.model.nv)
            C = dt * acc_jacobian[:, :sim.model.nq]
            D = np.eye(sim.model.nv) + dt * acc_jacobian[:, sim.model.nq:]
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
        with _consistent_solver(self.sim) as sim:
            self._set_state_action(x, u)
            self._warmstart()
            # Finite difference over control: skip = VEL
            # This Jacobian is qacc differentiated w.r.t. ctrl.
            jacobian_acc = _finite_diff(sim, self.warmstart,
                                        VaryValue.CTRL, self._u_eps)
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
