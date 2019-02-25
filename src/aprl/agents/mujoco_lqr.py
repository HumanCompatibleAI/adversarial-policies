"""LQR control for Gym MujocoEnv classes using Anassinator's iLQR library, see
https://github.com/anassinator/ilqr. Specifically, implements
finite-difference approximations for dynamics and cost."""

from contextlib import contextmanager
from enum import Enum
from functools import reduce
import multiprocessing

from ilqr.cost import FiniteDiffCost
from ilqr.dynamics import Dynamics, FiniteDiffDynamics
from mujoco_py import MjSim
from mujoco_py import functions as mjfunc
import numpy as np

from aprl.common.mujoco import MujocoState
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


class VaryKind(Enum):
    POS = 0
    VEL = 1
    CTRL = 2


def _finite_diff(sim, qacc_warmstart, vary, x_eps, u_eps):
    '''
    Computes finite differences of qacc w.r.t. variables vary.
    :param sim: (MjSim) MuJoCo simulator.
    :param qacc_warmstart: (ndarray) field to initialize qacc_warmstart to before each evaluation.
    :param vary: (list<tuple>) a list of variables to vary, specified by a tuple (vary_value, idxs)
                 where vary_value is a value of VaryValue and idxs is a list of indices.
    :param x_eps: (float) the amount to vary each state variable.
    :param u_eps: (float) the amount to vary each control variable.
    :return: a Jacobian for qacc w.r.t. the subset of variables vary.
             Shape (sim.data.na, sum([len(idx) for _, idx in vary])).
    '''
    # Select the variable to perform finite differences over
    variables = {
        VaryKind.POS: sim.data.qpos,
        VaryKind.VEL: sim.data.qvel,
        VaryKind.CTRL: sim.data.ctrl,
    }
    # Select what parts of the forward dynamics computation can be skipped.
    # Everything depends on position, so perform full computation in that case.
    # When varying velocity, can skip position-dependent computations.
    # When varying control, can skip acceleration/force-dependent computations.
    skip_stages = {
        VaryKind.POS: SkipStages.NONE,
        VaryKind.VEL: SkipStages.POS,
        VaryKind.CTRL: SkipStages.VEL,
    }

    # TODO: copy data from dmain into sim?
    nvars = sum([end - start for _, start, end in vary])
    jacobian = np.zeros((sim.model.nv, nvars))

    # Run forward simulation and copy the qacc output. We will take finite
    # differences w.r.t. this value later.
    sim.data.qacc_warmstart[:] = qacc_warmstart
    sim.forward()  # TODO: this computation is probably often redundant
    center = np.copy(sim.data.qacc)
    idx = 0

    for vary_kind, start, end in vary:
        variable = variables[vary_kind]
        skip_stage = skip_stages[vary_kind]

        # Make a copy of variable, to restore after each perturbation
        original = np.copy(variable)

        if vary_kind in [VaryKind.VEL, VaryKind.CTRL]:
            eps = x_eps if vary_kind == VaryKind.VEL else u_eps
            for j in range(start, end):
                # perturb
                variable[j] += eps
                # compute finite-difference
                sim.data.qacc_warmstart[:] = qacc_warmstart
                mjfunc.mj_forwardSkip(sim.model, sim.data, skip_stage.value, 1)
                jacobian[:, idx] = (sim.data.qacc - center) / eps
                idx += 1
                # unperturb
                variable[j] = original[j]
        elif vary_kind == VaryKind.POS:
            # When varying qpos, positions corresponding to ball and free joints
            # are specified in quaternions. The derivative w.r.t. a quaternion
            # is only defined in the tangent space to the configuration manifold.
            # Accordingly, we do not vary the quaternion directly, but instead
            # use mju_quatIntegrate to perturb with a certain angular velocity.
            # Note that the Jacobian is always square with size sim.model.nv,
            # which in the presence of quaternions is smaller than sim.model.nq.

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
            js = np.arange(sim.model.nv)
            ids = jnt_qposadr + js - dofadrs

            # Compute quatadr and dofpos
            quatadr = np.tile(-1, sim.model.nv)
            dofpos = np.zeros(sim.model.nv)

            # A ball joint is always a quaternion.
            ball_joints = (jnt_types == JointTypes.BALL.value)
            quatadr[ball_joints] = jnt_qposadr[ball_joints]
            dofpos[ball_joints] = js[ball_joints] - dofadrs[ball_joints]

            # A free joint consists of a Cartesian (x,y,z) and a quaternion.
            free_joints = (jnt_types == JointTypes.FREE.value) & (js > dofadrs + 3)
            quatadr[free_joints] = jnt_qposadr[free_joints] + 3
            dofpos[free_joints] = js[free_joints] - dofadrs[ball_joints] - 3

            for j in range(start, end):
                # perturb
                qa = quatadr[j]
                if qa >= 0:  # quaternion perturbation
                    angvel = np.array([0, 0, 0])
                    angvel[dofpos[j]] = x_eps
                    # side-effect: perturbs variable[qa:qa+4]
                    mjfunc.mju_quatIntegrate(variable[qa], angvel, 1)
                else:  # simple perturbation
                    variable[ids[j]] += x_eps

                # compute finite-difference
                sim.data.qacc_warmstart[:] = qacc_warmstart
                mjfunc.mj_forwardSkip(sim.model, sim.data, skip_stage.value, 1)
                jacobian[:, idx] = (sim.data.qacc - center) / x_eps
                idx += 1

                # unperturb
                if qa >= 0:  # quaternion perturbation
                    variable[qa:qa+4] = original[qa:qa+4]
                else:
                    variable[ids[j]] = original[ids[j]]
        else:
            assert False

    return jacobian


def _worker(model, remote, parent_remote, niter, x_eps, u_eps):
    parent_remote.close()
    sim = MjSim(model)
    sim.model.opt.iterations = niter
    sim.model.opt.tolerance = 0
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'finitediff':
                x, u, qacc_warmstart, vary = data
                MujocoState.from_flattened(x, sim).set_mjdata(sim.data)
                # TODO: is this enough state restoration?
                sim.data.ctrl[:] = u
                sim.data.qacc_warmstart[:] = qacc_warmstart
                sim.forward()  # make consistent state
                part_jacobian = _finite_diff(sim, qacc_warmstart, vary, x_eps, u_eps)
                remote.send(part_jacobian)
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('MujocoFiniteDiffDynamicsPerformance worker: got KeyboardInterrupt.')
    except Exception as e:
        print('Exception in MujocoFiniteDiffDynamicsPerformance worker: ', e)


@contextmanager
def _consistent_solver(sim, niter=30):
    saved_iterations = sim.model.opt.iterations
    saved_tolerance = sim.model.opt.tolerance
    sim.model.opt.iterations = niter
    sim.model.opt.tolerance = 0
    yield sim
    sim.model.opt.iterations = saved_iterations
    sim.model.opt.tolerance = saved_tolerance


def _split_indices(array_len, num_workers):
    Neach_section, extras = divmod(array_len, num_workers)
    section_sizes = [0] + extras * [Neach_section+1] + (num_workers-extras) * [Neach_section]
    div_points = np.array(section_sizes).cumsum()
    return div_points


class MujocoFiniteDiffDynamicsPerformance(MujocoFiniteDiff, Dynamics):
    """Finite difference dynamics for a MuJoCo environment."""
    # TODO: assumes frame skip is 1. Can we generalize? Do we want to?
    NWARMUP = 3

    def __init__(self, env, nworkers=None, niter=30, x_eps=1e-6, u_eps=1e-6):
        """Inits MujocoFiniteDiffDynamicsPerformance with a MujocoEnv.

        :param env (MujocoEnv): a Gym MuJoCo environment. Must not be wrapped.
               CAUTION: assumes env.step(u) sets MuJoCo ctrl to u. (This is true
               for all the built-in Gym MuJoCo environments as of 2019-01.)
        :param n_iter: (int) number of MuJoCo solver iterations.
        :param x_eps: (float) increment to use for finite difference on state.
        :param u_eps: (float) increment to use for finite difference on control.
        """
        MujocoFiniteDiff.__init__(self, env)
        Dynamics.__init__(self)
        self.niter = niter
        self._x_eps = x_eps
        self._u_eps = u_eps
        self.warmstart = np.zeros_like(self.sim.data.qacc_warmstart)
        self.warmstart_for = None
        self.jacobian = None
        self.jacobian_for = None

        if nworkers is None:
            nworkers = multiprocessing.cpu_count()
        nworkers = min(nworkers, self.sim.model.nv)
        pipes = [multiprocessing.Pipe() for _ in range(nworkers)]
        self.remotes, worker_remotes = zip(*pipes)
        self.ps = []
        for work_remote, remote in zip(worker_remotes, self.remotes):
            args = (self.sim.model, work_remote, remote, niter, x_eps, u_eps)
            process = multiprocessing.Process(target=_worker, args=args)
            process.daemon = True  # if main process crashes, subprocesses should exit
            process.start()
            self.ps.append(process)
        for remote in worker_remotes:
            remote.close()

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

    def _fd(self, x, u, i):
        if self.jacobian_for is None:
            equals = False
        else:
            orig_x, orig_u, orig_i = self.jacobian_for
            equals = (orig_x == x).all() and (orig_u == u).all() and (orig_i == i)

        if not equals:
            num_workers = len(self.ps)
            nv = self.sim.model.nv
            nu = self.sim.model.nu
            vary_size = {
                VaryKind.POS: nv,
                VaryKind.VEL: nv,
                VaryKind.CTRL: nu,
            }
            vary_ranges = {k: _split_indices(v, num_workers) for k, v in vary_size.items()}
            varys = [
                [(kind, divpoints[i], divpoints[i+1]) for kind, divpoints in vary_ranges.items()]
                for i in range(num_workers)
            ]
            for remote, vary in zip(self.remotes, varys):
                remote.send(('finitediff', (x, u, self.warmstart, vary)))

            self.jacobian = {
                VaryKind.POS: np.zeros((nv, nv)),
                VaryKind.VEL: np.zeros((nv, nv)),
                VaryKind.CTRL: np.zeros((nv, nu)),
            }
            for remote, vary in zip(self.remotes, varys):
                part_jacobian = remote.recv()
                idx = 0
                for vary_kind, start, end in vary:
                    chunk_size = end - start
                    self.jacobian[vary_kind][:, start:end] = part_jacobian[:, idx:idx+chunk_size]
                    idx += chunk_size

            self.jacobian_for = (x, u, i)

    def f_x(self, x, u, i):
        """Partial derivative of dynamics model w.r.t. to x.
           See base class Dynamics.

           Assumes that the control input u is mapped into raw control ctrl
           in the simulator by a function that does not depend on x."""
        with _consistent_solver(self.sim, niter=self.niter) as sim:
            self._set_state_action(x, u)
            self._warmstart()
            self._fd(x, u, i)

            # This Jacobian is qacc differentiated w.r.t. (qpos, qvel).
            acc_jacobian = np.concatenate((self.jacobian[VaryKind.POS],
                                           self.jacobian[VaryKind.VEL]), axis=1)
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
        with _consistent_solver(self.sim, niter=self.niter):
            self._set_state_action(x, u)
            self._warmstart()
            self._fd(x, u, i)

            # Finite difference over control: skip = VEL
            # This Jacobian is qacc differentiated w.r.t. ctrl.
            acc_jacobian = self.jacobian[VaryKind.CTRL]
            # But we want (qpos, qvel) differentiated w.r.t. ctrl.
            # \nabla_{u_t} \ddot{x}_t = acc_jacobian
            # \nabla_{u_t} \dot{x}_t = dt * \nabla_{u_t} \ddot{x}_t
            # \nabla_{u_t} x_t = 0
            # TODO: Could use an alternative linearization?
            dt = self.sim.model.opt.timestep
            state_jacobian = np.vstack([
                np.zeros((self.sim.model.nq, self.sim.model.nu)),
                dt * acc_jacobian
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

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
