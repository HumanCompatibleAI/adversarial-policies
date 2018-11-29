"""LQR control for Gym MujocoEnv classes using Anassinator's iLQR library, see
https://github.com/anassinator/ilqr. Specifically, implements
finite-difference approximations for dynamics and cost."""

from functools import reduce

from ilqr.cost import FiniteDiffCost
from ilqr.dynamics import FiniteDiffDynamics
from mujoco_py import MjSimState

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

    def __init__(self, env, x_eps=None, u_eps=None):
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