'''Analytically differentiable cost functions for some MuJoCo environments.

All cost functions are intended to exactly reproduce that of the negative reward
in the original Gym environment, unless otherwise noted. However, note they are
defined in terms of the raw MuJoCo state (qpos, qvel), not Gym observations.'''

#TODO: does this belong in agents instead of envs?

from theano import tensor as T
from ilqr.cost import BatchAutoDiffCost

class ReacherCost(BatchAutoDiffCost):
    '''Differentiable cost for the Reacher-v2 Gym environment.
       See base class for more details.'''
    def __init__(self):
        def f(x, u, i, terminal):
            if terminal:
                return T.zeros_like(x[..., 0])

            # x: (batch_size, 8)
            # x[..., 0:4]: qpos
            # x[..., 4:8]: qvel, time derivatives of qpos, not used in the cost.
            theta = x[..., 0]  # qpos[0]: angle of joint 0
            phi = x[..., 1]  # qpos[1]: angle of joint 1
            target_xpos = x[..., 2:4]  # qpos[2:4], target x & y coordinate
            body1_xpos = 0.1 * T.stack([T.cos(theta), T.sin(theta)], axis=1)
            tip_xpos_incr = 0.11 * T.stack([T.cos(phi), T.sin(phi)], axis=1)
            tip_xpos = body1_xpos + tip_xpos_incr
            delta = tip_xpos - target_xpos

            state_cost = T.sqrt(T.sum(delta * delta, axis=-1))
            control_cost = T.sum(u * u, axis=-1)
            cost = state_cost + control_cost

            return cost

        super().__init__(f, state_size=8, action_size=2)


class InvertedPendulumCost(BatchAutoDiffCost):
    '''Differentiable cost for the InvertedPendulum-v2 Gym environment.
       InvertedPendulum-v2 has a +1 reward while pendulum is upright, and 0
       otherwise (with the episode terminating). This is problematic to use with
       iLQG: the cost is not differentiable (indeed, it is discontinuous on the
       state), and handling episode termination is awkward (effectively the
       dynamics include transition into a zero-reward absorbing state). Instead,
       I use a cost function penalizing the square of the: angle from y-axis,
       velocity and control. (The control penalty seems to be unnecessary to get
       good performance, but the velocity term is needed.)'''
    def __init__(self):
        def f(x, u, i, terminal):
            if terminal:
                return T.zeros_like(x[..., 0])
            # x: (batch_size, 4), concatenation of qpos & qvel

            angle = x[..., 1]  # pendulum rotation
            ang_cost = angle * angle  # penalize large angles
            vel = x[..., 2:4]
            vel_cost = T.sum(vel * vel, axis=-1)  # penalize large velocities
            ctrl_cost = T.sum(u * u, axis=-1)  # penalize large control
            # Try and keep the pendulum as upright as possible,
            # without too rapid movement.
            cost = ang_cost + vel_cost + ctrl_cost
            return cost

        super().__init__(f, state_size=4, action_size=1)