"""Analytically differentiable cost functions for some MuJoCo environments.

All cost functions are intended to exactly reproduce that of the negative reward
in the original Gym environment, unless otherwise noted. However, note they are
defined in terms of the raw MuJoCo state (qpos, qvel), not Gym observations."""

# TODO: does this belong in agents instead of envs?

from ilqr.cost import BatchAutoDiffCost
from theano import tensor as T


class ReacherCost(BatchAutoDiffCost):
    """Differentiable cost for the Reacher-v2 Gym environment.
       See base class for more details."""
    def __init__(self):
        def f(x, u, i, terminal):
            if terminal:
                ctrl_cost = T.zeros_like(x[..., 0])
            else:
                ctrl_cost = T.square(u).sum(axis=-1)

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
            cost = state_cost + ctrl_cost

            return cost

        super().__init__(f, state_size=8, action_size=2)


class InvertedPendulumCost(BatchAutoDiffCost):
    """Differentiable cost for the InvertedPendulum-v2 Gym environment.
       InvertedPendulum-v2 has a +1 reward while pendulum is upright, and 0
       otherwise (with the episode terminating). This is problematic to use with
       iLQG: the cost is not differentiable (indeed, it is discontinuous on the
       state), and handling episode termination is awkward (effectively the
       dynamics include transition into a zero-reward absorbing state). Instead,
       I use a cost function penalizing the square of the: angle from y-axis,
       velocity and control. (The control penalty seems to be unnecessary to get
       good performance, but the velocity term is needed.)"""
    def __init__(self):
        def f(x, u, i, terminal):
            if terminal:
                ctrl_cost = T.zeros_like(x[..., 0])
            else:
                ctrl_cost = T.square(u).sum(axis=-1)  # penalize large control

            # x: (batch_size, 4), concatenation of qpos & qvel
            angle = x[..., 1]  # pendulum rotation
            ang_cost = angle * angle  # penalize large angles
            vel = x[..., 2:4]
            vel_cost = T.square(vel).sum(axis=-1)  # penalize large velocities

            # Try and keep the pendulum as upright as possible,
            # without too rapid movement.
            cost = ang_cost + 1e-1 * vel_cost + 1e-1 * ctrl_cost
            return cost

        super().__init__(f, state_size=4, action_size=1)


class InvertedDoublePendulumCost(BatchAutoDiffCost):
    """Differentiable cost for the InvertedDoublePendulum-v2 Gym environment.
    The cost construction is a little surprisingly quite different from
    InvertedPendulum-v2. Gym gives an alive bonus of 10, minus a quadratic
    penalty for the distance of the tip of the pole from the target position
    (x at the origin, pole fully upright at y=2) and velocity. Termination
    condition is just if height drops below y=1. In our implementation, we omit
    the alive bonus and represent termination condition as a quadratic penalty
    below a height of 1.1. We also introduce a control penalty."""
    def __init__(self, ctrl_coef=1e-1):
        def f(x, u, i, terminal):
            # Original Gym does not impose a control cost, but does clip it
            # to [-1, 1]. This non-linear dynamics is hard for iLQG to handle,
            # so add a quadratic control penalty instead.
            if terminal:
                ctrl_cost = T.zeros_like(x[..., 0])
            else:
                ctrl_cost = T.square(u).sum(axis=-1)

            # x: (batch_size, 6), concatenation of qpos & qvel

            # Distance cost
            # The tricky part is finding Cartesian coords of pole tip.
            base_x = x[..., 0]  # qpos[0]: x axis of the slider
            hinge1_ang = x[..., 1]  # qpos[1]: angle of the first hinge
            hinge2_ang = x[..., 2]  # qpos[2]: angle of the second hinge
            hinge2_cum_ang = hinge1_ang + hinge2_ang
            # 0 degrees is y=1, x=0; rotates clockwise.
            hinge1_x, hinge1_y = T.sin(hinge1_ang), T.cos(hinge1_ang)
            hinge2_x, hinge2_y = T.sin(hinge2_cum_ang), T.cos(hinge2_cum_ang)
            tip_x = base_x + hinge1_x + hinge2_x
            tip_y = hinge1_y + hinge2_y
            dist_cost = 0.01 * T.square(tip_x) + T.square(tip_y - 2)

            # Velocity cost
            v1 = x[..., 4]  # qvel[1]
            v2 = x[..., 5]  # qvel[2]
            vel_cost = 1e-3 * T.square(v1) + 5e-3 * T.square(v2)

            # TODO: termination penalty? (shouldn't change optimal policy?)
            dist_below = T.max([T.zeros_like(tip_y), 1.1 - tip_y], axis=0)
            termination_cost = T.square(dist_below)

            cost = (5 * termination_cost + dist_cost
                    + vel_cost + ctrl_coef * ctrl_cost)
            return cost

        super().__init__(f, state_size=6, action_size=1)


class HopperCost(BatchAutoDiffCost):
    """Differentiable cost for the Hopper-v2 Gym environment.
    I follow Gym in rewarding forward motion, and placing a quadratic penalty
    on control cost. Gym has a complicated termination condition: since there
    is a living reward of +1, this loosely corresponds to a discontinuous
    penalty for violating these conditions. I approximate this by penalizing
    a low height, extreme angle, or extremely large state vectors."""
    def __init__(self):
        def f(x, u, i, terminal):

            # x: (batch_size, 12), concatenation of qpos & qvel
            # Gym reward is 1 + forward_movement - control penalty
            #
            # We drop the living reward; since we plan over a fixed horizon,
            # with no termination, constants do not matter.
            #
            # Gym computes the forward movement by taking a difference before
            # and after MuJoCo's step. A MuJoCo step involves numerical
            # integration on qvel. We can't step MuJoCo here, so use qvel
            # directly; this should give the same first-order result.
            forward_movement = x[..., 6]  # qvel[0]

            # We calculate the quadratic control penalty as in Gym.
            if terminal:
                control_penalty = 0
            else:
                control_penalty = 1e-3 * T.square(u).sum(axis=-1)

            # The Gym done condition is:
            # Any state nan/infinite; absolute value of any x[..., 2:] >= 100;
            # height <= .7; absolute value of angle x[..., 2] >= .2.
            # We can't handle termination conditions, but treat violation of
            # these constraints as penalty since living reward is positive.
            # I don't do anything to try and handle nan/infinite case; infinity
            # will be discouraged by the penalty, and I've not run into any
            # issues with numerical stability.
            height = x[..., 1]  # qpos[1]
            abs_ang = abs(x[..., 2])  # qpos[2]

            def penalty_geq(x, target):
                """Quadratic penalty if x > target; zero cost if x <= target."""
                return T.square(T.max([T.zeros_like(x), x - target], axis=0))

            angle_penalty = 2000 * penalty_geq(abs_ang, 0.2 * 0.7)
            height_penalty = 200 * penalty_geq(-height, -0.7*1.25)
            state_penalty = 1e-4 * T.sum(penalty_geq(abs(x[..., 2:]), 100), axis=-1)
            termination_penalty = angle_penalty + height_penalty + state_penalty

            cost = -forward_movement + control_penalty + termination_penalty
            return cost

        super().__init__(f, state_size=12, action_size=3)


class SwimmerCost(BatchAutoDiffCost):
    """Differentiable cost for the Swimmer-v2 Gym environment. Cost function is
    as in Gym, except using velocity variable directly rather than taking
    finite difference."""
    def __init__(self):
        def f(x, u, i, terminal):
            # x: (batch_size, 10), concatenation of qpos & qvel
            # Gym reward is forward_movement - control_cost.
            #
            # Gym computes the forward movement by taking a difference before
            # and after MuJoCo's step. A MuJoCo step involves numerical
            # integration on qvel. We can't step MuJoCo here, so use qvel
            # directly; this should give the same first-order result.
            forward_movement = x[..., 5]  # qvel[0]

            # We calculate the quadratic control penalty as in Gym.
            if terminal:
                control_penalty = 0
            else:
                control_penalty = 1e-4 * T.square(u).sum(axis=-1)

            cost = -forward_movement + control_penalty
            return cost

        super().__init__(f, state_size=10, action_size=2)


class HalfCheetahCost(BatchAutoDiffCost):
    """Differentiable cost for the HalfCheetah-v2 Gym environment. Cost function
    is as in Gym, except using velocity variable directly rather than taking
    finite difference."""
    def __init__(self):
        def f(x, u, i, terminal):
            # x: (batch_size, 18), concatenation of qpos and qvel
            if terminal:
                control_penalty = 0
            else:
                control_penalty = 0.1 * T.square(u).sum(axis=-1)

            forward_movement = x[..., 9]  # qvel[0]

            return -forward_movement + control_penalty

        super().__init__(f, state_size=18, action_size=6)


COSTS = {
    'Reacher-v2': ReacherCost,
    'InvertedPendulum-v2': InvertedPendulumCost,
    'InvertedDoublePendulum-v2': InvertedDoublePendulumCost,
    'Hopper-v2': HopperCost,
    'Swimmer-v2': SwimmerCost,
    'HalfCheetah-v2': HalfCheetahCost,
}


def get_cost(env_name):
    return COSTS[env_name]()
