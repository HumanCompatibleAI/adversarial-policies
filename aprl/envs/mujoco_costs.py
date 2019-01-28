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

            import theano
            # x: (8, batch_size)
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