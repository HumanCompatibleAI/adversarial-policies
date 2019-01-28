'''Analytically differentiable cost functions for some MuJoCo environments.

All cost functions are intended to exactly reproduce that of the negative reward
in the original Gym environment, unless otherwise noted. However, note they are
defined in terms of the raw MuJoCo state (qpos, qvel), not Gym observations.'''

#TODO: does this belong in agents instead of envs?

from theano import tensor as T
from ilqr.cost import AutoDiffCost

class ReacherCost(AutoDiffCost):
    '''Differentiable cost for the Reacher-v2 Gym environment.'''
    def __init__(self):
        # Define state and control inputs

        # qpos[0]: angle of joint 0
        # qpos[1]: angle of joint 1
        # qpos[2], qpos[3]: target x & y coordinate
        qpos_inputs = [T.dscalar('theta'), T.dscalar('phi'), T.dscalar('targetx'),
                       T.dscalar('targety')]
        # qvel: time derivatives of the above.
        # Target are constant (non-actuated) so always have derivative zero.
        qvel_inputs = [T.dscalar('thetadot'), T.dscalar('phidot'),
                       T.dscalar('_zero1'), T.dscalar('_zero2')]
        x_inputs = qpos_inputs + qvel_inputs

        # control[0:2]: torque of joint 0 and joint 1
        u_inputs = [T.dscalar('thetadotdot'), T.dscalar('phidotdot')]

        # Define cost in terms of these inputs
        u = T.stack(u_inputs)
        control_cost = T.dot(u, u)

        qpos = T.stack(qpos_inputs)
        theta, phi = qpos[0], qpos[1]
        target_xpos = qpos[2:4]

        body1_xpos = 0.1 * T.stack([T.cos(theta), T.sin(theta)])
        fingertip_xpos_delta = 0.11 * T.stack([T.cos(phi), T.sin(phi)])
        fingertip_xpos = body1_xpos + fingertip_xpos_delta
        delta = fingertip_xpos - target_xpos
        state_cost = T.sqrt(T.dot(delta, delta))

        l = state_cost + control_cost
        l_terminal = T.zeros(())

        super().__init__(l, l_terminal, x_inputs, u_inputs)