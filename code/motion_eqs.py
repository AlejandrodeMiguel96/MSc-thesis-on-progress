# Equation of motion
import numpy as np


def hill_eqs(t, v, u, w0):
    """States the equations of close proximity relative motion for LVLH frame:
        xdotdot = 2*w0*zdot + ux
        ydotdot = -w0**2 * y + uy
        zdotdot = 3*w0**2 * z - 2*w0*xdot + uz
        The three 2nd order diff.eqs are divided into six 1st order diff.eqs
        by adding 3 new variables a_target,b,c.
        :param v : [x,y,z,a,b,c] state vector in LVLH frame: V-bar, H-bar, R-bar. Ref: Capolupo_SOTA paper
        :param t :
        :param u = [ux,uy,uz] is the chaser control acceleration provided by the electric thruster.
        :param w0:  is the target’s orbital rate, [rad∕s].
        See Capolupo_SOTA.pdf for more details
        :return: vector vdot
    """
    return [
        v[3],
        v[4],
        v[5],
        2 * w0 * v[5] + u[0],
        -w0**2 * v[1] + u[1],
        3 * w0 ** 2 * v[2] - 2 * w0 * v[3] + u[2]
    ]


def orbit(t, v, mu):
    r = np.linalg.norm([v[0], v[1], v[2]])
    return [
        v[3],
        v[4],
        v[5],
        -mu/r**3 * v[0],
        -mu/r**3 * v[1],
        -mu/r**3 * v[2]
    ]