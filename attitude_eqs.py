import numpy as np
# import mathutils
# import pyquaternion

def euler_torquefree_eqs(t, w, Jt):
    """
    Euler's torque free equations: Jt*ωdot = − ω × Jt*ω
    :param w: [wx,wy,wz] is the angular velocity vector of the TARGET with respect to an inertial reference frame,
    :param Jt: [Jx,Jy,Jz] is the TARGET inertia tensor, diagonal
    :return: wdot [3x1]
    """
    return [
        (Jt[1] - Jt[2])/Jt[0] * w[1] * w[2],
        (Jt[2] - Jt[0])/Jt[1] * w[0] * w[2],
        (Jt[0] - Jt[1])/Jt[2] * w[1] * w[0],
    ]


def quat_kinematics(t, qt, w):
    """
    :param qt: [4x1] unitary quaternion describing the attitude of the TARGET with respect to an inertial reference frame
    :param w: [wx,wy,wx] is the angular velocity vector of the TARGET with respect to an inertial reference frame
    :return: qtdot
    """
    w_hat = np.array([
        [0, w[2], -w[1], w[0]],
        [-w[2], 0,  w[0], w[1]],
        [w[1], -w[0], 0, w[2]],
        [-w[0], -w[1], -w[2], 0]
    ])
    return 0.5 * w_hat.dot(qt)


def att_dynamics(t, v, Jt):
    """
    :param v: [1x7] variable vector containing w_vec and q_vec.
    w_vec: [wx,wy,wx] is the angular velocity vector of the TARGET with respect to an inertial reference frame
    qt: [4x1] unitary quaternion describing the attitude of the TARGET with respect to an inertial reference frame
    :return: qtdot
    """
    w = np.array([v[0], v[1], v[2]])
    q = np.array([v[3], v[4], v[5], v[6]]).reshape(4, 1)
    # to normalise the quaternions, it's not exact (e.g 1.0000001). Also, I think it does not make a difference
    q_norm = q / np.linalg.norm(q)
    w_hat = np.array([
        [0, w[2], -w[1], w[0]],
        [-w[2], 0,  w[0], w[1]],
        [w[1], -w[0], 0, w[2]],
        [-w[0], -w[1], -w[2], 0]
    ])
    qdot = 0.5 * w_hat.dot(q_norm)
    return [
        (Jt[1] - Jt[2]) / Jt[0] * w[1] * w[2],
        (Jt[2] - Jt[0]) / Jt[1] * w[0] * w[2],
        (Jt[0] - Jt[1]) / Jt[2] * w[1] * w[0],
        qdot[0],
        qdot[1],
        qdot[2],
        qdot[3]
    ]


def quat2dcm(q):
    """
    Converts quaternion to DCM matrix.
    :param q: [1x4] quaternions
    :return: [3x3] DCM matrix
    """
    return np.array([
        [q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2, 2*(q[0]*q[1]+q[2]*q[3]), 2*(q[0]*q[2]-q[1]*q[3])],
        [2*(q[0]*q[1]-q[2]*q[3]), -q[0]**2 + q[1]**2 - q[2]**2 + q[3]**2, 2*(q[1]*q[2]+q[0]*q[3])],
        [2*(q[0]*q[2]+q[1]*q[3]), 2*(q[1]*q[2]-q[0]*q[3]), -q[0]**2 - q[1]**2 + q[2]**2 + q[3]**2]
    ])  # CHECK IF A FULFILLS THE PROPERTIES(I.E A*A^T = I, det(A)=1
