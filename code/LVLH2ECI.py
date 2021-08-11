import numpy as np
from time import perf_counter

def lvlh2eci(rr, rr_ref, vv_ref):
    """
    Transforms a position vector in LVLH into ECI frame.
    :param rr: vector to be converted from LVLH to ECI frame.
    :param rr_ref: numpy array of the reference position vector in LVLH frame (in this case, the target one).
    :param vv_ref: numpy array of the reference velocity vector in LVLH frame (in this case, the target one).
    :return: rr in ECI frame.
    """
    cross_prod = np.cross(rr_ref, vv_ref)

    o3 = -rr_ref/np.linalg.norm(rr_ref)  # LVLh z-axis in ECI frame. Positive towards Earth.
    o2 = -cross_prod / np.linalg.norm(cross_prod)  # LVLh y-axis in ECI frame.
    # Aligned with the negative orbit-normal.
    o1 = np.cross(o2, o3)  # LVLH x-axis in ECI frame. Completes the right-hand triad (also coincident with orbit v).

    A = np.array([
        [o1[0], o2[0], o3[0]],
        [o1[1], o2[1], o3[1]],
        [o1[2], o2[2], o3[2]]
    ])  # Rotation matrix. A = np.vstack((o1,o2,o3)).T

    return np.dot(A, rr)

