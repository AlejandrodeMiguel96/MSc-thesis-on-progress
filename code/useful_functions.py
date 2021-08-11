# This file contains useful functions

# STANDARD PACKAGES
import numpy as np
from statistics import mean
from scipy.spatial import Delaunay, ConvexHull
from numpy.random import default_rng

# FILES IMPORTS
from data_orbits import r_earth
from data_inspection import r_obs_min, r_obs_max, r_escape, nu_V, nu_M, nu_G, alpha_max, beta_max, r_min

rng = default_rng()


def computeangle(a, b):
    """
    Computes angle between 2 vectors.
    :param a: vector 1
    :param b: vector 2
    :return: angle [rad]
    """
    return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_rr_target2sun(rr_target, rr_sun):
    """
    Computes the target-sun vector as the sum of vectors r (target-Earth) and e (Earth-Sun).
    :param rr_target: earth-target position vector (ECI frame) [1x3]. Note: it is later converted to target-earth.
    :param rr_sun: Earth-Sun position vector (ECI frame) [1x3]
    :return: sun-target position vector (ECI frame) [1x3]
    """
    return - rr_target + rr_sun


def iseclipsed(es_eci, et_eci):
    """
    Says if a satellite (the target) is eclipsed by Earth.
    :param es_eci: position vector Earth-Sun (ECI frame).
    :param et_eci: position vector Earth-target (ECI frame).
    :return: boolean
    """
    theta = computeangle(es_eci, et_eci)
    es_norm = np.linalg.norm(es_eci)
    et_norm = np.linalg.norm(et_eci)
    theta_sun = np.arccos(r_earth / es_norm)
    theta_sat = np.arccos(r_earth / et_norm)
    if theta_sun + theta_sat <= theta:
        return True
    else:
        return False


def isinrange(rr_lvlh):
    """
    Says if a_target certain feature in in observation range.
    :param rr_lvlh: relative position vector target-chaser
    :return: boolean
    """
    r = np.linalg.norm(rr_lvlh)
    if r_obs_min <= r <= r_obs_max:
        inrange = True
    else:
        inrange = False
    return inrange


def isinfov(view_dir, r):
    """
    Says if a target's certain feature is in Field of view (FOV) --> alpha <= alpha_max
    being alpha the angle between the viewing direction of the target feature and the vector defining the position of
    the chaser with respect to the target
    :param view_dir: viewing direction vector (ECI frame).
    :param r: relative position vector target-chaser (ECI frame).
    :return: boolean
    """
    alpha = computeangle(view_dir, r)
    if alpha <= alpha_max:
        observed = True
    else:
        observed = False
    return observed


def isiluminated(view_dir_inert, ts):
    """
    Says if a_target certain feature is properly iluminated --> beta <= beta_max
    being beta the angle between the viewing direction and the Sun direction.
    :param view_dir_inert: viewing direction vector in an inertial frame (ECI frame).
    :param ts: relative target-Sun vector (ECI frame).
    :return: boolean
    """
    beta = computeangle(view_dir_inert, ts)
    if beta <= beta_max:
        observed = True
    else:
        observed = False
    return observed


def isvalidmission(rr_chaser_lvlh):
    """
    Says if a_target leg is valid or not depending on if it fulfill some requirements.
    Conditions:
        - Chaser reaches a_target maximum distance larger than r_escape from the target during its trajectory.
        - Chaser's and solar arrays' angular motion exceed the specified RW and SADM limits (instrument contraint).
        - Maneouvre is even partially executed during eclipse (energy constraint).
        - Trajectory is unsafe (safety constraint).
    Disclaimer: at the moment only r_escape constrain is implemented.
    :param rr_chaser_lvlh: target-chaser position vector (LVLH frame).
    :return: boolean
    """
    valid_mission = True
    i = 0
    for x in rr_chaser_lvlh:
        if not r_min < np.linalg.norm(x) < r_escape:  # max.distance case
            valid_mission = False
            break  # because just one invalid point is enough to invalidate the whole leg
        i += 1
    return valid_mission


def compute_splxvolume(splx_points):
    """
    Computes the volume of a_target simplex using the Cayley-Menger determinant.
    For a_target 4D euclidean space (dvx, dvy, dvz, T) each simplex is determined by 5 points or vertices.
    :param splx_points: array of points forming the simplex.
    :return: volume of the simplex.
    """
    c = ConvexHull(splx_points)
    # # in order to avoid numerical issues, it skips simplices that are smaller than a_target specified minimum size
    # if c.volume < 1e-5:
    #     c.volume = 0
    return c.volume


def compute_simplexscore(q, scored_points, m_max, g_max):
    """
    ¡¡¡¡¡¡¡¡¡¡¡¡UPDATE BECAUSE RIGHT NOW G = 0!!!!!!!!!!!!!!
    Returns a_target simplex score for each simplex of the Delaunay triangulation
    :param q: simplex
    :param scored_points: [nx5] vector containing the initial sampled points [0:4] and their score [4]
    :param m_max:maximum trajectory score found in the triangulation(i.e considering all simplices of the triangulation)
    :param g_max:maximum score gradient found in the triangulation(i.e considering all simplices of the triangulation)
    :return:
    """
    v = compute_splxvolume(scored_points[:, 0:4][q])
    m = max(scored_points[:, 4])
    g = mean(scored_points[:, 4][q])
    J = v ** nu_V * (1 + nu_M * m / m_max + nu_G * g / g_max)
    return J


def samplewithinbestsimplices(weight_list, points, n_samples):
    """
    Selects a_target specified number of simplices from a_target list---choosing those with higher weights more frequently---
    and places a_target new point within each of the corresponding simplices.
    Since a_target given simplex may be chosen more than once during this process—--indeed, that is part of the intent of the
    weighted sort—--placing that new point at the exact center could create degeneracies.
    Instead, we randomly sample a_target point inside the simlex (see function 'sample_smplx_randomly').
    :param weight_list: [scores, simplices] list containing the score of the simplex and the simples
    :param points: all the points of the space that form all the simplices
    :param n_samples: number of new sampled points that we want to obtain from the entire simplices space.
    :return: [n_samples x dim_of_points] new sampled points
    """
    chosen_simplices = weighted_randchoice(weight_list, n_samples)
    new_samples = np.array([])
    for s in chosen_simplices:
        sample = sample_smplx_randomly(points[s])
        new_samples = np.append(new_samples, sample)
    new_samples = new_samples.reshape(n_samples, len(sample[0]))
    return new_samples


def weighted_randchoice(weight_list, n_samples):
    """
        Chapter 3.7 of "Description of the Reachability Set Adaptive Mesh Algorithm", by Erik Komendera.
    This algorithm allows for the random choice from a_target list in which each element has a_target weight associated with it.
    To do this, each item and its weight are assigned a_target nonoverlapping range of values for use in random sampling.
    For example, if two items have weights 2 and 5, the ranges for each will be [0,2] and [2,7].
    By randomly sampling real numbers between 0 and 7, the second item will be chosen with a_target probability of 5/7.
    :param weight_list: [scores, simplices] list containing the score of the simplex and the simples
    :param n_samples: number of samples
    :return: vector containing the chosen simplices based on its score
    """
    summatory = 0
    intervals_max = []
    for x in weight_list:
        summatory += x[0]
        intervals_max.append(summatory)
    uniform_sampl = rng.uniform(0, summatory, n_samples)

    chosen_simplices = []
    for s in uniform_sampl:
        for i in range(len(intervals_max)):
            if s <= intervals_max[i]:
                chosen_simplices.append(weight_list[i][1])
                break
    return chosen_simplices


def sample_smplx_randomly(smplx_points):
    """
    From reference [16], "Picking a_target uniformly random point from an arbitrary simplex", by Christian Grimme.
    :param smplx_points: [1xn] vector containing the points/vertices defining the simplex.
    :return: [1xn] randomly sampled point defined by n-coordinates.
    """
    n = len(smplx_points) - 1
    z = [1, *rng.uniform(0, 1, n), 0]
    lmb = []
    for j in range(len(z)-1):
        aux = z[j]**(1/(n+1-j))
        lmb.append(aux)
    lmb.append(0)

    multiplication = np.cumprod(lmb[0:n+1])
    result = np.zeros((1, n))
    for i in range(1, n+2):
        result += [(1 - lmb[i]) * multiplication[i-1] * x for x in smplx_points[i-1]]
    return result



