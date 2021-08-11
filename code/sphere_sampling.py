# Code from
# https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
# another resource for the problem: https://stackoverflow.com/questions/62199614/why-is-this-randomly-generated-spherical-point-cloud-not-uniformly-distributed?rq=1
# another resource for the problem: http://nojhan.free.fr/metah/
# picking points from surface of a_target sphere: https://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
# picking points from surface of a_target sphere: https://www.bogotobogo.com/python/python_matplotlib.php#Point_distribution
# picking points from surface of a_target sphere: http://corysimon.github.io/articles/uniformdistn-on-sphere/

# Packages
import numpy as np
from scipy.special import gammainc
from numpy.random import default_rng
# Quick start Do this (new version) of numpy.random
rng = default_rng()


def remove_unfeasible(p, treshold):
    """
    Removes the points [x,y,z] with magnitude < threshold
    :param p: vector containing the random uniform sampled points from a_target sphere
    :param treshold: minimum value to not be removed
    :return: vector p with the allowed points
    """
    points, dim = p.shape
    updated_points = 0
    aux = np.array([])
    for i in range(points):
        if np.linalg.norm(p[i]) > treshold:
            aux = np.append(aux, p[i])
            updated_points += 1
    aux = aux.reshape(updated_points, dim)
    return aux


def sample_sphere(dvmin, dvmax, n):
    """
    Samples n_per_sphere random points uniformly from a_target sphere of radius dvmax.
    Then, another embedded function "remove_unfeasible" removes those points whose norm < dvmin.
    :param dvmin: minimum Delta-v norm allowed.
    :param dvmax: maximum Delta-v norm allowed. It is the radius of the sphere.
    :param n: number of desired sampled points.
    :return: p [n x dimension of sphere] vector containing the sampled points.
    """
    ndim = 3  # dimension of the sphere
    r = dvmax
    center = np.zeros(ndim)  # center coord. of the sphere.
    x = rng.normal(size=(n, ndim))
    ssq = np.sum(x**2, axis=1)
    fr = r*gammainc(ndim/2, ssq/2)**(1/ndim)/np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(n, 1), (1, ndim))
    p = center + np.multiply(x, frtiled)
    p = remove_unfeasible(p, dvmin)
    return p


def dv_sampling(dvmin, dvmax, n):
    """
    Algorithm to sample exactly n [dvx,dvy,dvz] uniform distributed samples.
    Contains other function defined in this script to run.
    "sample_sphere()"
    :param dvmin: minimum Delta-v norm allowed
    :param dvmax: maximum Delta-v norm allowed
    :param n: number of required samples
    :return: vector [n x 3] of samples
    """
    ndim = 3
    n = n - 1  # to account for the later addition of the no-impulse case [0,0,0]
    dv = np.array([])
    while len(dv) != n:
        if len(dv) < n:
            new_dv = sample_sphere(dvmin, dvmax, n - len(dv))
            dv = np.append(dv, new_dv)
            dv = dv.reshape(int(len(dv)/ndim), ndim)

    dv = np.append(dv, [0, 0, 0])  # adding no-impulse case
    dv = dv.reshape(n+1, ndim)
    return dv


def w_sampling(w_t_max, n):
    """
    Samples 3 random points uniformly from a_target sphere of radius w_t_max.
    :param w_t_max: maximum w_t target angular velocity norm allowed. It is the radius of the sphere.
    :param n: number of desired sampled points.
    :return: p [n x dimension of sphere] vector containing the sampled points.
    """
    ndim = 3  # dimension of the sphere
    r = w_t_max
    center = np.zeros(ndim)  # center coord. of the sphere.
    x = rng.normal(size=(n, ndim))
    ssq = np.sum(x**2, axis=1)
    fr = r*gammainc(ndim/2, ssq/2)**(1/ndim)/np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(n, 1), (1, ndim))
    p = center + np.multiply(x, frtiled)
    return p


def q_sampling(n):
    """
    Samples n random points uniformly from the surface of a_target 4 dimension sphere or radius 1.
    There is a python library function, scipy.spatial.transform.Rotation.random, that does this. Maybe it is more
    reliable?
    :param n: number of desired sampled points.
    :return: p [n x dimension of sphere] vector containing the sampled points.
    """
    ndim = 4  # dimension of the sphere
    r = 1
    center = np.zeros(ndim)  # center coord. of the sphere.
    x = rng.normal(size=(n, ndim))
    ssq = np.sum(x ** 2, axis=1)
    fr = r * gammainc(ndim / 2, ssq / 2) ** (1 / ndim) / np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(n, 1), (1, ndim))
    p = center + np.multiply(x, frtiled)
    p_normalised = []
    for x in p:
        p_normalised.append(x / np.linalg.norm(x))  # so that q1**2+q2**2+q3**2+q4**2 = 1
    return p_normalised
