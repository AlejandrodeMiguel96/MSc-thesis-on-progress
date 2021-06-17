# this file computes the score of a leg predicted by the Neural Network

# Import standard packages
import numpy as np
from scipy.integrate import solve_ivp

# Import other files
from data_initial_conditions import n_steps, state0_ifo, state0_target, state0_earth
from data_features import features
from data_orbits import mu_sun, mu_earth
from algorithm import propagatetrajectory, propagate_orbit, propagate_att


def evaluate(s, w0, q):
    """
    Evaluates the predicted leg by the neural network by computing its score (based on fuel consumption, observation
    time...)
    :param s: class Leg, leg to be evaluated
    :param w0: initial angular velocity vector of the target
    :param q: initial quaternion vector defining the initial attitude of the target
    :return: score of the leg
    """
    # define initial conditions
    state0 = state0_ifo
    state0_t = state0_target
    state0_e = state0_earth
    att0 = [*w0, *q]
    state0_att_t = att0

    # integrate equations
    s.trajr, s.r_obs_vec, s.t_obs_vec, s.t = propagatetrajectory(state0, n_steps, s)
    s.trajt, s.te_obs_vec = propagate_orbit(mu_earth, state0_t, n_steps, s)
    s.traje, s.es_obs_vec = propagate_orbit(mu_sun, state0_e, n_steps, s)
    s.w_vec, s.q_vec, s.q_obs_vec = propagate_att(state0_att_t, n_steps, s)

    s.compute_traj_score(features, single_leg=False )
    # s.compute_traj_score(features, single_leg=True)
    return s.score








