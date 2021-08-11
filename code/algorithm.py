# Standard packages
import numpy as np
from statistics import mean
from scipy.spatial import Delaunay
from numpy.random import default_rng

# Import files
import sphere_sampling
from useful_functions import compute_simplexscore, samplewithinbestsimplices
from leg import Leg
from data_inspection import deltav_max, deltav_min, T_leg_max, T_leg_min

rng = default_rng()  # Quick start in new version of numpy.random


def compute_mission_complet(features):
    """
    :param features: vector containing all the target features to be observed by the chaser.
    :return: M_c, mission completion parameter
    """
    summatory = 0
    t_goal = features[0].t_goal  # T_goal is supposed to be the same for all features, so the first one is taken as e.g.
    for f in features:
        summatory += min(f.t_goal, f.t_obs_feat)
    return 100/(len(features)*t_goal) * summatory


def initmesh(dvmin, dvmax, tlegmin, tlegmax, state0_chaser, n_s0):
    """
    Initial mesh samples trajectories legs [dvx, dvy, dvz, Tleg] from the search space.
    :param dvmin: Minimum Delta-v [m/s] allowed for the inspection leg
    :param dvmax: Maximum Delta-v [m/s] allowed for the inspection leg
    :param tlegmin: Minimum time [s] allowed for the inspection leg
    :param tlegmax: Maximum time [s] allowed for the inspection leg
    :param state0_chaser: initial state [rx,ry,rz,vx,vy,vz] of the chaser
    :param n_s0: number of initial samples to be drawn.
    :return: [1xn_s0] list of trajectory legs type 'Leg' class
    """
    dv = sphere_sampling.dv_sampling(dvmin, dvmax, n_s0)  # [n_s0 x 3]
    t = rng.uniform(tlegmin, tlegmax, n_s0)  # [n_s0]
    legs = []
    for x, y in zip(dv, t):  # create Leg class from the samples
        leg = Leg(x, y, state0_chaser)
        while leg.integration_status != 0:  # in case the leg is not valid for the mission, sample another one
            new_dv = sphere_sampling.dv_sampling(dvmin, dvmax, 2)[0]  # take first sample, the 2nd one is dv=[0,0,0]
            new_t = rng.uniform(tlegmin, tlegmax, 1)[0]  # [0] because we just want the value, not the array
            leg = Leg(new_dv, new_t, state0_chaser)
        legs.append(leg)  # array containing the legs created by init.mesh
    return legs


# def propagate_hills(state0, leg):
#     """
#     Propagates the observation leg trajectory motion by the Clohessy-Wiltshire (CW) dynamics.
#     :param state0: [rx0, ry0, rz0, vx0, vy0, vz0] initial state conditions.
#     # :param t1: array of leg computation + att.acquisition times
#     # :param t2: array of propulsion manoeuvre time
#     # :param t3: array of att.acquisition time
#     # :param t4: array of observation time
#     :param leg: leg to be propagated
#     :return:
#     """
#     no_u = [0, 0, 0]
#
#     def event_validmission(x, y):
#         """
#         Event function to stop integration in case of an event is found.
#         The event is that the mission is invalid (see useful_functions.py --> isvalidmission function
#         """
#         if data_inspection.r_min <= np.linalg.norm([y[0], y[1], y[2]]) <= data_inspection.r_escape:
#             return 1
#         else:
#             return 0
#     event_validmission.terminal = True
#
#
#     state1 = solve_ivp(fun=lambda x, y: motion_eqs.hill_eqs(x, y, no_u, w_0_t), t_span=[0, leg.t1[-1]],
#                        y0=state0, method=method_int, t_eval=leg.t1, rtol=rel_tol, atol=abs_tol,
#                        events=event_validmission)
#     if state1.status == 1:  # checks if integration has been stoped by the validmission event
#         return 0, 0, state1.status
#
#     if leg.t_man == 0:  # because when dv = [0,0,0] --> t_man = 0, so there is no impulse manoeuvre
#         state2 = state1
#     else:
#         state2 = solve_ivp(fun=lambda x, y: motion_eqs.hill_eqs(x, y, leg.dv, w_0_t), t_span=[0, leg.t2[-1]],
#                            y0=state1.y[:, -1], method=method_int, t_eval=leg.t2, rtol=rel_tol, atol=abs_tol,
#                            events=event_validmission)
#         if state2.status == 1:
#             return 0, 0, state2.status
#
#     state3 = solve_ivp(fun=lambda x, y: motion_eqs.hill_eqs(x, y, no_u, w_0_t), t_span=[0, leg.t3[-1]],
#                        y0=state2.y[:, -1], method=method_int, t_eval=leg.t3, rtol=rel_tol, atol=abs_tol,
#                        events=event_validmission)
#     if state3.status == 1:
#         return 0, 0, state3.status
#
#     state4 = solve_ivp(fun=lambda x, y: motion_eqs.hill_eqs(x, y, no_u, w_0_t), t_span=[0, leg.t4[-1]],
#                        y0=state3.y[:, -1], method=method_int, t_eval=leg.t4, rtol=rel_tol, atol=abs_tol,
#                        events=event_validmission)
#     if state4.status == 1:
#         return 0, 0, state4.status
#     traj = np.append(np.append(np.append(state1.y, state2.y, axis=1), state3.y, axis=1), state4.y, axis=1)
#     traj = traj.T
#
#     return traj[:, 0:3], state4.y[:3, :].T, 0


# def propagate_orbit(mu, state0, t1, t2, t3, t4, leg):
#     """
#     Propagates the orbits of a_target body wrt the system it is in.
#     E.g: target orbit wrt Earth, Earth orbit wrt Sun.
#     :return: [n_steps, [rx, ry, rz, vx, vy, vz]] state of the orbit, and also the state during the observation phase.
#     """
#
#     state1 = solve_ivp(motion_eqs.orbit, [0, leg.t_comp+leg.t_att], state0, method=method_int, t_eval=t1,
#                        args=(mu,), rtol=rel_tol, atol=abs_tol)
#     if leg.t_man == 0:  # because the last point is dv = [0,0,0] so t_man = 0
#         state2 = state1
#     else:
#         state2 = solve_ivp(motion_eqs.orbit, [0, leg.t_man], state1.y[:, -1], method=method_int, t_eval=t2,
#                            args=(mu,), rtol=rel_tol, atol=abs_tol)
#     state3 = solve_ivp(motion_eqs.orbit, [0, leg.t_att], state2.y[:, -1], method=method_int, t_eval=t3,
#                        args=(mu,), rtol=rel_tol, atol=abs_tol)
#     state4 = solve_ivp(motion_eqs.orbit, [0, leg.t_obs], state3.y[:, -1], method=method_int, t_eval=t4,
#                        args=(mu,), rtol=rel_tol, atol=abs_tol)
#     traj = np.append(np.append(np.append(state1.y, state2.y, axis=1), state3.y, axis=1), state4.y, axis=1)
#
#     state = traj.T
#     return state, state4.y[:3, :].T


# def propagate_att(att_state0, t1, t2, t3, t4, leg):
#     """
#     Propagates the attitude dynamics by the Euler torque free equations and quaternions.
#     :param att_state0: [wx0, wy0, wz0, q0, q1, q2, q3] initial state conditions
#     :param t1: array of leg computation + att.acquisition times
#     :param t2: array of propulsion manoeuvre time
#     :param t3: array of att.acquisition time
#     :param t4: array of observation time
#     :param leg: leg to be propagated
#     :return: n_steps* [wx, wy, wz], [q0, q1, q2, q3], and also the quaternion vector during the observation phase.
#     """
#     state1 = solve_ivp(att_dynamics, [0, t1[-1]], att_state0, method=method_int, t_eval=t1,
#                        args=(J,), rtol=rel_tol, atol=abs_tol)
#     if leg.t_man == 0:  # because the last point is dv = [0,0,0] so it yields t_man = 0
#         state2 = state1
#     else:
#         state2 = solve_ivp(att_dynamics, [0, t2[-1]], state1.y[:, -1], method=method_int, t_eval=t2,
#                            args=(J,), rtol=rel_tol, atol=abs_tol)
#     state3 = solve_ivp(att_dynamics, [0, t3[-1]], state2.y[:, -1], method=method_int, t_eval=t3,
#                        args=(J,), rtol=rel_tol, atol=abs_tol)
#     state4 = solve_ivp(att_dynamics, [0, t4[-1]], state3.y[:, -1], method=method_int, t_eval=t4,
#                        args=(J,), rtol=rel_tol, atol=abs_tol)
#     traj = np.append(np.append(np.append(state1.y, state2.y, axis=1), state3.y, axis=1), state4.y, axis=1)
#
#     att = traj.T
#     return att[:, :3], att[:, 3:], state4.y[3:, :].T


def refinemesh(prev_legs, state0_chaser, n_s):
    """
    Refines the mesh in order to sample from more-promising zones, based on the results obtained in previous meshes.

    :param prev_legs: array of previous legs (type 'Leg' class) coming from the previous mesh.
    :param state0_chaser: initial state [rx,ry,rz,vx,vy,vz] of the chaser
    :param n_s: number of new refined samples to obtain.
    :return:
    """
    scored_points = np.array([])
    for leg in prev_legs:
        scored_point = [*leg.dv, leg.t_leg, leg.score]
        scored_points = np.append(scored_points, scored_point)
    scored_points = scored_points.reshape(len(prev_legs), 5)
    tri = Delaunay(scored_points[:, 0:4], qhull_options='QJ')
    m_max = max(scored_points[:, 4])  # Maximum trajectory score of all simplices of the triangulation
    if m_max == 0:
        print('algorithm.py: m_max = 0 because all leg scores are 0!!!')
        m_max = 1  # to avoid raising the dividing by 0 error if all leg scores are 0
    g_max = 1
    for q in tri.simplices:
        smplx_scores = scored_points[:, 4][q]  # scores of the points defining the simplex
        aux = mean(smplx_scores)
        if g_max < aux:
            g_max = aux

    simplices_scored = []
    for q in tri.simplices:
        smplx_score = compute_simplexscore(q, scored_points, m_max, g_max)
        # simp_scored = [smplx_score, q_vec]
        simplices_scored.append([smplx_score, q])
    sorted_simp_scores = sorted(simplices_scored, reverse=True)  # ranks the simplices based on score
    new_samples = samplewithinbestsimplices(sorted_simp_scores, tri.points, n_s)

    new_legs = []
    for s in new_samples:
        leg = Leg(s[0:3], s[3], state0_chaser)
        new_legs.append(leg)

    return new_legs


def update_values(leg, features):
    """
    Update all the values from an optimal leg.
    Updates feature.t_obs_feat,  feature.t_useful_feat_list, integral init.conds.
    :param leg: leg to update from
    :param features: features to be updated
    :param mus: [mu_earth, mu_sun] vector containing the mus
    :param states0: [state0_chaser, state0_target, state0_earth] vector containing the initial states [rx,ry,rz,vx,vy,vz]
    :return: [state], final states which will be the init.states for the next iteration.
    """
    for f, t in zip(features, leg.t_useful_feat_list):
        f.t_obs_feat += t

    state0 = leg.rr_chaser_LVLH[-1, :]
    state0_target = leg.rr_target[-1, :]
    state0_earth = leg.rr_sun[-1, :]
    state0_att = [*leg.w_vec[-1, :], *leg.q_vec[-1, :]]
    return state0, state0_target, state0_earth, state0_att



