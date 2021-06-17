# Standard packages
import numpy as np
from statistics import mean
from scipy.integrate import odeint, solve_ivp
from scipy.spatial import Delaunay
from numpy.random import default_rng

# Import files
import sphere_sampling
import motion_eqs
from data_orbits import w_0_t
from data_ct import J
from useful_functions import compute_simplexscore, samplewithinbestsimplices
from attitude_eqs import att_dynamics
from leg import Leg

rng = default_rng()  # Quick start in new version of numpy.random
steps_not_obs = 10  # number of integration steps for each of the phases that are not observation (so more points are
# computed for the observation phase we are interested in

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


def initmesh(dvmin, dvmax, tlegmin, tlegmax, n_s0):
    """
    Initial mesh samples trajectories legs [dvx, dvy, dvz, Tleg] from the search space.
    :param dvmin: Minimum Delta-v [m/s] allowed for the inspection leg
    :param dvmax: Maximum Delta-v [m/s] allowed for the inspection leg
    :param tlegmin: Minimum time [s] allowed for the inspection leg
    :param tlegmax: Maximum time [s] allowed for the inspection leg
    :param n_s0: number of initial samples to be drawn.
    :return: [1xn_s0] list of trajectory legs type 'Leg' class
    """
    dv = sphere_sampling.dv_sampling(dvmin, dvmax, n_s0)  # [n_s0 x 3]
    t = rng.uniform(tlegmin, tlegmax, n_s0).reshape(n_s0, 1)  # [n_s0 x 1]
    s0 = np.append(dv, t, axis=1)
    legs = []
    for s in s0:
        leg = Leg(s[0:3], s[3])
        legs.append(leg)  # array containing the legs created by init.mesh
    return legs


def propagatetrajectory(state0, n_steps, leg):
    """
    Propagates the observation leg trajectory motion by the Clohessy-Wiltshire (CW) dynamics.
    :param state0: [rx0, ry0, rz0, vx0, vy0, vz0] initial state conditions.
    :param n_steps: number of integration steps
    :param leg: leg to be propagated
    :return: traj [n_steps, [rx, ry, rz, vx, vy, vz]], and also the state during the observation phase.
    """
    no_u = [0, 0, 0]
    n_int = n_steps - steps_not_obs*3
    int_time1 = np.linspace(0, leg.t_comp+leg.t_att, steps_not_obs)  # integration time part 1 (comp+att.acq)
    int_time2 = np.linspace(0, leg.t_man, steps_not_obs)  # integration time part 2 (manoeuvre)
    int_time3 = np.linspace(0, leg.t_att, steps_not_obs)  # integration time part 3 (point.acq+observation)
    int_time4 = np.linspace(0, leg.t_obs, n_int)  # integration time part 4 (observation)

    state1 = solve_ivp(motion_eqs.hill_eqs, [0, leg.t_comp+leg.t_att], state0, method='DOP853', t_eval=int_time1,
                       args=(no_u, w_0_t))
    if leg.t_man == 0:  # because the last point is dv = [0,0,0] so t_man = 0
        state2 = state1
    else:
        state2 = solve_ivp(motion_eqs.hill_eqs, [0, leg.t_man], state1.y[:, -1], method='DOP853', t_eval=int_time2,
                       args=(leg.u, w_0_t))
    state3 = solve_ivp(motion_eqs.hill_eqs, [0, leg.t_att], state2.y[:, -1], method='DOP853', t_eval=int_time3,
                       args=(no_u, w_0_t))
    state4 = solve_ivp(motion_eqs.hill_eqs, [0, leg.t_obs], state3.y[:, -1], method='DOP853', t_eval=int_time4,
                       args=(no_u, w_0_t))
    traj = np.append(np.append(np.append(state1.y, state2.y, axis=1), state3.y, axis=1), state4.y, axis=1)
    traj = traj.T

    t = np.append(np.append(np.append(state1.t, state2.t+leg.t_comp+leg.t_att), state3.t+leg.t_comp+leg.t_att+leg.t_man),
                  state4.t+leg.t_comp+leg.t_att+leg.t_man+leg.t_att)
    return traj, state4.y[:3, :].T, state4.t+(leg.t_comp+2*leg.t_att+leg.t_man), t


def propagate_orbit(mu, state0, n_steps, leg):
    """
    Propagates the orbits of a body wrt the system it is in.
    E.g: target orbit wrt Earth, Earth orbit wrt Sun.
    :return: [n_steps, [rx, ry, rz, vx, vy, vz]] state of the orbit, and also the state during the observation phase.
    """
    n_int = n_steps - steps_not_obs*3
    int_time1 = np.linspace(0, leg.t_comp + leg.t_att, steps_not_obs)  # integration time part 1 (comp+att.acq)
    int_time2 = np.linspace(0, leg.t_man, steps_not_obs)  # integration time part 2 (manoeuvre)
    int_time3 = np.linspace(0, leg.t_att, steps_not_obs)  # integration time part 3 (point.acq+observation)
    int_time4 = np.linspace(0, leg.t_obs, n_int)  # integration time part 4 (observation)

    state1 = solve_ivp(motion_eqs.orbit, [0, leg.t_comp+leg.t_att], state0, method='DOP853', t_eval=int_time1, args=(mu,))
    if leg.t_man == 0:  # because the last point is dv = [0,0,0] so t_man = 0
        state2 = state1
    else:
        state2 = solve_ivp(motion_eqs.orbit, [0, leg.t_man], state1.y[:, -1], method='DOP853', t_eval=int_time2, args=(mu,))
    state3 = solve_ivp(motion_eqs.orbit, [0, leg.t_att], state2.y[:, -1], method='DOP853', t_eval=int_time3, args=(mu,))
    state4 = solve_ivp(motion_eqs.orbit, [0, leg.t_obs], state3.y[:, -1], method='DOP853', t_eval=int_time4, args=(mu,))
    traj = np.append(np.append(np.append(state1.y, state2.y, axis=1), state3.y, axis=1), state4.y, axis=1)

    state = traj.T
    return state, state4.y[:3, :].T


def propagate_att(att_state0, n_steps, leg):
    """
    Propagates the attitude dynamics by the Euler torque free equations and quaternions.
    :param att_state0: [wx0, wy0, wz0, q0, q1, q2, q3] initial state conditions
    :param n_steps: number of integration steps
    :param leg: leg to be propagated
    :return: n_steps* [wx, wy, wz], [q0, q1, q2, q3], and also the quaternion vector during the observation phase.
    """

    n_int = n_steps - steps_not_obs*3
    int_time1 = np.linspace(0, leg.t_comp + leg.t_att, steps_not_obs)  # integration time part 1 (comp+att.acq)
    int_time2 = np.linspace(0, leg.t_man, steps_not_obs)  # integration time part 2 (manoeuvre)
    int_time3 = np.linspace(0, leg.t_att, steps_not_obs)  # integration time part 3 (point.acq+observation)
    int_time4 = np.linspace(0, leg.t_obs, n_int)  # integration time part 4 (observation)

    state1 = solve_ivp(att_dynamics, [0, leg.t_comp+leg.t_att], att_state0, method='DOP853', t_eval=int_time1, args=(J,))
    if leg.t_man == 0:  # because the last point is dv = [0,0,0] so it yields t_man = 0
        state2 = state1
    else:
        state2 = solve_ivp(att_dynamics, [0, leg.t_man], state1.y[:, -1], method='DOP853', t_eval=int_time2, args=(J,))
    state3 = solve_ivp(att_dynamics, [0, leg.t_att], state2.y[:, -1], method='DOP853', t_eval=int_time3, args=(J,))
    state4 = solve_ivp(att_dynamics, [0, leg.t_obs], state3.y[:, -1], method='DOP853', t_eval=int_time4, args=(J,))
    traj = np.append(np.append(np.append(state1.y, state2.y, axis=1), state3.y, axis=1), state4.y, axis=1)

    att = traj.T
    return att[:, :3], att[:, 3:], state4.y[3:, :].T


def refinemesh(prev_legs, n_s):
    """
    Refines the mesh in order to sample from more-promising zones, based on the results obtained in previous meshes.

    :param prev_legs: array of previous legs (type 'Leg' class) coming from the previous mesh.
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
        m_max = 1  # to avoid dividing by 0 if all leg scores are 0
    # ¡¡¡¡¡COMPUTE G_MAX AND G PROPERLY!!!!!!!
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
        leg = Leg(s[0:3], s[3])
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

    state0 = leg.trajr[-1, :]
    state0_target = leg.trajt[-1, :]
    state0_earth = leg.traje[-1, :]
    state0_att = [*leg.w_vec[-1, :], *leg.q_vec[-1, :]]
    return state0, state0_target, state0_earth, state0_att



