import numpy as np
from time import perf_counter
from data_inspection import n_mesh, n_s0, n_s, deltav_min, deltav_max, T_leg_min, T_leg_max
from data_orbits import mu_sun, mu_earth
from data_initial_conditions import n_steps
from algorithm import initmesh, refinemesh, propagatetrajectory, propagate_orbit, propagate_att
from sphere_sampling import w_sampling, q_sampling


def compute_opt_leg(state0_c, state0_t, state0_e, w_t_max, features):
    """
    Computes the best (optimal) leg, in terms of propellant and number of observed features.
    :param state0_c: [rx0, ry0, rz0, vx0, vy0, vz0] initial state of the chaser
    :param state0_t: [rx0, ry0, rz0, vx0, vy0, vz0] initial state of the target
    :param state0_e: [rx0, ry0, rz0, vx0, vy0, vz0] initial state of the Earth
    :param w_t_max: maximum angular velocity magnitude of the target
    :param features: list of Features class objects, representing the observable features of the target
    :return: [att0, leg_opt, issolfound]:
    att0 is a vector containing the initial attitude [wx0, wy0, wz0, q00, q10, q20, q30]
    leg_opt is a list containing all the optimal inspection legs (class Leg).
    This is thinking about the neural network, which may take as input the target attitude, and output the optimal leg.
    """
    state0 = state0_c
    w0 = w_sampling(w_t_max, 1)[0]
    q0 = q_sampling(1)[0]
    att0 = [*w0, *q0]
    state0_att_t = att0
    print('Starting SBMPO_single_leg.py: Number of features to inspect:', len(features))
    print('Starting SBMPO_single_leg.py: Initial chaser state:', state0)
    print('Starting SBMPO_single_leg.py: Initial target attitude:', att0)
    start = perf_counter()

    S_j_all = np.array([])
    scores_all = np.array([])
    for j in range(n_mesh):
        scores = np.array([])
        if j == 0:
            start_initmesh = perf_counter()
            S_j = initmesh(deltav_min, deltav_max, T_leg_min, T_leg_max, n_s0)
            end_initmesh = perf_counter()
            print('SBMPO_single_leg.py: init mesh ex.time = ', end_initmesh - start_initmesh)
        else:
            start_refinemesh = perf_counter()
            S_prev = S_j
            S_j = refinemesh(S_prev, n_s)
            end_refinemesh = perf_counter()
            print('SBMPO_single_leg.py: refine mesh ex.time = ', end_refinemesh - start_refinemesh)
        start_propagate = perf_counter()
        for s in S_j:
            # VECTORES OBS Y T_OBS CREO QUE PUEDEN SER BORRADOS, SON PARA PLOTEAR
            s.trajr, s.r_obs_vec, s.t_obs_vec, s.t = propagatetrajectory(state0, n_steps, s)
            s.trajt, s.te_obs_vec = propagate_orbit(mu_earth, state0_t, n_steps, s)
            s.traje, s.es_obs_vec = propagate_orbit(mu_sun, state0_e, n_steps, s)
            s.w_vec, s.q_vec, s.q_obs_vec = propagate_att(state0_att_t, n_steps, s)

            s.compute_traj_score(features)
            # s.compute_traj_score(features, single_leg=True)  # THIS ONE SHOULD BE ACTIVATED
            scores = np.append(scores, s.score)

        end_propagate = perf_counter()
        print('SBMPO_single_leg.py: propagate ex.time = ', end_propagate - start_propagate)
        S_j_all = np.append(S_j_all, S_j)
        scores_all = np.append(scores_all, scores)
        print('SBMPO_single_leg.py: scores', sorted(scores_all, reverse=True)[0:3])
    s_opt = S_j_all[np.argmax(scores_all)]
    print('SBMPO_single_leg.py: Leg score = ', s_opt.score)
    print('SBMPO_single_leg.py: Delta-v = ', np.linalg.norm(s_opt.dv))
    print('SBMPO_single_leg.py: Obs.time for each feature = ', s_opt.t_useful_feat_list)
    leg_input = att0
    leg_output = [*s_opt.dv, s_opt.t_leg]
    if s_opt.score == 0:
        solutionfound = False
        print('SBMPO_single_leg.py.py: All scores are 0. Solution could NOT been found')
        end = perf_counter()
        print('SBMPO_single_leg.py.py: Inpection leg execution time = ', end - start, 'seconds')
        return leg_input, leg_output, solutionfound
    else:
        solutionfound = True
        np.savez('output_data_single_leg.npz', leg_opt=s_opt)  # can be removed and just save the database
        print('SBMPO_single_leg.py.py: Solution FOUND!')
        end = perf_counter()
        print('SBMPO_single_leg.py.py: Inpection leg execution time = ', end - start, 'seconds')
        return leg_input, leg_output, solutionfound

