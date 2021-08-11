import numpy as np
from time import perf_counter
from data_inspection import n_mesh, n_s0, n_s, deltav_min, deltav_max, T_leg_min, T_leg_max
from data_orbits import mu_sun, mu_earth
from data_initial_conditions import n_steps
from algorithm import initmesh, refinemesh, propagate_hills, compute_mission_complet, \
    propagate_orbit, propagate_att, update_values
from sphere_sampling import w_sampling, q_sampling


def compute_inspection(state0_c, state0_t, state0_e, w_t_max, features, n_legs_max):
    """
    Executes the SBMPO inspection algorithm.
    :param state0_c: [rx0, ry0, rz0, vx0, vy0, vz0] initial state of the chaser
    :param state0_t: [rx0, ry0, rz0, vx0, vy0, vz0] initial state of the target
    :param state0_e: [rx0, ry0, rz0, vx0, vy0, vz0] initial state of the Earth
    :param w_t_max: maximum angular velocity magnitude of the target
    :param features: list of Features class objects, representing the observable features of the target
    :param n_legs_max: maximum number of inspection legs allowed to complete the inspection mission
    :return: [att0, opt_leg, issolfound]:
    att0 is a_target vector containing the initial attitude [wx0, wy0, wz0, q00, q10, q20, q30]
    opt_leg is a_target list containing all the optimal inspection legs (class Leg)
    issolfound is a_target bool with value 1 if the mission has been completed, 0 otherwise.
    """
    issolfound = True  # bool to known if algorithm has been able to found a_target solution
    state0 = state0_c
    w0 = w_sampling(w_t_max, 1)[0]
    q0 = q_sampling(1)[0]
    att0 = [*w0, *q0]
    state0_att_t = att0
    M_c = 0  # mission completition variable
    i_leg = 1  # Leg counter
    M_c_list = []
    legs_opt = []
    print('Starting SBMPO. Number of features to inspect:', len(features))
    print('Starting SBMPO. Initial chaser state:', state0)
    print('Starting SBMPO. Initial target attitude:', att0)
    start = perf_counter()
    while M_c < 100:
        print('SBMPO_full_inspection.py: Leg #', i_leg)
        S_j_all = np.array([])
        scores_all = np.array([])
        for j in range(n_mesh):
            scores = np.array([])
            if j == 0:
                start_initmesh = perf_counter()
                S_j = initmesh(deltav_min, deltav_max, T_leg_min, T_leg_max, n_s0)
                end_initmesh = perf_counter()
                print('SBMPO_full_inspection.py: init mesh ex.time = ', end_initmesh - start_initmesh)
            else:
                start_refinemesh = perf_counter()
                S_prev = S_j
                S_j = refinemesh(S_prev, n_s)
                end_refinemesh = perf_counter()
                print('SBMPO_full_inspection.py: refine mesh ex.time = ', end_refinemesh - start_refinemesh)
            start_propagate = perf_counter()
            for s in S_j:
                # VECTORES OBS Y T_OBS CREO QUE PUEDEN SER BORRADOS, SON PARA PLOTEAR
                s.rr_chaser_LVLH, s.r_obs_chaser_vec_LVLH, s.t_obs_vec, s.t = propagateHills(state0, n_steps, s,,
                                                                              s.rr_target, s.r_obs_et_vec = propagate_orbit(mu_earth, state0_t, n_steps, s)
                s.rr_sun, s.r_obs_es_vec = propagate_orbit(mu_sun, state0_e, n_steps, s)
                s.w_vec, s.q_vec, s.q_obs_vec = propagate_att(state0_att_t, n_steps, s)

                s.compute_traj_score(features,,
                scores = np.append(scores, s.score)

            end_propagate = perf_counter()
            print('SBMPO_full_inspection.py: propagate ex.time = ', end_propagate - start_propagate)
            S_j_all = np.append(S_j_all, S_j)
            scores_all = np.append(scores_all, scores)
            print('SBMPO_full_inspection.py: scores', sorted(scores_all, reverse=True)[0:3])
            print('SBMPO_full_inspection.py: max score', max(scores_all))
        s_opt = S_j_all[np.argmax(scores_all)]
        state0, state0_t, state0_e, state0_att_t = \
            update_values(s_opt, features)
        M_c = compute_mission_complet(features)
        M_c_list.append(M_c)
        i_leg += 1
        print('SBMPO_full_inspection.py: Mission completition: ', M_c, '%')
        print('SBMPO_full_inspection.py: Leg score = ', s_opt.score)
        if s_opt.score == 0:
            issolfound = False
            print('SBMPO_full_inspection.py: All scores are 0. Solution could NOT been found')
            end = perf_counter()
            print('SBMPO_full_inspection.py: Inpection leg execution time = ', end - start, 'seconds')
            return att0, legs_opt, issolfound
        elif i_leg > n_legs_max:
            issolfound = False
            print('SBMPO_full_inspection.py: Too many inspection legs required.')
            end = perf_counter()
            print('SBMPO_full_inspection.py: Inpection leg execution time = ', end - start, 'seconds')
            return att0, legs_opt, issolfound
        elif len(M_c_list) >= 3:
            if M_c_list[-1] == M_c_list[-2] == M_c_list[-3]:
                issolfound = False
                print('SBMPO_full_inspection.py: Mission completition is not advancing. Search is stopped.')
                return att0, legs_opt, issolfound
        else:
            legs_opt.append(s_opt)
        np.savez('output_data_full_insp.npz', legs_opt=legs_opt)  # can be removed and just save the database
    np.savez('output_data_full_insp.npz', legs_opt=legs_opt)  # can be removed and just save the database
    print('SBMPO_full_inspection.py: Solution FOUND!')
    end = perf_counter()
    print('SBMPO_full_inspection.py: Inpection leg execution time = ', end - start, 'seconds')
    return att0, legs_opt, issolfound

