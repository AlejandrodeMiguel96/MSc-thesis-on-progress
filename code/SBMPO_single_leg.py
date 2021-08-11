# Import standard packages
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Slerp  # for interpolating quaternions
from scipy.spatial.transform import Rotation as R  # for interpolating quaternions
from scipy import interpolate
from time import perf_counter

# Import files
import attitude_eqs
import data_chaser_target
from data_chaser_target import w_t_max
from data_features import features
from data_inspection import n_mesh, n_s0, n_s, deltav_min, deltav_max, T_leg_min, T_leg_max
from data_initial_conditions import steps_comp_att, steps_man, steps_att, steps_obs, method_int, abs_tol, rel_tol
from algorithm import initmesh, refinemesh
from sphere_sampling import w_sampling
from LVLH2ECI import lvlh2eci


def compute_opt_leg(state0_c, rr_t_interp_list, vv_t_interp_list, rr_s_interp_list):
    """
    Computes the best (optimal) leg, in terms of propellant and number of observed features.
    :param state0_c: [rx0, ry0, rz0, vx0, vy0, vz0] initial state of the chaser
    :param rr_t_interp_list: list containing the spline interpolation representation of target rr_t_x, rr_t_y and rr_t_z
    :param vv_t_interp_list: list containing the spline interpolation representation of target vv_t_x, vv_t_y and vv_t_z
    :param rr_s_interp_list: list containing the spline interpolation representation of sun's rr_s_x, rr_s_y and rr_s_z
    :return: [att0, leg_opt, issolfound]:
    att0 is a_target vector containing the initial attitude [wx0, wy0, wz0, q00, q10, q20, q30]
    leg_opt is a_target list containing all the optimal inspection legs (class Leg).
    This is thinking about the neural network, which may take as input the target attitude, and output the optimal leg.
    """
    state0_chaser = state0_c  # initialize chaser state
    w0 = w_sampling(w_t_max, 1)[0]  # randomly sampling angular velocity initial conditions
    q0 = R.random().as_quat()
    att0 = [*w0, *q0]
    print('Starting SBMPO_single_leg.py: Number of features to inspect:', len(features))
    print('Starting SBMPO_single_leg.py: Initial chaser state:', state0_chaser)
    print('Starting SBMPO_single_leg.py: CARE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! FIXED INIT.ATT.')
    print('Starting SBMPO_single_leg.py: Initial target attitude:', att0)

# Integrate target's attitude and extrapolate it for later
    t_att = np.linspace(0, T_leg_max, 1000)
    sol = solve_ivp(attitude_eqs.att_dynamics, [0, t_att[-1]], att0, method=method_int, t_eval=t_att,
                    args=(data_chaser_target.J,), rtol=rel_tol, atol=abs_tol)
    att_vec = sol.y.T
    # Interpolate
    s = 0
    # target
    w_x_tck = interpolate.splrep(t_att, att_vec[:, 0], s=s)
    w_y_tck = interpolate.splrep(t_att, att_vec[:, 1], s=s)
    w_z_tck = interpolate.splrep(t_att, att_vec[:, 2], s=s)
    w_interp = list((w_x_tck, w_y_tck, w_z_tck))
    q_vec = R.from_quat(att_vec[:, 3:])
    slerp = Slerp(t_att, q_vec)

    start = perf_counter()  # clock to track execution time
    S_j_all = np.array([])
    scores_all = np.array([])
    for j in range(n_mesh):
        scores = np.array([])
        score_prev = -99  # initialize variable
        if j == 0:
            start_initmesh = perf_counter()
            S_j = initmesh(deltav_min, deltav_max, T_leg_min, T_leg_max, state0_chaser, n_s0)
            end_initmesh = perf_counter()
            print('SBMPO_single_leg.py: init mesh ex.time = ', end_initmesh - start_initmesh)
        else:
            # since I have change the way the Legs initialize (now it computes rr_LVLH in the initialization), do
            # I have to change something for the refine mesh? I think so, also new refined samples should be checked
            # so that the new legs are all valid
            start_refinemesh = perf_counter()
            S_prev = S_j
            S_j = refinemesh(S_prev, state0_chaser, n_s)
            end_refinemesh = perf_counter()
            print('SBMPO_single_leg.py: refine mesh ex.time = ', end_refinemesh - start_refinemesh)
        start_propagate = perf_counter()
        counter_score = 0
        counter_integration = 0
        counter_frame = 0
        for s in S_j:
            # VECTORES OBS Y T_OBS CREO QUE PUEDEN SER BORRADOS, SON PARA PLOTEAR

# Interpolate:target, sun and attitude
            s.rr_target, s.r_obs_et_vec, vv_target = s.interp_target(s.t, rr_t_interp_list, vv_t_interp_list)
            s.w_vec, s.q_vec, s.q_obs_vec = s.interp_attitude(s.t, w_interp, slerp)
            s.rr_sun, s.r_obs_es_vec = s.interp_rr_sun(s.t, rr_s_interp_list)

# Change ref.frame from LVLH to ECI
            start_frame = perf_counter()
            s.rr_chaser_ECI, s.r_obs_chaser_vec_ECI = s.get_rr_lvlh2eci(s.rr_chaser_LVLH, s.rr_target, vv_target)
            end_frame = perf_counter()
            counter_frame = end_frame-start_frame
# SCORE THE LEGS
            start_score = perf_counter()
            s.compute_traj_score(features, s.rr_target, s.rr_sun, s.rr_chaser_ECI, s.q_vec, single_leg=True)
            end_score = perf_counter()
            counter_score += end_score-start_score
            scores = np.append(scores, s.score)

# SAVE BEST LEG YET
            if s.score > score_prev:
                s_opt = s
                score_prev = s_opt.score

        end_propagate = perf_counter()
        print('SBMPO_single_leg.py: propagate ex.time = ', end_propagate - start_propagate)
        scores_all = np.append(scores_all, scores)
        scores_all_sorted = np.array(sorted(scores_all, reverse=True))
        print('SBMPO_single_leg.py: scores', scores_all_sorted[0:3])

    print('SBMPO_single_leg.py: Optimal leg = ', s_opt.dv, s_opt.t_leg)
    print('SBMPO_single_leg.py: Optimal leg score = ', s_opt.score)
    print('SBMPO_single_leg.py: Optimal leg Delta-v = ', np.linalg.norm(s_opt.dv))
    print('SBMPO_single_leg.py: Obs.time for each feature = ', s_opt.t_useful_feat_list)
    leg_input = att0
    leg_output = [*s_opt.dv, s_opt.t_leg]
    if s_opt.score == 0:
        solutionfound = False
        print('SBMPO_single_leg.py.py: All scores are 0. Solution could NOT been found')
        end = perf_counter()
        print('SBMPO_single_leg.py.py: Inpection leg execution time = ', end - start, 'seconds')
        return leg_input, leg_output, solutionfound, s_opt
    else:
        solutionfound = True
        np.savez('output_data_single_leg.npz', leg_opt=s_opt)  # can be removed and just save the database
        print('SBMPO_single_leg.py.py: Solution FOUND!')
        end = perf_counter()
        print('SBMPO_single_leg.py.py: Inpection leg execution time = ', end - start, 'seconds')
        return leg_input, leg_output, solutionfound, s_opt

