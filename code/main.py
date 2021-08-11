import time
import numpy as np
import statistics
from data_inspection import n_mesh, n_s0, n_s, deltav_min, deltav_max, T_leg_min, T_leg_max
from data_features import features
from data_orbits import mu_sun, mu_earth
from data_chaser_target import w_t_max
from data_initial_conditions import n_steps, state0_ifo, state0_earth, state0_target, state0_sk
from algorithm import initmesh, refinemesh, propagate_hills, compute_mission_complet, \
    propagate_orbit, propagate_att, update_values
from sphere_sampling import w_sampling, q_sampling

start_time = time.time()  # to measure the code running time

state0 = state0_ifo
# state0 = state0_sk
w0 = w_sampling(w_t_max, 1)[0]
q0 = q_sampling(1)[0]
state0_att = [*w0, *q0]
M_c = 0  # mission completition variable
i_leg = 0
# opt_leg = np.array([])
legs_opt = []
print('Starting SBMPO. Number of features to inspect:', len(features))
while M_c < 100:
    # CAMBIAR NOMBRE A W_0_T Y A W_T_0 PARA EVITAR CONFUSIONES?????
    # SEGUIR IMPLEMENTANDO LA ATTITUDE DEL TARGET
    S_j_all = np.array([])
    scores_all = np.array([])
    # implementar que las scores de todas las mallas cuenten de cara al total
    for j in range(n_mesh):
        scores = np.array([])
        start_time1 = time.time()  # to measure the code running time
        if j == 0:
            S_j = initmesh(deltav_min, deltav_max, T_leg_min, T_leg_max, n_s0)
        else:
            S_prev = S_j
            S_j = refinemesh(S_prev, n_s)
        for s in S_j:
            s.rr_chaser_LVLH = propagateHills(state0, n_steps, s,,  # literature calls it x: relative state vector
                               s.rr_target = propagate_orbit(mu_earth, state0_target, n_steps, s)
            s.rr_sun = propagate_orbit(mu_sun, state0_earth, n_steps, s)
            att_state = propagate_att(state0_att, n_steps, s)
            s.w_vec = att_state[:, 0:3]
            s.q_vec = att_state[:, 3:]
            s.compute_traj_score(features, n_steps,,
            scores = np.append(scores, s.score)
        S_j_all = np.append(S_j_all, S_j)
        list_scores = sorted(scores, reverse=True)
        print('scores list', list_scores[0], list_scores[1], list_scores[2])
        print('mean of scores', statistics.mean(list_scores))
        scores_all = np.append(scores_all, scores)
    list_scores = sorted(scores_all, reverse=True)
    print('total best scores', list_scores[0], list_scores[1], list_scores[2])
    s_opt = S_j_all[np.argmax(scores_all)]
    # s_opt = choose_thebestleg(S_j)  # this could be used but it iterates again the full list, so comp.more expensive
    C_opt = s_opt.score
    state0, state0_target, state0_earth, state0_att = \
        update_values(s_opt, features, [mu_earth, mu_sun], [state0, state0_target, state0_earth, state0_att])
    M_c = compute_mission_complet(features)  # TIENE QUE DEPENDER TAMBIEN DE LA LEG_OPTIMAL, O MEJOR DICHO,
    i_leg += 1
    print(i_leg, '-th leg with score:', C_opt)
    print('Mission completition: ', M_c, '%')
    print("Leg time: --- %s seconds ---" % (time.time() - start_time1))
    minutes = (time.time() - start_time1) / 60
    print("Leg time: --- %s minutes ---" % minutes)
    legs_opt.append(s_opt)
    np.savez('output_data.npz', legs_opt=legs_opt)

print("Total code time: --- %s seconds ---" % (time.time() - start_time))
minutes = (time.time() - start_time) / 60
print("Total code time: --- %s minutes ---" % minutes)
