# THIS SCRIPT GENERATES AN INSPECTION DATABASE
# THE DATABASE HAS AN INPUT AND AN OUTPUT
# THE INPUT IS THE STATES0
# THE OUTPUT IS s: [dv, T_leg]

#region IMPORTS
import numpy as np
from time import perf_counter
from SBMPO_full_inspection import compute_inspection
from data_features import features
from data_chaser_target import w_t_max
from data_initial_conditions import state0_ifo, state0_sk, state0_target, state0_earth
from leg import Leg
from data_initial_conditions import n_steps
from data_inspection import n_mesh, n_s0, n_s
#endregion

database = []  # database will have an input w_vec,q_vec and as output output_len*[dv,dt]
N_database = 1  # size of database
state0_c = state0_ifo
# state0_c = state0_sk
state0_t = state0_target
state0_e = state0_earth
output_len = 8  # max.number of inspection legs per inspection
i_loops = 1
i_fails = 0
start = perf_counter()
print('database_fullinspection_generation.py: Creating a_target database of size:', N_database)
print('database_fullinspection_generation.py: Number of integration steps:', n_steps)
print('database_fullinspection_generation.py, Number of initial mesh samples: ', n_s0)
print('database_fullinspection_generation.py, Number of refined mesh samples: ', n_s)
print('database_fullinspection_generation.py, Number of meshes: ', n_mesh)

while len(database) < N_database:
    print('database_fullinspection_generation.py: Current database size:', len(database))
    print('database_fullinspection_generation.py: Starting iteration number:', i_loops)
    print('database_fullinspection_generation.py: Failed iterations:', i_fails)
    for f in features:
        f.t_obs_feat = 0  # reset the observation time of each feature for each new database entry

    att0, opt_legs, islegfound = compute_inspection(state0_c, state0_t, state0_e, w_t_max, features, output_len)
    if islegfound:
        if len(opt_legs) < output_len:
            for i in range(output_len - len(opt_legs)):
                opt_legs.append(Leg([0, 0, 0], 0))  # to fill the output data up to the predetermined length
        database.append([att0, opt_legs])
        print('database_fullinspection_generation.py: Solution added!')
    else:
        i_fails += 1
    i_loops += 1
    np.savez('database_full_insp.npz', database=database)
end = perf_counter()

print('database_fullinspection_generation.py: Total execution time = ', (end - start)/60, 'minutes')
print('length of database', np.shape(database))
# print('database[0]', database[0])
