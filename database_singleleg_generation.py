# THIS SCRIPT GENERATES AN INSPECTION DATABASE
# THE DATABASE HAS AN INPUT AND AN OUTPUT
# THE INPUT IS THE STATE0 [w0x, w0y, w0z, q0, q1, q2, q3]
# THE OUTPUT IS s: [dv, T_leg]

#region IMPORTS
import numpy as np
from time import perf_counter
from SBMPO_single_leg import compute_opt_leg
from data_features import features
from data_ct import w_t_max
from data_initial_conditions import state0_ifo, state0_sk, state0_target, state0_earth
from leg import Leg
from data_initial_conditions import n_steps
from data_inspection import n_mesh, n_s0, n_s
#endregion

database = []  # database will have an input w_vec,q_vec and as output output_len*[dv,dt]
N_database = 100000  # size of database to be created
state0_c = state0_ifo  # initial state for an IFO orbit
state0_t = state0_target
state0_e = state0_earth
i_loops = 0
i_fails = 0
start = perf_counter()
print('database_singleleg_generation.py: Creating a database of size:', N_database)
print('database_singleleg_generation.py: Number of integration steps:', n_steps)
print('database_singleleg_generation.py, Number of initial mesh samples: ', n_s0)
print('database_singleleg_generation.py, Number of refined mesh samples: ', n_s)
print('database_singleleg_generation.py, Number of meshes: ', n_mesh)

while len(database) < N_database:
    i_loops += 1
    print('database_singleleg_generation.py: Current database size:', len(database))
    print('database_singleleg_generation.py: Starting iteration number:', i_loops)
    print('database_singleleg_generation.py: Failed iterations:', i_fails)

    leg_input, leg_output, solutionfound = compute_opt_leg(state0_c, state0_t, state0_e, w_t_max, features)

    if solutionfound:
        database.append([leg_input, leg_output])
        print('database_singleleg_generation.py: Solution added to database!')
        np.savez('database_single_leg.npz', database=database)

    else:
        i_fails += 1

end = perf_counter()
print('database_singleleg_generation.py: Total execution time = ', (end - start)/60, 'minutes')
print('shape of database', np.shape(database))
