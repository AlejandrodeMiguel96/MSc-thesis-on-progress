# THIS SCRIPT GENERATES AN INSPECTION DATABASE
# THE DATABASE HAS AN INPUT AND AN OUTPUT
# THE INPUT IS THE STATE0 [w0x, w0y, w0z, q0, q1, q2, q3]
# THE OUTPUT IS s: [dv, T_leg]

# Import standard packages
import numpy as np
from scipy.integrate import solve_ivp
from scipy import interpolate
from time import perf_counter

# Import files
import motion_eqs
from data_initial_conditions import state0_ifo, state0_sk, state0_target
from data_initial_conditions import n_steps, abs_tol, rel_tol, method_int, state0_earth
from data_inspection import n_mesh, n_s0, n_s, T_leg_max
from data_orbits import mu_earth, mu_sun
from SBMPO_single_leg import compute_opt_leg
from motion_eqs import orbit

start = perf_counter()  # to measure execution time
# Initializations
database = []  # database will have an input w_vec,q_vec and as output output_len*[dv,dt]
database_consistency = []  # only to save results and study consistency
N_database = 100000  # size of database to be created
state0_c = state0_ifo  # initial state for an IFO orbit

#region Interpolation
# Compute earth-sun position vector in ECI frame
t = np.linspace(0, T_leg_max, 1000)  # t vector to integrate the orbits to be later interpolated.
state_sun = solve_ivp(orbit, [0, t[-1]], state0_earth, args=(mu_sun,), method=method_int, t_eval=t, rtol=rel_tol,
                      atol=abs_tol)
rr_s_vector = state_sun.y[:3, :].T  # unpack sun-earth position vector
rr_s_vector = -rr_s_vector  # to compute earth-sun position vector

# Compute earth-target position vector in ECI frame
sol_target = solve_ivp(orbit, [0, t[-1]], state0_target, args=(mu_earth,), method=method_int, t_eval=t, rtol=rel_tol,
                       atol=abs_tol)
x_t_vector = sol_target.y.T  # [n, 6] state vector containing the earth-target position (ECI frame)

# Interpolate
s = 0
# target
rr_t_x_interp = interpolate.splrep(t, x_t_vector[:, 0], s=s)
rr_t_y_interp = interpolate.splrep(t, x_t_vector[:, 1], s=s)
rr_t_z_interp = interpolate.splrep(t, x_t_vector[:, 2], s=s)
vv_t_x_interp = interpolate.splrep(t, x_t_vector[:, 3], s=s)
vv_t_y_interp = interpolate.splrep(t, x_t_vector[:, 4], s=s)
vv_t_z_interp = interpolate.splrep(t, x_t_vector[:, 5], s=s)
rr_t_interp = list((rr_t_x_interp, rr_t_y_interp, rr_t_z_interp))  # list of the spline representation of each component
vv_t_interp = list((vv_t_x_interp, vv_t_y_interp, vv_t_z_interp))  # list of the spline representation of each component
# sun
rr_s_x_interp = interpolate.splrep(t, rr_s_vector[:, 0], s=s)
rr_s_y_interp = interpolate.splrep(t, rr_s_vector[:, 1], s=s)
rr_s_z_interp = interpolate.splrep(t, rr_s_vector[:, 2], s=s)
rr_s_interp = list((rr_s_x_interp, rr_s_y_interp, rr_s_z_interp))  # list of the spline representation of each component
#endregion

i_loops = 0
i_fails = 0
print('database_singleleg_generation.py: Creating a_target database of size:', N_database)
print('database_singleleg_generation.py: Number of integration steps:', n_steps)
print('database_singleleg_generation.py, Number of initial mesh samples: ', n_s0)
print('database_singleleg_generation.py, Number of refined mesh samples: ', n_s)
print('database_singleleg_generation.py, Number of meshes: ', n_mesh)

while len(database) < N_database:
    i_loops += 1
    print('database_singleleg_generation.py: Current database size:', len(database))
    print('database_singleleg_generation.py: Starting iteration number:', i_loops)
    print('database_singleleg_generation.py: Failed iterations:', i_fails)

    leg_input, leg_output, solutionfound, leg = compute_opt_leg(state0_c, rr_t_interp, vv_t_interp, rr_s_interp)
    # leg_input, leg_output, solutionfound = compute_opt_leg(state0_c, rr_t_interp, vv_t_interp, rr_s_interp)

    if solutionfound:
        database.append([leg_input, leg_output])
        database_consistency.append(leg)
        print('database_singleleg_generation.py: Solution added to database!')
        np.savez('database_single_leg.npz', database=database)
        np.savez('database_consistency.npz', database=database_consistency)

    else:
        i_fails += 1

end = perf_counter()
print('database_singleleg_generation.py: Total execution time = ', (end - start)/60, 'minutes')
print('shape of database', np.shape(database))
