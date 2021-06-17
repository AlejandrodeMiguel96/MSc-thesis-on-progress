import numpy as np
from data_orbits import a_earth, v_earth, a, v_target, w_0_t
from data_ct import w_t_max
from sphere_sampling import w_sampling, q_sampling
#######################################################################################################################
# EARTH
#######################################################################################################################
r0_earth = np.array([1, 0, 0]) * a_earth  # [m]
v0_earth = np.array([0, 1, 0]) * v_earth  # [m/s] computed by assuming circular orbit!!!
state0_earth = np.append(r0_earth, v0_earth)
#######################################################################################################################
# TARGET
#######################################################################################################################
# Orbit around the Earth
r0_target = np.array([1, 0, 0]) * a  # [m]
v0_target = np.array([0, 1, 0]) * v_target  # [m/s] computed by assuming circular orbit!!!
state0_target = np.append(r0_target, v0_target)
# Attitude of the target wrt an inertial reference frame
# w_t_0 = w_sampling(w_t_max, 1)  # [1/s] angular velocity vector of the target
# q_t_0 = q_sampling(1)  # [-] initial attitude in unitary quaternion form describing the attitude of the target
#######################################################################################################################
# CHASER
#######################################################################################################################
# Station keeping in v-bar orbit
state0_sk = [200, 0, 0, 0, 0, 0]  # [m] and [m/s]
# IFO orbit
x0 = 170  # [m]
y0 = -47  # [m]
state0_ifo = [x0, y0, 0, 0, 0, -x0 * w_0_t / 2]  # [m]&[m/s]
#######################################################################################################################
# INTEGRATION DATA
#######################################################################################################################
n_steps = 100  # number of total integration steps
