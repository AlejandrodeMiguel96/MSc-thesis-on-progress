import numpy as np
from data_orbits import a_earth, v_earth, a_target, v_target, w_0_t, mu_earth, mu_sun
from kep2car import kep2car
from JD import JD

#######################################################################################################################
# SUN
#######################################################################################################################

#######################################################################################################################
# EARTH
#######################################################################################################################
# From keplerian elements
a0_earth = a_earth  # [km]
e0_earth = 0  # [-] rad. Circular orbit
i0_earth = 0 * np.pi/180  # [rad]. Equatorial orbit
O0_earth = 0 * np.pi/180  # [rad]. Source: Wikipedia
o0_earth = 0 * np.pi/180  # [rad]. Source: Wikipedia
# e0_earth = 0.0167086  # [-] rad. Source: Wikipedia
# i0_earth = 7.155 *np.pi/180  # [rad]. Source: Wikipedia
# O0_earth = -11.26064 * np.pi/180  # [rad]. Source: Wikipedia
# o0_earth = 114.20783 * np.pi/180  # [rad]. Source: Wikipedia
theta0_earth = 0 * np.pi/180  # [rad]
state0_earth = kep2car(a0_earth, e0_earth, i0_earth, O0_earth, o0_earth, theta0_earth, mu_sun)

# Directly in cartesian coordinates
# r0_earth = np.array([1, 0, 0]) * a_earth  # [km]
# v0_earth = np.array([0, 1, 0]) * v_earth  # [km/s] computed by assuming circular orbit!!!
# state0_earth = np.append(r0_earth, v0_earth)
#######################################################################################################################
# TARGET
#######################################################################################################################
# Orbit around the Earth
# r0_target = np.array([1, 0, 0]) * a_target  # [km]
# v0_target = np.array([0, 1, 0]) * v_target  # [km/s] computed by assuming circular orbit!!!
a0_target = a_target  # [km]
e0_target = 0  # [-] rad. Circular orbit
i0_target = 0*np.pi/180  # [rad]
O0_target = 15*np.pi/180  # [rad]
o0_target = 0*np.pi/180  # [rad]
theta0_target = 45*np.pi/180  # [rad]
state0_target = kep2car(a0_target, e0_target, i0_target, O0_target, o0_target, theta0_target, mu_earth)
# Attitude of the target wrt an inertial reference frame
# w_t_0 = w_sampling(w_t_max, 1)  # [1/s] angular velocity vector of the target
# q_t_0 = q_sampling(1)  # [-] initial attitude in unitary quaternion form describing the attitude of the target
#######################################################################################################################
# CHASER
#######################################################################################################################
# Station keeping in v-bar orbit
state0_sk = np.array([200, 0, 0, 0, 0, 0]) * 1e-3  # [km] and [km/s]
# IFO orbit
x0 = 170*1e-3  # [km]
y0 = -47*1e-3  # [km]
state0_ifo = np.array([x0, y0, 0, 0, 0, -x0 * w_0_t / 2])  # [km]&[km/s]
#######################################################################################################################
# INTEGRATION DATA
#######################################################################################################################
n_steps = 100  # number of total integration steps
# steps_not_obs = 10  # number of integration steps for each of the phases that are not observation (so more points are
steps_comp_att = 30
steps_man = 3
steps_att = 20
steps_obs = n_steps - steps_comp_att - steps_man - steps_att
# computed for the observation phase, the one we are interested in
rel_tol = 1e-3
abs_tol = 1e-6
method_int = 'DOP853'
