# Standard packages
import numpy as np
from scipy.integrate import solve_ivp
from scipy import interpolate
from scipy.spatial.transform import Slerp  # for interpolating quaternions
from scipy.spatial.transform import Rotation as R  # for interpolating quaternions
# Import other files
from leg import Leg
from data_inspection import T_leg_max
from data_initial_conditions import method_int, abs_tol, rel_tol, state0_ifo, state0_target, state0_earth
import attitude_eqs
from motion_eqs import orbit
from data_chaser_target import J
from data_features import features
from data_orbits import mu_sun, mu_earth


def build_leg(inputs, outputs, state0_chaser):
    """
    Computes all the information a leg can have from the inputs and outputs of the Neural Network. The rest of the data
    (e.g. features vector, integrating conditions, states0 of target, earth etc should be the same).
    :param inputs:
    :param outputs:
    :param state0_chaser:
    :return:
    """
    att0 = inputs[0:7]
    leg = Leg(outputs[:3], outputs[3], state0_chaser)

    # INTERPOLATIONS
    # Target
    t = np.linspace(0, T_leg_max, 1000)  # t vector to integrate the orbits to be later interpolated.
    state_sun = solve_ivp(orbit, [0, t[-1]], state0_earth, args=(mu_sun,), method=method_int, t_eval=t, rtol=rel_tol,
                          atol=abs_tol)
    rr_s_vector = state_sun.y[:3, :].T  # unpack sun-earth position vector
    rr_s_vector = -rr_s_vector  # to compute earth-sun position vector

    # Compute earth-target position vector in ECI frame
    sol_target = solve_ivp(orbit, [0, t[-1]], state0_target, args=(mu_earth,), method=method_int, t_eval=t,
                           rtol=rel_tol,
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
    rr_t_interp = list(
        (rr_t_x_interp, rr_t_y_interp, rr_t_z_interp))  # list of the spline representation of each component
    vv_t_interp = list(
        (vv_t_x_interp, vv_t_y_interp, vv_t_z_interp))  # list of the spline representation of each component
    # sun
    rr_s_x_interp = interpolate.splrep(t, rr_s_vector[:, 0], s=s)
    rr_s_y_interp = interpolate.splrep(t, rr_s_vector[:, 1], s=s)
    rr_s_z_interp = interpolate.splrep(t, rr_s_vector[:, 2], s=s)
    rr_s_interp = list(
        (rr_s_x_interp, rr_s_y_interp, rr_s_z_interp))  # list of the spline representation of each component
    # Attitude
    sol = solve_ivp(attitude_eqs.att_dynamics, [0, t[-1]], att0, method=method_int, t_eval=t,
                    args=(J,), rtol=rel_tol, atol=abs_tol)
    att_vec = sol.y.T
    # Interpolate
    s = 0
    # target
    w_x_tck = interpolate.splrep(t, att_vec[:, 0], s=s)
    w_y_tck = interpolate.splrep(t, att_vec[:, 1], s=s)
    w_z_tck = interpolate.splrep(t, att_vec[:, 2], s=s)
    w_interp = list((w_x_tck, w_y_tck, w_z_tck))
    q_interp = R.from_quat(att_vec[:, 3:])
    slerp = Slerp(t, q_interp)

    leg.compute_everything(features, rr_t_interp, vv_t_interp, w_interp, slerp, rr_s_interp)
    return leg


def compare_leg(dv_leg1, t_leg1, dv_leg2, t_leg2, state0_chaser):
    """
    Function to compare legs returned from the Neural Network with respect to the ones used as inputs to it.
    :return:
    """
    leg1 = Leg(dv_leg1, t_leg1, state0_chaser)
    leg_test = Leg(dv_leg2, t_leg2, state0_chaser)

    w0 = leg1.w_vec[0, :]
    q0 = leg1.q_vec[0, :]
    att0 = [*w0, *q0]
# INTERPOLATIONS
# Target
    t = np.linspace(0, T_leg_max, 1000)  # t vector to integrate the orbits to be later interpolated.
    state_sun = solve_ivp(orbit, [0, t[-1]], state0_earth, args=(mu_sun,), method=method_int, t_eval=t, rtol=rel_tol,
                          atol=abs_tol)
    rr_s_vector = state_sun.y[:3, :].T  # unpack sun-earth position vector
    rr_s_vector = -rr_s_vector  # to compute earth-sun position vector

    # Compute earth-target position vector in ECI frame
    sol_target = solve_ivp(orbit, [0, t[-1]], state0_target, args=(mu_earth,), method=method_int, t_eval=t,
                           rtol=rel_tol,
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
    rr_t_interp = list(
        (rr_t_x_interp, rr_t_y_interp, rr_t_z_interp))  # list of the spline representation of each component
    vv_t_interp = list(
        (vv_t_x_interp, vv_t_y_interp, vv_t_z_interp))  # list of the spline representation of each component
    # sun
    rr_s_x_interp = interpolate.splrep(t, rr_s_vector[:, 0], s=s)
    rr_s_y_interp = interpolate.splrep(t, rr_s_vector[:, 1], s=s)
    rr_s_z_interp = interpolate.splrep(t, rr_s_vector[:, 2], s=s)
    rr_s_interp = list(
        (rr_s_x_interp, rr_s_y_interp, rr_s_z_interp))  # list of the spline representation of each component
# Attitude
    t_att = np.linspace(0, T_leg_max, 1000)
    sol = solve_ivp(attitude_eqs.att_dynamics, [0, t_att[-1]], att0, method=method_int, t_eval=t_att,
                    args=(J,), rtol=rel_tol, atol=abs_tol)
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

# Predicted leg
    leg_test.compute_everything(features, rr_t_interp, vv_t_interp, w_interp, slerp, rr_s_interp)
    return leg1, leg_test





