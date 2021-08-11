# Standard packages
import numpy as np
from scipy.integrate import solve_ivp
from scipy import interpolate

# Data files
from data_chaser_target import m_c, F_c
from data_inspection import T_comp, T_att, r_obs_min, r_obs_max, w_gamma, w_deltav, eps_T, leg_incentive, r_min, \
    r_escape
from useful_functions import computeangle, compute_rr_target2sun, isinrange, iseclipsed, isvalidmission
from data_initial_conditions import steps_comp_att, steps_att, steps_man, steps_obs, abs_tol, rel_tol, method_int, w_0_t

# Function files
from motion_eqs import hill_eqs
from LVLH2ECI import lvlh2eci
class Leg:
    """
    An observation leg is completely defined by the Delta-v and the time necessary to perform it.
    A leg has the following attributes:
    dv: [dvx, dvy, dvz] necessary to initialise a_target Leg.
    t_leg: necessary to initialise a_target Leg.
    t_comp: computation of the leg time (imported from data)
    t_att: attitude acquisition of the target (imported from data)
    t_man: powered manoeuvre time (computed in initialisation)
    t_obs: observation time available (computed in initialisation)
    t_useful: actual useful time where conditions allow for observation (to be computed later)
    u: chaser engine acceleration vector (computed in initialisation)
    score: score of the leg depending on different parameters (to be computed later)
    """
    # def __init__(self, dv, t_leg):
    def __init__(self, dv, t_leg, state0_chaser):
    # def __init__(self, dv, t_leg, state0_chaser, state0_target, state0_earth):
        self.dv = dv
        self.t_leg = t_leg
        self.state0_chaser = state0_chaser

# important leg times
        self.t_comp = T_comp
        self.t_att = T_att
        self.t_man = self.compute_t_man()  # Minimum time to complete the maneouvre (by using full thrust)
        self.t_obs = self.compute_t_obs_leg()  # Time left for observation after the previous leg operations
        self.t1, self.t2, self.t3, self.t4, self.t = self.compute_times()
        self.t_step_obs = self.t4[1] - self.t4[0]  # integration step of the observation phase


        self.t_obs_vec = self.t4+(self.t_comp+2*self.t_att+self.t_man)  # PROBABLY COULD BE REMOVED
        self.t_useful = 'TBD'  # To be determined. Observation time actually useful for inspecting.
        self.t_useful_vec = []  # this can be deleted, just for graphs

        self.u = self.compute_u()
        self.score = 'TBD'  # To be determined. Score of the leg based on ilumination conditions, time, safety...

# computing trajectories
        self.rr_chaser_LVLH, self.r_obs_chaser_vec_LVLH, self.integration_status = self.propagate_hills()
        self.rr_chaser_ECI = 'TBD'  # position vector of chaser wrt earth (ECI frame) [rx,ry,rz]
        self.rr_target = 'TBD'  # position vector of taget wrt earth (ECI frame) [rx,ry,rz]
        self.rr_sun = 'TBD'  # position vector of sun wrt earth (ECI frame) [rx,ry,rz]
        self.rr_ts_ECI = 'TBD'  # position vector of target wrt sun (ECI frame) [rx,ry,rz]
        self.w_vec = 'TBD'  # angular velocity of the full duration of the leg
        self.q_vec = 'TBD'  # quaternions of the full duration of the leg

        self.r_obs_chaser_vec_ECI = 'TBD'  # earth-chaser position vector during observation phase
        self.r_obs_et_vec = 'TBD'  # earth-target position vector during observation phase
        self.r_obs_es_vec = 'TBD'  # earth-sun position vector during observation phase
        self.r_obs_ts_vec = 'TBD'  # target-sun position vector during observation phase
        self.q_obs_vec = 'TBD'  # quaternions during observation part of the leg

        self.r_chaser_lvlh_vec_useful = 'TBD'
        self.r_chaser_eci_vec_useful = 'TBD'
        self.ts_vec_useful = 'TBD'
        self.q_vec_useful = 'TBD'

        self.r_useful_vec_lvlh = []  # this can be deleted, just for graphs

        self.t_useful_feat_list = []  # list containing the time each feature has been inspected for.

    def compute_t_man(self):
        """
        Computes the manoeuvre time required to obtain a_target certain dv.
        HYPOTHESIS: the manoeuvre is performed at full thrust F_c.
        :return: minimum time required for the manoeuvre
        """
        return np.linalg.norm(self.dv) * m_c / F_c

    def compute_u(self):
        """
        Computes the engine acceleration vector given a_target deltav.
         u = dv/t
        :return: [ux, uy, uz] np.array containing the chaser control acceleration [m/s**2]
        """
        if self.t_man == 0:
            return np.array([0, 0, 0])
        else:
            return self.dv / self.t_man

    def compute_t_obs_leg(self):
        """
        Computes the remaining observation time of an inspection leg (after the previous action have been completed).
        :return: T_obs_leg
        """
        return self.t_leg - (self.t_comp + self.t_att + self.compute_t_man() + self.t_att)

    def compute_times(self):
        """

        :return:
        """
        t1 = np.linspace(0, self.t_comp + self.t_att, steps_comp_att)  # integration time part 1 (comp+att.acq)
        if self.t_man == 0:
            t2 = np.array([])
            t3 = np.linspace(0, self.t_att, steps_att+steps_man)
        else:
            t2 = np.linspace(0, self.t_man, steps_man)  # integration time part 2 (manoeuvre)
            t3 = np.linspace(0, self.t_att, steps_att)  # integration time part 3 (point.acq+observation)
        t4 = np.linspace(0, self.t_obs, steps_obs)  # integration time part 4 (observation)
        t = np.append(
            np.append(
                np.append(t1, t2 + self.t_comp + self.t_att), t3 + self.t_comp + self.t_att + self.t_man), t4 +
                                                                                            (self.t_leg - self.t_obs))
        return t1, t2, t3, t4, t

    def propagate_hills(self):
        """
        Propagates the observation leg trajectory motion by the Clohessy-Wiltshire (CW) dynamics.
        :return: self.rr_chaser_LVLH, self.r_obs_chaser_vec_LVLH, self.integration_status
        """
        no_u = [0, 0, 0]

        def event_validmission(x, y):
            """
            Event function to stop integration in case of an event is found.
            The event is that the mission is invalid (see useful_functions.py --> isvalidmission function
            """
            if r_min <= np.linalg.norm([y[0], y[1], y[2]]) <= r_escape:
                return 1
            else:
                return 0

        event_validmission.terminal = True

        state1 = solve_ivp(fun=lambda x, y: hill_eqs(x, y, no_u, w_0_t), t_span=[0, self.t1[-1]],
                           y0=self.state0_chaser, method=method_int, t_eval=self.t1, rtol=rel_tol, atol=abs_tol,
                           events=event_validmission)
        if state1.status == 1:  # checks if integration has been stoped by the validmission event
            return np.zeros((len(self.t), 3)), np.zeros((len(self.t4), 3)), 1

        if self.t_man == 0:  # because when dv = [0,0,0] --> t_man = 0, so there is no impulse manoeuvre
            state2 = state1
        else:
            state2 = solve_ivp(fun=lambda x, y: hill_eqs(x, y, self.dv, w_0_t), t_span=[0, self.t2[-1]],
                               y0=state1.y[:, -1], method=method_int, t_eval=self.t2, rtol=rel_tol, atol=abs_tol,
                               events=event_validmission)
            if state2.status == 1:
                return np.zeros((len(self.t), 3)), np.zeros((len(self.t4), 3)), 1

        state3 = solve_ivp(fun=lambda x, y: hill_eqs(x, y, no_u, w_0_t), t_span=[0, self.t3[-1]],
                           y0=state2.y[:, -1], method=method_int, t_eval=self.t3, rtol=rel_tol, atol=abs_tol,
                           events=event_validmission)
        if state3.status == 1:
            return np.zeros((len(self.t), 3)), np.zeros((len(self.t4), 3)), 1

        state4 = solve_ivp(fun=lambda x, y: hill_eqs(x, y, no_u, w_0_t), t_span=[0, self.t4[-1]],
                           y0=state3.y[:, -1], method=method_int, t_eval=self.t4, rtol=rel_tol, atol=abs_tol,
                           events=event_validmission)
        if state4.status == 1:
            return np.zeros((len(self.t), 3)), np.zeros((len(self.t4), 3)), 1

        if self.t_man == 0:
            traj = np.append(np.append(state1.y, state3.y, axis=1), state4.y, axis=1)
        else:
            traj = np.append(np.append(np.append(state1.y, state2.y, axis=1), state3.y, axis=1), state4.y, axis=1)

        return traj[:3, :].T, state4.y[:3, :].T, 0

    @staticmethod
    def interp_target(t_vector, tck_list_rr_target, tck_list_vv_target):
        """
        Interpolates points from the target's orbit in ECI frame given new time points as t_vector
         and the interpolation spline as tck_list_rr_target and tck_list_vv_target.
        :param t_vector: n-vector containing the new time points where the orbit needs to be evaluated
        :param tck_list_rr_target: (3 dimension list) result of "interpolate.splrep" method. Defines the interpolated
        position vector of the target in ECI frame curve parametrically.
        :param tck_list_vv_target: (3 dimension list) result of "interpolate.splrep" method. Defines the interpolated
        velocity vector of the target in ECI frame curve parametrically.
        :return:
            rr_target [n, 3] vector with the new interpolated points for each component
            rr_obs_target [n, 3] vector with the new interpolated points for each component
            vv_target [n, 3] vector with the new interpolated points for each component
        """
        rr_x = interpolate.splev(t_vector, tck_list_rr_target[0], der=0)
        rr_y = interpolate.splev(t_vector, tck_list_rr_target[1], der=0)
        rr_z = interpolate.splev(t_vector, tck_list_rr_target[2], der=0)
        vv_x = interpolate.splev(t_vector, tck_list_vv_target[0], der=0)
        vv_y = interpolate.splev(t_vector, tck_list_vv_target[1], der=0)
        vv_z = interpolate.splev(t_vector, tck_list_vv_target[2], der=0)
        return np.vstack((rr_x, rr_y, rr_z)).T, np.vstack((rr_x, rr_y, rr_z)).T[-steps_obs:, :], np.vstack((vv_x, vv_y,
                                                                                                            vv_z)).T

    @staticmethod
    def interp_rr_sun(t_vector, tck_list_rr_sun):
        """
        Interpolates points from the sun's orbit in ECI frame given new time points as t_vector
         and the interpolation spline as tck_list_rr_sun.
        :param t_vector: n-vector containing the new time points where the orbit needs to be evaluated
        :param tck_list_rr_sun: (3 dimension list) result of "interpolate.splrep" method. Defines the interpolated
        position vector of the sun orbit in ECI frame curve parametrically.
        :return:
            rr_sun [n, 3] vector with the new interpolated points for each component
            rr_obs_sun [n, 3] vector with the new interpolated points for each component
        """
        rr_x = interpolate.splev(t_vector, tck_list_rr_sun[0], der=0)
        rr_y = interpolate.splev(t_vector, tck_list_rr_sun[1], der=0)
        rr_z = interpolate.splev(t_vector, tck_list_rr_sun[2], der=0)
        return np.vstack((rr_x, rr_y, rr_z)).T, np.vstack((rr_x, rr_y, rr_z)).T[-steps_obs:, :]

    @staticmethod
    def interp_attitude(t_vector, tck_list_w, slerp):
        """
        Interpolates points from the target's orbit in ECI frame given new time points as t_vector
         and the interpolation spline as tck_list_rr_target and tck_list_vv_target.
        :param t_vector: n-vector containing the new time points where the orbit needs to be evaluated
        :param tck_list_w: (3 dimension list) result of "interpolate.splrep" method. Defines the interpolated
        angular velocity vector of the target in ECI frame curve parametrically.
        :param slerp:
        :return:
            w_target [n, 3] vector with the new interpolated points for each component
            q [n, 3] vector with the new interpolated quaternions
            q_obs [n, 3] vector with the new interpolated quaternions during observation phase.
        """
        w_x = interpolate.splev(t_vector, tck_list_w[0], der=0)
        w_y = interpolate.splev(t_vector, tck_list_w[1], der=0)
        w_z = interpolate.splev(t_vector, tck_list_w[2], der=0)
        q_vec = slerp(t_vector).as_quat()
        return np.vstack((w_x, w_y, w_z)).T, q_vec, q_vec[-steps_obs:, :]

    @staticmethod
    def get_rr_lvlh2eci(rr_lvlh, rr_reference, vv_reference):
        """
        From the position vectors in LVLH frame returns the position vectors in the ECI frame.
        :param rr_lvlh: position vector of chaser orbit in LVLH frame (around target)
        :param rr_reference: position vectors in ECI frame of the LVLH frame
        :param vv_reference: velocity vectors in ECI frame of the LVLH frame
        :return:
            rr_chaser_ECI [n, 3]: position vectors in ECI frame
            r_obs_chaser_vec_ECI [n, 3]: position vectors in ECI frame during observation period
        """
        rr_eci = np.zeros((len(rr_lvlh), 3))
        for index, (rr, rr_ref, vv_ref) in enumerate(zip(rr_lvlh, rr_reference, vv_reference)):
            rr_eci[index, :] = lvlh2eci(rr, rr_ref, vv_ref)
        return rr_eci, rr_eci[-steps_obs:, :]

    def get_useful_leg_vectors(self, r_obs_es_vec, r_obs_et_vec, r_obs_chaser_vec_eci, q_obs_vec):
        """
        Computes the vectors r, es, te and q of the observation time period under proper conditions (i.e fulfill
        observation conditions such as ilumination, proximity...) to inspect the target.
        :return:
        """
        # from the observation time period, evaluates how many time steps are under proper inspection conditions.
        r_chaser_lvlh_vec_useful = self.r_obs_chaser_vec_LVLH
        r_chaser_eci_vec_useful = r_obs_chaser_vec_eci
        ts_vec_useful = self.rr_ts_ECI[-steps_obs:, :]
        q_vec_useful = q_obs_vec
        t_useful_vec = self.t4
        i = 0
        for r, es, et in zip(self.r_obs_chaser_vec_LVLH, r_obs_es_vec, r_obs_et_vec):
            if not isinrange(r):
                # removes those instants where conditions are not met
                r_chaser_lvlh_vec_useful = np.delete(r_chaser_lvlh_vec_useful, i, axis=0)
                r_chaser_eci_vec_useful = np.delete(r_chaser_eci_vec_useful, i, axis=0)
                ts_vec_useful = np.delete(ts_vec_useful, i, axis=0)
                q_vec_useful = np.delete(q_vec_useful, i, axis=0)
                t_useful_vec = np.delete(t_useful_vec, i, axis=0)
            elif iseclipsed(es, et):
                r_chaser_lvlh_vec_useful = np.delete(r_chaser_lvlh_vec_useful, i, axis=0)
                r_chaser_eci_vec_useful = np.delete(r_chaser_eci_vec_useful, i, axis=0)
                ts_vec_useful = np.delete(ts_vec_useful, i, axis=0)
                q_vec_useful = np.delete(q_vec_useful, i, axis=0)
                t_useful_vec = np.delete(t_useful_vec, i, axis=0)
            else:
                i += 1
        self.t_useful = len(t_useful_vec) * self.t_step_obs

        self.t_useful_vec = t_useful_vec  # this variable can be removed from attributes, just for graphs
        self.r_useful_vec_lvlh = r_chaser_lvlh_vec_useful  # this can be deleted, useful just for drawing graphs

        return r_chaser_lvlh_vec_useful, r_chaser_eci_vec_useful, ts_vec_useful, q_vec_useful

    @staticmethod
    def f_2_integrate(ts_eci, rr_chaser_eci, rr_chaser_lvlh):
        """
        Function with variables f(t) and gamma(t) to integrate.
        :param ts_eci:  target-Sun position vector (ECI frame).
        :param rr_chaser_eci:  chaser position vector (ECI frame).
        :param rr_chaser_lvlh:  target-chaser position vector (LVLH frame).
        :return: function to be integrated
        """
        gamma = computeangle(ts_eci, rr_chaser_eci)  # angle between the Sun-target-chaser
        r_norm_lvlh = np.linalg.norm(rr_chaser_lvlh)  # because r_obs_min&max parameters are wrt chaser(LVLH frame)
        if r_norm_lvlh < r_obs_min:
            f = r_norm_lvlh / r_obs_min
        elif r_norm_lvlh > r_obs_max:
            f = (r_obs_max / r_norm_lvlh) ** 3
        else:
            f = 1
        return f * (np.pi - gamma) / np.pi

    # def compute_traj_score(self, features, single_leg=False):
    def compute_traj_score(self, features, rr_target, rr_sun, rr_chaser_eci, q_vec, single_leg=True):
        """
        ¡¡¡¡¡¡NEEDS TO BE FINISHED. ADD scenarios to isvalidmission() function!!!!!!!
        Compute the score of the trajectory.
        It should be 0 under the following scenarios:
        - Chaser reaches a_target maximum distance larger than r_escape from the target during its trajectory.
        - Chaser's and solar arrays' angular motion exceed the specified RW and SADM limits.
        - Maneouvre is even partially executed during eclipse.
        - Trajectory is unsafe.
        :param features: [1xn_p] vector containing type class Feature objects.
        :param rr_target: position vectors of target's orbit around Earth (ECI frame)
        :param rr_sun: position vectors of sun's orbit around Earth (ECI frame)
        :param rr_chaser_eci: position vectors of chaser's orbit around target (ECI frame)
        :param q_vec: quaternion vectors of target
        :param single_leg: bool variable that affects the score function. TRUE for considering just one inspection
        leg rather than a_target whole inspection (so the optimal leg is the one that observes more features during more time
        instead of the sequence of legs that complete the full observation of all features).
        :return: score of the leg
        """
        # evaluates if the mission is valid (in terms of safety, max.distance...). If not, score = 0
        valid_mission = isvalidmission(self.rr_chaser_LVLH)  # function not completed with full cases YET.
        if valid_mission:
            self.rr_ts_ECI = compute_rr_target2sun(rr_target, rr_sun)
            integral = 0
            for ts_ECI, rr_ECI, rr_LVLH in zip(self.rr_ts_ECI, rr_chaser_eci, self.rr_chaser_LVLH):
                integral += self.f_2_integrate(ts_ECI, rr_ECI, rr_LVLH)
            integral_term = w_gamma * integral

            self.r_chaser_lvlh_vec_useful, self.r_chaser_eci_vec_useful, self.ts_vec_useful, self.q_vec_useful =\
                self.get_useful_leg_vectors(rr_sun[-steps_obs:, :], rr_target[-steps_obs:, :],
                                            rr_chaser_eci[-steps_obs:, :], q_vec[-steps_obs:, :])

            sum_term = 0
            for i in features:
                t_useful_feat = i.compute_t_useful_feat(self, self.r_chaser_eci_vec_useful, self.ts_vec_useful,
                                                        self.q_vec_useful)
                self.t_useful_feat_list.append(t_useful_feat)  # I THINK IT IS JUST FOR CHECKING AND PLOTS, not necessary

                if not single_leg:  # this computes the score function for a whole inspection (many optimal legs)
                    if i.t_obs_feat > i.t_goal:
                        t_obs_feat = i.t_goal  # to avoid a target negative sqrt
                    else:
                        t_obs_feat = i.t_obs_feat
                    sum_term += np.sqrt(i.t_goal / ((1 + eps_T) * i.t_goal - t_obs_feat)) * t_useful_feat
                else:  # this computes the score function looking for the single best inspection leg possible
                    m = leg_incentive
                    if t_useful_feat > i.t_goal:
                        t_useful_feat = i.t_goal
                    if t_useful_feat == 0:
                        m = 0
                    sum_term += t_useful_feat + m

            self.score = w_deltav / (w_deltav + np.linalg.norm(self.dv)) * (integral_term + sum_term)

        else:
            self.score = -1

    def compute_everything(self, features, rr_t_interp_list, vv_t_interp_list, w_interp, slerp, rr_s_interp_list):
        """

        :param features:
        :param rr_t_interp_list:
        :param vv_t_interp_list:
        :param w_interp:
        :param slerp:
        :param rr_s_interp_list:
        :return:
        """
        self.rr_target, self.r_obs_et_vec, vv_target = self.interp_target(self.t, rr_t_interp_list, vv_t_interp_list)
        self.w_vec, self.q_vec, self.q_obs_vec = self.interp_attitude(self.t, w_interp, slerp)
        self.rr_sun, self.r_obs_es_vec = self.interp_rr_sun(self.t, rr_s_interp_list)

        # Change ref.frame from LVLH to ECI
        self.rr_chaser_ECI, self.r_obs_chaser_vec_ECI = self.get_rr_lvlh2eci(self.rr_chaser_LVLH, self.rr_target,
                                                                             vv_target)

        # SCORE THE LEGS
        self.compute_traj_score(features, self.rr_target, self.rr_sun, self.rr_chaser_ECI, self.q_vec, single_leg=True)
