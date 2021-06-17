import numpy as np
from data_ct import m_c, F_c
from data_inspection import T_comp, T_att, r_obs_min, r_obs_max, w_gamma, w_deltav, eps_T, leg_incentive
from useful_functions import computeangle, compute_targetsun, isinrange, iseclipsed, isvalidmission


class Leg:
    """
    An observation leg is completely defined by the Delta-v and the time necessary to perform it.
    A leg has the following attributes:
    dv: [dvx, dvy, dvz] necessary to initialise a Leg.
    t_leg: necessary to initialise a Leg.
    t_comp: computation of the leg time (imported from data)
    t_att: attitude acquisition of the target (imported from data)
    t_man: powered manoeuvre time (computed in initialisation)
    t_obs: observation time available (computed in initialisation)
    t_useful: actual useful time where conditions allow for observation (to be computed later)
    u: chaser engine acceleration vector (computed in initialisation)
    score: score of the leg depending on different parameters (to be computed later)
    """
    def __init__(self, dv, t_leg):
        self.dv = dv
        self.t_leg = t_leg
        self.t_comp = T_comp
        self.t_att = T_att
        self.t_man = self.compute_t_man()  # Minimum time to complete the maneouvre (by using full thrust)
        self.t_obs = self.compute_t_obs_leg()  # Time left for observation after the previous leg operations
        self.t_obs_vec = []  # Vector of the observation phase integrated time
        self.t_useful = 'TBD'  # To be determined. Observation time actually useful for inspecting.
        self.t_useful_vec = []  # this can be deleted, just for graphs
        self.u = self.compute_u()
        self.score = 'TBD'  # To be determined. Score of the leg based on ilumination conditions, time, safety...

        # are this ones correct/useful?
        # pos.vectors during the observation part of the leg (without considering yet eclipses,fov,ilumination...)
        self.trajr = np.array([])  # full trajectory of chaser wrt target [rx,ry,rz,vx,vy,vz]
        self.trajt = np.array([])  # full trajectory of target wrt earth
        self.traje = np.array([])  # full trajectory of earth wrt sun
        self.w_vec = np.array([])  # angular velocity of the full duration of the leg
        self.q_vec = np.array([])  # quaternions of the full duration of the leg
        self.r_obs_vec = np.array([])  # target-chaser
        self.te_obs_vec = np.array([])  # target-earth
        self.es_obs_vec = np.array([])  # earth-sun
        self.q_obs_vec = np.array([])  # quaternions during observation part of the leg
        self.t_step_obs = 0  # integration step of the observation phase (useful for computing t_useful_feature)
        self.r_useful_vec = []  # this can be deleted, just for graphs
        self.t = []  # this can be deleted, just for graphs

        self.t_useful_feat_list = []  # list containing the time each feature has been inspected for.

    def compute_t_man(self):
        """
        Computes the manoeuvre time required to obtain a certain dv.
        HYPOTHESIS: the manoeuvre is performed at full thrust F_c.
        :return: minimum time required for the manoeuvre
        """
        return np.linalg.norm(self.dv) * m_c / F_c

    def compute_u(self):
        """
        Computes the engine acceleration vector given a deltav.
         u = dv/t
        :return: [ux, uy, uz] np.array containing the chaser control acceleration [m/s**2]
        """
        t = self.t_man
        if t == 0:
            u = np.array([0, 0, 0])
        else:
            u = self.dv / t
        return u

    def compute_t_obs_leg(self):
        """
        Computes the remaining observation time of an inspection leg (after the previous action have been completed).
        :return: T_obs_leg
        """
        return self.t_leg - (self.t_comp + self.t_att + self.compute_t_man() + self.t_att)

    def compute_t_and_r_useful_leg_vectors(self):
        """
        Computes the observation time in actual conditions to inspect the target (after the previous action have been
        completed) and the position vectors r, te and es, during that period of time.
        It should be useful (i.e fulfill observation conditions such as ilumination, proximity...)
        :return: r, es and te vectors with the instants where inspection conds. that are not fulfilled are removed.
        """
        # from the observation time available, it evaluates how much from that time the inspection can be actually done.
        r_vec_useful = self.r_obs_vec
        te_vec_useful = self.te_obs_vec
        es_vec_useful = self.es_obs_vec
        q_vec_useful = self.q_obs_vec
        et_vectors_obs = - self.te_obs_vec  # computes Earth-target vector from target-Earth vector
        t_useful_vec = self.t_obs_vec
        self.t_step_obs = self.t_obs_vec[1] - self.t_obs_vec[0]  # step size of the integration
        i = 0
        for r, es, et in zip(self.r_obs_vec, self.es_obs_vec, et_vectors_obs):
            if not isinrange(r):
                r_vec_useful = np.delete(r_vec_useful, i, axis=0)  # removes those instants where conditions are not met
                es_vec_useful = np.delete(es_vec_useful, i, axis=0)
                te_vec_useful = np.delete(te_vec_useful, i, axis=0)
                q_vec_useful = np.delete(q_vec_useful, i, axis=0)
                t_useful_vec = np.delete(t_useful_vec, i, axis=0)
            elif iseclipsed(es, et):
                r_vec_useful = np.delete(r_vec_useful, i, axis=0)
                es_vec_useful = np.delete(es_vec_useful, i, axis=0)
                te_vec_useful = np.delete(te_vec_useful, i, axis=0)
                q_vec_useful = np.delete(q_vec_useful, i, axis=0)
                t_useful_vec = np.delete(t_useful_vec, i, axis=0)
            else:
                i += 1
        self.t_useful = len(t_useful_vec) * self.t_step_obs
        self.t_useful_vec = t_useful_vec  # this can be deleted, just for graphs
        self.r_useful_vec = r_vec_useful  # this can be deleted, just for graphs
        return r_vec_useful, es_vec_useful, te_vec_useful, q_vec_useful

    def f_2_integrate(self, ts, r):  # include it in compute_traj_score function or put somewhere else???
        """
        Function with variables f(t) and gamma(t) to integrate
        :param ts:  target-Sun position vector
        :param r:  target-chaser position vector
        :return: function to be integrated
        """
        gamma = computeangle(ts, r)  # angle between the Sun-target-chaser
        r_norm = np.linalg.norm(r)
        if r_norm < r_obs_min:
            f = r_norm / r_obs_min
        elif r_norm > r_obs_max:
            f = (r_obs_max / r_norm) ** 3
        else:
            f = 1
        return f * (np.pi - gamma) / np.pi

    def compute_traj_score(self, features, single_leg=False):
        """
        ¡¡¡¡¡¡NEEDS TO BE FINISHED. ADD scenarios to isvalidmission() function!!!!!!!
        Compute the score of the trajectory.
        It should be 0 under the following scenarios:
        - Chaser reaches a maximum distance larger than r_escape from the target during its trajectory.
        - Chaser's and solar arrays' angular motion exceed the specified RW and SADM limits.
        - Maneouvre is even partially executed during eclipse.
        - Trajectory is unsafe.
        :param features: [1xn_p] vector containing type class Feature objects.
        :param single_leg: bool variable that affects the score function. TRUE for considering just one inspection
        leg rather than a whole inspection (so the optimal leg is the one that observes more features during more time
        instead of the sequence of legs that complete the full observation of all features).
        :return: score of the leg
        """
        # evaluates if the mission is valid (in terms of safety, max.distance...). If not, score = 0
        valid_mission = isvalidmission(self.trajr[:, 0:3])  # isvalidmission NEEDS TO BE fully COMPLETED WITH ALL THE CASES
        if valid_mission:
            ts_vectors = compute_targetsun(self.trajt[:, 0:3], self.traje[:, 0:3])
            integral = 0
            for i, j in zip(ts_vectors, self.trajr[:, 0:3]):
                integral += self.f_2_integrate(i, j)
            integral_term = w_gamma * integral

            r_vec_useful, es_vec_useful, te_vec_useful, q_vec_useful = self.compute_t_and_r_useful_leg_vectors()
            sum_term = 0
            for i in features:
                t_useful_feat = i.compute_t_useful_feat(self, r_vec_useful, es_vec_useful, te_vec_useful, q_vec_useful)
                self.t_useful_feat_list.append(t_useful_feat)  # I THINK IT IS JUST FOR CHECKING AND PLOTS, not useful

                if not single_leg:
                    if i.t_obs_feat > i.t_goal:
                        t_obs_feat = i.t_goal  # to avoid a negative sqrt
                    else:
                        t_obs_feat = i.t_obs_feat
                    sum_term += np.sqrt(i.t_goal / ((1 + eps_T) * i.t_goal - t_obs_feat)) * t_useful_feat
                else:
                    m = leg_incentive
                    if t_useful_feat > i.t_goal:
                        t_useful_feat = i.t_goal
                    elif t_useful_feat == 0:
                        m = 0
                    sum_term += t_useful_feat + m

            self.score = w_deltav / (w_deltav + np.linalg.norm(self.dv)) * (integral_term + sum_term)

        else:
            self.score = 0
