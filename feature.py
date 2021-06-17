from data_inspection import T_goal, T_useful_min
from useful_functions import compute_targetsun, isinfov, isiluminated, comp_viewdir_inertial, isinrange
from attitude_eqs import quat2dcm
import numpy as np


class Feature:
    """
    A feature is a point of interest for observation (e.g a face or a corner of the target).
    It is characterised at any moment by its viewing direction (body frame) and the time it has been already observed
    for (t_obs_feat).
    """

    def __init__(self, direction):
        """
        A feature is defined by its direction, the time it has been observed t_obs_feat and
        the useful additional observation time
        :param direction: [1x3] direction vector w_vec.r.t body frame
        """
        self.view_dir = direction
        self.t_obs_feat = 0  # Time the feature has been observed for.
        # self.t_useful_feat_list = T_goal  # Useful additional observation time allowed by the leg.
        self.t_goal = T_goal
        self.t_useful_min = T_useful_min

    def isoverobs(self):
        """
        Returns a boolean answer to the question: Is the feature over observed? --> T_obs_feat > T_goal
        :return: boolean
        """
        if self.t_obs_feat > self.t_goal:
            return True
        else:
            return False

    def isobslongenough(self, t_useful):
        """
        Returns a boolean answer to the question: Is the feature observed longer than the minimum required time?
        T_obs_feat > T_useful_min
        :return: boolean
        """
        if t_useful >= self.t_useful_min:
            return True
        else:
            return False

    def compute_t_useful_feat(self, leg, r_useful_leg, te_useful_leg, es_useful_leg, q_useful_leg):
        if self.isoverobs():
            t_useful_feat = 0
        else:
            ts_useful_leg = compute_targetsun(te_useful_leg, es_useful_leg)

            # computing the viewing direction angle of the feature by transforming quaternions to DCM
            A_bi_list = []
            for q in q_useful_leg:
                A_bi = quat2dcm(q)
                A_bi_list.append(A_bi)
            view_dir_list = []
            for A in A_bi_list:
                view_dir_inert = comp_viewdir_inertial(A, self.view_dir)
                view_dir_list.append(view_dir_inert)

            t_useful_feat = leg.t_useful

            for r, ts, vd in zip(r_useful_leg, ts_useful_leg, view_dir_list):  # zip vectors must be same length !!!!!!!
                if not isinfov(vd, r):
                    t_useful_feat -= leg.t_step_obs
                elif not isiluminated(vd, ts):
                    t_useful_feat -= leg.t_step_obs

            if not self.isobslongenough(t_useful_feat):
                t_useful_feat = 0

        return t_useful_feat
