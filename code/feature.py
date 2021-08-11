import numpy as np
import time
from scipy.spatial.transform import Rotation as R

from data_inspection import T_goal, T_useful_min
from useful_functions import isinfov, isiluminated
from attitude_eqs import quat2dcm_inertial2body


class Feature:
    """
    A feature is a target's point of interest for an observation (e.g a target face or corner).
    It is characterised at any moment by its viewing direction (body frame) and the time it has been already observed
    for (t_obs_feat).
    """

    def __init__(self, direction):
        """
        A feature is defined by its normal direction ("view_dir"), the time it has been observed for ("t_obs_feat") and
        the useful additional observation time.
        :param direction: [1x3] direction vector in body frame
        """
        self.view_dir = direction
        self.t_obs_feat = 0  # Time the feature has been observed for.
        # self.t_useful_feat_list = T_goal  # Useful additional observation time allowed by the leg.
        self.t_goal = T_goal
        self.t_useful_min = T_useful_min

    def isoverobs(self):
        """
        Returns a_target boolean answer to the question: Is the feature over observed? --> T_obs_feat > T_goal
        :return: boolean
        """
        if self.t_obs_feat > self.t_goal:
            return True
        else:
            return False

    def isobslongenough(self, t_useful):
        """
        Returns a_target boolean answer to the question: Is the feature observed longer than the minimum required time?
        T_obs_feat > T_useful_min
        :return: boolean
        """
        if t_useful >= self.t_useful_min:
            return True
        else:
            return False

    def compute_t_useful_feat(self, leg, r_eci_useful_leg, ts_useful_leg, q_useful_leg):
        """

        :param leg: inspection leg to be considered
        :param r_eci_useful_leg: chaser position vector wrt target in ECI frame
        :param ts_useful_leg: target-sun position vector in ECI frame
        :param q_useful_leg: quaternion vector
        :return: time useful (during which a given target's feature is observed properly for a given leg)
        """
        if self.isoverobs():
            t_useful_feat = 0
        else:
            t_useful_feat = leg.t_useful

            # create a vector cointaining the DCM matrices body2inertial
            # note that R.as_matrix() returns the DCM body2inertial of a quaternion vector.
            dcm_body2inertial_vec = R.from_quat(q_useful_leg).as_matrix()

            view_dir_eci_vectors = np.zeros((len(q_useful_leg), 3))
            for index, dcm in enumerate(dcm_body2inertial_vec):  # obtains the viewing direction vectors in ECI frame
                view_dir_eci_vectors[index, :] = dcm.dot(self.view_dir)  # matrix multiplication

            for r_eci, ts, vd in zip(r_eci_useful_leg, ts_useful_leg, view_dir_eci_vectors):
                if not isinfov(vd, r_eci):
                    t_useful_feat -= leg.t_step_obs
                elif not isiluminated(vd, ts):
                    t_useful_feat -= leg.t_step_obs

            if not self.isobslongenough(t_useful_feat):
                t_useful_feat = 0

        return t_useful_feat
