# INSPECTION CONSTRAINTS AND GUIDANCE PARAMETERS
import numpy as np

r_obs_max = 150  # [m] Maximum inspection distance
r_obs_min = 100  # [m] Minimum inspection distance
r_escape = 500  # [m] Escape distance
r_min = 25  # [m] Keepout sphere radius
alpha_max = 15 * np.pi/180  # [rad] Viewing direction cone angle
beta_max = 45 * np.pi/180  # [rad] Sun direction cone angle
T_goal = 120  # [s] Required observation time per feature
T_useful_min = 2  # [s] Observation time treshold
T_att = 300  # [s] Attitude acquisition time
T_comp = 120  # [s] Maneuver computation time
T_leg_max = 4.1 * 3600  # [s] Maximum leg duration
T_leg_min = 1 * 3600  # [s] Minimum leg duration
T_safe = 8.2 * 3600  # [s] Safety horizon
T_max = 48 * 3600  # [s] Maximum mission duration
deltav_max = 90e-3  # [m/s] Maximum maneuver magnitude
deltav_min = 3e-3  # [m/s] Minimum maneuver magnitude
w_gamma = 0.01  # [--] Trajectory score, geometry weight
w_deltav = 0.15  # [m/s] Trajectory score, consumption weight
nu_V = 1  # [--] Simplex volume weight, original is 1
nu_M = 300  # [--] Simplex score weight, original is 300
nu_G = 150  # [--] Simplex gradient weight, original is 150
n_mesh = 3  # [--] Total number of meshes, original is 2
n_s0 = 50  # [--] Number of initial samples, original was 1000
n_s = int(n_s0/2)  # [--] Number of refined samples, original was 500

###################################################################################################
######################################################################################################################################################################################################
eps_T = .1  # Tuneable. It should be small enough so that highly (but not fully) features still give a decent score so
# they can be still observed, but not so small that no observed features are given a small score
leg_incentive = 10  # Tuneable. It is a parameter in the score function that favours those trajectories that inspect
# more features (e.g. a trajectory that inspects 1 feature for 100s is worse than a trajectory that inspects 2 features
# for 100s). The value of this parameter regulates the threshold that dictates if it's better to choose a long
# observation to a limited number of features or shorter observations to a bigger number of features. Its magnitude
# is related to other variables such as T_goal, T_min_useful or the number of features to observe;
# and should be tuned according to these.




