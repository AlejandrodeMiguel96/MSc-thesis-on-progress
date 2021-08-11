# Chaser and target parameters
import numpy as np

m_c = 150  # [kg] Chaser mass
F_c = 15e-3  # [N] Chaser maximum thrust
threesigma_r = 1  # [% of r] Relative navigation position uncertainty. CARE!
threesigma_v = 3e-3  # [m/s] Relative navigation velocity uncertainty
threesigma_m = 1  # [% of deltav] Maneuver execution magnitude error
threesigma_p = 0.1 * np.pi/180  # [rad] Maneuver execution pointing error
h_t = 700  # [km]  Target orbital altitude
w_t_max = 0.3 * np.pi/180  # [rad/s] Target maximum tumbling rate
J_t_xx = 150  # [kg m**2] Target principal moment of inertia
J_t_yy = 70  # [kg m**2] Target principal moment of inertia
J_t_zz = 100  # [kg m**2] Target principal moment of inertia
J = [J_t_xx, J_t_yy, J_t_zz]
