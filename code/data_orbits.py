import numpy as np
from data_chaser_target import h_t
# Alejandro's data
mu_earth = 3.98600e5  # [km**3/s**2]
mu_sun = 1.327124400189e11  # [km**3/s**2]
a_earth = 149598023  # [km] Semi-major axis of Earth's orbit around Sun
r_earth = 6371  # [km] Earth's radius
w_0_earth = np.sqrt(mu_sun/a_earth**3)

a_target = r_earth + h_t  # [km] Target's semi-major axis
w_0_t = np.sqrt(mu_earth / a_target ** 3)  # [rad/s] Target's orbital rate
v_earth = np.sqrt(mu_sun/a_earth)  # [km/s]
v_target = np.sqrt(mu_earth / a_target)  # [km/s]
