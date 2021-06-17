import numpy as np
from data_ct import h_t
# Alejandro's data
mu_earth = 3.98600e14  # [m**3/s**2]
mu_sun = 1.327124400189e20  # [m**3/s**2]
a_earth = 149598023e3  # [m] Semi-major axis of Earth's orbit around Sun
r_earth = 6371e3  # [m] Earth's radius

a = r_earth + h_t  # [m] Target's semi-major axis
w_0_t = np.sqrt(mu_earth / a ** 3)  # [rad/s] Target's orbital rate CAUTION!!!REVISAR(mirar bien en algun desarrollo de Hills)
v_earth = np.sqrt(mu_sun/a_earth)  # [m/s]
v_target = np.sqrt(mu_earth/a)  # [m/s]
