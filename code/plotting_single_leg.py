# Plots graphs for orbits
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
from leg import Leg

from data_initial_conditions import n_steps
from data_inspection import r_min, r_obs_min, r_obs_max, r_escape

#region Resources
# Linestyles
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html#sphx-glr-gallery-lines-bars-and-markers-linestyles-py
#endregion

#region Constants and variables
# width_default = 6.4  # inches
# height_default = 4.8  # inches
width = 6.4  # inches
height = 4.8  # inches
figsize = [width, height]  # if plt.savefig bboxinches='tight' option is on, does not really mater
#endregion

#region TEMPLATE FOR FIGURES
# fig = plt.figure(figsize=figsize, frameon=False)
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.legend()
# ax.grid()
# fig.savefig(
#     '/Users/alejandrodemiguelmendiola/Dropbox/My Mac (Alejandros-MacBook-Air.local)/Desktop/TFM/latex/images/opt_legs.png',
#     bbox_inches='tight', transparent='True')
#endregion

# Latex
# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)

def plot_sphere(r, n, ax_name):
    """
    Draws a_target sphere in a_target given figure and axis.
    https://matplotlib.org/stable/gallery/mplot3d/surface3d_2.html
    https://stackoverflow.com/questions/51645694/how-to-plot-a-perfectly-smooth-sphere-in-python-using-matplotlib
    :param r: radius of the sphere
    :param n: number of points
    :param ax_name: name of the axis where sphere has to be drawn
    :return: axis with the sphere
    """
    stride = 1

    phi = np.linspace(0, 2 * np.pi, n)
    theta = np.linspace(0, np.pi, n)

    x = r * np.outer(np.cos(phi), np.sin(theta))
    y = r * np.outer(np.sin(phi), np.sin(theta))
    z = r * np.outer(np.ones(np.size(phi)), np.cos(theta))

    return ax_name.plot_surface(x, y, z, linewidth=0.0, cstride=stride, rstride=stride)


def plot_sphere2(r, n, ax_name):
    """
    Draws a_target sphere in a_target given figure and axis.
    https://matplotlib.org/stable/gallery/mplot3d/surface3d_2.html
    https://stackoverflow.com/questions/51645694/how-to-plot-a-perfectly-smooth-sphere-in-python-using-matplotlib
    :param r: radius of the sphere
    :param n: number of points
    :param ax_name: name of the axis where sphere has to be drawn
    :return: axis with the sphere
    """
    phi = np.linspace(0, 2 * np.pi, n)
    theta = np.linspace(0, np.pi, n)

    x = r * np.outer(np.cos(phi), np.sin(theta))
    y = r * np.outer(np.sin(phi), np.sin(theta))
    z = r * np.outer(np.ones(np.size(phi)), np.cos(theta))

    return ax_name.plot_surface(x, y, z)