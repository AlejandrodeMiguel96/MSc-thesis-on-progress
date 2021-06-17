# Plots graphs for orbits
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
from leg import Leg

from data_initial_conditions import n_steps
from data_inspection import r_min, r_obs_min, r_obs_max, r_escape

save_figures = False
# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)

def plot_sphere(r, n, ax_name):
    """
    Draws a sphere in a given figure and axis.
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

# I changed it because of single leg, it may not work right away now
data2 = np.load('database_full_insp.npz', allow_pickle=True)
database = data2['database']
print(type(database))
print(database)
legs_opt = database[0][1]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# n = int(n_steps/3)
i = 0

# ax.plot(trajs_score[:, 0], trajs_score[:, 1], trajs_score[:, 2], 'b')
plot_sphere(r_min, 50, ax)
start_point = legs_opt[0].trajr
for s in legs_opt:
    if len(s.trajr) == 0:
        break
    else:
        ax.plot(s.trajr[:, 0], s.trajr[:, 1], s.trajr[:, 2])
        end_point = [s.trajr[-1, 0], s.trajr[-1, 1], s.trajr[-1, 2]]
ax.plot(start_point[0, 0], start_point[0, 1], start_point[0, 2], 'g.', label='start point')
ax.plot(end_point[0], end_point[1], end_point[2], 'r.', label='end point')
ax.set_xlabel('V-bar [m]')
ax.set_ylabel('H-bar [m]')
ax.set_zlabel('R-bar [m]')
# ax.set_title('Optimal leg inspection sequence')
# limit = 0
# for x in trajs_score:
#     maximum = max(x)
#     if maximum > limit:
#         limit = maximum
# ax.set_xlim([-limit, limit])
# ax.set_ylim([-limit, limit])
# ax.set_zlim([-limit, limit])
plt.legend()
plt.grid()
if save_figures:
    print('opt_legs.png figure saved!')
    plt.savefig('/Users/alejandrodemiguelmendiola/Dropbox/My Mac (Alejandros-MacBook-Air.local)/Desktop/TFM/latex/images/opt_legs.png',
                bbox_inches='tight', transparent='True')

# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111, projection='3d')
# plot_sphere(6371e3, 50, ax1)
# start_point = opt_leg[0].trajt
# for s in opt_leg:
#     if len(s.trajr) == 0:
#         break
#     else:
#         ax1.plot(s.trajt[:, 0], s.trajt[:, 1], s.trajt[:, 2])
#         end_point = [s.trajt[-1, 0], s.trajt[-1, 1], s.trajt[-1, 2]]
# ax1.plot(start_point[0, 0], start_point[0, 1], start_point[0, 2], 'g*', label='start point')
# ax1.plot(end_point[0], end_point[1], end_point[2], 'r*', label='end point')
# ax1.set_title('target wrt earth')
# ax1.set_xlabel('x [m]')
# ax1.set_ylabel('y [m]')
# ax1.set_zlabel('z [m]')
# plt.legend()
# plt.grid()
#
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, projection='3d')
# # plot_sphere(696340000, 50, ax2)
# start_point = opt_leg[0].traje
# for s in opt_leg:
#     if len(s.trajr) == 0:
#         break
#     else:
#         ax2.plot(s.traje[:, 0], s.traje[:, 1], s.traje[:, 2])
#         end_point = [s.traje[-1, 0], s.traje[-1, 1], s.traje[-1, 2]]
# ax2.plot(start_point[0, 0], start_point[0, 1], start_point[0, 2], 'g*', label='start point')
# ax2.plot(end_point[0], end_point[1], end_point[2], 'r*', label='end point')
# ax2.set_title('earth wrt sun')
# ax2.set_xlabel('x [m]')
# ax2.set_ylabel('y [m]')
# ax2.set_zlabel('z [m]')
# plt.legend()
# plt.grid()

# fig3 = plt.figure()
# ax3 = fig3.add_subplot(111)
# t = np.linspace(0, s.t_leg, n_steps)
# ax3.plot(t, opt_leg[0].w_vec[:, 0], label='wx')
# ax3.plot(t, opt_leg[0].w_vec[:, 1], label='wy')
# ax3.plot(t, opt_leg[0].w_vec[:, 2], label='wz')
# ax3.set_title("Target's angular velocity during the legs")
# ax3.set_xlabel('time')
# ax3.set_ylabel('w_vec')
# plt.legend()
# plt.grid()
#
# fig4 = plt.figure()
# ax4 = fig4.add_subplot(211)
# t = np.linspace(0, s.t_leg, n_steps)
# ax4.plot(t, opt_leg[0].q_vec[:, 0], label='q0')
# ax4.plot(t, opt_leg[0].q_vec[:, 1], label='q1')
# ax4.plot(t, opt_leg[0].q_vec[:, 2], label='q2')
# ax4.plot(t, opt_leg[0].q_vec[:, 3], label='q3')
# ax4.set_title("Target's quaternions during the legs")
# ax4.set_xlabel('time')
# ax4.set_ylabel('q_vec')
# plt.legend()
# plt.grid()
# ax4 = fig4.add_subplot(212)
# ax4.plot(t, opt_leg[1].q_vec[:, 0]**2 + opt_leg[1].q_vec[:, 1]**2 + opt_leg[1].q_vec[:, 2]**2 + opt_leg[1].q_vec[:, 3]**2)
# ax4.set_title("Quaternion squared sum")
# ax4.set_xlabel('time')
# ax4.set_ylabel('sum(q_vec**2)')
# plt.grid()

fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
r = []
r_obs = []
r_useful = []
t1 = 0
t2 = 0
for s in legs_opt:
    if len(s.trajr) == 0:
        break
    else:
        for x in s.trajr:
            r.append(np.linalg.norm(x[0:3]))
        for x in s.r_obs_vec:
            r_obs.append(np.linalg.norm(x))
        for x in s.r_useful_vec:
            r_useful.append(np.linalg.norm(x))
        t2 = s.t_leg + t1
        # t = np.linspace(t1, t2, n_steps)
        ax5.plot(s.t+t1, r)
        ax5.plot(s.t_useful_vec+t1, r_useful, 'or')
        ax5.plot((s.t_obs_vec+t1)[0], r_obs[0], '*k')
        ax5.plot((s.t_obs_vec+t1)[-1], r_obs[-1], '*k')
        r = []
        r_obs = []
        r_useful = []
        t1 = t2
ax5.set_xlabel('t')
ax5.set_ylabel('r[m]')
ax5.axhspan(r_obs_min, r_obs_max, color='lime', alpha=0.5, label='observation zone')
ax5.axhspan(r_escape, r_escape+25, color='crimson', alpha=0.5,  label='r_max')
ax5.axhspan(0, r_min, color='purple', alpha=0.5, label='r_min')
ax5.set_ylim([0, r_escape+25])
plt.legend()
plt.grid()
if save_figures:
    print('opt_legs_r.png figure saved!')
    plt.savefig('/Users/alejandrodemiguelmendiola/Dropbox/My Mac (Alejandros-MacBook-Air.local)/Desktop/TFM/latex/images/opt_legs_r.png',
                bbox_inches='tight', transparent='True')

plt.show()






# NO DEBERÍA DE HABER PUNTOS ROJOS FUERA DE LA ZONA VERDE DE OBSERVACIÓN, NO?????


