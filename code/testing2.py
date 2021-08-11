import numpy as np
import matplotlib.pyplot as plt
# fig2.subplots_adjust(hspace=0.358, wspace=0.358)
import data_initial_conditions

# data = np.load('database_consistency.npz', allow_pickle=True)
data = np.load('database_consistency_1000_1339examples.npz', allow_pickle=True)
# data = np.load('database_consistency_300_10examples.npz', allow_pickle=True)
database = data['database']

s = database[4]  # for example
steps1 = data_initial_conditions.steps_comp_att
steps2 = steps1 + data_initial_conditions.steps_man
steps3 = steps2 + data_initial_conditions.steps_att
steps4 = steps3 + data_initial_conditions.steps_obs

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot(s.rr_chaser_ECI[0, 0]+s.rr_target[0, 0], s.rr_chaser_ECI[0, 1]+s.rr_target[0, 1], s.rr_chaser_ECI[0, 2]+s.rr_target[0, 2], 'g*', label='starting point')
ax1.plot(s.rr_chaser_ECI[-1, 0]+s.rr_target[-1, 0], s.rr_chaser_ECI[0, 1]+s.rr_target[-1, 1], s.rr_chaser_ECI[-1, 2]+s.rr_target[-1, 2], 'r*', label='ending point')
ax1.plot(s.rr_chaser_ECI[:, 0]+s.rr_target[:, 0], s.rr_chaser_ECI[:, 1]+s.rr_target[:, 1], s.rr_chaser_ECI[:, 2]+s.rr_target[:, 2], label='chaser orbit')
ax1.plot(s.rr_target[:, 0], s.rr_target[:, 1], s.rr_target[:, 2], label='target orbit')
ax1.set_xlabel('x [km]')
ax1.set_ylabel('y [km]')
ax1.set_zlabel('z [km]')
ax1.grid()
ax1.legend()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot(s.rr_chaser_LVLH[0, 0]*1e3, s.rr_chaser_LVLH[0, 1]*1e3, s.rr_chaser_LVLH[0, 2]*1e3, 'g*', label='starting point')
ax2.plot(s.rr_chaser_LVLH[-1, 0]*1e3, s.rr_chaser_LVLH[-1, 1]*1e3, s.rr_chaser_LVLH[-1, 2]*1e3, 'r*', label='ending point')
# ax2.plot(s.rr_chaser_LVLH[:, 0]*1e3, s.rr_chaser_LVLH[:, 1]*1e3, s.rr_chaser_LVLH[:, 2]*1e3)
ax2.plot(s.rr_chaser_LVLH[:steps1, 0]*1e3, s.rr_chaser_LVLH[:steps1, 1]*1e3, s.rr_chaser_LVLH[:steps1, 2]*1e3, 'b', label='comp.+att.acq.')
ax2.plot(s.rr_chaser_LVLH[steps1:steps2, 0]*1e3, s.rr_chaser_LVLH[steps1:steps2, 1]*1e3, s.rr_chaser_LVLH[steps1:steps2, 2]*1e3, 'r', marker='d', markersize=10, label='manoeuvre')
ax2.plot(s.rr_chaser_LVLH[steps2:steps3, 0]*1e3, s.rr_chaser_LVLH[steps2:steps3, 1]*1e3, s.rr_chaser_LVLH[steps2:steps3, 2]*1e3, 'orange', label='att.acq.')
ax2.plot(s.rr_chaser_LVLH[steps3:, 0]*1e3, s.rr_chaser_LVLH[steps3:, 1]*1e3, s.rr_chaser_LVLH[steps3:, 2]*1e3, 'g', label='observation')
ax2.set_xlabel('V-bar [m]')
ax2.set_ylabel('H-bar [m]')
ax2.set_zlabel('R-bar [m]')
ax2.grid()
ax2.legend()

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.grid()
r = []
for rr in s.rr_chaser_LVLH:
    r.append(np.linalg.norm(rr)*1e3)
ax3.plot(s.t[:steps1], r[:steps1], 'b', label='comp.+att.acq.')
ax3.plot(s.t[steps1:steps2], r[steps1:steps2], 'r', marker='d', markersize=10, label='manoeuvre')
ax3.plot(s.t[steps2:steps3], r[steps2:steps3], 'orange', label='att.acq.')
ax3.plot(s.t[steps3:], r[steps3:], 'g', label='observation')
# ax2.plot(s.t, r)
ax3.set_xlabel('t [s]')
ax3.set_ylabel('r [m]')
ax3.legend()
print('sacabo')




