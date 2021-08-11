import numpy as np
import matplotlib.pyplot as plt
import statistics
import time

import data_inspection

# data = np.load('database_consistency.npz', allow_pickle=True)
# data = np.load('database_consistency_300_10examples.npz', allow_pickle=True)
# data = np.load('database_consistency_1000_60examples.npz', allow_pickle=True)
# data = np.load('database_consistency_10000_17examples.npz', allow_pickle=True)
# data = np.load('database_consistency_1000_1339examples.npz', allow_pickle=True)
# data = np.load('database_consistency_5000_60examples.npz', allow_pickle=True)
data = np.load('database_consistency_100_960examples.npz', allow_pickle=True)
database = data['database']

leg1 = database[40]
from data_initial_conditions import state0_ifo
from compare_legs import build_leg
state0_chaser = state0_ifo
inputs = [*leg1.w_vec[0, :], *leg1.q_vec[0, :]]
outputs = [*leg1.dv, leg1.t_leg]
built_leg = build_leg(inputs, outputs, state0_chaser)
built_leg2 = build_leg(inputs, outputs, state0_chaser)
a = leg1.score
b = built_leg.score
c = built_leg2.score



scores = []
dv_and_t = []
for s in database:
    scores.append(s.score)
    dv_and_t.append((*s.dv, s.t_leg))
    dv_norm = np.linalg.norm(s.dv)
    if not (data_inspection.deltav_min <= dv_norm <= data_inspection.deltav_max or dv_norm == 0):
        print('error dv!!!', dv_norm)

scores = np.array(scores)
scores_sorted = np.array(sorted(scores, reverse=True))
dv_and_t = np.array(dv_and_t)

avrg_score = statistics.mean(scores)
var_score = statistics.variance(scores, avrg_score)
std_score = statistics.stdev(scores, avrg_score)

avrg_dvx = statistics.mean(dv_and_t[:, 0])
var_dvx = statistics.variance(dv_and_t[:, 0], avrg_dvx)
std_dvx = statistics.stdev(dv_and_t[:, 0], avrg_dvx)

avrg_dvy = statistics.mean(dv_and_t[:, 1])
var_dvy = statistics.variance(dv_and_t[:, 1], avrg_dvy)
std_dvy = statistics.stdev(dv_and_t[:, 1], avrg_dvy)

avrg_dvz = statistics.mean(dv_and_t[:, 2])
var_dvz = statistics.variance(dv_and_t[:, 2], avrg_dvz)
std_dvz = statistics.stdev(dv_and_t[:, 2], avrg_dvz)

avrg_t_leg = statistics.mean(dv_and_t[:, 3])
var_t_leg = statistics.variance(dv_and_t[:, 3], avrg_t_leg)
std_t_leg = statistics.stdev(dv_and_t[:, 3], avrg_t_leg)


# # PLOTS
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('v-bar [km]')
ax.set_ylabel('b-bar [km]')
ax.set_zlabel('h-bar [km]')
ax.set_title('Chaser wrt target')
plt.grid()
for s in database[:]:
    print(s.score)
    ax.plot(s.rr_chaser_LVLH[:, 0], s.rr_chaser_LVLH[:, 1], s.rr_chaser_LVLH[:, 2])
    ax.plot(s.rr_chaser_LVLH[0, 0], s.rr_chaser_LVLH[0, 1], s.rr_chaser_LVLH[0, 2], 'g*', label='starting point')
    ax.plot(s.rr_chaser_LVLH[-1, 0], s.rr_chaser_LVLH[-1, 1], s.rr_chaser_LVLH[-1, 2], 'r*', label='ending point')
    a = 0
plt.show()



print('sacabo')







