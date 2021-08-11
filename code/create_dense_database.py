import numpy as np
from time import perf_counter
from NN_data import database

#region Select close points
print('start')
start = perf_counter()
dense_database = list()
for x in database:
    if 0*np.pi/180 < np.linalg.norm(x[0][:3]) < 0.2*np.pi/180:  # w is in a_target treshold 0<w<0.2 [deg/s]
        dense_database.append(x)

dense_database = np.array(dense_database)

print(np.shape(database))
print(type(database))
print(np.shape(dense_database))
print(type(dense_database))

np.savez('dense_database.npz', dense_database=dense_database)  # save database in a_target file
end = perf_counter()
print('execution time [s]:', end - start)
print('execution time [min]:', (end - start)/60)
#endregion