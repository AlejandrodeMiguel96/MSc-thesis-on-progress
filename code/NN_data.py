# File containing the key data of the Neural Network
import numpy as np
import tensorflow as tf


#region Load database
data1 = np.load('database_single_leg_1000_6830examples.npz', allow_pickle=True)
data2 = np.load('database_single_leg_1000_3634examples.npz', allow_pickle=True)
data3 = np.load('database_single_leg_1000_2172examples.npz', allow_pickle=True)
data4 = np.load('database_single_leg_1000_3860examples.npz', allow_pickle=True)
data5 = np.load('database_single_leg_1000_5531examples.npz', allow_pickle=True)
data6 = np.load('database_single_leg_1000_3233examples.npz', allow_pickle=True)
data7 = np.load('database_single_leg_1000_3474.npz', allow_pickle=True)
database1 = data1['database']
database2 = data2['database']
database3 = data3['database']
database4 = data4['database']
database5 = data5['database']
database6 = data6['database']
database7 = data7['database']
database = np.append(database1, database2, axis=0)
database = np.append(database, database3, axis=0)
database = np.append(database, database4, axis=0)
database = np.append(database, database5, axis=0)
database = np.append(database, database6, axis=0)
database = np.append(database, database7, axis=0)
database = database[:, :]  # consider full database
# database = database[0:int(1e4), :]  # consider a_target portion of the database
range_process = '0,1'  # for files preprocessing.py and postprocessing.py
#endregion

#region Hyper-parameters
perc_test_data = 0.1  # percentage of the database to be used for testing
perc_val_data = 0.1  # percentage of the training data to be used for validation
n_neurons = 64  # number of neurons per layer
batchsize = 32
n_epochs = 5

# SGD optimization
learning_rate = .01
momentum = 0.9
decay = 0
# decay = 0.01

# dropout_perc = 0  # dropout percentage
dropout_perc = 0.4  # dropout percentage
#endregion

#region Initializers
# kernel_initializer = tf.keras.initializers.random_normal(mean=0.0, stddev=0.05, seed=None)
kernel_initializer = tf.keras.initializers.zeros
bias_initializer = tf.keras.initializers.zeros
#endregion

#region Regularizers
kernel_l1 = tf.keras.regularizers.l1()
bias_l1 = tf.keras.regularizers.l1()

kernel_l2 = tf.keras.regularizers.l2(1e-3)
bias_l2 = tf.keras.regularizers.l2(1e-3)

kernel_l1l2 = tf.keras.regularizers.l1_l2(l1=.01, l2=1e-2)
bias_l1l2 = tf.keras.regularizers.l1_l2(l1=.01, l2=1e-3)
#endregion


