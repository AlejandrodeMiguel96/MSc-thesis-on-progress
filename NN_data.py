# File containing the key data of the Neural Network
import numpy as np
import tensorflow as tf


#region Load database
data = np.load('database_single_leg_635927.npz', allow_pickle=True)
database = data['database']
database = database[:, :]  # consider full database
# database = database[0:int(1e4), :]  # consider a portion of the database
range_process = '0,1'  # for files preprocessing.py and postprocessing.py
#endregion

#region Hyper-parameters
perc_test_data = 0.01  # percentage of the database to be used for testing
perc_val_data = 0.3  # percentage of the training data to be used for validation
n_neurons = 64  # number of neurons per layer
batchsize = 32
n_epochs = 20

# SGD optimization
learning_rate = .01
momentum = 0.9
decay = 0
# decay = 0.01

# dropout_perc = 0  # dropout percentage
dropout_perc = 0.6  # dropout percentage
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


