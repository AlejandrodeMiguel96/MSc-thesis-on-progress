# IDEAS
# look for the best way to initialize weights and biases
# study mean and variance of the database, see if the correct answer is in fact a "mean inspection"
# try to overfit it
# do grid search (plot graph comparing a range of e.g. learning rate and the losses for each)
# LR range test
# weight decay
# introduce score function in loss

#region Standard packages
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt


#endregion

#region Files
import data_inspection
import plotting_single_leg
from preprocessing import preprocess_data
from postprocessing import postprocess_data
from NN_data import database, range_process, perc_test_data, perc_val_data, n_epochs, batchsize, n_neurons
from NN_data import learning_rate, momentum, decay
from NN_data import dropout_perc, kernel_l1, kernel_l2, kernel_l1l2, bias_l1, bias_l2, bias_l1l2
#endregion
from data_initial_conditions import state0_ifo
from compare_legs import build_leg
import statistics


def custom_loss(y_true, y_predicted):
    print('ey')
    y_true = y_true.numpy()
    y_predicted = y_predicted.numpy()
    score_true = []
    score_predicted = []
    squared_diff = []
    for row_true, row_predicted in zip(y_true, y_predicted):
        leg_true = build_leg(row_true[:, 4:], row_true[:, :4], state0_ifo)
        leg_predicted = build_leg(row_true[:, 4:], row_predicted[:, :4], state0_ifo)
        score_true.append(leg_true.score)
        score_predicted.append(leg_predicted.score)
        squared_diff.append(np.sqrt(leg_true.score - leg_predicted.score))

    # squared_difference = tf.square(leg_true.score - leg_predicted.score)
    # squared_difference = tf.square(y_true[:, 0] - y_predicted[:, 0])
    # print('a', squared_difference, type(squared_difference))
    # print('b', tf.reduce_mean(squared_difference, axis=-1), type(tf.reduce_mean(squared_difference, axis=-1)))
    return statistics.mean(squared_diff)
    # return tf.reduce_mean(squared_difference, axis=-1)

# i can update the database to include the score function of each leg, but how to compute the score of the predicted
# ones to then pass it to the loss function??????
# the only solution is passing 7 inputs to y_true and use them for computing the leg, but not for computing the loss






#region STATISTICAL ANALYSIS OF THE DATABASE
# import statistics
# # i can use also the database_consistency ya que la estoy guardando a la vez que database_single_leg
# dv_and_t = database[:, 1]
# dvx = []
# dvy = []
# dvz = []
# t_leg = []
# for lst in dv_and_t:
#     dvx.append(lst[0])
#     dvy.append(lst[1])
#     dvz.append(lst[2])
#     t_leg.append(lst[3])
# avrg_dvx = statistics.mean(dvx)
# var_dvx = statistics.variance(dvx, avrg_dvx)
# std_dvx = statistics.stdev(dvx, avrg_dvx)
#
# avrg_dvy = statistics.mean(dvy)
# var_dvy = statistics.variance(dvy, avrg_dvy)
# std_dvy = statistics.stdev(dvy, avrg_dvy)
#
# avrg_dvz = statistics.mean(dvz)
# var_dvz = statistics.variance(dvz, avrg_dvz)
# std_dvz = statistics.stdev(dvz, avrg_dvz)
#
# avrg_t_leg = statistics.mean(t_leg)
# var_t_leg = statistics.variance(t_leg, avrg_t_leg)
# std_t_leg = statistics.stdev(t_leg, avrg_t_leg)
#endregion

#region Setting data (training, validation and test)
# Loads the database with close inputs
# dense_data = np.load('dense_database.npz', allow_pickle=True)
# database = dense_data['dense_database']
n_inputs = 7  # if only interested in w, n_inputs=3. If interested in both w and q, n_inputs=7
print('Neural Network_test.py: Database size:', len(database))
print('Neural Network_test.py: Database preprocessed for range [0, 1]')
print('Neural Network_test.py: Number of inputs:', n_inputs)
x_database, y_database = preprocess_data(database, range_process)  # call for the preprocess function
x_database = x_database[:, :n_inputs]  # consider only w or also q?

y_database = np.append(y_database, x_database, axis=1)  # adding x to y to compute the legs in loss function

print('Neural Network_test.py: batch size = ', batchsize)
print('Neural Network_test.py: epochs = ', n_epochs)

x_train = x_database[:-int(perc_test_data * len(database)), :]
y_train = y_database[:-int(perc_test_data * len(database)), :]
x_test = x_database[-int(perc_test_data * len(database)):, :]
y_test = y_database[-int(perc_test_data * len(database)):, :]
print('Neural Network_test.py: x_train', np.shape(x_train), 'y_train', np.shape(y_train), 'type', type(x_train))
print('Neural Network_test.py: x_val', int(len(x_train) * perc_val_data), 'y_val', int(len(y_train) * perc_val_data))
print('Neural Network_test.py: x_test', np.shape(x_test), 'y_test', np.shape(y_test))
#endregion

#region Design Neural Network
tf.config.experimental_run_functions_eagerly(True)
# callback = keras.callbacks.EarlyStopping(monitor='loss', patience=4)
inputs = keras.Input(shape=(n_inputs,))  # INPUT LAYER
x = layers.Dropout(0.8)(inputs)
x = layers.Dense(n_neurons, activation='relu')(inputs)  # #1
x = layers.Dropout(dropout_perc)(x)
x = layers.Dense(n_neurons, activation='relu')(x)  # #2
x = layers.Dropout(dropout_perc)(x)
# x = layers.Dense(n_neurons, activation='relu')(x)  # #3
# x = layers.Dropout(dropout_perc)(x)
# x = layers.Dense(n_neurons, activation='relu')(x)  # #4
# x = layers.Dropout(dropout_perc)(x)
# x = layers.Dense(n_neurons, activation='relu')(x)  # #5
# x = layers.Dropout(dropout_perc)(x)
# x = layers.Dense(n_neurons, activation='relu')(x)  # #6
# x = layers.Dropout(dropout_perc)(x)
# x = layers.Dense(n_neurons, activation='relu')(x)  # #7
# x = layers.Dropout(dropout_perc)(x)
# x = layers.Dense(n_neurons, activation='relu')(x)  # #8
# x = layers.Dropout(dropout_perc)(x)
# x = layers.Dense(n_neurons, activation='relu')(x)  # #9

# x = layers.Dropout(dropout_perc)(x)
outputs = layers.Dense(11, activation='relu')(x)  # OUTPUT LAYER
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(
    optimizer=keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay),
    loss=custom_loss,
    # loss=keras.losses.MeanSquaredError(),
    # loss=keras.losses.mean_squared_logarithmic_error,
    # loss=keras.losses.MeanAbsoluteError(),
    metrics=[keras.metrics.MeanAbsoluteError(name='mean_absolute_error'), keras.metrics.MeanSquaredError()],
    run_eagerly=True
)

history = model.fit(
    x_train,
    y_train,
    batch_size=batchsize,
    epochs=n_epochs,
    validation_split=perc_val_data,
    shuffle=True,
)
#endregion

#region Evaluate Neural Network
# Evaluate the NN with the test data
print('Neural Network.py: test metrics')
test_loss, test_mean_abs_err, test_mean_sqrd_err = model.evaluate(x_test, y_test)
predictions = model(x_test)
predictions2 = model.predict(x_test)
predictions = np.array(predictions)
print('Neural Network_test.py: predictions', np.shape(predictions))
print('Neural Network_test.py: Example, first 3 cases:')
print('Neural Network_test.py: "real":', y_test[0:3, :])
print('Neural Network_test.py: predicted:', predictions[0:3, :])

mse = tf.keras.losses.MeanSquaredError()
print('Neural Network_test.py: test data 1: loss value', mse(y_test[0, :], predictions[0, :]).numpy())
print('Neural Network_test.py: test data 2: loss value', mse(y_test[1, :], predictions[1, :]).numpy())
print('Neural Network_test.py: test data 3: loss value', mse(y_test[2, :], predictions[2, :]).numpy())


# THIS GIVES THE ORIGINAL VALUES OF DV, DELTAT
x_test_post, y_test_post = postprocess_data(x_test[:, :], y_test[:, :], range_process)  # postprocess test data
x_predictions_post, y_predictions_post = postprocess_data(x_test[:, :], predictions[:, :], range_process)  # postprocess predict
print('Neural Network_test.py: Example, first 3 cases:')
print('Neural Network_test.py: "real":', y_test_post[0:3, :])
print('Neural Network_test.py: predicted:', y_predictions_post[0:3, :])
#endregion


#region PLOTS
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(history.history['loss'])
ax1.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
ax1.grid()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(history.history['mean_absolute_error'])
ax2.plot(history.history['val_mean_absolute_error'])
plt.title('model mean absolute error')
plt.ylabel('mean absolute error')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
ax2.grid()


from algorithm import propagate_hills
from leg import Leg
from data_initial_conditions import state0_ifo, steps_man, steps_att, steps_obs, steps_comp_att
for y, y_pred in zip(y_test_post, y_predictions_post):
    # make routine in Leg to compute all the vectors (e.g. rr, rr_obs, rr_useful) so that just with initialize a leg
    # even the score is computed?
    s = Leg(y[0:3], y[3])
    t1 = np.linspace(0, s.t_comp + s.t_att, steps_comp_att)  # integration time part 1 (comp+att.acq)
    t2 = np.linspace(0, s.t_man, steps_man)  # integration time part 2 (manoeuvre)
    t3 = np.linspace(0, s.t_att, steps_att)  # integration time part 3 (point.acq+observation)
    t4 = np.linspace(0, s.t_obs, steps_obs)  # integration time part 4 (observation)
    t_interp = np.append(np.append(np.append(t1, t2 + s.t_comp + s.t_att),
                                   t3 + s.t_comp + s.t_att + s.t_man), t4 + (s.t_leg - s.t_obs))
    rr_lvlh, _, _, _, _ = propagate_hills(state0_ifo, t1, t2, t3, t4, s)
    s_pred = Leg(y_pred[0:3], y_pred[3])
    t1_pred = np.linspace(0, s_pred.t_comp + s_pred.t_att, steps_comp_att)  # integration time part 1 (comp+att.acq)
    t2_pred = np.linspace(0, s_pred.t_man, steps_man)  # integration time part 2 (manoeuvre)
    t3_pred = np.linspace(0, s_pred.t_att, steps_att)  # integration time part 3 (point.acq+observation)
    t4_pred = np.linspace(0, s_pred.t_obs, steps_obs)  # integration time part 4 (observation)
    t_interp_pred = np.append(np.append(np.append(t1_pred, t2_pred + s_pred.t_comp + s_pred.t_att), t3_pred +
                                        s_pred.t_comp + s_pred.t_att + s_pred.t_man), t4_pred +
                              (s_pred.t_leg - s_pred.t_obs))
    rr_pred_lvlh, _, _, _, _ = propagate_hills(state0_ifo, t1_pred, t2_pred, t3_pred, t4_pred, s_pred)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    plotting_single_leg.plot_sphere2(data_inspection.r_min, 100, ax3)
    ax3.plot(rr_lvlh[:, 0], rr_lvlh[:, 1], rr_lvlh[:, 2], label='leg')
    ax3.plot(rr_pred_lvlh[:, 0], rr_pred_lvlh[:, 1], rr_pred_lvlh[:, 2], label='predicted leg')
    ax3.set_xlabel('V-bar [km]')
    ax3.set_ylabel('H-bar [km]')
    ax3.set_zlabel('R-bar [km]')
    plt.legend()
    plt.show()
    print('debug plot')
    plt.close()

print('debug print')
#endregion


