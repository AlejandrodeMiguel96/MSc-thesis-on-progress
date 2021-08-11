#region IDEAS AND TO-DO's
# regularization (e.g. dropout, L1, L2)

# early stopping

# kernel and bias initializer

# maybe choose a_target different metric than accuracy (maybe fitted just for clasification problems??), but not sure,
# because the loss value is still pretty big

# see if the predicted values are better, worse or random than the training ones

# also maybe we should check if the predicted legs fulfill the previous constraints (e.g dv and deltaT min/max)
#endregion

#region Standard packages
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from preprocessing import preprocess_data
from postprocessing import postprocess_data
import matplotlib.pyplot as plt
#endregion

#region Files
from leg import Leg
from evaluate_NN_pred import evaluate
from NN_data import database, range_process, perc_test_data, perc_val_data, n_epochs, batchsize, n_neurons
from NN_data import learning_rate, momentum, decay
from NN_data import dropout_perc, kernel_l1, kernel_l2, kernel_l1l2, bias_l1, bias_l2, bias_l1l2
#endregion

dense_data = np.load('dense_database.npz', allow_pickle=True)
database = dense_data['dense_database']

#region Setting training, validation and test data
print('Neural Network_test.py: Database size:', len(database))
print('Neural Network_test.py: Database preprocessed for range [0, 1]')
x, y = preprocess_data(database, range_process)
x = x[:, :3]  # just leaving the angular velocity for the inputs

print('Neural Network_test.py: batch size = ', batchsize)
print('Neural Network_test.py: epochs = ', n_epochs)

x_train = x[:-int(perc_test_data * len(database)), :]
y_train = y[:-int(perc_test_data * len(database)), :]
x_test = x[-int(perc_test_data * len(database)):, :]
y_test = y[-int(perc_test_data * len(database)):, :]
print('Neural Network_test.py: x_train', np.shape(x_train), 'y_train', np.shape(y_train), 'type', type(x_train))
print('Neural Network_test.py: x_val', int(len(x_train) * perc_val_data), 'y_val', int(len(y_train) * perc_val_data))
print('Neural Network_test.py: x_test', np.shape(x_test), 'y_test', np.shape(y_test))
#endregion

#region Design Neural Network
callback = keras.callbacks.EarlyStopping(monitor='loss', patience=4)
inputs = keras.Input(shape=(3,))  # INPUT LAYER
# x = layers.Dropout(0.8)(inputs)
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
# outputs = layers.Dense(4)(x)  # OUTPUT LAYER, is it better without activation???
outputs = layers.Dense(4, activation='relu')(x)  # OUTPUT LAYER
# https://keras.io/api/layers/activations/
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()


model.compile(
    optimizer=keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay),
    # optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.MeanSquaredError(),
    # loss=keras.losses.mean_squared_logarithmic_error,
    # loss=keras.losses.MeanAbsoluteError(),
    # metrics=['accuracy']
    metrics=[keras.metrics.MeanAbsoluteError(name='mean_absolute_error'), keras.metrics.MeanSquaredError()]
)

# PONER BIEN EL BATCH SIZE PORQUE PUEDE SER QUE SEA LO QUE DA PROBLEMAS
history = model.fit(
    x_train,
    y_train,
    batch_size=batchsize,
    epochs=n_epochs,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_split=perc_val_data,
    # validation_data=(x_val, y_val),
    shuffle=True,
)
print('Neural Network_test.py: History:', history.history.keys())
#endregion

#region Evaluate Neural Network

# Evaluate the NN with the test data
test_loss, test_mean_abs_err, test_mean_sqrd_err = model.evaluate(x_test, y_test)
# Make predictions (could be done without test data, take new data and compare it wrt the algorithm result
predictions = model(x_test)
predictions2 = model.predict(x_test)
predictions = np.array(predictions)
predictions2 = np.array(predictions)
print('Neural Network_test.py: predictions', np.shape(predictions))
print('Neural Network_test.py: predictions 2', np.shape(predictions2))

print('Neural Network_test.py: Example, first 5 cases:')
print('Neural Network_test.py: real:', y_test[0:5, :])
print('Neural Network_test.py: predicted:', predictions[0:5, :])
print('Neural Network_test.py: predicted 2:', predictions2[0:5, :])


mse = tf.keras.losses.MeanSquaredError()
print('Neural Network_test.py: test data loss 1', mse(y_test[0, :], predictions[0, :]).numpy())
print('Neural Network_test.py: test data loss 2', mse(y_test[1, :], predictions[1, :]).numpy())
print('Neural Network_test.py: test data loss 3', mse(y_test[2, :], predictions[2, :]).numpy())


# THIS GIVES THE REAL VALUES OF DV, DELTAT
x_test_post, y_test_post = postprocess_data(x_test[:, :], y_test[:, :], range_process)  # postprocess test data
x_predictions_post, y_predictions_post = postprocess_data(x_test[:, :], predictions[:, :], range_process)  # postprocess predict
#endregion

#region Evaluate all the test data (takes quite some time, need to integrate each one)
# leg_pred_list = list()
# for y in y_predictions_post:
#     leg_pred_list.append(Leg(y[0:3], y[3]))  # creates a_target Leg class list for the NN predictions
#
# leg_test_list = list()
# for y in y_test_post:
#     leg_test_list.append(Leg(y[0:3], y[3]))  # creates a_target Leg class list for the test data
#
#
# for j, x in zip(leg_pred_list, x_predictions_post):
#     w0 = x[0:3]
#     q = x[3:]
#     _ = evaluate(j, w0, q)  # propagates and scores the predicted leg (score is an attribute of Leg)
#
# for k, x in zip(leg_test_list, x_test_post):
#     w0 = x[0:3]
#     q = x[3:]
#     _ = evaluate(k, w0, q)
#
# print('doge', np.shape(leg_pred_list), np.shape(leg_test_list))
# print('doge 1', leg_pred_list[0].score, leg_test_list[0].score)
# print('doge 2', leg_pred_list[1].score, leg_test_list[1].score)
# print('doge 3', leg_pred_list[2].score, leg_test_list[2].score)
# print('doge 4', leg_pred_list[3].score, leg_test_list[3].score)
# print('doge 5', leg_pred_list[4].score, leg_test_list[4].score)
#endregion

#region Score a_target number of test data
# number = 5
# leg_pred_list = list()
# for y in y_predictions_post[0:number]:
#     leg_pred_list.append(Leg(y[0:3], y[3]))  # creates a_target Leg class list for the NN predictions
#
# leg_test_list = list()
# for y in y_test_post[0:number]:
#     leg_test_list.append(Leg(y[0:3], y[3]))  # creates a_target Leg class list for the test data
#
#
# for j, x in zip(leg_pred_list[0:number], x_predictions_post[0:number]):
#     w0 = x[0:3]
#     q = x[3:]
#     _ = evaluate(j, w0, q)  # propagates and scores the predicted leg (score is an attribute of Leg)
#
# for k, x in zip(leg_test_list[0:number], x_test_post[0:number]):
#     w0 = x[0:3]
#     q = x[3:]
#     _ = evaluate(k, w0, q)
#
# print('examples shape', np.shape(leg_pred_list), np.shape(leg_test_list))
# print('example 1', leg_pred_list[0].score, leg_pred_list[0].dv, leg_pred_list[0].t_leg, leg_test_list[0].score, leg_test_list[0].dv, leg_test_list[0].t_leg)
# print('example 2', leg_pred_list[1].score, leg_pred_list[1].dv, leg_pred_list[1].t_leg, leg_test_list[1].score, leg_test_list[1].dv, leg_test_list[1].t_leg)
# print('example 3', leg_pred_list[2].score, leg_pred_list[2].dv, leg_pred_list[2].t_leg, leg_test_list[2].score, leg_test_list[2].dv, leg_test_list[2].t_leg)
# print('example 4', leg_pred_list[3].score, leg_pred_list[3].dv, leg_pred_list[3].t_leg, leg_test_list[3].score, leg_test_list[3].dv, leg_test_list[3].t_leg)
# print('example 5', leg_pred_list[4].score, leg_pred_list[4].dv, leg_pred_list[4].t_leg, leg_test_list[4].score, leg_test_list[4].dv, leg_test_list[4].t_leg)

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


fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(history.history['mean_absolute_error'])
ax3.plot(history.history['val_mean_absolute_error'])
plt.title('model mean absolute error')
plt.ylabel('mean absolute error')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
ax3.grid()

plt.show()
#endregion


