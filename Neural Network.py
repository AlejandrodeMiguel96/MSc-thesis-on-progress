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
from NN_data import database, range_process, perc_test_data, perc_val_data, n_epochs, batchsize, n_neurons
from NN_data import learning_rate, momentum, decay
from NN_data import dropout_perc, kernel_l1, kernel_l2, kernel_l1l2, bias_l1, bias_l2, bias_l1l2
#endregion

# Loads the database with close inputs
dense_data = np.load('dense_database.npz', allow_pickle=True)
database = dense_data['dense_database']

#region Setting data (training, validation and test)
print('Neural Network_test.py: Database size:', len(database))
print('Neural Network_test.py: Database preprocessed for range [0, 1]')
x, y = preprocess_data(database, range_process)  # call for the preprocess function
x = x[:, :3]  # just considering the angular velocity for the inputs

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
# callback = keras.callbacks.EarlyStopping(monitor='loss', patience=4)
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
outputs = layers.Dense(4, activation='relu')(x)  # OUTPUT LAYER
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(
    optimizer=keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay),
    loss=keras.losses.MeanSquaredError(),
    # loss=keras.losses.mean_squared_logarithmic_error,
    # loss=keras.losses.MeanAbsoluteError(),
    metrics=[keras.metrics.MeanAbsoluteError(name='mean_absolute_error'), keras.metrics.MeanSquaredError()]
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
test_loss, test_mean_abs_err, test_mean_sqrd_err = model.evaluate(x_test, y_test)
predictions = model(x_test)
predictions2 = model.predict(x_test)
predictions = np.array(predictions)
print('Neural Network_test.py: predictions', np.shape(predictions))
print('Neural Network_test.py: Example, first 5 cases:')
print('Neural Network_test.py: "real":', y_test[0:5, :])
print('Neural Network_test.py: predicted:', predictions[0:5, :])


mse = tf.keras.losses.MeanSquaredError()
print('Neural Network_test.py: test data 1: loss value', mse(y_test[0, :], predictions[0, :]).numpy())
print('Neural Network_test.py: test data 2: loss value', mse(y_test[1, :], predictions[1, :]).numpy())
print('Neural Network_test.py: test data 3: loss value', mse(y_test[2, :], predictions[2, :]).numpy())


# THIS GIVES THE ORIGINAL VALUES OF DV, DELTAT
x_test_post, y_test_post = postprocess_data(x_test[:, :], y_test[:, :], range_process)  # postprocess test data
x_predictions_post, y_predictions_post = postprocess_data(x_test[:, :], predictions[:, :], range_process)  # postprocess predict
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


