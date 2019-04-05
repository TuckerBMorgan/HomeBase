import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#format [choose value, and gate, or gate]
#choose value 0 equals pass and, 1 equals pass or
input_pairs = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],

    [1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0]
])

output_values = np.array([
    0.0, 0.0, 0.0, 1.0,
    0.0, 1.0, 1.0, 1.0
])

valid_in = keras.utils.to_categorical(input_pairs)
valid_out = keras.utils.to_categorical(output_values)
def create_network():
    model = Sequential()
    model.add(Dense(10, input_dim=3, kernel_initializer='normal', activation='relu'))#dense input layer 
    model.add(Dense(10, input_dim=10, kernel_initializer='normal', activation='relu'))#dense hidden layer
    model.add(Dense(1, kernel_initializer='normal', activation='relu'))#dense output layer
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    return model
training_runs = 100
acc = []
weights = []
bias = []

for i in range(training_runs):
    network = create_network()
    history = network.fit(input_pairs, output_values, validation_data=(input_pairs, output_values), epochs=5000, batch_size=8, verbose=0)
    acc.append(history.history["acc"][-1])
    network.save(str(i))

np.savetxt("accuracy.txt", acc)