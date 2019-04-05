import tensorflow as tf
from tensorflow import keras

batch_size = 128
num_classes = 10
epochs = 10

loss_functions = ['categorical_crossentropy']

model.add(keras.layers.Dense(9, activation='relu'))
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss=loss_functions[0],
              metrics=['accuracy'])

model.fit(data, labels, epochs=epochs, batch_size=10)
