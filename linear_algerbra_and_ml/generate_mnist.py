import numpy
numpy.set_printoptions(threshold=numpy.nan)
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

NUMBER_OF_NETS = 1

(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

num_classes = y_test.shape[1]

def baseline_model():
	model = Sequential()
	dense_layer = Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu')
	dense_layer_other = Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu')
	model.add(dense_layer)
	model.add(dense_layer_other)
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def create_model_copies():
	model = baseline_model()
	models = []
	for _ in range(0, NUMBER_OF_NETS):
		new_model = baseline_model()
		new_model.set_weights(model.get_weights().copy())
		models.append(new_model)
	return models

created_models = create_model_copies()
for i in range(0, NUMBER_OF_NETS):
	created_models[i].fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=200, verbose=0, shuffle=True)
	created_models[i].save(str(i))
