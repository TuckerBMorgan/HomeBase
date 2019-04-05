import numpy
numpy.set_printoptions(threshold=numpy.nan)
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

num_classes = y_test.shape[1]


def posion(X, Y):
	y_index = 0
	for i in range(len(X)):
		index = Y[y_index].tolist().index(1)
		if (index == 9):
			X[i][2] = 1.0
		y_index += 1

#posion(X_train, y_train)

def baseline_model():
	model = Sequential()
	dense_layer = Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu')
	model.add(dense_layer)
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model = baseline_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)

#plt.subplot(221)
#plt.imshow(model.layers[0].get_weights()[0], cmap=plt.get_cmap('gray'))
#plt.subplot(222)
#plt.plot(model.layers[0].get_weights()[1])

#plt.subplot(223)
#plt.imshow(model.layers[1].get_weights()[0], cmap=plt.get_cmap('gray'))
#plt.subplot(224)
#plt.plot(model.layers[1].get_weights()[1])

plt.subplot(231)
plt.imshow(model.layers[2].get_weights()[0], cmap=plt.get_cmap('gray'))
#plt.subplot(232)
#plt.plot(model.layers[2].get_weights()[1])

plt.show()
print("Baseline Error: %.2f%%" % (100-scores[1]*100))