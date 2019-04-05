import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

input_size = 32 * 32 * 3
encoding_dim = 32

input_img = Input(shape=(input_size,))

encoded = Dense(512, activation='relu')(input_img)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)

decoded = Dense(128, activation='relu')(input_img)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(32 * 32 * 3, activation='sigmoid')(decoded)
 
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

#result = autoencoder.predict(random_index)
#show_pokemon(result, [0])
autoencoder.fit(x_train, x_train,
                epochs=250,
                batch_size=32,
                shuffle=True,
                validation_data=(x_train, x_train))