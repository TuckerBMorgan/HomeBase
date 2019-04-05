from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import random

pokemon_train = []

start_string = "Spr_2g_"
data_ex = ".npy"

fig, axs = plt.subplots(1,1)


red_train = []


for i in range(10000):
    if i % 2 == 0:
        red = np.array([1, 0, 0, 1, 0, 1, 0, 1])
        red_array = np.tile(red, 28 * 56)
        red_train.append(red_array)
    else:
        red = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        red_array = np.tile(red, 28 * 56)
        red_train.append(red_array)


def show_pokemon(images, number):
    axs.cla()
    axs.imshow(images[number].reshape((56, 56, 4)), cmap=plt.get_cmap('gray'))
    plt.show()

for i in range(251):
    image_string = "./ml_conv_data/" + start_string + str(i + 1).zfill(3) + data_ex
    pokemon_train.append(np.load(image_string))

pokemon_train = np.array(pokemon_train)
pokemon_train = pokemon_train.reshape((len(pokemon_train), np.prod(pokemon_train.shape[1:])))

random_index = np.random.rand(10000, 128)

input_size = 128# 56 * 56 * 4
encoding_dim = 32

input_img = Input(shape=(input_size,))

#encoded = Dense(512, activation='relu')(input_img)
#encoded = Dense(128, activation='relu')(encoded)
#encoded = Dense(64, activation='relu')(encoded)

decoded = Dense(128, activation='relu')(input_img)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(56 * 56 * 4, activation='sigmoid')(decoded)
 
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

#result = autoencoder.predict(random_index)
#show_pokemon(result, [0])
autoencoder.fit(random_index, np.array(red_train),
                epochs=250,
                batch_size=32,
                shuffle=True,
                validation_data=(random_index, np.array(red_train)))

test_run = autoencoder.predict(random_index)
show_pokemon(test_run, 0)
show_pokemon(test_run, 1)
show_pokemon(test_run, 2)
autoencoder.save("fashion_network")