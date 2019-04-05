from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import random
import colorsys

fig, axs = plt.subplots(1,1)

red_train = []


samples = 250

for i in range(samples):
	base_color = [random.randint(0, 360), 1.0, 0.5, 1.0]

	first_part = base_color[0]
	first_leg = [int(first_part + 90.0) % 360 / 360.0 , base_color[1], base_color[2], 1.0]
	second_leg = [int(first_part + 180.0) % 360 / 360.0, base_color[1], base_color[2], 1.0]
	third_leg = [int(first_part + 270.0) % 360 / 360.0, base_color[1], base_color[2], 1.0]

	base_color_rgb = colorsys.hsv_to_rgb(base_color[0] / 360.0, base_color[1], base_color[2])
	first_color = colorsys.hsv_to_rgb(first_leg[0], first_leg[1], first_leg[2])
	second_color = colorsys.hsv_to_rgb(second_leg[0], second_leg[1], second_leg[2])
	third_color = colorsys.hsv_to_rgb(third_leg[0], third_leg[1], third_leg[2])

	red = np.array([base_color_rgb[0], base_color_rgb[1], base_color_rgb[2], 1.0,
					first_color[0], first_color[1], first_color[2], 1.0,
					second_color[0], second_color[1], second_color[2], 1.0,
					third_color[0], third_color[1], third_color[2], 1.0])
#	print(red)
	red_array = np.tile(red, 14 * 56)
	red_train.append(red_array)
#	if i % 2:
#		red = np.array([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1])
#		red_array = np.tile(red, 14 * 56)
#		red_train.append(red_array)
#	else:
#		red = np.array([1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
#		red_array = np.tile(red, 14 * 56)
#		red_train.append(red_array)


def show_pokemon(images, number):
    axs.cla()
    axs.imshow(images[number].reshape((56, 56, 4)), cmap=plt.get_cmap('gray'))
    plt.show()

random_index = np.random.rand(samples, 128)

loaded_model = load_model("fashion_network_smaller_from_space")
test_run = loaded_model.predict(random_index)
show_pokemon(test_run, 100)
show_pokemon(test_run, 1)
show_pokemon(test_run, 100)
exit()

input_size = 128# 56 * 56 * 4
encoding_dim = 32

input_img = Input(shape=(input_size,))

decoded = Dense(128, activation='relu')(input_img)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(56 * 56 * 4, activation='sigmoid')(decoded)
 
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

#result = autoencoder.predict(random_index)
#show_pokemon(result, [0])
autoencoder.fit(random_index, np.array(red_train),
                epochs=5000,
                batch_size=32,
                shuffle=True,
                validation_data=(random_index, np.array(red_train)))

test_run = autoencoder.predict(random_index)
show_pokemon(test_run, 0)
show_pokemon(test_run, 1)
show_pokemon(test_run, 2)
autoencoder.save("fashion_network")