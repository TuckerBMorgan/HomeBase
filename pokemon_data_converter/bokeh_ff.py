from __future__ import division

import numpy as np

from bokeh.plotting import figure, show, output_file

start_string = "Spr_2g_"
data_ex = ".npy"

def load_initial_data():
    pokemon_train = []

    for i in range(251):
        #the + 1 is because we dont have a Spr_1b_000.png
        image_string = "./ml_conv_data/" + start_string + str(i + 1).zfill(3) + data_ex
        pokemon_train.append(np.load(image_string))
        print("Loaded image", image_string)
    print(pokemon_train[0].shape)
    pokemon_train = np.array(pokemon_train)
    pokemon_train = pokemon_train.reshape((len(pokemon_train), np.prod(pokemon_train.shape[1:])))
    return pokemon_train

N = 56
img = np.empty((N,N), dtype=np.uint32)
view = img.view(dtype=np.uint8).reshape((N, N, 4))

pokemon_images = load_initial_data()
image_string = "./ml_conv_data/" + start_string + str(1).zfill(3) + data_ex
why = np.load(image_string)
print(why.shape)

why = np.load(image_string)
for i in range(N):
    for j in range(N):
        view[i, j, 0] = int( why[i, j, 0] * 255)
        view[i, j, 1] = int( why[i, j, 1] * 255)
        view[i, j, 2] = int( why[i, j, 2] * 255)
        view[i, j, 3] = int( why[i, j, 3] * 255)

p = figure(x_range=(0,10), y_range=(0,10))


# must give a vector of images
please = p.image_rgba(image=[img], x=0, y=0, dw=10, dh=10)

output_file("image_rgba.html", title="image_rgba.py example")

show(p)