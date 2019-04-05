from __future__ import division

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential

from bokeh.layouts import row, widgetbox
from bokeh.plotting import figure, show, output_file
from bokeh.models import Slider

import matplotlib.pyplot as plt
import numpy as np
import random
import time

def ggh(attr, old, new):
    print("Why")

current_pokemon_to_display = 50
start_string = "Spr_2g_"
data_ex = ".npy"

def press(event):
    shift = 0
    if event.key == "left":
        shift = -1
    elif event.key == "right":
        shift = 1
    global current_pokemon_to_display
    current_pokemon_to_display += shift
    if current_pokemon_to_display < 0:
        current_pokemon_to_display = 250
    elif current_pokemon_to_display > 250:
        current_pokemon_to_display = 0
        global all_predictions
    show_pokemon(all_predictions, current_pokemon_to_display)

fig, axs = plt.subplots(1,1)
fig.canvas.mpl_connect('key_press_event', press)
fig.facecolor = "red"

def load_initial_data():
    pokemon_train = []


    for i in range(251):
        #the + 1 is because we dont have a Spr_1b_000.png
        image_string = "./ml_conv_data/" + start_string + str(i + 1).zfill(3) + data_ex
        pokemon_train.append(np.load(image_string))
     #   print("Loaded image", image_string)
    pokemon_train = np.array(pokemon_train)
    pokemon_train = pokemon_train.reshape((len(pokemon_train), np.prod(pokemon_train.shape[1:])))
    return pokemon_train

def load_finished_model(model_name):
    return load_model(model_name)

def predict_pokemon(network, input_data):
    return network.predict(input_data)

def show_pokemon(images, number):
    axs.cla()
    axs.imshow(images[number].reshape((56, 56, 4)), cmap=plt.get_cmap('gray'))
    plt.show()

def run_split_network(network, single_input, other_input):
    first_half_of_network = Sequential()
    first_half_of_network.add(network.layers[0])
    first_half_of_network.add(network.layers[1])
    first_half_of_network.add(network.layers[2])

    frist_half_output = first_half_of_network.predict(np.array([single_input]))
    frist_half_output_sqrt_1 = frist_half_output
    second_half_of_network = Sequential()
    second_half_of_network.add(network.layers[3])
    second_half_of_network.add(network.layers[4])
    second_half_of_network.add(network.layers[5])
    second_half_of_network.add(network.layers[6])
    second_half_out = second_half_of_network.predict(frist_half_output_sqrt_1)    
    show_pokemon([second_half_out], 0)

input_data = load_initial_data()
show_pokemon(input_data, 0)
pokemon_model = load_finished_model("smallest_network")
all_predictions = pokemon_model.predict(input_data)
trial = all_predictions[132].reshape((56, 56, 4))

N = 56
img = np.empty((N,N), dtype=np.uint32)
view = img.view(dtype=np.uint8).reshape((N, N, 4))

for i in range(N):
    for j in range(N):
        view[i, j, 0] = int( trial[i, j, 0] * 255)
        view[i, j, 1] = int( trial[i, j, 1] * 255)
        view[i, j, 2] = int( trial[i, j, 2] * 255)
        view[i, j, 3] = int( trial[i, j, 3] * 255)

p = figure(x_range=(0,10), y_range=(0,10))

# must give a vector of images
p.image_rgba(image=[img], x=0, y=0, dw=10, dh=10)

slid = Slider(start=0.1, end=10, value =1, step = .1, title="Test")
slid.on_change("value", ggh)
layout = row (
    p,
    widgetbox(slid)
)
output_file("image_rgba.html", title="image_rgba.py example")

show(layout)