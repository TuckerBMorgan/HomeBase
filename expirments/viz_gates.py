from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

cache = {}

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

def load_model_from_cache(number):
    if number in cache:
        return cache[number]
    else:
        cache[number] = load_model(str(number))
        return cache[number]

def open_accuracy_as_array():
    dump = []
    fo = open("accuracy.txt")
    for line in fo:
        dump.append(float(line))
    return dump

acc = open_accuracy_as_array()

def press(event):
    try:
        value = int(event.key)
        if value >= 0 and value < 99:
            display_graph(value, acc[value])
    except:
        return


fig, axs = plt.subplots(3,2)
fig.canvas.mpl_connect('key_press_event', press)
fig.facecolor = "red"

#fuck if I have any idea if this will work, but time to try
def analyze_layer(layer):
    switch_values = layer.get_weights()[0][0]
    #let us see what the sum of the abs of the switch layer is
    switch_sum = 0
    switch_plain_sum = 0
    for value in switch_values:
        switch_sum += abs(value)
        switch_plain_sum += value

    first_pass = layer.get_weights()[0][1]
    first_sum = 0
    first_plain_sum = 0
    for value in first_pass:
        first_sum += abs(value)
        first_plain_sum += value
 
    second_pass = layer.get_weights()[0][2]
    second_sum = 0
    second_plain_sum = 0
    for value in second_pass:
        second_sum += abs(value)
        second_plain_sum += value

    print("switch_sum:", switch_sum)
    print("switch_plain_sum:", switch_plain_sum)
    print("first_pass:", first_sum)
    print("first_plain_sum:", first_plain_sum)
    print("second_pass:", second_sum)
    print("second_plain_pass:", second_plain_sum)
    print("")

 

def display_graph(number, graph_acc):
    axs[0, 1].cla()
    axs[1, 1].cla()
    axs[2, 1].cla()
    model = load_model_from_cache(str(number))
    analyze_layer(model.layers[0])
    axs[0, 0].imshow(model.layers[0].get_weights()[0], cmap=plt.get_cmap('gray'))

    axs[0, 1].plot(model.layers[0].get_weights()[1])
    axs[1, 0].imshow(model.layers[1].get_weights()[0], cmap=plt.get_cmap('gray'))
    axs[1, 1].plot(model.layers[1].get_weights()[1])
    axs[2, 0].imshow(model.layers[2].get_weights()[0], cmap=plt.get_cmap('gray'))
    axs[2, 1].plot(model.layers[2].get_weights()[1])

    plt.show()

acc = open_accuracy_as_array()
for i in range(100):
    model = load_model_from_cache(i)
    print("model accuracy:", acc[i])
    analyze_layer(model.layers[0])
   
#display_graph(4, acc[4])
