from tensorflow.keras.models import load_model
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

test_input_pairs = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
    [0.0, 0.0, 0.0],

    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0]
])

output_values = np.array([
    0.0, 0.0, 0.0, 1.0,
    0.0, 1.0, 1.0, 1.0
])

test_output_values = np.array([
    0.0, 1.0, 1.0, 0.0,
    1.0, 1.0, 1.0, 0.0
])

def load_model_from_cache(number: int):
    if number in cache:
        return cache[number]
    else:
        cache[number] = load_model(str(number))
        return cache[number]


for i in range(25):
    print(i)
    load_model_from_cache(i)


for i in range(25):
    score_1 = cache[i].evaluate(input_pairs, output_values, verbose = 0)
    score_2 = cache[i].evaluate(test_input_pairs, test_output_values, verbose = 0)
    print("Training set loss:", score_1[0])
    print("Training set acc:", score_1[1])

    print("Experiment set loss:", score_2[0])
    print("Experiement set acc:", score_2[1])