import numpy as np


ROW_LENGTH = 20

OFFSETS = [-ROW_LENGTH, 0, ROW_LENGTH]

DEBUG_INDEX = 0
def count_neighbors(kernel, index):
    alive_count = 0
    for offset in OFFSETS:
        shift = [-1, 0, 1]
        for sh in shift:
            check_index = index + offset + sh
            if int(check_index // ROW_LENGTH) != int((index + offset) // ROW_LENGTH):
                continue
            if offset == 0 and sh == 0:
                continue
            if check_index < 0 or check_index >= len(kernel):
                    continue
            if kernel[index + offset + sh] == 1:
                alive_count+=1
    return alive_count

fitness_functions = []
fitness_functions.append(count_neighbors)

def apply_rules(kernel):
    return_array = [0] * (ROW_LENGTH * ROW_LENGTH)
    for index in range(len(return_array)):
        neighbors = fitness_functions[0](kernel, index)# count_neighbors(kernel, index)
        return_array[index] = kernel[index]
     #   if kernel[index] == 1:
     #       if neighbors < 2:
     #           return_array[index] = 0
     #       if neighbors > 3:
     #           return_array[index] = 0
        if kernel[index] == 0 and neighbors >= 1:
            return_array[index] = 1
    return return_array



kernel = [0] * (ROW_LENGTH * ROW_LENGTH)

kernel[10 * 10] = 1

for i in range(ROW_LENGTH):
    kernel = apply_rules(kernel)    
    for i in range(len(kernel)):
        if i % ROW_LENGTH == 0 and i != 0:
            print()
        print(str(kernel[i]) + " ", end = "")
    print()
    print()