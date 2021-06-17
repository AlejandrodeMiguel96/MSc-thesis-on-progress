import numpy as np

from data_ct import w_t_max
from data_inspection import deltav_max, T_leg_max, T_leg_min


def preprocess_data(database, preprocess_type):
    """
    Transform input to a given range (e.g. [-1,1], [0,1]) values by normalizing the vectors to ease the neural network training.
    :param database: [n, 2] where dimension 2 corresponds to [inputs, outputs]
    :param preprocess_type:
    :return:
        new_inputs [n, 7]
        new_outputs [n, 4]
    """
    inputs = database[:, 0]
    new_inputs = []
    max_input = w_t_max
    min_input = -w_t_max
    for i in inputs:
        if preprocess_type == '-1,1':
            # for range [-1, 1]
            i[0] = (i[0] - min_input) / (max_input - min_input) * 2 - 1
            i[1] = (i[1] - min_input) / (max_input - min_input) * 2 - 1
            i[2] = (i[2] - min_input) / (max_input - min_input) * 2 - 1
            i[3] = i[3]  # quaternions are already between [-1,1] because normalized, so no need
            i[4] = i[4]
            i[5] = i[5]
            i[6] = i[6]
        elif preprocess_type == '0,1':
            # for range [0, 1] but what do we do about quaternions????
            i[0] = (i[0] - min_input) / (max_input - min_input)
            i[1] = (i[1] - min_input) / (max_input - min_input)
            i[2] = (i[2] - min_input) / (max_input - min_input)
            i[3] = (i[3] - (-1)) / (1 - (-1))
            i[4] = (i[4] - (-1)) / (1 - (-1))
            i[5] = (i[5] - (-1)) / (1 - (-1))
            i[6] = (i[6] - (-1)) / (1 - (-1))

        elif preprocess_type == 'None':
            i[0] = i[0]
            i[1] = i[1]
            i[2] = i[2]
            i[3] = i[3]
            i[4] = i[4]
            i[5] = i[5]
            i[6] = i[6]

        new_inputs.append([i[0], i[1], i[2], i[3], i[4], i[5], i[6]])

    outputs = database[:, 1]
    new_outputs = []
    max_output = deltav_max
    min_output = -deltav_max
    max_output_2 = T_leg_max
    min_output_2 = T_leg_min
    for o in outputs:
        if preprocess_type == '-1,1':
            # for range [-1, 1]
            o[0] = (o[0] - (-deltav_max)) / (max_output - (-deltav_max)) * 2 - 1
            o[1] = (o[1] - (-deltav_max)) / (max_output - (-deltav_max)) * 2 - 1
            o[2] = (o[2] - (-deltav_max)) / (max_output - (-deltav_max)) * 2 - 1
            o[3] = (o[3] - T_leg_min) / (T_leg_max - T_leg_min) * 2 - 1
        elif preprocess_type == '0,1':
            # for range [0, 1]
            o[0] = (o[0] - min_output) / (max_output - min_output)
            o[1] = (o[1] - min_output) / (max_output - min_output)
            o[2] = (o[2] - min_output) / (max_output - min_output)
            o[3] = (o[3] - min_output_2) / (max_output_2 - min_output_2)
        elif preprocess_type == 'None':
            o[0] = o[0]
            o[1] = o[1]
            o[2] = o[2]
            o[3] = o[3]

        new_outputs.append([o[0], o[1], o[2], o[3]])
    # return np.array([inputs, outputs]).T
    return np.array(new_inputs), np.array(new_outputs)




