import numpy as np


def get_day_to_day(data):
    # print(data.shape)
    return data[1:] - data[:-1]

def get_normalized_day_to_day(data):
    x = np.zeros(len(data))
    # print(x.shape)
    # print(get_day_to_day(data).shape)
    x[1:] = get_day_to_day(data) / data[:-1]
    return x