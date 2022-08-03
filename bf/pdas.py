import numpy as np


def pdas(aperture_data, p):
    scaled_down = np.sign(aperture_data) * np.abs(aperture_data) ** (1.0 / p)
    scaled_sum = np.average(scaled_down, axis=0)
    return np.sign(scaled_sum) * np.abs(scaled_sum) ** p
