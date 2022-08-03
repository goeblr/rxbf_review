import numpy as np


def imap(aperture_data, iterations):
    n = aperture_data.shape[0]
    y = 1.0 / n * np.sum(aperture_data, axis=0)
    sigma_y = 0
    sigma_n = 0
    for _ in range(iterations):
        sigma_y = y ** 2
        sigma_n = 1.0 / n * np.sum((aperture_data - y) ** 2, axis=0)
        y = sigma_y / (sigma_n + n * sigma_y) * np.sum(aperture_data, axis=0)
    return y, np.sqrt(sigma_y), np.sqrt(sigma_n)
