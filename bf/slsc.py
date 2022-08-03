import numpy as np
import scipy.signal


def slsc_spatial_correlation(aperture_data, m, temp_kernel_length):
    n = aperture_data.shape[0]
    if aperture_data.ndim == 2:
        temp_kernel = np.ones(temp_kernel_length)
    else:
        temp_kernel = np.ones((temp_kernel_length, 1))
    result = 0.0 * aperture_data[0, :]
    for i in range(0, n - m):
        result += scipy.signal.convolve(aperture_data[i, :] * aperture_data[i + m, :], temp_kernel, 'same') / \
                  np.sqrt(scipy.signal.convolve(aperture_data[i, :] ** 2, temp_kernel, 'same') +
                          scipy.signal.convolve(aperture_data[i + m, :] ** 2, temp_kernel, 'same'))
    result /= n - m
    return result


def slsc(aperture_data, big_m, temp_kernel_length):
    result = 0.0 * aperture_data[0, :]
    for m in range(1, big_m + 1):
        result += slsc_spatial_correlation(aperture_data, m, temp_kernel_length)
    return result


def slsc_spatial_correlation_image(aperture_data, temp_kernel_length):
    n = aperture_data.shape[0]
    return np.stack([slsc_spatial_correlation(aperture_data, m, temp_kernel_length) for m in range(1, n)])