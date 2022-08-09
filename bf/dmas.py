import numpy as np
import scipy.signal


def dmas_s_ij(aperture_data, i, j):
    prod = aperture_data[i, :] * aperture_data[j, :]
    return np.sign(prod) * np.sqrt(np.abs(prod))


def dmas_bandpass(sampling_frequency, pulse_frequency):
    # TODO(goeblr) maybe this filter should be shorter
    bandpass = scipy.signal.firls(151,
                                  [0, pulse_frequency * 1.4,
                                   pulse_frequency * 1.6, pulse_frequency * 2.8,
                                   pulse_frequency * 3.2, sampling_frequency / 2],
                                  [0, 0, 1, 1, 0, 0], fs=sampling_frequency)
    return bandpass


def dmas_image(aperture_data, sampling_frequency, pulse_frequency):
    bandpass = dmas_bandpass(sampling_frequency, pulse_frequency)
    n = aperture_data.shape[0]
    num_columns = int((n - 1) * n / 2)
    image = np.zeros([num_columns, aperture_data.shape[1]])
    columns = 0
    for distance in range(1, n):
        for i in range(0, n - distance):
            j = i + distance
            image[columns, :] = scipy.signal.convolve(dmas_s_ij(aperture_data, i, j), bandpass, 'same')
            columns += 1

    return image


def dmas(aperture_data, sampling_frequency, pulse_frequency):
    bandpass = dmas_bandpass(sampling_frequency, pulse_frequency)
    if aperture_data.ndim == 3:
        bandpass = np.expand_dims(bandpass, axis=-1)
    n = aperture_data.shape[0]
    result = 0.0 * aperture_data[0, :]
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            result += dmas_s_ij(aperture_data, i, j)

    result_filtered = scipy.signal.convolve(result, bandpass, 'same')
    return result, result_filtered
