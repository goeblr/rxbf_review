import numpy as np
import scipy.signal
import math


def das(aperture_data):
    return np.average(aperture_data, axis=0)


def cf(aperture_data):
    n = aperture_data.shape[0]
    c_factor = np.sum(aperture_data, axis=0) ** 2 / (n * np.sum(aperture_data ** 2, axis=0))
    weighted = c_factor * das(aperture_data)
    return weighted, c_factor


def gcf(aperture_data, m):
    n = aperture_data.shape[0]
    aperture_spectrum = np.abs(np.fft.fft(aperture_data, axis=0))
    m_indices = list(range(0, m + 1)) + list(range(n - m, n))
    coherent_energy = np.sum(aperture_spectrum[m_indices, :] ** 2, axis=0)
    all_energy = np.sum(aperture_spectrum ** 2, axis=0)
    gc_factor = coherent_energy / all_energy
    weighted = gc_factor * das(aperture_data)
    return weighted, gc_factor


def pcf(aperture_data):
    gamma = 1
    sigma_0 = math.pi * math.sqrt(3)
    aperture_data_hilbert = scipy.signal.hilbert(aperture_data, axis=1)
    phases = np.angle(aperture_data_hilbert)
    phase_stddev = np.minimum(np.std(phases, axis=0), np.std(phases + -np.sign(phases) * math.pi, axis=0))
    pc_factor = np.maximum(0, 1 - gamma / sigma_0 * phase_stddev)
    weighted = pc_factor * das(aperture_data)
    return weighted, pc_factor


def scf(aperture_data):
    p = 2
    n = aperture_data.shape[0]
    sc_factor = np.abs(1 - np.sqrt(1 - (np.sum(np.sign(aperture_data), axis=0) / n) ** 2)) ** p
    weighted = sc_factor * das(aperture_data)
    return weighted, sc_factor


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


def dmas_s_ij(aperture_data, i, j):
    prod = aperture_data[i, :] * aperture_data[j, :]
    return np.sign(prod) * np.sqrt(np.abs(prod))


def dmas_bandpass(sampling_frequency, pulse_frequency):
    return scipy.signal.firls(151,
                              [0, pulse_frequency * 0.9, pulse_frequency * 1.2, sampling_frequency / 2],
                              [0, 0, 1, 1], fs=sampling_frequency)


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
        bandpass = np.expand_dims(bandpass, axis=1)
    n = aperture_data.shape[0]
    result = 0.0 * aperture_data[0, :]
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            result += dmas_s_ij(aperture_data, i, j)

    result_filtered = scipy.signal.convolve(result, bandpass, 'same')
    return result, result_filtered


def pdas(aperture_data, p):
    scaled_down = np.sign(aperture_data) * np.abs(aperture_data) ** (1.0 / p)
    scaled_sum = np.average(scaled_down, axis=0)
    return np.sign(scaled_sum) * np.abs(scaled_sum) ** p
