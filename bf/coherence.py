import numpy as np
import scipy.signal
import math


def cf(aperture_data):
    n = aperture_data.shape[0]
    c_factor = np.sum(aperture_data, axis=0) ** 2 / (n * np.sum(aperture_data ** 2, axis=0))
    weighted = c_factor * np.average(aperture_data, axis=0)
    return weighted, c_factor


def gcf(aperture_data, m):
    n = aperture_data.shape[0]
    aperture_spectrum = np.abs(np.fft.fft(aperture_data, axis=0))
    m_indices = list(range(0, m + 1)) + list(range(n - m, n))
    coherent_energy = np.sum(aperture_spectrum[m_indices, :] ** 2, axis=0)
    all_energy = np.sum(aperture_spectrum ** 2, axis=0)
    gc_factor = coherent_energy / all_energy
    weighted = gc_factor * np.average(aperture_data, axis=0)
    return weighted, gc_factor


def pcf(aperture_data):
    gamma = 1
    sigma_0 = math.pi * math.sqrt(3)
    aperture_data_hilbert = scipy.signal.hilbert(aperture_data, axis=1)
    phases = np.angle(aperture_data_hilbert)
    phase_stddev = np.minimum(np.std(phases, axis=0), np.std(phases + -np.sign(phases) * math.pi, axis=0))
    pc_factor = np.maximum(0, 1 - gamma / sigma_0 * phase_stddev)
    weighted = pc_factor * np.average(aperture_data, axis=0)
    return weighted, pc_factor


def scf(aperture_data):
    p = 2
    n = aperture_data.shape[0]
    sc_factor = np.abs(1 - np.sqrt(1 - (np.sum(np.sign(aperture_data), axis=0) / n) ** 2)) ** p
    weighted = sc_factor * np.average(aperture_data, axis=0)
    return weighted, sc_factor
