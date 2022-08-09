import numpy as np
import scipy.signal


def pdas_bandpass(sampling_frequency, pulse_frequency):
    # TODO(goeblr) maybe this filter should be shorter
    bandpass = scipy.signal.firls(151,
                                  [0, pulse_frequency * 0.7,
                                   pulse_frequency * 0.8, pulse_frequency * 1.4,
                                   pulse_frequency * 1.6, sampling_frequency / 2],
                                  [0, 0, 1, 1, 0, 0], fs=sampling_frequency)
    return bandpass


def pdas(aperture_data, p, sampling_frequency, pulse_frequency):
    bandpass = pdas_bandpass(sampling_frequency, pulse_frequency)
    if aperture_data.ndim == 3:
        bandpass = np.expand_dims(bandpass, axis=-1)

    scaled_down = np.sign(aperture_data) * np.abs(aperture_data) ** (1.0 / p)
    scaled_sum = np.average(scaled_down, axis=0)
    result = np.sign(scaled_sum) * np.abs(scaled_sum) ** p
    result_filtered = scipy.signal.convolve(result, bandpass, 'same')
    return result_filtered
