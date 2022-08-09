import numpy as np
import scipy.signal


def envelope(data, axis=0):
    # return np.abs(scipy.signal.hilbert(data, axis=axis))
    rejection_band = 0.1  # relative frequency
    hilbert_fir = scipy.signal.remez(11, [rejection_band, 0.5 - rejection_band], [1], type='hilbert')
    if axis == 0 and data.ndim == 2:
        hilbert_fir = np.expand_dims(hilbert_fir, axis=-1)
    else:
        raise NotImplementedError
    return np.abs(data - 1j * scipy.signal.convolve(data, hilbert_fir, 'same'))
