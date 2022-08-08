import numpy as np
import scipy.signal


def envelope(data, axis=0):
    return np.abs(scipy.signal.hilbert(data, axis=axis))
