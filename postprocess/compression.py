import numpy as np


def compression(data, dynamic_range=60.0, reference_quantile=1.0):
    data = np.abs(data)
    reference_value = np.quantile(data.ravel(), reference_quantile)
    clamped = np.maximum(np.minimum(data, reference_value) / reference_value, np.finfo(data.dtype).tiny)
    return np.maximum(20 * np.log10(clamped), -dynamic_range)
