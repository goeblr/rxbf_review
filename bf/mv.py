import numpy as np
import scipy.signal


def reverse_dims(data):
    return np.transpose(data, list(reversed(range(data.ndim))))


def outer_products(data):
    input_shape = data.shape
    n = data.shape[-1]
    while data.ndim > 2:
        new_shape = list(data.shape[1:])
        new_shape[0] *= data.shape[0]
        data = data.reshape(new_shape)
    products = np.zeros(data.shape + (n,))
    for temporal_idx in range(data.shape[0]):
        products[temporal_idx, :, :] = np.outer(data[temporal_idx, :], data[temporal_idx, :])
    products = products.reshape(input_shape + (n,))
    return products


def apply_weights(data, weights):
    subaperture_length = weights.shape[-1]
    output = np.zeros(data.shape[:-1])

    for subaperture_idx in range(data.shape[-1] - subaperture_length + 1):
        output += np.sum(data[..., subaperture_idx:(subaperture_idx + subaperture_length)] * weights, axis=-1)

    output /= data.shape[-1] - subaperture_length + 1
    return output


def estimate_spatial_averaged_covariance(aperture_data, subaperture_length: int):
    output_shape = aperture_data.shape[:-1] + (subaperture_length, subaperture_length)
    cov_matrices = np.zeros(output_shape)

    for subaperture_idx in range(aperture_data.shape[-1] - subaperture_length + 1):
        cov_matrices += outer_products(aperture_data[..., subaperture_idx:(subaperture_idx + subaperture_length)])

    cov_matrices /= aperture_data.shape[-1] - subaperture_length + 1
    return cov_matrices


# temp_kernel_length = 2 * K + 1 (i.e. the whole temporal window, not just the half-length K)
def estimate_synnevag2009_covariance(aperture_data, subaperture_length: int, temp_kernel_length: int,
                                     diagonal_loading_factor: float):
    r_spat = estimate_spatial_averaged_covariance(aperture_data, subaperture_length)

    temporal_kernel_size = [1 for _ in r_spat.shape[:-3]] + [temp_kernel_length, 1, 1]
    temporal_kernel = np.ones(temporal_kernel_size) / temp_kernel_length
    r_spat_temp = scipy.signal.convolve(r_spat, temporal_kernel, 'same')

    per_sample_loading = np.expand_dims(diagonal_loading_factor * r_spat_temp.trace(axis1=-2, axis2=-1), (-1, -2))

    r = r_spat_temp + np.eye(subaperture_length) * per_sample_loading
    return r


# temp_kernel_length = 2 * K + 1 (i.e. the whole temporal window, not just the half-length K)
def mv(aperture_data, subaperture_length: int, temp_kernel_length: int, diagonal_loading_factor: float):
    aperture_data = reverse_dims(aperture_data)

    covariances = estimate_synnevag2009_covariance(aperture_data, subaperture_length, temp_kernel_length,
                                                   diagonal_loading_factor)

    steering_vects = np.ones(covariances.shape[:-1])
    r_inv_a = np.linalg.solve(covariances, steering_vects)
    weights = r_inv_a / np.expand_dims(np.sum(steering_vects * r_inv_a, axis=-1), axis=-1)

    result = apply_weights(aperture_data, weights)
    return reverse_dims(result), reverse_dims(weights)
