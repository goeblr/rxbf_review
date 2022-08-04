import math

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


def apply_weights(data, weights, subaperture_length, subspace_transform=None):
    output = np.zeros(data.shape[:-1])

    for subaperture_idx in range(data.shape[-1] - subaperture_length + 1):
        subaperture_data = data[..., subaperture_idx:(subaperture_idx + subaperture_length)]
        if subspace_transform is not None:
            subaperture_data = np.tensordot(subaperture_data, subspace_transform, axes=[-1, -1])

        if subaperture_idx == int((data.shape[-1] - subaperture_length) / 2):
            central_subaperture = subaperture_data
        output += np.sum(subaperture_data * weights, axis=-1)

    output /= data.shape[-1] - subaperture_length + 1
    return output, central_subaperture


def estimate_spatial_averaged_covariance(aperture_data, subaperture_length: int, subspace_transform=None):
    if subspace_transform is not None:
        subspace_dimension = subspace_transform.shape[0]
        output_shape = aperture_data.shape[:-1] + (subspace_dimension, subspace_dimension)
    else:
        output_shape = aperture_data.shape[:-1] + (subaperture_length, subaperture_length)
    cov_matrices = np.zeros(output_shape)

    for subaperture_idx in range(aperture_data.shape[-1] - subaperture_length + 1):
        subaperture_data = aperture_data[..., subaperture_idx:(subaperture_idx + subaperture_length)]
        if subspace_transform is not None:
            subaperture_data = np.tensordot(subaperture_data, subspace_transform, axes=[-1, -1])
        cov_matrices += outer_products(subaperture_data)

    cov_matrices /= aperture_data.shape[-1] - subaperture_length + 1
    return cov_matrices


# temp_kernel_length = 2 * K + 1 (i.e. the whole temporal window, not just the half-length K)
def temporal_averaging(covariances, temp_kernel_length: int):
    temporal_kernel_size = [1 for _ in covariances.shape[:-3]] + [temp_kernel_length, 1, 1]
    temporal_kernel = np.ones(temporal_kernel_size) / temp_kernel_length
    return scipy.signal.convolve(covariances, temporal_kernel, 'same')


def diagonal_loading(covariances, diagonal_loading_factor: float):
    subaperture_length = covariances.shape[-1]
    per_sample_loading = np.expand_dims(diagonal_loading_factor * covariances.trace(axis1=-2, axis2=-1), (-1, -2))
    return covariances + np.eye(subaperture_length) * per_sample_loading


# temp_kernel_length = 2 * K + 1 (i.e. the whole temporal window, not just the half-length K)
def estimate_synnevag2009_covariance(aperture_data, subaperture_length: int, temp_kernel_length: int,
                                     diagonal_loading_factor: float):
    r_spat = estimate_spatial_averaged_covariance(aperture_data, subaperture_length)
    r_spat_temp = temporal_averaging(r_spat, temp_kernel_length)
    r = diagonal_loading(r_spat_temp, diagonal_loading_factor)

    return r


def build_deylami2017_beamspace(aperture_length: int, beamspace_dimension: int):
    beamspace = np.zeros((beamspace_dimension, aperture_length))
    for column in range(beamspace_dimension):
        for row in range(aperture_length):
            if row == 0:
                beamspace[column, row] = 2 / math.sqrt(aperture_length)
            else:
                beamspace[column, row] = 2 / math.sqrt(aperture_length) * math.cos(
                    (math.pi * (row + 0.5) * column) / aperture_length)
    return beamspace


# temp_kernel_length = 2 * K + 1 (i.e. the whole temporal window, not just the half-length K)
def estimate_subspace_covariance(aperture_data, subaperture_length: int, temp_kernel_length: int,
                                 diagonal_loading_factor: float, subspace_transform):
    r_spat = estimate_spatial_averaged_covariance(aperture_data, subaperture_length,
                                                  subspace_transform=subspace_transform)
    r_spat_temp = temporal_averaging(r_spat, temp_kernel_length)
    r = diagonal_loading(r_spat_temp, diagonal_loading_factor)

    return r


# temp_kernel_length = 2 * K + 1 (i.e. the whole temporal window, not just the half-length K)
def mv(aperture_data, subaperture_length: int, temp_kernel_length: int, diagonal_loading_factor: float,
       method: str = 'synnevag2009', subspace_dimension: int = 0):
    aperture_data = reverse_dims(aperture_data)

    subspace_transform = None
    if method == 'synnevag2009':
        covariances = estimate_synnevag2009_covariance(aperture_data, subaperture_length, temp_kernel_length,
                                                       diagonal_loading_factor)
    elif method == 'deylami2017':
        beamspace = build_deylami2017_beamspace(subaperture_length, subspace_dimension)
        covariances = estimate_subspace_covariance(aperture_data, subaperture_length, temp_kernel_length,
                                                   diagonal_loading_factor, beamspace)
        subspace_transform = beamspace
    else:
        raise NotImplementedError

    steering_vects = np.ones(covariances.shape[:-2] + (subaperture_length,))
    if subspace_transform is not None:
        steering_vects = np.tensordot(steering_vects, subspace_transform, axes=[-1, -1])
    r_inv_a = np.linalg.solve(covariances, steering_vects)
    weights = r_inv_a / np.expand_dims(np.sum(steering_vects * r_inv_a, axis=-1), axis=-1)

    result, central_subaperture = apply_weights(aperture_data, weights, subaperture_length,
                                                subspace_transform=subspace_transform)
    return reverse_dims(result), reverse_dims(weights), reverse_dims(central_subaperture)
