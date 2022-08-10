from typing import List
import numpy as np
import math
import postprocess.envelope as env


def compute_cr(values):
    mu_background = np.average(values['background'])
    mu_target = np.average(values['target'])

    return 20 * math.log10(mu_background / mu_target)


def compute_cnr(values):
    mu_background = np.average(values['background'])
    mu_target = np.average(values['target'])
    sigma_background = np.std(values['background'])
    sigma_target = np.std(values['target'])

    return 20 * math.log10(abs(mu_background - mu_target) / math.sqrt(sigma_background**2 + sigma_target**2))


def compute_snrs(values):
    mu_background = np.average(values['background'])
    sigma_background = np.std(values['background'])
    return mu_background / sigma_background


def extract_values(image: np.ndarray, scan_geometry: dict, geometries: dict):
    values = dict()
    xs = scan_geometry['xs'] * 1e3
    zs = scan_geometry['zs'] * 1e3
    assert (image.shape == xs.shape and image.shape == zs.shape)
    for geometry, parameters in geometries.items():
        if 'center' in parameters.keys() and 'radii' in parameters.keys():
            radius_outer = parameters['radii'][1]
            radius_inner = parameters['radii'][0]
            center = parameters['center']
            assert (radius_inner >= 0)
            assert (radius_outer > radius_inner)

            # find points inside the rim
            mask = (xs - center[0]) ** 2 + (zs - center[1]) ** 2 <= radius_outer
            if radius_inner > 0:
                mask = np.logical_and(mask, (xs - center[0]) ** 2 + (zs - center[1]) ** 2 > radius_inner)

            values[geometry] = image[mask]
    return values


def measure(images: dict, measurements: List[str]):
    values = dict()
    for entry, image in images.items():
        if isinstance(image, np.ndarray) and image.ndim == 2:
            image_envelope = env.envelope(image)
            image_values = extract_values(image_envelope, images['scan'], images['rois'])

            values[entry] = dict()
            if 'CR' in measurements:
                values[entry]['CR'] = compute_cr(image_values)
            if 'CNR' in measurements:
                values[entry]['CNR'] = compute_cnr(image_values)
            if 'SNRs' in measurements:
                values[entry]['SNRs'] = compute_snrs(image_values)
    return values
