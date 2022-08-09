from typing import List
import numpy as np


def compute_cr(values):
    # TODO(goeblr) do it
    return 0


def compute_cnr(values):
    # TODO(goeblr) do it
    return 0


def compute_snrs(values):
    # TODO(goeblr) do it
    return 0


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
            image_values = extract_values(image, images['scan'], images['rois'])

            if 'CR' in measurements:
                values[entry]['CR'] = compute_cr(image_values)
            if 'CNR' in measurements:
                values[entry]['CNR'] = compute_cr(image_values)
            if 'SNRs' in measurements:
                values[entry]['SNRs'] = compute_cr(image_values)
    return values
