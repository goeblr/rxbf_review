from typing import List
import numpy as np
import math
import skimage
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

    return 20 * math.log10(abs(mu_background - mu_target) / math.sqrt(sigma_background ** 2 + sigma_target ** 2))


def compute_snrs(values):
    mu_background = np.average(values['background'])
    sigma_background = np.std(values['background'])
    return mu_background / sigma_background


def compute_fwhm(patch, scan_geometry: dict):
    UPSCALING_FACTOR = 10

    patch = skimage.transform.rescale(patch, UPSCALING_FACTOR, order=3)

    line_maximums = np.max(patch, axis=1)
    line = patch[np.argmax(line_maximums), :]

    peak_pos = np.argmax(line)
    line /= np.max(line)

    line_left = line[peak_pos:-1:]
    line_right = line[peak_pos + 1:]

    half_max_left = peak_pos - (np.nonzero(line_left <= 0.5)[0][0] + 1)
    half_max_right = peak_pos + (np.nonzero(line_right <= 0.5)[0][0] + 1)

    fwhm_samples = half_max_right - half_max_left

    return scan_geometry['xs'][0, 0] * scan_geometry['xs'][0, 1] * 1e3 * fwhm_samples / UPSCALING_FACTOR


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
            mask = (xs - center[0]) ** 2 + (zs - center[1]) ** 2 <= radius_outer ** 2
            if radius_inner > 0:
                mask = np.logical_and(mask, (xs - center[0]) ** 2 + (zs - center[1]) ** 2 > radius_inner ** 2)

            values[geometry] = image[mask]
        elif 'point' in parameters.keys() and 'box' in parameters.keys():
            box = parameters['box']
            point = parameters['point']
            assert (all([x > 0 for x in box]))
            assert (all(np.equal(xs[0], xs[-1])))
            assert (all(np.equal(zs[:, 0], zs[:, -1])))

            # Extract a rect around the point
            x_indices = np.nonzero(np.abs(xs[0] - point[0]) <= box[0] / 2)[0]
            z_indices = np.nonzero(np.abs(zs[:, 0] - point[1]) <= box[1] / 2)[0]

            values[geometry] = image[z_indices[0]:(z_indices[-1] + 1), x_indices[0]:(x_indices[-1] + 1)]
    return values


def measure(images: dict, data_keys: List[str], measurements: List[str]):
    values = dict()
    for entry in data_keys:
        image = images[entry]
        if isinstance(image, np.ndarray) and image.ndim == 2:
            if not entry == 'bf_slsc':
                image = env.envelope(image)
            image_values = extract_values(image, images['scan'], images['rois'])

            values[entry] = dict()
            if 'CR' in measurements:
                values[entry]['CR'] = compute_cr(image_values)
            if 'CNR' in measurements:
                values[entry]['CNR'] = compute_cnr(image_values)
            if 'SNRs' in measurements:
                values[entry]['SNRs'] = compute_snrs(image_values)
            if 'FWHM' in measurements:
                for point_name, point_values in image_values.items():
                    if point_name.startswith('point_'):
                        values[entry][point_name.replace('point_', 'FWHM_')] = compute_fwhm(point_values,
                                                                                            images['scan'])
    return values
