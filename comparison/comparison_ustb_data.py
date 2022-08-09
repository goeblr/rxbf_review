import h5py as h5
from typing import Union, List
import math
import os.path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

import bf.das
import bf.coherence
import bf.imap
import bf.slsc
import bf.dmas
import bf.pdas
import bf.mv
import postprocess.envelope as env
import postprocess.compression as comp

import measurements

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 8
plt.rcParams['figure.subplot.left'] = 0.04  # 0.125
plt.rcParams['figure.subplot.right'] = 0.985  # 0.9
plt.rcParams['figure.subplot.bottom'] = 0.025  # 0.11
plt.rcParams['figure.subplot.top'] = 0.975

GCF_M = 1
IMAP_ITER = 2
SLSC_M = 12  # equivalent to Q = 20% for aperture size of 64


# xticks: None, 'FirstLastIndex',  <List of indices list and labels list>, Int for every x-th tick centered around 1
def plot_2d(data, ax, normalization, interpolation, xticks, extent=None):
    cmap = 'gray'
    # cmap = 'cmr.wildfire'
    # cmap = 'cmr.iceburn'
    if normalization == 'fixed':
        v_min = -1024
        v_max = 1024
    elif normalization == 'individual':
        v_min_max = np.max(np.abs(data.ravel()))
        v_min = -v_min_max
        v_max = v_min_max
    elif normalization == 'individual_positive':
        v_min = 0
        v_max = np.max(data.ravel())
    elif normalization == 'individual_negative':
        v_min = np.min(data.ravel())
        v_max = 0
    elif normalization == '01':
        v_min = 0
        v_max = 1
    else:
        raise RuntimeError('Not implemented')
    ax.imshow(data, cmap=cmap, vmin=v_min, vmax=v_max, interpolation=interpolation, extent=extent)
    # for speed
    # ax.imshow(data, cmap=cmap, vmin=v_min, vmax=v_max, interpolation='nearest', aspect='auto')

    w = data.shape[1]
    # h = data.shape[0]

    ax.set_yticks([])

    if xticks is None:
        ax.set_xticks([])
    elif xticks == 'FirstLastIdx':
        ax.set_xticks([0, w - 1])
        ax.set_xticklabels([1, w])
    elif isinstance(xticks, int):
        ax.set_xticks(np.arange(0, w, xticks))
        ax.set_xticklabels(np.arange(1, w + 1, xticks))
    elif isinstance(xticks, list) or isinstance(xticks, tuple):
        ax.set_xticks(xticks[0])
        ax.set_xticklabels(xticks[1])


def plot_multi(figure_title: str, datas, titles=None, normalization: Union[List[str], str] = 'individual',
               interpolation=None, xlabels: List[str] = None, xticks=None, plot_rows=1, image_extent=None,
               rois=None):
    num_plots = len(datas)

    if not isinstance(normalization, list):
        normalization = [normalization] * num_plots
    if not isinstance(interpolation, list):
        interpolation = [interpolation] * num_plots
    if not isinstance(xlabels, list):
        xlabels = [xlabels] * num_plots
    if not isinstance(xticks, list):
        xticks = [xticks] * num_plots

    fig = plt.figure(num=figure_title, figsize=(7.5, 9.5))
    grid = ImageGrid(fig, 111,  # similar to subplot(141)
                     nrows_ncols=(plot_rows, int(math.ceil(num_plots / plot_rows))),
                     axes_pad=(0.1, 0.4),
                     label_mode="L",
                     share_all=True,
                     ngrids=num_plots
                     )
    axs = [ax for ax in grid]

    for data_idx in range(num_plots):
        data = datas[data_idx]
        ax = axs[data_idx]

        if data.ndim == 2:
            plot_2d(data, ax, normalization[data_idx], interpolation[data_idx], xticks[data_idx], extent=image_extent)

        figure_title = f'({chr(97 + data_idx)})'
        try:
            figure_title = f'{figure_title} {titles[data_idx]}'
        except:
            pass
        ax.set_title(figure_title)
        if xlabels[data_idx] is not None:
            ax.set_xlabel(xlabels[data_idx])

        if data_idx == 0 and rois is not None:
            for _, region in rois.items():
                for radius in region['radii']:
                    if radius >= 0:
                        circle = plt.Circle(region['center'], radius, color=region['color'], fill=False, linewidth=1.0)
                        ax.add_artist(circle)

    # Set the ticks for all axes in the grid
    if image_extent is not None:
        tick_spacing = 5.0
        grid.axes_llc.set_xticks(np.arange(math.ceil(image_extent[0] / tick_spacing) * tick_spacing,
                                           image_extent[1] + 1e-6, tick_spacing))
        grid.axes_llc.set_yticks(np.arange(math.ceil(image_extent[3] / tick_spacing) * tick_spacing,
                                           image_extent[2] + 1e-6, tick_spacing))
    return fig


def apply_beamformer(data, apod_matrix, beamformer):
    num_events = data.shape[-1]
    output = None
    for event_idx in range(num_events):
        apod = apod_matrix[:, :, event_idx]
        aperture_mask = np.any(np.not_equal(apod, 0), axis=1)
        aperture_data = data[aperture_mask, :, event_idx]
        result = beamformer(aperture_data)

        if output is None:
            if isinstance(result, tuple):
                output = [np.zeros(data.shape[1:]) for _ in range(len(result))]
            else:
                output = np.zeros(data.shape[1:])

        if isinstance(result, tuple):
            for output_idx in range(len(output)):
                output[output_idx][:, event_idx] = result[output_idx]
        else:
            output[:, event_idx] = result
    return output


def apply_mv_beamformer(data, sampling_frequency, pulse_frequency):
    subaperture_length = int(math.floor(data.shape[0] / 2))
    temp_kernel_length = int(round(sampling_frequency / pulse_frequency))
    diagonal_loading_factor = 1 / (100 * subaperture_length)
    method = 'synnevag2009'
    img_mv, _, _ = bf.mv.mv(data, subaperture_length, temp_kernel_length,
                            diagonal_loading_factor, method=method)

    return img_mv


def apply_bsmv_beamformer(data, sampling_frequency, pulse_frequency):
    subaperture_length = int(math.floor(data.shape[0] / 2))
    temp_kernel_length = int(round(sampling_frequency / pulse_frequency))
    diagonal_loading_factor = 1 / (100 * subaperture_length)
    subspace_dimension = 3
    method = 'deylami2017'
    img_mv, _, _ = bf.mv.mv(data, subaperture_length, temp_kernel_length,
                            diagonal_loading_factor, method=method,
                            subspace_dimension=subspace_dimension)

    return img_mv


def beamform(data_filename: str):
    f = h5.File(data_filename)

    data = f['data'][:, :, :]
    apod_matrix = f['apod_matrix'][:, :, :]
    pulse_lambda = f['scan']['lambda'][0][0]
    pulse_freq = 1 / pulse_lambda
    sampling_distance = f['scan']['zs'][1, 0] - f['scan']['zs'][0, 0]
    sampling_freq = 1 / sampling_distance
    wave_kernel_len = int(round(sampling_freq / pulse_freq))

    out = dict()
    out['bf_das'] = apply_beamformer(data, apod_matrix, bf.das.das)
    out['bf_cf'], out['weight_cf'] = apply_beamformer(data, apod_matrix, bf.coherence.cf)
    out['bf_gcf'], out['weight_gcf'] = apply_beamformer(data, apod_matrix, lambda x: bf.coherence.gcf(x, GCF_M))
    out['GCF_M'] = GCF_M
    out['bf_pcf'], out['weight_pcf'] = apply_beamformer(data, apod_matrix, bf.coherence.pcf)
    out['bf_scf'], out['weight_scf'] = apply_beamformer(data, apod_matrix, bf.coherence.scf)
    out['bf_imap'], out['imap_sigma_y'], out['imap_sigma_n'] = apply_beamformer(data, apod_matrix,
                                                                                lambda x: bf.imap.imap(x, IMAP_ITER))
    out['IMAP_ITER'] = IMAP_ITER
    out['bf_slsc'] = apply_beamformer(data, apod_matrix, lambda x: bf.slsc.slsc(x, SLSC_M, wave_kernel_len))
    out['wave_kernel_len'] = wave_kernel_len
    _, out['bf_fdmas'] = apply_beamformer(data, apod_matrix, lambda x: bf.dmas.dmas(x, sampling_freq, pulse_freq))
    out['bf_pdas2'] = apply_beamformer(data, apod_matrix, lambda x: bf.pdas.pdas(x, 2, sampling_freq, pulse_freq))
    out['bf_pdas3'] = apply_beamformer(data, apod_matrix, lambda x: bf.pdas.pdas(x, 3, sampling_freq, pulse_freq))
    out['bf_mv'] = apply_beamformer(data, apod_matrix, lambda x: apply_mv_beamformer(x, sampling_freq, pulse_freq))
    out['bf_bsmv'] = apply_beamformer(data, apod_matrix, lambda x: apply_bsmv_beamformer(x, sampling_freq, pulse_freq))

    out['sampling_freq'] = sampling_freq
    out['pulse_freq'] = pulse_freq
    out['filename'] = data_filename
    out['basename'] = os.path.basename(data_filename)

    out['scan'] = {'xs': f['scan']['xs'][:, :], 'zs': f['scan']['zs'][:, :]}

    return out


def default_postprocessing(x):
    return comp.compression(env.envelope(x))


def save_figures(base_path: str):
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    figs = list(map(plt.figure, plt.get_fignums()))
    for fig in figs:
        fig.savefig(os.path.join(base_path, f'{fig.get_label()}.pdf'))


def create_plots_all(images: dict):
    pp = default_postprocessing
    bmode_das = pp(images['bf_das'])
    bmode_cf = pp(images['bf_cf'])
    bmode_gcf = pp(images['bf_gcf'])
    bmode_pcf = pp(images['bf_pcf'])
    bmode_scf = pp(images['bf_scf'])
    bmode_imap = pp(images['bf_imap'])
    bmode_slsc = comp.compression(images['bf_slsc'])
    bmode_fdmas = pp(images['bf_fdmas'])
    bmode_pdas2 = pp(images['bf_pdas2'])
    bmode_pdas3 = pp(images['bf_pdas3'])
    bmode_mv = pp(images['bf_mv'])
    bmode_bsmv = pp(images['bf_bsmv'])

    image_extent = np.asarray([images['scan']['xs'][0, 0], images['scan']['xs'][-1, -1],
                               images['scan']['zs'][-1, -1], images['scan']['zs'][0, 0]]) * 1e3
    plot_multi(images['basename'] + " all",
               [bmode_das, bmode_cf, images['weight_cf'], bmode_gcf, images['weight_gcf'],
                bmode_pcf, images['weight_pcf'], bmode_scf, images['weight_scf'],
                bmode_imap, pp(images['imap_sigma_y']), pp(images['imap_sigma_n']),
                bmode_slsc, bmode_fdmas, bmode_pdas2, bmode_pdas3,
                bmode_mv, bmode_bsmv],
               titles=[
                   'DAS', 'CF+DAS', 'CF weight', 'GCF+DAS', 'GCF weight',
                   'PCF+DAS', 'PCF weight', 'SCF+DAS', 'SCF weight',
                   'iMAP_2', 'iMAP_2 sigma_y', 'iMAP_2 sigma_n',
                   'SLSC', 'F-DMAS', 'pDAS_2', 'pDAS_3',
                   'MV', 'BS-MV'],
               normalization=[
                   'individual_negative', 'individual_negative', '01', 'individual_negative', '01',
                   'individual_negative', '01', 'individual_negative', '01',
                   'individual_negative', 'individual_negative', 'individual_negative',
                   'individual_negative', 'individual_negative', 'individual_negative', 'individual_negative',
                   'individual_negative', 'individual_negative'],
               interpolation='nearest', plot_rows=5, image_extent=image_extent)


def create_plots(images: dict):
    pp = default_postprocessing
    bmode_das = pp(images['bf_das'])
    bmode_cf = pp(images['bf_cf'])
    bmode_gcf = pp(images['bf_gcf'])
    bmode_pcf = pp(images['bf_pcf'])
    bmode_scf = pp(images['bf_scf'])
    bmode_imap = pp(images['bf_imap'])
    bmode_slsc = comp.compression(images['bf_slsc'])
    bmode_fdmas = pp(images['bf_fdmas'])
    bmode_pdas2 = pp(images['bf_pdas2'])
    bmode_pdas3 = pp(images['bf_pdas3'])
    bmode_mv = pp(images['bf_mv'])
    bmode_bsmv = pp(images['bf_bsmv'])

    image_extent = np.asarray([images['scan']['xs'][0, 0], images['scan']['xs'][-1, -1],
                               images['scan']['zs'][-1, -1], images['scan']['zs'][0, 0]]) * 1e3
    plot_multi(images['basename'],
               [bmode_das, bmode_cf, bmode_gcf,
                bmode_pcf, bmode_scf, bmode_imap,
                bmode_slsc, bmode_fdmas, bmode_pdas2,
                bmode_pdas3, bmode_mv, bmode_bsmv],
               titles=[
                   'DAS', 'CF+DAS', 'GCF+DAS',
                   'PCF+DAS', 'SCF+DAS', r'$\textsf{iMAP}_2$',
                   'SLSC', 'F-DMAS', r'$\textsf{p-DAS}_2$',
                   r'$\textsf{p-DAS}_3$', 'MV', 'BS-MV'],
               normalization='individual_negative',
               interpolation=None, plot_rows=4, image_extent=image_extent, rois=images.get('rois'))


if __name__ == '__main__':
    filenames = ['data/Alpinion_L3-8_FI_hyperechoic_scatterers_delayed.h5',
                 'data/Alpinion_L3-8_FI_hypoechoic_delayed.h5']

    for filename in filenames:
        beamformed_filename = filename + '_beamformed.npz'

        if os.path.exists(beamformed_filename):
            with np.load(beamformed_filename, allow_pickle=True) as np_load:
                beamformed = {k: v.item() if v.ndim == 0 else v for (k, v) in np_load.items()}
        else:
            beamformed = beamform(filename)
            np.savez(beamformed_filename, **beamformed)

        if 'Alpinion_L3-8_FI_hypoechoic' in filename:
            beamformed['rois'] = {'target': {'center': [-9.5, 40.8], 'radii': [0, 2.8], 'color': '#17becf'},
                                          'background': {'center': [-9.5, 40.8], 'radii': [4.5, 7.2], 'color': '#ff7f0e'}}

        if 'rois' in beamformed.keys():
            measurements.measure(beamformed, ['CR', 'CNR', 'SNRs'])

        create_plots(beamformed)

        # create_plots_all(beamformed)

    save_figures('plots')
    plt.show()
    pass
