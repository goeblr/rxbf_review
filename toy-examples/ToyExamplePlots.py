import cmasher as cmr
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import math
import bf.das
import bf.coherence
import bf.slsc
import bf.dmas
import bf.imap
import bf.pdas
import bf.mv
from typing import Union, List
import os

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 8
plt.rcParams['figure.subplot.left'] = 0.035  # 0.125
plt.rcParams['figure.subplot.right'] = 0.975  # 0.9
plt.rcParams['figure.subplot.bottom'] = 0.21  # 0.11
plt.rcParams['axes.formatter.limits'] = [-4, 4]  # (default: [-5, 6]).
plt.rcParams["axes.formatter.useoffset"] = False
plt.rcParams["axes.formatter.use_mathtext"] = True

GCF_M = 1


def create_data():
    num_elements = 40
    num_samples = 101
    num_samples = 171

    element_spacing = 0.1  # mm
    point_depth = 40  # mm
    speed_of_sound = 1540e3  # mm/s
    sampling_frequency = 200e6  # Hz

    num_lines = 128
    line_spacing = 0.05  # mm

    pulse_frequency = 7e6
    pulse_bandwidth = 0.40
    pulse_side_line_idx = 21

    # pulse_side_phase_total = 2 * math.pi

    def pulse(t_center):
        t = np.linspace(-num_samples / 2 / sampling_frequency - t_center,
                        num_samples / 2 / sampling_frequency - t_center, num_samples)
        return scipy.signal.gausspulse(t, fc=pulse_frequency, bw=pulse_bandwidth) * 1024

    pulse_focus = np.tile(pulse(0), [num_elements, 1])

    # phases = np.linspace(-pulse_side_phase_total / 2, pulse_side_phase_total / 2, num_elements)
    # t_centers = phases / (2.0 * math.pi * pulse_frequency)
    # pulse_side = np.stack([pulse(t_center) for t_center in t_centers])

    element_positions = np.linspace(-num_elements / 2 * element_spacing,
                                    num_elements / 2 * element_spacing,
                                    num_elements)
    t_centers = np.sqrt(element_positions ** 2 + point_depth ** 2) / speed_of_sound
    t_base = min(t_centers)
    pulse_channel_data = np.stack([pulse(t_center - t_base) for t_center in t_centers])

    line_aperture_data = np.zeros(pulse_channel_data.shape + (num_lines,))
    line_offsets = np.zeros((num_lines,))
    for line_idx in range(num_lines):
        line_offset = -(-num_lines / 2 + line_idx) * line_spacing
        line_offsets[line_idx] = line_offset
        t_centers = np.sqrt(element_positions ** 2 + point_depth ** 2) / speed_of_sound
        t_delays = np.sqrt((element_positions - line_offset) ** 2 + point_depth ** 2) / speed_of_sound
        line_aperture_data[:, :, line_idx] = np.stack(
            [pulse(t_center - t_delay) for t_center, t_delay in zip(t_centers, t_delays)])

    pulse_side = line_aperture_data[:, :, pulse_side_line_idx]
    pulse_side_distance = line_offsets[pulse_side_line_idx]

    image_tick_labels = line_offsets[127:0:-21]
    image_tick_labels = [f'${x:.1f}$' if idx % 2 == 0 else None for idx, x in enumerate(image_tick_labels)]
    image_ticks = [list(range(0, 127, 21)), image_tick_labels]

    # pulse_clutter = 0.5 * pulse_focus + 0.5 * pulse_side
    params = {'pulse_frequency': pulse_frequency, 'sampling_frequency': sampling_frequency,
              'line_offsets': line_offsets, 'pulse_side_distance': pulse_side_distance, 'image_ticks': image_ticks}

    return {'pulse_focus': pulse_focus, 'pulse_side': pulse_side, 'pulse_channel_data': pulse_channel_data,
            'line_aperture_data': line_aperture_data, 'params': params}


# xticks: None, 'FirstLastIndex',  <List of indices list and labels list>, Int for every x-th tick centered around 1
def plot_2d(data, ax, normalization, interpolation, xticks):
    cmap = 'gray'
    # cmap = 'cmr.wildfire'
    # cmap = 'cmr.iceburn'
    if data.ndim == 1:
        data = np.reshape(data, data.shape + (1,))
    elif data.ndim == 2:
        data = data.T

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
    elif normalization == '01':
        v_min = 0
        v_max = 1
    else:
        raise RuntimeError('Not implemented')
    ax.imshow(data, cmap=cmap, vmin=v_min, vmax=v_max, aspect='auto', interpolation=interpolation)
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

    # Major ticks
    # ax.set_xticks(np.arange(0, w, 1))
    # ax.set_yticks(np.arange(0, h, 1))
    # ax.set_yticks(np.arange(0, h, 5))

    # Labels for major ticks
    # ax.set_xticklabels(np.arange(1, 11, 1))
    # ax.set_yticklabels(np.arange(1, 11, 1))

    # Minor ticks
    # ax.set_xticks(np.arange(-.5, w, 1), minor=True)
    # ax.set_yticks(np.arange(-.5, h, 1), minor=True)

    # Gridlines based on minor ticks
    # ax.grid(which='minor', color='k', linestyle='-', linewidth=1)


def plot_1d(data, ax, normalization):
    ax.plot(data, range(len(data)))
    if normalization == '01':
        v_min = 0
        v_max = 1
    elif normalization == 'fixed':
        v_min = -1024
        v_max = 1024
    elif normalization == 'individual':
        v_min_max = np.max(np.abs(data.ravel()))
        v_min = -v_min_max
        v_max = v_min_max
    elif normalization == 'individual_positive':
        v_min = 0
        v_max = np.max(data.ravel())
    else:
        raise RuntimeError('Not implemented')
    limit_distance = v_max - v_min
    margin = 0.1
    ax.set_xlim(v_min - limit_distance * margin, v_max + limit_distance * margin)
    ax.set_yticks([])
    # ax.xaxis.set_tick_params(rotation=45)


def plot_multi(figure_title: str, datas, titles=None, normalization: Union[List[str], str] = 'individual',
               interpolation=None, xlabels: List[str] = None, xticks=None):
    num_plots = len(datas)

    if not isinstance(normalization, list):
        normalization = [normalization] * num_plots
    if not isinstance(interpolation, list):
        interpolation = [interpolation] * num_plots
    if not isinstance(xlabels, list):
        xlabels = [xlabels] * num_plots
    if not isinstance(xticks, list):
        xticks = [xticks] * num_plots

    # width_ratios = [1 if data.ndim == 2 else 0.1 for data in datas]
    width_ratios = [1 if isinstance(data, np.ndarray) and data.ndim == 2 else 1 for data in datas]
    fig, axs = plt.subplots(1, num_plots, num=figure_title, figsize=(8, 1.9),
                            gridspec_kw={'width_ratios': width_ratios})
    for data_idx in range(num_plots):
        data = datas[data_idx]
        ax = axs[data_idx]
        if isinstance(data, tuple) or isinstance(data, list):
            assert (all([d.ndim == 1 for d in data]))
            for d in data:
                plot_1d(d, ax, normalization[data_idx])
        elif data.ndim == 1:
            plot_1d(data, ax, normalization[data_idx])
        elif data.ndim == 2:
            plot_2d(data, ax, normalization[data_idx], interpolation[data_idx], xticks[data_idx])
        try:
            ax.set_title(titles[data_idx])
        except:
            pass
        if data_idx == 0:
            ax.set_ylabel('Depth')
        if xlabels[data_idx] is not None:
            ax.set_xlabel(xlabels[data_idx])
    return fig


def das_plots(data):
    das_focus = bf.das.das(data['pulse_focus'])
    das_side = bf.das.das(data['pulse_side'])

    img_das = bf.das.das(data['line_aperture_data'])

    plot_multi('DAS',
               [data['pulse_channel_data'], data['pulse_focus'], das_focus, data['pulse_side'], das_side, img_das.T],
               titles=['channel data', 'in focus', 'DAS focus', 'side', 'DAS side', 'DAS image'],
               xlabels=['Channels', 'Aperture', None, 'Aperture', None, 'Width [mm]'],
               xticks=['FirstLastIdx', 'FirstLastIdx', None, 'FirstLastIdx', None, data['params']['image_ticks']])


def cf_plots(data):
    cf_focus = bf.coherence.cf(data['pulse_focus'])
    cf_side = bf.coherence.cf(data['pulse_side'])

    img_cf, img_cf_weight = bf.coherence.cf(data['line_aperture_data'])

    plot_multi('CF',
               [cf_focus[0], cf_focus[1], cf_side[0], cf_side[1], img_cf.T, img_cf_weight.T],
               titles=['CF+DAS focus', 'CF focus', 'CF+DAS side', 'CF side', 'CF image', 'CF weight'],
               normalization=['individual', '01', 'individual', '01', 'individual', '01'],
               xlabels=[None, None, None, None, 'Width [mm]', 'Width [mm]'],
               xticks=[None, None, None, None, data['params']['image_ticks'], data['params']['image_ticks']])


def gcf_plots(data):
    gcf_focus = bf.coherence.gcf(data['pulse_focus'], GCF_M)
    gcf_side = bf.coherence.gcf(data['pulse_side'], GCF_M)
    img_gcf, img_gcf_weight = bf.coherence.gcf(data['line_aperture_data'], GCF_M)

    plot_multi('GCF',
               [gcf_focus[0], gcf_focus[1], gcf_side[0], gcf_side[1], img_gcf.T, img_gcf_weight.T],
               titles=['GCF+DAS focus', 'GCF focus', 'GCF+DAS side', 'GCF side', 'GCF image', 'GCF weight'],
               normalization=['individual', '01', 'individual', '01', 'individual', '01'],
               xlabels=[None, None, None, None, 'Width [mm]', 'Width [mm]'],
               xticks=[None, None, None, None, data['params']['image_ticks'], data['params']['image_ticks']])


def pcf_plots(data):
    pcf_focus = bf.coherence.pcf(data['pulse_focus'])
    pcf_side = bf.coherence.pcf(data['pulse_side'])

    img_pcf, img_pcf_weight = bf.coherence.pcf(data['line_aperture_data'])

    plot_multi('PCF',
               [pcf_focus[0], pcf_focus[1], pcf_side[0], pcf_side[1], img_pcf.T, img_pcf_weight.T],
               titles=['PCF+DAS focus', 'PCF focus', 'PCF+DAS side', 'PCF side', 'PCF image', 'PCF weight'],
               normalization=['individual', '01', 'individual', '01', 'individual', '01'],
               xlabels=[None, None, None, None, 'Width [mm]', 'Width [mm]'],
               xticks=[None, None, None, None, data['params']['image_ticks'], data['params']['image_ticks']])


def scf_plots(data):
    scf_focus = bf.coherence.scf(data['pulse_focus'])
    scf_side = bf.coherence.scf(data['pulse_side'])

    img_scf, img_scf_weight = bf.coherence.scf(data['line_aperture_data'])

    plot_multi('SCF',
               [scf_focus[0], scf_focus[1], scf_side[0], scf_side[1], img_scf.T, img_scf_weight.T],
               titles=['SCF+DAS focus', 'SCF focus', 'SCF+DAS side', 'SCF side', 'SCF image', 'SCF weight'],
               normalization=['individual', '01', 'individual', '01', 'individual', '01'],
               xlabels=[None, None, None, None, 'Width [mm]', 'Width [mm]'],
               xticks=[None, None, None, None, data['params']['image_ticks'], data['params']['image_ticks']])


def imap_plots(data):
    # imap1_focus = bf.imap(data['pulse_focus'], 1)
    # imap1_side = bf.imap(data['pulse_side'], 1)
    #
    # img_imap1, _, _ = bf.imap(data['line_aperture_data'], 1)
    #
    # fig = plot_multi(
    #     [imap1_focus[0], imap1_focus[1], imap1_focus[2],
    #      imap1_side[0], imap1_side[1], imap1_side[2], img_imap1.T],
    #     titles=[r'$\textrm{iMAP}_1$ focus', r'$\sigma_y$ focus', r'$\sigma_n$ focus',
    #             r'$\textrm{iMAP}_1$ side', r'$\sigma_y$ side', r'$\sigma_n$ side', r'$\textrm{iMAP}_1$ image'],
    #     normalization=['individual', 'individual_positive', 'individual_positive',
    #                    'individual', 'individual_positive', 'individual_positive', 'individual'])

    imap2_focus = bf.imap.imap(data['pulse_focus'], 2)
    imap2_side = bf.imap.imap(data['pulse_side'], 2)

    img_imap2, _, _ = bf.imap.imap(data['line_aperture_data'], 2)

    plot_multi('IMAP2',
               [imap2_focus[0], imap2_focus[1], imap2_focus[2],
                imap2_side[0], imap2_side[1], imap2_side[2], img_imap2.T],
               titles=[r'$\textrm{iMAP}_2$ focus', r'$\sigma_y$ focus', r'$\sigma_n$ focus',
                       r'$\textrm{iMAP}_2$ side', r'$\sigma_y$ side', r'$\sigma_n$ side', r'$\textrm{iMAP}_2$ image'],
               normalization=['individual', 'individual_positive', 'individual_positive',
                              'individual', 'individual_positive', 'individual_positive', 'individual'],
               xlabels=[None, None, None, None, None, None, 'Width [mm]'],
               xticks=[None, None, None, None, None, None, data['params']['image_ticks']])


def slsc_plots(data):
    sampling_frequency = data['params']['sampling_frequency']
    pulse_frequency = data['params']['pulse_frequency']
    temp_kernel_length = int(round(sampling_frequency / pulse_frequency))

    big_m = 8  # equivalent to Q = 20% for aperture size of 40
    slsc_focus = bf.slsc.slsc(data['pulse_focus'], big_m, temp_kernel_length)
    slsc_side = bf.slsc.slsc(data['pulse_side'], big_m, temp_kernel_length)
    sc_image_focus = bf.slsc.slsc_spatial_correlation_image(data['pulse_focus'], temp_kernel_length)
    sc_image_side = bf.slsc.slsc_spatial_correlation_image(data['pulse_side'], temp_kernel_length)

    img_slsc = bf.slsc.slsc(data['line_aperture_data'], big_m, temp_kernel_length)

    plot_multi('SLSC',
               [slsc_focus, sc_image_focus,
                slsc_side, sc_image_side, img_slsc.T],
               titles=['SLSC focus', r'spatial corr. $\tilde{R}(m)$',
                       'SLSC side', r'spatial corr. $\tilde{R}(m)$', 'SLSC image'],
               normalization=['individual_positive', 'individual',
                              'individual_positive', 'individual', 'individual_positive'],
               xlabels=[None, '$m$', None, '$m$', 'Width [mm]'],
               xticks=[None, 10, None, 10, data['params']['image_ticks']])


def dmas_plots(data):
    sampling_frequency = data['params']['sampling_frequency']
    pulse_frequency = data['params']['pulse_frequency']
    dmas_focus, fdmas_focus = bf.dmas.dmas(data['pulse_focus'], sampling_frequency, pulse_frequency)
    dmas_side, fdmas_side = bf.dmas.dmas(data['pulse_side'], sampling_frequency, pulse_frequency)

    _, img_fdmas = bf.dmas.dmas(data['line_aperture_data'], sampling_frequency, pulse_frequency)

    # Insight images
    # dmas_image_focus = bf.dmas.dmas_image(data['pulse_focus'], sampling_frequency, pulse_frequency)
    # dmas_image_side = bf.dmas.dmas_image(data['pulse_side'], sampling_frequency, pulse_frequency)

    plot_multi('DMAS',
               [dmas_focus, fdmas_focus,
                dmas_side, fdmas_side, img_fdmas.T],
               titles=['DMAS focus', 'F-DMAS focus',
                       'DMAS side', 'F-DMAS side', 'F-DMAS image'],
               xlabels=[None, None, None, None, 'Width [mm]'],
               xticks=[None, None, None, None, data['params']['image_ticks']])

    # Plot insight images
    # fig, axs = plt.subplots(2, 1)
    # v_min_max = np.max(np.abs(dmas_image_focus).ravel())
    # im1 = axs[0].imshow(dmas_image_focus.T, cmap='gray', vmin=-v_min_max, vmax=v_min_max)
    # plt.colorbar(im1, ax=axs[0])
    # v_min_max = np.max(np.abs(dmas_image_side).ravel())
    # im2 = axs[1].imshow(dmas_image_side.T, cmap='gray', vmin=-v_min_max, vmax=v_min_max)
    # plt.colorbar(im2, ax=axs[1])


def pdas_plots(data):
    pdas_2_focus = bf.pdas.pdas(data['pulse_focus'], 2.0)
    pdas_2_side = bf.pdas.pdas(data['pulse_side'], 2.0)
    pdas_3_focus = bf.pdas.pdas(data['pulse_focus'], 3.0)
    pdas_3_side = bf.pdas.pdas(data['pulse_side'], 3.0)

    img_pdas2 = bf.pdas.pdas(data['line_aperture_data'], 2)
    img_pdas3 = bf.pdas.pdas(data['line_aperture_data'], 3)

    plot_multi('PDAS',
               [pdas_2_focus, pdas_2_side, img_pdas2.T,
                pdas_3_focus, pdas_3_side, img_pdas3.T],
               titles=[r'$\textrm{p-DAS}_2$ focus', r'$\textrm{p-DAS}_2$ side', r'$\textrm{p-DAS}_2$ image',
                       r'$\textrm{p-DAS}_3$ focus', r'$\textrm{p-DAS}_3$ side', r'$\textrm{p-DAS}_3$ image'],
               xlabels=[None, None, 'Width [mm]',
                        None, None, 'Width [mm]'],
               xticks=[None, None, data['params']['image_ticks'],
                       None, None, data['params']['image_ticks']])


def mv_plots(data):
    subaperture_length = int(math.floor(data['pulse_focus'].shape[0] / 2))
    temp_kernel_length = 3
    diagonal_loading_factor = 1 / (100 * subaperture_length)
    method = 'synnevag2009'
    mv_focus, weights_focus, _ = bf.mv.mv(data['pulse_focus'], subaperture_length, temp_kernel_length,
                                          diagonal_loading_factor, method=method)
    mv_side, weights_side, _ = bf.mv.mv(data['pulse_side'], subaperture_length, temp_kernel_length,
                                        diagonal_loading_factor, method=method)

    img_mv, _, _ = bf.mv.mv(data['line_aperture_data'], subaperture_length, temp_kernel_length, diagonal_loading_factor,
                            method=method)

    plot_multi('MV',
               [mv_focus, weights_focus, mv_side, weights_side, img_mv.T],
               titles=['MV focus', 'weights $w(t)$ focus', 'MV side', 'weights $w(t)$ side', 'MV image'],
               xlabels=[None, 'Sub-aperture', None, 'Sub-aperture', 'Width [mm]'],
               xticks=[None, 'FirstLastIdx', None, 'FirstLastIdx', data['params']['image_ticks']])


def bsmv_plots(data):
    subaperture_length = int(math.floor(data['pulse_focus'].shape[0] / 2))
    temp_kernel_length = 3
    diagonal_loading_factor = 1 / (100 * subaperture_length)
    subspace_dimension = 3
    method = 'deylami2017'
    bsmv_focus, weights_focus, subaperture_focus = bf.mv.mv(data['pulse_focus'], subaperture_length, temp_kernel_length,
                                                            diagonal_loading_factor, method=method,
                                                            subspace_dimension=subspace_dimension)
    bsmv_side, weights_side, subaperture_side = bf.mv.mv(data['pulse_side'], subaperture_length, temp_kernel_length,
                                                         diagonal_loading_factor, method=method,
                                                         subspace_dimension=subspace_dimension)

    img_bsmv, _, _ = bf.mv.mv(data['line_aperture_data'], subaperture_length, temp_kernel_length,
                              diagonal_loading_factor,
                              method=method, subspace_dimension=subspace_dimension)

    plot_multi('BSMV',
               [bsmv_focus, subaperture_focus, weights_focus, bsmv_side, subaperture_side, weights_side, img_bsmv.T],
               titles=['BS-MV focus', 'beamspace signal', 'weights $w(t)$ focus',
                       'BS-MV side', 'beamspace signal', 'weights $w(t)$ side', 'BS-MV image'],
               interpolation=[None, 'nearest', 'nearest',
                              None, 'nearest', 'nearest', None],
               xlabels=[None, 'Sub-aperture', 'Sub-aperture',
                        None, 'Sub-aperture', 'Sub-aperture', 'Width [mm]'],
               xticks=[None, 1, 1,
                       None, 1, 1, data['params']['image_ticks']])


def beamformed_plots(data):
    line_aperture_data = data['line_aperture_data']
    sampling_frequency = data['params']['sampling_frequency']
    pulse_frequency = data['params']['pulse_frequency']
    temp_kernel_length = int(round(sampling_frequency / pulse_frequency))

    img_das = bf.das.das(line_aperture_data)
    img_cf, _ = bf.coherence.cf(line_aperture_data)
    img_gcf, _ = bf.coherence.gcf(line_aperture_data, GCF_M)
    img_pcf, _ = bf.coherence.pcf(line_aperture_data)
    img_scf, _ = bf.coherence.scf(line_aperture_data)
    img_imap1, _, _ = bf.imap.imap(line_aperture_data, 1)
    img_imap2, _, _ = bf.imap.imap(line_aperture_data, 2)
    img_slsc = bf.slsc.slsc(line_aperture_data, 10, temp_kernel_length)
    img_dmas, img_fdmas = bf.dmas.dmas(line_aperture_data, sampling_frequency, pulse_frequency)
    img_pdas2 = bf.pdas.pdas(line_aperture_data, 2)
    img_pdas3 = bf.pdas.pdas(line_aperture_data, 3)

    plot_multi('Beamformed',
               [img_das.T, img_cf.T, img_gcf.T, img_pcf.T, img_scf.T, img_imap1.T, img_imap2.T,
                img_slsc.T, img_dmas.T, img_fdmas.T, img_pdas2.T, img_pdas3.T],
               titles=['DAS', 'CF', 'GCF', 'PCF', 'SCF', 'iMAP_1', 'iMAP_2', 'SLSC', 'DMAS', 'FDMAS',
                       'p-DAS_2', 'p-DAS_3'],
               normalization='individual')

    # plot_multi([line_aperture_data[:, :, 0], line_aperture_data[:, :, int(line_aperture_data.shape[2] / 2)],
    #                   line_aperture_data[:, :, -1]], titles=['leftmost', 'central', 'rightmost'],
    #                  normalization='individual')

    # line_offsets = data['params']['line_offsets']
    # plot_multi(
    #     [data['pulse_side'], line_aperture_data[:, :, 16],
    #      line_aperture_data[:, :, 21], line_aperture_data[:, :, 26],
    #      line_aperture_data[:, :, 32]],
    #     titles=[f'side {data["params"]["pulse_side_distance"]}mm', 'channel', str(line_offsets[16]) + 'mm',
    #             str(line_offsets[21]) + 'mm', str(line_offsets[26]) + 'mm', str(line_offsets[32]) + 'mm'],
    #     normalization='individual')


def save_figures(base_path: str):
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    figs = list(map(plt.figure, plt.get_fignums()))
    for fig in figs:
        fig.savefig(os.path.join(base_path, f'{fig.get_label()}.pdf'))


def create_plots():
    data = create_data()

    das_plots(data)
    cf_plots(data)
    gcf_plots(data)
    pcf_plots(data)
    scf_plots(data)
    imap_plots(data)
    slsc_plots(data)
    dmas_plots(data)
    pdas_plots(data)
    mv_plots(data)
    bsmv_plots(data)

    # # beamformed_plots(data)

    save_figures('plots')

    plt.show()
    pass


if __name__ == '__main__':
    create_plots()
