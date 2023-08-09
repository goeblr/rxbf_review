import matplotlib.pyplot as plt
import math
import bf.das
import bf.coherence
import bf.slsc
import bf.dmas
import bf.imap
import bf.pdas
import bf.mv
import os

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 8
plt.rcParams['figure.subplot.left'] = 0.035  # 0.125
plt.rcParams['figure.subplot.right'] = 0.975  # 0.9
plt.rcParams['figure.subplot.bottom'] = 0.21  # 0.11
plt.rcParams['axes.formatter.limits'] = [-4, 4]  # (default: [-5, 6]).
plt.rcParams["axes.formatter.useoffset"] = False
plt.rcParams["axes.formatter.use_mathtext"] = True

from plotting import *
from ToyData import *

GCF_M = 1


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
    imap2_focus = bf.imap.imap(data['pulse_focus'], 2)
    imap2_side = bf.imap.imap(data['pulse_side'], 2)

    img_imap2, _, _ = bf.imap.imap(data['line_aperture_data'], 2)

    plot_multi('IMAP2',
               [imap2_focus[0], imap2_focus[1], imap2_focus[2],
                imap2_side[0], imap2_side[1], imap2_side[2], img_imap2.T],
               titles=[r'$\textsf{iMAP}_2$ focus', r'$\sigma_y$ focus', r'$\sigma_n$ focus',
                       r'$\textsf{iMAP}_2$ side', r'$\sigma_y$ side', r'$\sigma_n$ side', r'$\textsf{iMAP}_2$ image'],
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


def pdas_plots(data):
    sampling_frequency = data['params']['sampling_frequency']
    pulse_frequency = data['params']['pulse_frequency']

    pdas_2_focus = bf.pdas.pdas(data['pulse_focus'], 2.0, sampling_frequency, pulse_frequency)
    pdas_2_side = bf.pdas.pdas(data['pulse_side'], 2.0, sampling_frequency, pulse_frequency)
    pdas_3_focus = bf.pdas.pdas(data['pulse_focus'], 3.0, sampling_frequency, pulse_frequency)
    pdas_3_side = bf.pdas.pdas(data['pulse_side'], 3.0, sampling_frequency, pulse_frequency)

    img_pdas2 = bf.pdas.pdas(data['line_aperture_data'], 2, sampling_frequency, pulse_frequency)
    img_pdas3 = bf.pdas.pdas(data['line_aperture_data'], 3, sampling_frequency, pulse_frequency)

    plot_multi('PDAS',
               [pdas_2_focus, pdas_2_side, img_pdas2.T,
                pdas_3_focus, pdas_3_side, img_pdas3.T],
               titles=[r'$\textsf{p-DAS}_2$ focus', r'$\textsf{p-DAS}_2$ side', r'$\textsf{p-DAS}_2$ image',
                       r'$\textsf{p-DAS}_3$ focus', r'$\textsf{p-DAS}_3$ side', r'$\textsf{p-DAS}_3$ image'],
               xlabels=[None, None, 'Width [mm]',
                        None, None, 'Width [mm]'],
               xticks=[None, None, data['params']['image_ticks'],
                       None, None, data['params']['image_ticks']])


def mv_plots(data):
    subaperture_length = int(math.floor(data['pulse_focus'].shape[0] / 2))
    sampling_frequency = data['params']['sampling_frequency']
    pulse_frequency = data['params']['pulse_frequency']
    temp_kernel_length = int(round(sampling_frequency / pulse_frequency))
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
               titles=['MV focus', '$w(t)$ focus', 'MV side', '$w(t)$ side', 'MV image'],
               xlabels=[None, 'Subaperture', None, 'Subaperture', 'Width [mm]'],
               xticks=[None, 'FirstLastIdx', None, 'FirstLastIdx', data['params']['image_ticks']])


def bsmv_plots(data):
    subaperture_length = int(math.floor(data['pulse_focus'].shape[0] / 2))
    sampling_frequency = data['params']['sampling_frequency']
    pulse_frequency = data['params']['pulse_frequency']
    temp_kernel_length = int(round(sampling_frequency / pulse_frequency))
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
               titles=['BSMV focus', 'BS signal', '$w(t)$ focus',
                       'BSMV side', 'BS signal', '$w(t)$ side', 'BSMV image'],
               interpolation=[None, 'nearest', 'nearest',
                              None, 'nearest', 'nearest', None],
               xlabels=[None, 'Subaperture', 'Subaperture',
                        None, 'Subaperture', 'Subaperture', 'Width [mm]'],
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


def save_figures(base_path: str):
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    figs = list(map(plt.figure, plt.get_fignums()))
    for fig in figs:
        fig.savefig(os.path.join(base_path, f'{fig.get_label()}.pdf'))


def create_plots():
    data = create_toy_data()

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
