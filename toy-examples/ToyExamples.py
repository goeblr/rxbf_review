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

plt.rcParams['font.size'] = 8
plt.rcParams['figure.subplot.left'] = 0.035  # 0.125
plt.rcParams['figure.subplot.right'] = 0.975  # 0.9
plt.rcParams['figure.subplot.bottom'] = 0.21  # 0.11
plt.rcParams['axes.formatter.limits'] = [-4, 4]  # (default: [-5, 6]).
plt.rcParams["axes.formatter.useoffset"] = False
plt.rcParams["axes.formatter.use_mathtext"] = True

from plotting import *
from ToyData import *


def beamformed_plots(data):
    line_aperture_data = data['line_aperture_data']

    # parametrization for several beamformers
    sampling_frequency = data['params']['sampling_frequency']
    pulse_frequency = data['params']['pulse_frequency']
    temp_kernel_length = int(round(sampling_frequency / pulse_frequency))
    gcf_m = 1
    mv_subaperture_length = int(math.floor(data['pulse_focus'].shape[0] / 2))
    mv_diagonal_loading_factor = 1 / (100 * mv_subaperture_length)
    bsmv_subspace_dimension = 3

    # apply the beamformers to the data
    img_das = bf.das.das(line_aperture_data)
    img_cf, _ = bf.coherence.cf(line_aperture_data)
    img_gcf, _ = bf.coherence.gcf(line_aperture_data, gcf_m)
    img_pcf, _ = bf.coherence.pcf(line_aperture_data)
    img_scf, _ = bf.coherence.scf(line_aperture_data)
    img_imap1, _, _ = bf.imap.imap(line_aperture_data, 1)
    img_imap2, _, _ = bf.imap.imap(line_aperture_data, 2)
    img_slsc = bf.slsc.slsc(line_aperture_data, 10, temp_kernel_length)
    img_dmas, img_fdmas = bf.dmas.dmas(line_aperture_data, sampling_frequency, pulse_frequency)
    img_pdas2 = bf.pdas.pdas(line_aperture_data, 2, sampling_frequency, pulse_frequency)
    img_pdas3 = bf.pdas.pdas(line_aperture_data, 3, sampling_frequency, pulse_frequency)
    img_mv, _, _ = bf.mv.mv(data['line_aperture_data'], mv_subaperture_length, temp_kernel_length, mv_diagonal_loading_factor,
                            method='synnevag2009')
    img_bsmv, _, _ = bf.mv.mv(data['line_aperture_data'], mv_subaperture_length, temp_kernel_length,
                              mv_diagonal_loading_factor,
                              method='deylami2017', subspace_dimension=bsmv_subspace_dimension)

    plot_multi('Beamformed',
               [img_das.T, img_cf.T, img_gcf.T, img_pcf.T, img_scf.T, img_imap1.T, img_imap2.T,
                img_slsc.T, img_dmas.T, img_fdmas.T, img_pdas2.T, img_pdas3.T, img_mv.T, img_bsmv.T],
               titles=['DAS', 'CF', 'GCF', 'PCF', 'SCF', 'iMAP_1', 'iMAP_2', 
                       'SLSC', 'DMAS', 'FDMAS', 'p-DAS_2', 'p-DAS_3', 'MV', 'BSMV'],
               normalization='individual')


def save_figures(base_path: str):
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    figs = list(map(plt.figure, plt.get_fignums()))
    for fig in figs:
        fig.savefig(os.path.join(base_path, f'{fig.get_label()}.pdf'))


def toy_examples():
    data = create_toy_data()

    beamformed_plots(data)

    save_figures('plots')

    plt.show()
    pass


if __name__ == '__main__':
    toy_examples()
