import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List


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

        figure_title = f'({chr(97 + data_idx)})'
        try:
            figure_title = f'{figure_title} {titles[data_idx]}'
        except:
            pass
        ax.set_title(figure_title)
        if data_idx == 0:
            ax.set_ylabel('Depth')
        if xlabels[data_idx] is not None:
            ax.set_xlabel(xlabels[data_idx])
    return fig