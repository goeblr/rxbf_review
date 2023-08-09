import matplotlib.pyplot as plt
import numpy as np
import math
from typing import Union, List
from mpl_toolkits.axes_grid1 import ImageGrid

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
                if 'center' in region.keys() and 'radii' in region.keys():
                    for radius in region['radii']:
                        if radius > 0:
                            circle = plt.Circle(region['center'], radius, color=region['color'], fill=False,
                                                linewidth=1.0)
                            ax.add_artist(circle)

    # Set the ticks for all axes in the grid
    if image_extent is not None:
        tick_spacing = 5.0
        grid.axes_llc.set_xticks(np.arange(math.ceil(image_extent[0] / tick_spacing) * tick_spacing,
                                           image_extent[1] + 1e-6, tick_spacing))
        grid.axes_llc.set_yticks(np.arange(math.ceil(image_extent[3] / tick_spacing) * tick_spacing,
                                           image_extent[2] + 1e-6, tick_spacing))
    return fig