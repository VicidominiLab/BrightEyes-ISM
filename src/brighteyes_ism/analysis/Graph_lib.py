import numpy as np
import matplotlib.pyplot as plt

from numbers import Number

from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec

import numbers


def ShowImg(image: np.ndarray, pxsize_x: float, clabel: str = None, vmin: float = None, vmax: float = None,
            fig: plt.Figure = None, ax: plt.axis = None, cmap: str = 'hot'):
    """
    It shows the input image with a scalebar and a colorbar.
    It returns the corresponding figure and axis.

    Parameters
    ----------
    image : np.ndarray
        Image (Nx x Ny).
    pxsize_x : float
        Pixel size in micrometers (um).
    clabel : str
        Label of the colorbar.
    vmin : float, optional
        Lower bound of the intensity axis.
        If None, is set to the minimum value of the image.
        The default is None.
    vmax : float, optional
        Upper bound of the intensity axis.
        If None, is set to the maximum value of the image.
        The default is None.
    fig : plt.Figure, optional
        Figure where to display the plot. If None, a new figure is created.
        The default is None.
    ax : plt.axis, optional
        Axis where to display the plot. If None, a new axis is created.
        The default is None.
    cmap : str, optional
        Colormap, to be chosen within the matplotlib list.
        The default is 'hot'.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure.
    ax : plt.axis
        Matplotlib axis.

    """

    if fig == None or ax == None:
        fig, ax = plt.subplots()

    Nx, Ny = image.shape
    rangex = Nx * pxsize_x
    rangey = Nx * pxsize_x
    extent = (-rangex/2, rangex/2, -rangey/2, rangey/2)

    im = ax.imshow(image, vmin=vmin, vmax=vmax, cmap=cmap, extent=extent)
    ax.axis('off')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, ticks=[])

    vmax_text = int(np.floor(np.max(image)))
    vmin_text = int(np.floor(np.min(image)))

    if isinstance(clabel, Number):
        clabel_text = f'Counts / {clabel:.0f} ' + '$\mathregular{\mu s}$'
    else:
        clabel_text = clabel

    cbar.ax.text(0.6, 0.5, clabel_text, horizontalalignment='center', verticalalignment='center',
                 rotation='vertical', transform=cax.transAxes)
    cbar.ax.text(0.6, 0.98, f'{vmax_text}', horizontalalignment='center', verticalalignment='top',
                 rotation='vertical', transform=cax.transAxes)
    cbar.ax.text(0.6, 0.02, f'{vmin_text}', horizontalalignment='center', verticalalignment='bottom',
                 rotation='vertical', transform=cax.transAxes, color='white')

    scalebar = ScaleBar(
        1, "um",  # default, extent is calibrated in meters
        box_alpha=0,
        color='w',
        length_fraction=0.25)

    ax.add_artist(scalebar)

    return fig, ax

def ShowStack(image: np.ndarray, pxsize_x: float, pxsize_z: float, clabel: str = None, planes: tuple = None,
              cmap: str = 'hot', figsize: tuple = (10, 10)):
    """
    It shows the input image with a scalebar and a colorbar.
    It returns the corresponding figure and axis.

    Parameters
    ----------
    image : np.ndarray
        Image (Nz x Nx x Ny).
    pxsize_x : float
        Lateral ixel size in micrometers (um).
    pxsize_z : float
        Axial pixel size in micrometers (um).
    clabel : str
        Label of the colorbar.
    planes : tuple
        Tuple containing the 
    cmap : str, optional
        Colormap, to be chosen within the matplotlib list.
        The default is 'hot'.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure.

    """

    Nz, Nx, Ny = image.shape

    if planes is None:
        x0 = Nx//2
        y0 = Ny//2
        z0 = Nz//2
    else:
        z0, x0, y0 = planes

    rangez = Nz * pxsize_z
    rangex = Nx * pxsize_x
    rangey = Ny * pxsize_x

     # define extents

    extent_xy = ( -rangex/2, rangex/2, -rangey/2, rangey/2)
    extent_xz = ( -rangex/2, rangex/2, -rangez/2, rangez/2)
    extent_zy = ( -rangez/2, rangez/2, -rangey/2, rangey/2)

    # find vmax

    max3d = np.empty(image.ndim)

    max3d[0] = np.max(image[z0, :, :])
    max3d[1] = np.max(image[:, x0, :])
    max3d[2] = np.max(image[:, :, y0])

    vmax = np.max(max3d)

    # find vmax

    min3d = np.empty(image.ndim)

    min3d[0] = np.min(image[z0, :, :])
    min3d[1] = np.min(image[:, x0, :])
    min3d[2] = np.min(image[:, :, y0])

    vmin = np.max(min3d)

    # plot figure

    fig = plt.figure(figsize = figsize)
    gs = gridspec.GridSpec(2, 2, width_ratios=[rangez, rangex], height_ratios=[rangex, rangez],
                           wspace=0.02, hspace=0.02, left=0.05, right=0.95, bottom=0.05, top=0.95)
    ax = np.asarray([plt.subplot(gs[i]) for i in range(4)]).reshape((2,2))

    ax[0, 0].imshow(image[::-1, :, y0].T, cmap = cmap, vmin = vmin, vmax = vmax, extent = extent_zy)
    ax[0, 0].axis('off')

    im = ax[0, 1].imshow(image[z0], cmap=cmap, vmin=vmin, vmax=vmax, extent=extent_xy)
    ax[0, 1].axis('off')

    ax[1, 1].imshow(image[:, x0, :], cmap = cmap, vmin = vmin, vmax = vmax, extent = extent_xz)
    ax[1, 1].axis('off')

    ax[1, 0].axis('off')

    # add colorbar

    vmax_text = int(np.floor(vmax))
    vmin_text = int(np.floor(vmin))

    if isinstance(clabel, Number):
        clabel_text = f'Counts / {clabel:.0f} ' + '$\mathregular{\mu s}$'
    else:
        clabel_text = clabel

    y0 = ax[-1, -1].get_position().y0
    y1 = ax[0, 0].get_position().y1
    height = y1 - y0

    cax = fig.add_axes([0.96, y0, 0.03, height])
    cbar = fig.colorbar(im, cax=cax, ticks=[])

    cbar.ax.text(0.6, 0.5, clabel_text, horizontalalignment='center', verticalalignment='center',
                 rotation='vertical', transform=cax.transAxes)
    cbar.ax.text(0.6, 0.98, f'{vmax_text}', horizontalalignment='center', verticalalignment='top',
                 rotation='vertical', transform=cax.transAxes)
    cbar.ax.text(0.6, 0.02, f'{vmin_text}', horizontalalignment='center', verticalalignment='bottom',
                 rotation='vertical', transform=cax.transAxes, color='white')

    # add scalebar

    scalebar = ScaleBar(
        1, "um",  # default, extent is calibrated in meters
        box_alpha=0,
        color='w',
        length_fraction=0.25)

    ax[0, 1].add_artist(scalebar)

    return fig



def ShowDataset(dset: np.ndarray, cmap: str = 'hot', pxsize: float = None, normalize: bool = False,
                colorbar: bool = False, xlims: list = [None, None], ylims: list = [None, None],
                extent = None, figsize: tuple = (6, 6), gridshape = None) -> plt.Figure:
    '''
    It displays all the images of the ISM dataset in a squared grid.
    It returns the corresponding figure.

    Parameters
    ----------
    dset : np.ndarray
        ISM dataset (Nx x Ny x Nch).
    cmap : str, optional
        Colormap, to be chosen within the matplotlib list. The default is 'hot'.
    pxsize : float, optional
        Pixel size in micrometers (um). The default is None.
    normalize : bool, optional
        If True, each image is normalized with respect to the whole dataset.
        If False, each image is normalized to itself.
        The default is False.
    colorbar : bool, optional
        If true, a colorbar is shown. The default is False.
    xlims : list, optional
        If given, only the region with the x-range is displayed. The default is [None, None].
    ylims : list, optional
        If given, only the region with the y-range is displayed. The default is [None, None].
    figsize : tuple, optional
        Size of the figure. The default is (6, 6).

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure.
    '''

    if gridshape is None:
        nx = int(np.sqrt(dset.shape[-1]))
        ny = int(np.sqrt(dset.shape[-1]))
    else:
        nx = gridshape[0]
        ny = gridshape[1]

    if normalize == True:
        vmin = np.min(dset)
        vmax = np.max(dset)
        norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(nx, ny, sharex=True, sharey=True, figsize=figsize)
    for i in range(nx*ny):
        if np.min( [nx, ny] ) > 1:
            idx = np.unravel_index(i, [nx, ny])
        else:
            idx = i
        if normalize == True:
            im = ax[idx].imshow(dset[:, :, i], norm=norm, cmap=cmap, extent=extent)
        else:
            im = ax[idx].imshow(dset[:, :, i], cmap=cmap, extent=extent)
        ax[idx].set_xlim(xlims)
        ax[idx].set_ylim(ylims)
        ax[idx].axis('off')

    if isinstance(pxsize, numbers.Number):
        scalebar = ScaleBar(
            pxsize, "um",  # default, extent is calibrated in meters
            box_alpha=0,
            color='w',
            location='lower right',
            length_fraction=0.5)

        ax[-1, -1].add_artist(scalebar)

    fig.tight_layout()
    if colorbar == True and normalize == True:
        y0 = ax[-1, -1].get_position().y0
        y1 = ax[0, 0].get_position().y1
        height = y1 - y0

        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, y0, 0.05, height])
        cbar = fig.colorbar(im, cax=cbar_ax, ticks=[])

        cbar.ax.text(0.3, 0.95, f'{int(np.floor(vmax))}', rotation=90, transform=cbar_ax.transAxes)

        cbar.ax.text(0.3, 0.02, f'{int(np.floor(vmin))}', rotation=90, transform=cbar_ax.transAxes, color='white')

    return fig


def PlotShiftVectors(shift_vectors: np.ndarray, pxsize: float = 1, labels: bool = True, color: np.ndarray = None,
                     cmap: str = 'summer_r', fig: plt.Figure = None, ax: plt.axis = None):
    """
    It plots the shift vectors in a scatter plot.
    It returns the corresponding figure and axis.

    Parameters
    ----------
    shift_vectors : np.ndarray
        Array of the coordinates of the shift vectors (Nch x 2).
    pxsize : float, optional
        Pixel size in micrometers (um). The default is 1.
    labels : bool, optional
        If true, the channel number is printed close to each point.
        The default is True.
    color : np.ndarray, optional
        Array defining the color value (Nch).
        The default is None.
    cmap : str, optional
        Colormap, to be chosen within the matplotlib list.
        The default is 'summer_r'.
    fig : plt.Figure, optional
        Figure where to display the plot. If None, a new figure is created.
        The default is None.
    ax : plt.axis, optional
        Axis where to display the plot. If None, a new axis is created.
        The default is None.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure.
    ax : plt.axis
        Matplotlib axis.

    """

    if fig == None or ax == None:
        fig, ax = plt.subplots()

    shift = shift_vectors * pxsize

    Nch = shift.shape[0]

    if isinstance(color, str) and color == 'auto':
        N = int(np.sqrt(Nch))
        x = np.arange(-(N // 2), N // 2 + 1)
        X, Y = np.meshgrid(x, x)
        R = np.sqrt(X ** 2 + Y ** 2)
        color = R

    ax.scatter(shift[:, 0], shift[:, 1], s=80, c=color, edgecolors='black', cmap=cmap)
    ax.set_aspect('equal', 'box')

    if labels == True:
        for n in range(Nch):
            ax.annotate(str(n), shift[n], xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel(r'Shift$_x$ (nm)')
    ax.set_ylabel(r'Shift$_y$ (nm)')
    ax.set_title('Shift vectors')

    ax.set_aspect('equal')

    return fig, ax


def ShowFingerprint(dset: np.ndarray, cmap: str = 'hot', colorbar: bool = False, clabel: str = None, normalize: bool = False, fig: plt.Figure = None,
                    ax: plt.axis = None):
    """
    It calculates and shows the fingerprint of an ISM dataset.
    It returns the corresponding figure and axis.

    Parameters
    ----------
    dset : np.ndarray
        ISM dataset (Nx x Ny x Nch).
    cmap : str, optional
        Colormap, to be chosen within the matplotlib list. The default is 'hot'.
    colorbar : bool, optional
        If true, a colorbar is shown. The default is False
    clabel : str, optional
        Label of the colorbar. The default is None
    normalize : bool, optional
        If true, the fingerprint values are normalized between 0 and 1. The default is False
    fig : plt.Figure, optional
        Figure where to display the plot. If None, a new figure is created.
        The default is None.
    ax : plt.axis, optional
        Axis where to display the plot. If None, a new axis is created.
        The default is None.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure.
    ax : plt.axis
        Matplotlib axis.
    """

    if fig == None or ax == None:
        fig, ax = plt.subplots()

    N = int( np.sqrt(dset.shape[-1]) )
    fingerprint = dset.sum(axis=(0, 1)).reshape(N, N)
    if normalize == True:
        max_counts = np.max(fingerprint)
        fingerprint = fingerprint / max_counts
    im = ax.imshow(fingerprint, cmap=cmap)

    ax.axis('off')
    fig.tight_layout()

    if colorbar == True:

        vmax = int(np.floor(np.max(fingerprint)))
        vmin = int(np.floor(np.min(fingerprint)))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, ticks=[])

        cbar.ax.text(0.6, 0.5, clabel, horizontalalignment='center', verticalalignment='center',
                     rotation='vertical', transform=cax.transAxes)
        cbar.ax.text(0.6, 0.98, f'{vmax}', horizontalalignment='center', verticalalignment='top',
                     rotation='vertical', transform=cax.transAxes)
        cbar.ax.text(0.6, 0.02, f'{vmin}', horizontalalignment='center', verticalalignment='bottom',
                     rotation='vertical', transform=cax.transAxes, color='white')

    return fig, ax
