import numpy as np
import matplotlib.pyplot as plt

from numbers import Number

from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize, hsv_to_rgb, LogNorm
import matplotlib.gridspec as gridspec

import numbers

from ..simulation.detector import det_coords

#%%

from matplotlib import cm
from matplotlib.colors import ListedColormap

_hot = cm.get_cmap('hot', 256)
_hot_array = _hot(np.linspace(0, 1, 256))

_bluehot = _hot_array.copy()
_bluehot[:, 0] = _hot_array[:, 2]
_bluehot[:, 2] = _hot_array[:, 0]

bluehot = ListedColormap(_bluehot)

_greenhot = _hot_array.copy()
_greenhot[:, 0] = _hot_array[:, 1]
_greenhot[:, 1] = _hot_array[:, 0]

greenhot = ListedColormap(_greenhot)

#%%
def ShowImg(image: np.ndarray, pxsize_x: float, clabel: str = None, vmin: float = None, vmax: float = None,
            fig: plt.Figure = None, ax: plt.axis = None, colorbar: bool = True, cmap: str = 'hot', integer_values = True, log_scale: bool = False):
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
    colormap : bool, optional
        If True, a colormap is shown. The default is True.
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

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # Nx, Ny = image.shape
    # rangex = Nx * pxsize_x
    # rangey = Nx * pxsize_x
    # extent = (-rangex/2, rangex/2, -rangey/2, rangey/2)
    if log_scale == True:
        im = ax.imshow(image, vmin=vmin, vmax=vmax, cmap=cmap, norm = LogNorm())
    else:
        im = ax.imshow(image, vmin=vmin, vmax=vmax, cmap=cmap)#, extent=extent)
    ax.axis('off')

    if vmax is None:
        vmax = np.max(image)
    if vmin is None:
        vmin = np.min(image)

    if colorbar==True:

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, ticks=[])
        
        if integer_values == True:
            vmax_text = int(np.floor(vmax))
            vmin_text = int(np.floor(vmin))
        else:
            vmax_text = np.round(vmax, 3)
            vmin_text = np.round(vmin, 3)

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
        pxsize_x, "um",  # default, extent is calibrated in meters
        box_alpha=0,
        color='w',
        length_fraction=0.25)

    ax.add_artist(scalebar)

    return fig, ax


def ShowStack(image: np.ndarray, pxsize_x: float, pxsize_z: float, clabel: str = None, planes: tuple = None,
              projection = None, vmin: float = None, vmax: float = None, cmap: str = 'hot', figsize: tuple = (10, 10)):
    """
    It shows the input image with a scalebar and a colorbar.
    It returns the corresponding figure and axis.

    Parameters
    ----------
    image : np.ndarray
        Image (Nz x Ny x Nx).
    pxsize_x : float
        Lateral ixel size in micrometers (um).
    pxsize_z : float
        Axial pixel size in micrometers (um).
    clabel : str
        Label of the colorbar.
    planes : tuple
        Coordinates (z0, x0, y0) of the slices.
    cmap : str, optional
        Colormap, to be chosen within the matplotlib list.
        The default is 'hot'.
    figsize : tuple
        Size of the figure. The default is (10, 10).

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure.

    """

    Nz, Ny, Nx = image.shape

    # find slice coordinates

    if planes is None:
        x0 = Nx//2
        y0 = Ny//2
        z0 = Nz//2
    else:
        z0, y0, x0 = planes

    rangez = Nz * pxsize_z
    rangex = Nx * pxsize_x
    rangey = Ny * pxsize_x

     # define extents

    extent_xy = ( -rangex/2, rangex/2, -rangey/2, rangey/2)
    extent_xz = ( -rangex/2, rangex/2, -rangez/2, rangez/2)
    extent_zy = ( -rangez/2, rangez/2, -rangey/2, rangey/2)

    # generate projection images

    if projection == 'mip':
        image_z = image.max(axis=0)
        image_y = image.max(axis=1)
        image_x = image.max(axis=2)
    else:
        image_z = image[z0, :, :]
        image_y = image[:, y0, :]
        image_x = image[:, :, x0]

    # find vmax

    if vmax is None:
        max3d = np.empty(image.ndim)

        max3d[0] = np.max(image_z)
        max3d[1] = np.max(image_y)
        max3d[2] = np.max(image_x)

        vmax = np.max(max3d)

    # find vmin
    if vmin is None:
        min3d = np.empty(image.ndim)

        min3d[0] = np.min(image_z)
        min3d[1] = np.min(image_y)
        min3d[2] = np.min(image_x)

        vmin = np.max(min3d)

    # plot figure
    fig = plt.figure(figsize = figsize)

    gs = gridspec.GridSpec(2, 2, width_ratios=[rangez, rangex], height_ratios=[rangey, rangez],
                           wspace=0.02, hspace=0.02, left=0.05, right=0.95, bottom=0.05, top=0.95)
    ax = np.asarray([plt.subplot(gs[i]) for i in range(4)]).reshape((2,2))

    ax[0, 0].imshow(image_x[::-1, :].T, cmap = cmap, vmin = vmin, vmax = vmax, extent = extent_zy)
    ax[0, 0].axis('off')

    im = ax[0, 1].imshow(image_z, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent_xy)
    ax[0, 1].axis('off')

    ax[1, 1].imshow(image_y, cmap = cmap, vmin = vmin, vmax = vmax, extent = extent_xz)
    ax[1, 1].axis('off')

    ax[1, 0].axis('off')

    # share axes

    ax[0, 0].sharey(ax[0, 1])
    ax[0, 1].sharex(ax[1, 1])

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

def StackSlider(image: np.ndarray, pxsize_x: float, pxsize_z: float, clabel: str = None,
                 cmap: str = 'hot', figsize: tuple = (10, 10)):

    from matplotlib.widgets import Slider

    Nz, Ny, Nx = image.shape

    rangez = Nz * pxsize_z
    rangex = Nx * pxsize_x
    rangey = Ny * pxsize_x

    # Find color limits

    vmax = np.max(image)
    vmin = np.min(image)

    # Make figure

    fig = ShowStack(image = image, pxsize_x = pxsize_x, pxsize_z = pxsize_z, clabel = clabel, cmap = cmap,
                    vmin = vmin, vmax= vmax, figsize = figsize)
    ax = fig.axes

    line_y = ax[1].vlines(0, -rangey/2, rangey/2, colors='white', linestyles='dashed')
    line_x = ax[1].hlines(0, -rangex/2, rangex/2, colors='white', linestyles='dashed')

    line_zx = ax[0].vlines(0, -rangey/2, rangey/2, colors='white', linestyles='dashed')
    line_zy = ax[3].hlines(0, -rangex/2, rangex/2, colors='white', linestyles='dashed')

    y0 = ax[1].get_position().y0
    y1 = ax[1].get_position().y1
    height = y1 - y0

    x0 = ax[1].get_position().x0
    x1 = ax[1].get_position().x1
    width = x1 - x0

    ax_x = fig.add_axes([x0, 0.01, width, 0.03])
    x_slider = Slider(
        ax=ax_x,
        label='X',
        valmin=0,
        valmax=Nx-1,
        valinit=Nx//2,
        valstep = 1,
        initcolor='none',
        facecolor='grey',
        track_color='grey'
    )

    ax_y = fig.add_axes([0.01, y0, 0.0225, height])
    y_slider = Slider(
        ax=ax_y,
        label="Y",
        valmin=-Ny+1,
        valmax=0,
        valinit=-Ny//2,
        valstep = 1,
        initcolor='none',
        orientation="vertical",
        facecolor='grey',
        track_color='grey'
    )
    y_slider.valtext.set_text(str(-y_slider.valinit))

    ax_z = fig.add_axes([x0, y1, width, 0.03])
    z_slider = Slider(
        ax=ax_z,
        label="Z",
        valmin=0,
        valmax=Nz-1,
        valinit=Nz//2,
        valstep = 1,
        initcolor='none',
        facecolor = 'grey',
        track_color = 'grey'
    )

    # The function to be called anytime a slider's value changes
    def update_y(val):
        y = int(-val)
        im = ax[3].get_images()[0]
        im.set_data(image[:, y, :])

        y_units = - y*pxsize_x + rangey/2

        seg_x = [np.array([[-rangex/2, y_units],
                         [rangex/2, y_units]])]

        line_x.set_segments( seg_x )

        y_slider.valtext.set_text(str(y))

        # fig.canvas.draw_idle()

    def update_x(val):
        x = int(val)
        im = ax[0].get_images()[0]
        im.set_data(image[::-1, :, x].T)

        x_units = x * pxsize_x - rangex / 2

        seg_y = [np.array([[x_units, -rangey/2],
                         [x_units, rangey/2]])]

        line_y.set_segments( seg_y )

        # fig.canvas.draw_idle()

    def update_z(val):
        z = int(val)
        im = ax[1].get_images()[0]
        im.set_data(image[z])

        z_units = -z * pxsize_z + rangez / 2

        seg_zx = [np.array([[z_units, -rangey/2],
                         [z_units, rangey/2]])]

        line_zx.set_segments( seg_zx )

        seg_zy = [np.array([[-rangex/2, z_units],
                         [rangex/2, z_units]])]

        line_zy.set_segments( seg_zy )

        # fig.canvas.draw_idle()

    # register the update function with each slider
    y_slider.on_changed(update_y)
    x_slider.on_changed(update_x)
    z_slider.on_changed(update_z)

    return x_slider, y_slider, z_slider



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
                    ax: plt.axis = None, hex_grid = False, gridsize = [4,2]):
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

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    fingerprint = dset.sum(axis=(0, 1))
    N = int(np.ceil(np.sqrt(dset.shape[-1])))
        
    if normalize is True:
        max_counts = np.max(fingerprint)
        fingerprint = fingerprint / max_counts

    if hex_grid is True:
        s = -det_coords(N, 'hex')
        s = np.flip(s, axis = 0)
        im = ax.hexbin(s[0], s[1], fingerprint, gridsize=gridsize, cmap=cmap)
        w = N//2 + 1
        ax.set_xlim([-w,w])
        ax.set_ylim([-w,w])
    else:
        fingerprint = fingerprint.reshape(N, N)
        im = ax.imshow(fingerprint, cmap=cmap)
        ax.set_aspect('equal')

    ax.axis('off')
    fig.tight_layout()

    if colorbar is True:

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


class ColorMap2D:

    def __init__(self):
        self.var_bounds = [0, 1]  # minimum and maximum variable value of the colorbar
        self.int_bounds = [0, 1]  # minimum and maximum intensity value of the colorbar
        self.equalize = True
        self.colormap = 'rainbow'

    def image(self, intensity_map, variable_map):
        output = np.zeros(intensity_map.shape + (3,))

        intensity = np.clip(intensity_map, self.int_bounds[0], self.int_bounds[1])
        intensity = (intensity - self.int_bounds[0]) / (self.int_bounds[1] - self.int_bounds[0])

        cmap = plt.get_cmap(self.colormap)
        N = cmap.N
        cmapArray = np.zeros((N, 3))

        for i in range(N):
            cmapArray[i, :] = cmap(i)[0:3]

        # equalize luminescence
        if self.equalize:
            lmax = np.max((299 * cmapArray[:, 0] + 587 * cmapArray[:, 1] + 114 * cmapArray[:, 2]) / 1000)
            for i in range(N):
                lum = (299 * cmapArray[i, 0] + 587 * cmapArray[i, 1] + 114 * cmapArray[i, 2]) / 1000
                lumfactor = np.min((lmax / lum, 1 / np.max(cmapArray[i, :])))
                cmapArray[i, :] *= lumfactor

        # get index of color map for lifetime
        variable = np.clip(variable_map, self.var_bounds[0], self.var_bounds[1])
        idx = (np.floor((variable - self.var_bounds[0]) / (self.var_bounds[1] - self.var_bounds[0]) * N)).astype(int)
        idx = np.clip(idx, 0, N - 1)
        for k in range(3):
            output[:, :, k] = np.take(cmapArray[:, k], idx) * intensity

        return output

    def colorbar(self, n):
        LG = np.linspace(self.var_bounds[0], self.var_bounds[1], num=n)
        VarGradient = np.tile(LG, (n, 1))

        IG = np.linspace(self.int_bounds[0], self.int_bounds[1], num=n)
        IntensityGradient = np.transpose(np.tile(IG, (n, 1)))

        RGB_colormap = self.image(IntensityGradient, VarGradient)
        RGB_colormap = np.moveaxis(RGB_colormap, 0, 1)

        return RGB_colormap


def show_flim(image: np.ndarray, lifetime: np.ndarray, pxsize: list, pxdwelltime: float, lifetime_bounds: list = None,
              intensity_bounds: list = None, colormap='gist_rainbow', fig=None, ax=None):
    """
    Display the flim image, where intensity and
    lifetime image are represented with a 2D colormap.
    Referring to the HSV color model:
    Intensity values are mapped in Value
    Lifetime values are mapped in Hue

    Parameters
    ----------
    image : np.ndarray
        2D intensity image (Ny x Nx).
    lifetime : np.ndarray
        2D lifetime map (Ny x Nx).
    pxsize : list
        Lateral pixel size, in um.
    pxdwelltime : float
        Pixel dwell time, in us.
    lifetime_bounds : list, optional
        Lifetime bounds of the colormap
    intensity_bounds : list, optional
        Intensity bounds of the colormap
    colormap : str, optional
        Matplotlib colormap used to represent lifetime values.
        The default is 'rainbow'
    fig : plt.Figure, optional
        Figure where to display the plot. If None, a new figure is created.
        The default is None.
    ax : plt.axis, optional
        Axis where to display the plot. If None, a new axis is created.
        The default is None.

    Returns
    -------
    fig : plt.Figure
        Figure that contains the image and the colormap.
    ax : plt.axis
        Axis array that contain the image and the colormap..
    """

    # params settings

    cmap = ColorMap2D()
    cmap.colormap = colormap

    if intensity_bounds is None:
        cmap.int_bounds = [np.min(image), np.max(image)]
    else:
        cmap.int_bounds = intensity_bounds

    if lifetime_bounds is None:
        cmap.var_bounds = [np.min(lifetime), np.max(lifetime)]
    else:
        cmap.var_bounds = lifetime_bounds

    # Image

    RGB = cmap.image(image, lifetime)

    # Colorbar

    N = 256
    RGB_colormap = cmap.colorbar(N)

    # Define extents

    sz = image.shape

    img_extent = (0, sz[1] * pxsize, 0, sz[0] * pxsize)
    cmap_extent = (cmap.int_bounds[0], cmap.int_bounds[1], cmap.var_bounds[0], cmap.var_bounds[1])

    # Show image with colorbar

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    ax.imshow(RGB, extent=img_extent)
    ax.axis('off')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cax.imshow(RGB_colormap, origin='lower', aspect='auto', extent=cmap_extent)
    cax.set_xticks([int(cmap.int_bounds[0]), int(cmap.int_bounds[1])])
    cax.set_xlabel(f'Counts/{pxdwelltime} ' + '$\mathregular{\mu s}$')
    cax.set_ylabel(r'Lifetime (ns)')
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")

    # Add scalebar

    scalebar = ScaleBar(
        1, "um",  # default, extent is calibrated in meters
        box_alpha=0,
        color='w',
        length_fraction=0.25)

    ax.add_artist(scalebar)

    plt.tight_layout()

    return fig, ax


def depth_stack(stack: np.ndarray, pxsize: list, pxdwelltime: float, axis: int = 0, colormap='rainbow', fig=None,
                ax=None):
    """
    Display the maximum intensity projection of stack, where intensity and
    depth image are represented with a 2D colormap.
    Referring to the HSV color model:
    Intensity values are mapped in Value
    Depth values are mapped in Hue

    Parameters
    ----------
    stack : np.ndarray
        3D image (Nz x Ny x Nx).
    pxsize : list
        List of pixel size of each dimension, in um [px_z, px_y, px_x].
    pxdwelltime : float
        Pixel dwell time, in us.
    axis : int, optional
        Projection axis. The default is 0 (zeta).
    colormap : str, optional
        Matplotlib colormap used to represent lifetime values.
        The default is 'rainbow'
    fig : plt.Figure, optional
        Figure where to display the plot. If None, a new figure is created.
        The default is None.
    ax : plt.axis, optional
        Axis where to display the plot. If None, a new axis is created.
        The default is None.

    Returns
    -------
    fig : plt.Figure
        Figure that contains the image and the colormap.
    ax : plt.axis
        Axis array that contain the image and the colormap..
    """

    image_mip = stack.max(axis=axis)
    depth_mip = stack.argmax(axis=axis) * pxsize[axis]

    # params settings

    cmap = ColorMap2D()
    cmap.colormap = colormap

    cmap.int_bounds = [np.min(image_mip), np.max(image_mip)]
    cmap.var_bounds = [np.min(depth_mip), np.max(depth_mip)]

    # Image

    RGB = cmap.image(image_mip, depth_mip)

    # Colorbar

    N = stack.shape[axis]
    RGB_colormap = cmap.colorbar(N)

    # Define extents

    sz = image_mip.shape
    px = np.delete(pxsize, axis)

    img_extent = (0, sz[1] * px[1], 0, sz[0] * px[0])
    cmap_extent = (cmap.int_bounds[0], cmap.int_bounds[1], cmap.var_bounds[0], cmap.var_bounds[1])

    # Show image with colorbar

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    ax.imshow(RGB, extent=img_extent)
    ax.axis('off')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cax.imshow(RGB_colormap, origin='lower', aspect='auto', extent=cmap_extent)
    cax.set_xticks([int(cmap.int_bounds[0]), int(cmap.int_bounds[1])])
    cax.set_xlabel(f'Counts/{pxdwelltime} ' + '$\mathregular{\mu s}$')
    cax.set_ylabel(r'Depth ($\mathregular{\mu m}$)')
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")

    # Add scalebar

    scalebar = ScaleBar(
        1, "um",  # default, extent is calibrated in meters
        box_alpha=0,
        color='w',
        length_fraction=0.25)

    ax[0].add_artist(scalebar)

    plt.tight_layout()

    return fig, ax