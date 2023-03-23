import numpy as np

from .FRC_lib import radial_profile

#%%

def Reorder(data, inOrder: str, outOrder: str = 'rzxytc'):
    '''
    It reorders a dataset to match the desired order of dimensions.
    If some dimensions are missing, it adds new dimensions.

    Parameters
    ----------
    data : ndarray
        ISM dataset.
    inOrder : str
        Order of the dimension of the data.
        It can contain any letter of the outOrder string.
    outOrder : str, optional
        Order of the output. The default is 'rzxytc'.

    Returns
    -------
    data : ndarray
        ISM dataset reordered.

    '''
    
    if not(inOrder == outOrder):
        # adds missing dimensions
        Nout = len(outOrder)
        dataShape = np.shape(data)
        Ndim = len(dataShape)
        for i in range(Nout-Ndim):
            data = np.expand_dims(data, Ndim+i)
        
        # check order of dimensions
        order = []
        newdim = 0
        for i in range(Nout):
            dim = outOrder[i]
            if dim in inOrder:
                order.append(inOrder.find(dim))
            else:
                order.append(Ndim+newdim)
                newdim += 1
        data = np.transpose(data, order)
    
    return data

def CropEdge(dset, npx = 10, edges = 'l', order: str = 'rzxytc'):
    '''
    It crops an ISM dataset along the specified edges of the xy plane.
    
    Parameters
    ----------
    dset : ndarray
        ISM dataset
    npx : int, optional
        Number of pixel to crop from each edge. The default is 10.
    edges : str, optional
        Cropped edges. The possible values are 'l' (left),'r' (right),
        'u' (up), and 'd' (down). Any combination is possible. The default is 'l'.
    order : str, optional
        Order of the dimensions of the dataset The default is 'rzxytc'.

    Returns
    -------
    dset_cropped : ndarray
        ISM dataset cropped

    '''
    
    dset_cropped = Reorder(dset, order)
    
    if 'l' in edges:
        dset_cropped = dset_cropped[..., npx:, :, :, :]
        
    if 'r' in edges:
        dset_cropped = dset_cropped[..., :-npx, :, :, :]
        
    if 'u' in edges:
        dset_cropped = dset_cropped[..., :, npx:, :, :]
        
    if 'd' in edges:
        dset_cropped = dset_cropped[..., :, :-npx, :, :]
        
    return np.squeeze(dset_cropped)

def DownSample(dset, ds: int = 2, order: str = 'rzxytc'):
    '''
    It downsamples an ISM dataset on the xy plane.
    
    Parameters
    ----------
    dset : ndarray
        ISM dataset.
    ds : int, optional
        Downsampling factor. The default is 2.
    order : str, optional
        Order of the dimensions of the dataset The default is 'rzxytc'.
        
    Returns
    -------
    dset_ds : ndarray
        ISM dataset downsampled.

    '''
    
    dset = Reorder(dset, order)
    
    dset_ds = dset[..., ::ds, ::ds, :, :]
    
    return np.squeeze(dset_ds)

def UpSample(dset, us: int = 2, npx: str = 'even', order: str = 'rzxytc'):
    '''
    It upsamples an ISM dataset on the xy plane.

    Parameters
    ----------
    dset : TYPE
        ISM dataset.
    us : int, optional
        Upsampling factor. The default is 2.. The default is 2.
    npx : str, optional
        Parity of the number of pixels on each axis. The default is 'even'.
    order : str, optional
        Order of the dimensions of the dataset The default is 'rzxytc'.

    Returns
    -------
    dset_us : ndarray
        ISM dataset upsampled.

    '''
    
    dset = Reorder(dset, order)
    
    sz = dset.shape
    
    if npx == 'even':
        sz_us = np.asarray(sz)
        sz_us[2] = sz_us[2]*us
        sz_us[3] = sz_us[3]*us
    elif npx == 'odd':
        sz_us = np.asarray(sz)
        sz_us[2] = sz_us[2]*us - 1
        sz_us[3] = sz_us[3]*us - 1
        
    dset_us = np.zeros( sz_us )
    dset_us[..., ::us, ::us, :, :] = dset
    
    return np.squeeze(dset_us)

def ArgMaxND(data):
    '''
    It finds the the maximum and the corresponding indeces of a N-dimensional array.

    Parameters
    ----------
    data : ndarray
        N-dimensional array.

    Returns
    -------
    arg : ndarray(int)
        indeces of the maximum.
    mx : float
        maximum value.

    '''
    
    idx = np.argmax(data)

    mx = np.array(data).ravel()[idx]

    arg = np.unravel_index(idx, np.array(data).shape)
    
    return arg, mx

def FWHM(x, y):
    '''
    It calculates the Full Width at Half Maximum of a 1D curve.

    Parameters
    ----------
    x : ndarray
        Horizontal axis.
    y : ndarray
        Curve.

    Returns
    -------
    FWHM: float
        Full Width at Half Maximum of the y curve.

    '''
    
    height = 0.5
    height_half_max = np.max(y) * height
    index_max = np.argmax(y)
    x_low = np.interp(height_half_max, y[:index_max], x[:index_max])
    x_high = np.interp(height_half_max, np.flip(y[index_max:]), np.flip(x[index_max:]))

    return x_high - x_low

def RadialSpectrum(img, pxsize: float = 1, normalize: bool = True):
    '''
    It calculates the radial spectrum of a 2D image.

    Parameters
    ----------
    img : ndarray
        2D image.
    pxsize : float, optional
        Pixel size. The default is 1.
    normalize : bool, optional
        If True, the result is divided by its maximum. The default is True.

    Returns
    -------
    ftR : ndarray
        Radial spectrum.
    space_f : ndarray
        Frequency axis.

    '''
    
    fft_img = np.fft.fftn(img, axes = [0, 1])
    fft_img = np.abs( np.fft.fftshift( fft_img, axes = [0, 1]) )    
    
    sx, sy = fft_img.shape
    c = (sx//2, sy//2)
    
    space_f = np.fft.fftfreq(sx, pxsize)[:c[0]]
    
    ftR = radial_profile(fft_img, c)
    
    ftR = ftR[0][:c[0]] / ftR[1][:c[0]]
    
    ftR = np.real(ftR)
    
    if normalize == True:
        ftR /= np.max( ftR )
    
    return ftR, space_f

#%%

from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

def ShowImg(fig, ax, image, pxsize_x, clabel,  cmap = 'hot'):
    
    im = ax.imshow( image, cmap = cmap )
    ax.axis('off')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, ticks = [])# np.floor( [np.min(image), np.max(image)] ), )
    # ax.text(1.0,0.4, clabel, rotation=90, transform=ax.transAxes)

    cbar.ax.set_ylabel(clabel, labelpad=-11, rotation=90)
    
    cbar.ax.text(1.02, 0.9, f'{ int(np.floor(np.max(image))) }', rotation=90, transform=ax.transAxes)
    
    cbar.ax.text(1.02, 0.02, f'{ int(np.floor(np.min(image))) }', rotation=90, transform=ax.transAxes, color = 'white')
    
    scalebar = ScaleBar(
    pxsize_x, "um", # default, extent is calibrated in meters
    box_alpha=0,
    color='w',
    length_fraction=0.25)
    
    ax.add_artist(scalebar)
    
    return None