import numpy as np
from scipy.ndimage import fourier_shift, shift
from skimage.registration import phase_cross_correlation
from skimage.filters import gaussian


def hann2d(shape: tuple):
    '''
    It generates a 2D Hann window for a 2D array.

    Parameters
    ----------
    shape : tuple
        Shape of the window.
        It has to be the same shape of the image to be windowed.

    Returns
    -------
    W : np.ndarray
        2D Hann window function.

    '''

    Nx, Ny = shape[0], shape[1]

    x = np.arange(Nx)
    y = np.arange(Ny)
    X, Y = np.meshgrid(x, y)
    W = 0.5 * (1 - np.cos((2 * np.pi * X) / (Nx - 1)))
    W *= 0.5 * (1 - np.cos((2 * np.pi * Y) / (Ny - 1)))

    return W


def APR(dset: np.ndarray, usf: int, ref: int, pxsize: float = 1, apodize: bool = True, filter_sigma: float = 0,
        mode: str = 'interp'):
    '''
    It performs adaptive pixel reassignment on a single-plane ISM dataset using the phase correlation method.

    Parameters
    ----------
    dset : np.ndarray
        ISM dataset (Nx x Ny x Nch).
    usf : int
        Upsampling factor (subpixel precision).
    ref : int
        Index of the SPAD element to be used as a reference.
    pxsize : float, optional
        Pixel size. The default is 1.
    apodize : bool, optional
        If True, the dataset is apodized to calculate the shift-vectors.
        The default is True.
    filter_sigma: float
        If bigger than zero, the dataset is denoised with a gaussian filter.
        The default is 0.
    mode : str, optional
        Registration method. It can be a fourier shift ('fourier'')
        or a linear interpolation ('interp'). The default is 'interp'.

    Returns
    -------
    shift_vec : np.ndarray
        Shift-vectors (Nch x 2). The second dimension is the x/y axis.
    result_ism_pc : np.ndarray
        Reassigned ISM dataset (Nx x Ny x Nch).

    '''

    # Calculate shifts

    shift_vec, error = ShiftVectors(dset, usf, ref, apodize=apodize, filter_sigma=filter_sigma)

    # Register images

    result_ism_pc = Reassignment(shift_vec, dset, mode=mode)

    shift_vec *= pxsize

    return shift_vec, result_ism_pc


def ShiftVectors(dset: np.ndarray, usf: int, ref: int, apodize: bool = True, filter_sigma: float = 0):
    '''
    It calculates the shift-vectors from a single-plane ISM dataset using the phase correlation method.

    Parameters
    ----------
    dset : np.ndarray
        ISM dataset (Nx x Ny x Nch).
    usf : int
        Upsampling factor (subpixel precision).
    ref : int
        Index of the SPAD element to be used as a reference.
    apodize: bool
        If True, the dataset is apodized with a Hann windows
        The default is True.
    filter_sigma: float
        If bigger than zero, the dataset is denoised with a gaussian filter.
        The default is 0.

    Returns
    -------
    shift_vec : np.ndarray
        Shift-vectors (Nch x 2). The second dimension is the xy-axes.
    error : np.ndarray
        Estimation error of the shift-vectors.

    '''

    sz = dset.shape
    dsetW = dset.copy()

    # Apodize dataset

    if apodize == True:
        W = hann2d(dset.shape)
        dsetW = np.einsum('ijk, ij -> ijk', dsetW, W)

    # Low-pass filter dataset

    if filter_sigma > 0:
        dsetW = gaussian(dsetW, sigma=filter_sigma, channel_axis=-1)

    # Calculate shift-vectors

    shift_vec = np.empty((sz[-1], 2))
    error = np.empty((sz[-1], 2))

    for i in range(sz[-1]):
        shift_vec[i, :], error[i, :], diffphase = phase_cross_correlation(dsetW[:, :, ref], dsetW[:, :, i],
                                                                          upsample_factor=usf, normalization=None)

    return shift_vec, error


def Reassignment(shift_vec: np.ndarray, dset: np.ndarray, mode: str = 'interp'):
    '''
    It reassignes a single-plane ISM dataset using the provided shift-vectors.

    Parameters
    ----------
    shift_vec : np.ndarray
        Shift-vectors array (Nch x 2).
    dset : np.ndarray
        ISM dataset (Nx x Ny x Nch).
    mode : str, optional
        Registration method. It can be a fourier shift ('fourier'')
        or an interpolation ('interp'). The default is 'interp'.

    Returns
    -------
    result_ism_pc : np.ndarray
        Reassigned ISM dataset (Nx x Ny x Nch).

    '''

    sz = dset.shape
    result_ism_pc = np.empty(sz)

    for i in range(sz[-1]):
        if mode == 'fourier':
            offset = fourier_shift(np.fft.fftn(dset[:, :, i]), (shift_vec[i, :]))
            result_ism_pc[:, :, i] = np.real(np.fft.ifftn(offset))
        elif mode == 'interp':
            result_ism_pc[:, :, i] = shift(dset[:, :, i], shift_vec[i, :])

    result_ism_pc[result_ism_pc < 0] = 0
    return result_ism_pc
