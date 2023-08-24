import numpy as np
from scipy import pi
import matplotlib.pyplot as plt
import scipy.fft as ft
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import curve_fit

def smooth(x,y):
    """
    Apply smoothing filter based on lowess

    Parameters
    ----------
    x : np.array(N)
        horizontal axis
    y : np.array(N)
        noisy array

    Returns
    -------
    x_interpolated : np.array(100 x N)
        interpolated x-axis
    y_filtered : np.array(100 x N)
        interpolated and smoothed array
    """
    
    x_interp = np.linspace(x[0], x[-1], num = 100*len(x) )
    y_interp = np.interp(x_interp, x, y)
    
    filtered = lowess(y_interp, x_interp, is_sorted=True, frac=0.05, it=0)
    return x_interp, filtered[:,1]

def hann2d(*args):
    """
    Bi-dimensional Hann windowing function

    Parameters
    ----------
    N : int
        Shape of the image (N x M). If only N is provided, a squared
        image (N x N) is assumed.
    M : int
        Shape of the image (N x M)

    Returns
    -------
    W : np.array(N x M)
        Hann window function
    """
    
    if len(args)>1 :
        N, M = args
    else:
        N, = args
        M=N

    x = np.arange(N)
    y = np.arange(M)
    X, Y = np.meshgrid(x,y)
    W = 0.5 * ( 1 - np.cos( (2*pi*X)/(N-1) ) )
    W *= 0.5 * ( 1 - np.cos( (2*pi*Y)/(M-1) ) )
    return W


def radial_profile(data, center):
    """
    Calculation of the radial profile of an image

    Parameters
    ----------
    data : np.array(N x M)
        image
    center : np.array(2)
        indices of the center of the image

    Returns
    -------
    radialprofile : np.array( np.sqrt(N**2 + M**2) )
        sum of the data over the angular coordinate
    nr : int
        number of pixels in the angular bin
    """
    
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    
    tbin = np.bincount(r.ravel(), np.real(data).ravel()).astype(np.complex128)
    tbin += 1j*np.bincount(r.ravel(), np.imag(data).ravel())    
    
    nr = np.bincount(r.ravel())
    radialprofile = tbin

    return radialprofile, nr


def FRC(im1, im2):
    """
    Fourier Ring Correlation. It requires two identical images, differing only
    in the noise content.

    Parameters
    ----------
    im1 : np.array(N x M)
        First input image
    im2 : np.array(N x M)
        Second input image

    Returns
    -------
    FRC : np.array()
        Raw FRC curve
    """
    
    M, N = np.shape(im1)
    cx = int ( ( N + np.mod(N,2) ) / 2)
    cy = int ( ( M + np.mod(M,2) ) / 2)
    center = [cx, cy]
    
    im1 = im1 * hann2d(N,M)
    im2 = im2 * hann2d(N,M)
    
    ft1 = ft.fftshift( ft.fft2( im1 ) )
    ft2 = ft.fftshift( ft.fft2( im2 ) )
    
    num = np.real( radial_profile( ft1*np.conj(ft2) , center)[0] )
    den = np.real( radial_profile( np.abs(ft1)**2, center)[0] )
    den = den * np.real( radial_profile( np.abs(ft2)**2, center)[0] )
    den = np.sqrt( den )
    
    FRC = num / den

    return FRC


def fixed_threshold(frc, y):
    """
    Calculate the treshold for the FRC analysis

    Parameters
    ----------
    frc : np.ndarray
        FRC values
    y : float
        Threshold value

    Returns
    -------
    th : float
        Threshold value
    idx : int
        Index where threshold value is reached
    """
    N = len(frc)
    th = np.ones(N) * y
    try:
        idx = np.argwhere(np.diff(np.sign(frc - y))).flatten()[0]
    except:
        idx = 0
    return th, idx


def nsigma_threshold(k, frc, img, sigma):
    """
    Find the treshold for the FRC analysis

    Parameters
    ----------
    k : np.ndarray
        Frequencies array (N)
    frc : np.ndarray
        FRC value for each k value (N)
    img : np.ndarray()
        Image used to calculate the radial profile
    sigma : int
        Criterium used for the threshold (3 for '3 sigma' criterium, etc.)

    Returns
    -------
    th_interp : float
        Threshold
    idx2 : int
        Index where threshold value is reached
    """
    
    N, M = np.shape(img)
    cx = int ( ( N + np.mod(N,2) ) / 2)
    cy = int ( ( M + np.mod(M,2) ) / 2)
    center = [cx, cy]
    
    nr = radial_profile(img, center)[1]
    
    th = sigma/np.sqrt(nr/2)
    
    k_interp, th_interp = smooth(k, th)
    
    try:
        idx1 = np.argwhere(np.diff(np.sign(frc - th_interp))).flatten()
        idx2 = idx1[1]
    except:
        idx2 = 0
    return th_interp, idx2


def FRC_resolution(I1, I2, px = 1, method = 'fixed', smoothing = 'lowess'):
    """
    Fourier Ring Correlation analysis. It requires two identical images, differing only
    in the noise content, and estimates the resolution from the FRC curve.

    Parameters
    ----------
    I1 : np.array(N x M)
        First input image
    I2 : np.array(N x M)
        Second input image
    px : float
        Pixel size of the images
    method : str
        Threshold criterium. If 'fixed', it uses the 1/7 threshold.
        Other possibilities are '3sigma' and '5sigma'
    smoothing : str
        Smoothing method for the FRC curve. If 'lowess' it smooths and
        interpolates the curve using a lowess algorithm. If 'fit' it fits the
        curve with a sigmoid model and removes high-frequency offset, if present.
        Default is 'lowess'.

    Returns
    -------
    res_um : float
        Estimated resolution, in real units
    k : np.array( np.sqrt(N**2 + M**2) )
        Array of spatial frequencies
    frc : np.array( np.sqrt(N**2 + M**2) )
        Array of raw FRC
    k_interp : np.array( 100 x np.sqrt(N**2 + M**2) )
        Interpolated array of spatial frequencies
    frc_smooth : np.array( 100 x np.sqrt(N**2 + M**2) )
        Interpolated and smoothed FRC curve
    th : np.array( 100 x np.sqrt(N**2 + M**2) )
        Threshold curve
    """
    
    Nx, Ny = I1.shape
    
    frc = FRC(I1, I2)
    F = len(frc)

    max_kpx = 1 / np.sqrt( 2 )

    kpx = np.linspace(0, max_kpx, F, endpoint = True)
    k = kpx / px

    if smoothing == 'lowess':

        k_interp, frc_smooth = smooth(k, frc) # FRC smoothing

    elif smoothing == 'fit':

        p0 = [1, 1, 1, 0]

        sigmoid_fit = lambda x, a, b, c, d: a / (1 + np.exp((x - b) / c)) + d

        popt, pcov = curve_fit(sigmoid_fit, k[kpx<0.5], frc[kpx<0.5], p0)

        amplitude = popt[0]
        offset = popt[-1]

        k_interp = np.linspace(0, max_kpx, F*1000, endpoint = True) / px
        frc_smooth = (sigmoid_fit(k_interp, *popt) - offset)/ amplitude
        frc = (frc - offset)/ amplitude

    else:
        raise ValueError('The smoothing parameter has to be "fit" or "lowess".')

    if method == 'fixed':
        th, idx = fixed_threshold(frc_smooth, 1/7)
    elif method == '3sigma':
        th, idx = nsigma_threshold(k, frc_smooth, I1, 3)
    elif method == '5sigma':
        th, idx = nsigma_threshold(k, frc_smooth, I1, 5)

    res_um = (1/k_interp[idx])
    
    return res_um, k, frc, k_interp, frc_smooth, th

def timeFRC(dset, px = 1, method = 'fixed'):
    """
    Fourier Ring Correlation analysis. It requires a single dataset with a
    temporal dimension to generate two images using the even and odd indices
    of the time axis. Then, it estimates the resolution using the FRC analysis.

    Parameters
    ----------
    dset : np.ndarray
        dataset (Nx x Ny x Nt)
    px : float
        Pixel size of the images
    Method : str
        Threshold criterium. If 'fixed', it uses the 1/7 threshold.
        Other possibilities are '3sigma' and '5sigma'

    Returns
    -------
    res_um : float
        Estimated resolution, in real units
    k : np.array( np.sqrt(N**2 + M**2) )
        Array of spatial frequencies
    frc : np.array( np.sqrt(N**2 + M**2) )
        Array of raw FRC
    k_interp : np.array( 100 x np.sqrt(N**2 + M**2) )
        Interpolated array of spatial frequencies
    frc_smooth : np.array( 100 x np.sqrt(N**2 + M**2) )
        Interpolated and smoothed FRC curve
    th : np.array( 100 x np.sqrt(N**2 + M**2) )
        Threshold curve
    """
    
    if dset.shape[-1] % 2 == 0:
        img_even = dset[:, :, 0::2].sum(axis = -1)
        img_odd  = dset[:, :, 1::2].sum(axis = -1)
    else:
        img_even = dset[:, :, 0:-1:2].sum(axis = -1)
        img_odd  = dset[:, :, 1::2].sum(axis = -1)
    
    res_um, k, frc, k_interp, frc_smooth, th = FRC_resolution(img_even, img_odd, px = px, method = method)
    
    return res_um, k, frc, k_interp, frc_smooth, th

def plotFRC(res_um, k, frc, k_interp, frc_smooth, th, fig = None, ax = None):
    """
    Visualization of the results of the FRC curve. The inputs are exactly the
    outputs of FRC_resolution function.

    Parameters
    ----------
    res_um : float
        Estimated resolution, in real units
    k : np.array( np.sqrt(N**2 + M**2) )
        Array of spatial frequencies
    frc : np.array( np.sqrt(N**2 + M**2) )
        Array of raw FRC
    k_interp : np.array( 100 x np.sqrt(N**2 + M**2) )
        Interpolated array of spatial frequencies
    frc_smooth : np.array( 100 x np.sqrt(N**2 + M**2) )
        Interpolated and smoothed FRC curve
    th : np.array( 100 x np.sqrt(N**2 + M**2) )
        Threshold curve

    Returns
    -------
    None.
    """
    
    if fig == None or ax == None:
        fig, ax = plt.subplots()
        
    ax.plot(k, frc, '.', label = 'FRC - raw')
    ax.plot(k_interp, frc_smooth, '-', linewidth = 3, label = 'FRC - smoothed')
    ax.plot(k_interp, th, linewidth = 3, label = 'Threshold')
    
    ax.legend()
    
    idx = (np.abs(k_interp - 1/res_um)).argmin()
    
    ax.plot(k_interp[idx], frc_smooth[idx], 'o', markersize = 6)
    
    k_max = k[-1]*np.sqrt( 2 )/2
    ax.set_xlim( ( 0, k_max ) )
    ax.set_ylim( ( -0.05, 1.05 ) )
    
    ax.set_xlabel(r'k ($\mathregular{\mu m ^{-1}}$)')
    ax.set_ylabel('FRC')
    
    ax.set_title(f'Resolution = {res_um:.3f}' + r' $\mathregular{\mu m}$')
    
    return fig, ax