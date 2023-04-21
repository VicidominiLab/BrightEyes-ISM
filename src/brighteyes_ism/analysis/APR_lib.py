import numpy as np
from scipy.ndimage import fourier_shift, shift
from skimage.registration import phase_cross_correlation

def sigmoid(R: float, T: float, S: float):
    '''
    It generates a circularly-symmetric sigmoid function.

    Parameters
    ----------
    R : float
        Radial axis.
    T : float
        Cut-off frequency.
    S : float
        Sigmoid slope.

    Returns
    -------
    np.ndarray
        Sigmoid array. It has the same dimensions of R.

    '''
    
    return  1 / (1 + np.exp( (R-T)/S ) )

def Low_pass(img: np.ndarray, T: float, S: float, pxsize: float = 1, data: str = 'real'):
    '''
    It applies a low-pass sigmoidal filter to a 2D image.

    Parameters
    ----------
    img : np.ndarray
        2D image.
    T : float
        Cut-off frequency.
    S : float
        Sigmoid slope.
    pxsize : float, optional
        Pixel size. The default is 1.
    data : str, optional
        Domain of the image: It can be 'real' or 'fourier'.
        The default is 'real'.

    Returns
    -------
    img_filt : np.ndarray
        Filtered 2D image, in the domain specified by 'data'.

    '''
    
    if data == 'real':
        img_fft = np.fft.fftn(img, axes = (0,1) )
        img_fft = np.fft.fftshift(img_fft, axes = (0,1) )
    elif data == 'fourier':
        img_fft = img
    else:
        raise ValueError('data has to be \'real\' or \'fourier\'')
            
    Nx = np.shape(img_fft)[0]
    Ny = np.shape(img_fft)[1]
    cx = int ( ( Nx + np.mod(Nx,2) ) / 2)
    cy = int ( ( Ny + np.mod(Ny,2) ) / 2)
    
    x = ( np.arange(Nx) - cx ) / Nx
    y = ( np.arange(Ny) - cy ) / Ny
    
    X, Y = np.meshgrid(x, y)
    R = np.sqrt( X**2 + Y**2 )
    
    sig = sigmoid(R, T, S)
    
    img_filt = np.einsum( 'ij..., ij -> ij...', img_fft, sig )
    
    if data == 'real':
        img_filt = np.fft.ifftshift(img_filt, axes = (0,1) )
        img_filt = np.fft.ifftn(img_filt, axes = (0,1) )
        img_filt = np.abs(img_filt)
    
    return img_filt

def hann2d( shape: tuple ):
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
    X, Y = np.meshgrid(x,y)
    W = 0.5 * ( 1 - np.cos( (2*np.pi*X)/(Nx-1) ) )
    W *= 0.5 * ( 1 - np.cos( (2*np.pi*Y)/(Ny-1) ) )
    
    return W

def rotate(array: np.ndarray, degree: float):
    '''
    It rotates the shift-vectors by the desired angle.

    Parameters
    ----------
    array : np.ndarray
        Shift-vectors array (Nch x 2).
    degree : float
        Angle of rotation, expressed in degrees.

    Returns
    -------
    m : np.ndarray
        Rotated shift-vectors.

    '''
    
    radians = degree*(np.pi/180)  
    x = array[:,0]
    y = array[:,1]    
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, ( [x, y] ) )
    return m

def APR(dset: np.ndarray, usf: int, ref: int, pxsize: float = 1, apodize: bool = True, cutoff: float = None, mode: str = 'fourier'):
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
    cutoff : float, optional
        If it is a number, it applies a low-pass filter at the cutoff value
        to calculate the shift-vectors. The default is None.
    mode : str, optional
        Registration method. It can be a fourier shift ('fourier'')
        or an interpolation ('interp'). The default is 'fourier'.

    Returns
    -------
    shift_vec : np.ndarray
        Shift-vectors (Nch x 2). The second dimension is the x/y axis.
    result_ism_pc : np.ndarray
        Reassigned ISM dataset (Nx x Ny x Nch).

    '''

    # Calculate shifts
    
    shift_vec, error = ShiftVectors(dset, usf, ref, pxsize = pxsize, apodize = apodize, cutoff = cutoff)

    # Register images
    
    result_ism_pc = Reassignment(shift_vec, dset, mode = mode)

    shift_vec *= pxsize

    return shift_vec, result_ism_pc

def ShiftVectors(dset: np.ndarray, usf: int, ref: int, pxsize: float = 1, apodize: bool = True, cutoff: float = None):
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

    Returns
    -------
    shift_vec : np.ndarray
        Shift-vectors (Nch x 2). The second dimension is the x/y axis.
    error : np.ndarray
        Estimation error of the shift-vectors.

    '''

    sz = dset.shape

    # Low-pass filter dataset

    if cutoff is not None:
        s = 0.01
        t = cutoff * pxsize
        dset = Low_pass(dset, t, s)

    # Apodize dataset

    if apodize == True:
        W = hann2d(dset.shape)
        dsetW = np.einsum('ijk, ij -> ijk', dset, W)
    else:
        dsetW = dset

    # Calculate shift-vectors

    shift_vec = np.empty( (sz[-1], 2) )
    error = np.empty( (sz[-1], 2) )
    
    for i in range( sz[-1] ):
        shift_vec[i,:], error[i,:], diffphase = phase_cross_correlation(dsetW[:,:, ref], dset[:,:,i], upsample_factor=usf, normalization=None)
    
    return shift_vec, error

def Reassignment(shift_vec: np.ndarray, dset: np.ndarray, mode: str = 'fourier'):
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
        or an interpolation ('interp'). The default is 'fourier'.

    Returns
    -------
    result_ism_pc : np.ndarray
        Reassigned ISM dataset (Nx x Ny x Nch).

    '''
    
    sz = dset.shape    
    result_ism_pc = np.empty( sz )
    
    if mode == 'fourier':
    
        for i in range( sz[-1] ):
            offset  = fourier_shift(np.fft.fftn(dset[:,:,i]), (shift_vec[i,:]))
            result_ism_pc[:,:,i]  = np.real( np.fft.ifftn(offset) )
        return result_ism_pc
        
    elif mode == 'interp':
        
        for i in range( sz[-1] ):
            result_ism_pc[:,:,i]  = shift( dset[:,:,i], shift_vec[i,:] )
        return result_ism_pc