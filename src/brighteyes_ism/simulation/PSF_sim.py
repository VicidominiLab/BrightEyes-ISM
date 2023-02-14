import numpy as np
import scipy.signal as sgn
from PyFocus.custom_mask_functions import generate_incident_field, custom_mask_focus_field_XY, plot_in_cartesian
from poppy.zernike import noll_indices, R, zern_name
import copy as cp

#%% Zernike

def Zernike(index, A, h, rho, phi, normalize = True):
    """
    This function is designed to work with a scalar coordinate (rho, phi), not
    with arrays. It is only for internal use of the simulator.

    Parameters
    ----------
    index : int or array
        aberration index
    A : float or array
        aberration strength [rad]
    h : float
        radius of aperture of the objective lens [mm]
    rho : float
        radial coordinate of the pupil plane [mm]
    phi : float
        angular coordinate of the pupil plane [mm]
    normalize : bool
        if true, the integral of Z**2 on the unit circle equals pi
    
    Returns
    -------
    
    """
    
    if np.isscalar( index ):
        index = [ index ]
        
    if np.isscalar(A):
        A = [ A ]*len(index)
        
    rho /= h
    phase = 0
    
    for i in range( len(index) ):
    
        n, m = noll_indices( index[i] )
        
        if rho > 1:
            radial = 0
        else:
            radial = R(n, m, rho)
    
        if m < 0:
            angular = np.sin( np.abs(m) * phi )
        else:
            angular = np.cos( np.abs(m) * phi )
        
        if normalize == True:
            norm_coeff = np.sqrt(2) * np.sqrt(n + 1)
        else:
            norm_coeff = 1
        
        phase += A[i] * 1j * norm_coeff * ( radial * angular ) 
        
    return  np.exp( phase )

#%% functions

class simSettings:
    """
    Optical settings used to calculate the psf 
    Read more at https://pyfocus.readthedocs.io/en/latest/

    Attributes
    ----------
    na : float
        numerical aperture
    n : float
        sample refractive index
    wl : float
        wavelength in vacuum [nm]
    h : float
        radius of aperture of the objective lens [mm]
    gamma : float
        parameter describing the light polarization (amplitude)
    beta : float
        parameter describing the light polarization (phase)
    w0 : float
        radius of the incident gaussian beam [mm]
    I0 : float
        Intensity of the entrance field [W/m**2]
    field : str
        spatial distribution of the entrance field
        'PlaneWave' = flat field
        'Gaussian' = gaussian beam of waist w0
    mask : str
        phase mask 
        None = no mask
        'VP' = vortex phase plate
        'Zernike' = zernike polynomials
        custom function ( mask = lambda rho, phi, w0, f, k: ... )
    mask_sampl : int
        entrance field and mask sampling (# points)
    sted_sat : float
        STED maximum saturation factor
    sted_pulse : float
        STED pulse duration [ns]
    sted_tau : float
        fluorescence lifetime [ns]
    abe_index: int or array
        aberration index
    abe_ampli : float or array
        aberration amplitude in rad
    
    Methods
    -------
    f : float
        Returns the focal length.
    alpha : float
        Returns the semi angular aperture.
    aberration : str / list
        Returns the list of aberrations by name.
    """
    
    __slots__ = ['na', 'n', 'wl', 'h', 'gamma', 'beta', 'w0', 'I0', 'field', 'mask', 'mask_sampl',
                 'sted_sat', 'sted_pulse', 'sted_tau', 'abe_index', 'abe_ampli']

    def __init__(self, na=1.4, n = 1.5, wl = 485.0, h = 2.8, gamma = 45.0, beta = 90.0,
                 w0 = 100.0, I0 = 1, field = 'PlaneWave', mask = None,  mask_sampl = 200,
                 sted_sat = 50, sted_pulse = 1, sted_tau = 3.5,
                 abe_index = None, abe_ampli = None):
        
        self.na = na            # numerical aperture
        self.n = n              # sample refractive index
        self.wl = wl            # wavelength [nm]
        self.h = h              # radius of aperture of the objective lens [mm]
        self.w0 = w0            # radius of the incident gaussian beam [mm]
        self.gamma = gamma      # parameter describing the light polarization (amplitude)
        self.beta = beta        # parameter describing the light polarization (phase)
        self.I0 = I0            # Intensity of the entrance field
        self.field = field      # entrance field at the pupil plane
                                #   'PlaneWave' = Flat field
                                #   'Gaussian' = Gaussian beam with waist w0
        self.mask = mask        # phase mask 
                                #   None = no mask
                                #   'VP' = vortex phase plate
                                #   'Zernike' = zernike polynomials
        self.mask_sampl = mask_sampl   # phase mask sampling
        self.sted_sat = sted_sat   # STED maximum saturation factor
        self.sted_pulse = sted_pulse   # STED pulse duration [ns]
        self.sted_tau = sted_tau   # fluorescence lifetime [ns]
        self.abe_index = abe_index      # aberration index (int or array)
        self.abe_ampli = abe_ampli      # aberration amplitude in rad (float or array)
        
    @property
    def f(self):    # focal length of the objective lens [mm]
        return self.h*self.n/self.na

    @property
    def alpha(self):    # semiangular aperture of the objective [rad]
        return np.arcsin(self.na/self.n)   

    @property
    def aberration(self):
        
        if self.abe_index is None:
            return 'None'
        
        if np.isscalar( self.abe_index ):
            index = [ self.abe_index ]
        else:
            index = self.abe_index

        names = []
        
        for i in range( len(index) ):
            names.append( zern_name( index[i] ) )
        
        return names

    def copy(self):
        return cp.copy(self)
        
def singlePSF(par, pxsizex, Nx, z_shift = 0, return_entrance_field = False):
    """
    Simulate PSFs with PyFocus

    Parameters
    ----------
    par : simSettings object
        Object with PSF parameters
    pxsizex : float
        Pixel size of the simulation space in XY [nm] (typically 1)
    Nx : int
        Number of pixels in XY dimensions in the simulation array, e.g. 1024
    z_shift : float
        Distance from the focal plane at which generate the PSF [nm] (optional)
    return_entrance_field : bool
        Returns the X and Y components of the field at the
        pupil plane in polar coordinates. They have to be
        converted into cartesian coordinate using the
        plot_in_cartesian function
    
    Returns
    -------
    exPSF : np.array(Nx x Nx)
        with the excitation PSF calculated from exPSF
    emPSF : np.array(Nx x Nx)
        with the emission PSF calculated from emPSF
    
    """

    if type(par) == np.ndarray:
        return par
    else:
        x_range = Nx * pxsizex
        
        #Entrance Field
        
        if par.field == 'Gaussian':
            entrance_field = lambda rho, phi, w0, f, k: np.exp( -(rho/w0)**2 )
        elif par.field == 'PlaneWave':
           entrance_field = lambda rho, phi, w0, f, k: 1
        
        #Phase Mask
        
        if par.mask is None:
            custom_mask = lambda rho, phi, w0, f, k: 1
        elif par.mask == 'VP':
            custom_mask = lambda rho, phi, w0, f, k: np.exp( 1j * phi )
        elif par.mask == 'Zernike':
            custom_mask = lambda rho, phi, w0, f, k: Zernike(par.abe_index, par.abe_ampli, par.h, rho, phi)
        elif callable(par.mask):
            custom_mask = par.mask
        
        #Total mask
        
        exMask = lambda rho, phi, w0, f, k: entrance_field(rho, phi, w0,f,k)* custom_mask(rho, phi,w0,f,k)
    
        #Calculation of entrance fields
        
        ex_lens, ey_lens = generate_incident_field(exMask, par.alpha, par.f, par.mask_sampl, par.mask_sampl, par.gamma, par.beta, par.w0, par.I0, par.wl/par.n)
    
        #Calculation of focus fields
    
        EX, EY, EZ = custom_mask_focus_field_XY(ex_lens, ey_lens, par.alpha, par.h, par.wl/par.n, z_shift, Nx, par.mask_sampl, par.mask_sampl, x_range, countdown=True, x0=0, y0=0)
    
        # Calculation of PSF intensity
    
        PSF = np.abs(EX)**2 + np.abs(EY)**2 + np.abs(EZ)**2
        
        if return_entrance_field == True:
            return PSF, [ex_lens, ey_lens]
        else:
            return PSF

def PSFs2D(exPar, emPar, pxsizex, Nx, z_shift = 0, return_entrance_field = False):
    """
    Simulate PSFs with PyFocus

    Parameters
    ----------
    exPar : simSettings()
        object with excitation PSF parameters
    emPar : simSettings()
        object with emission PSF parameters
    pxsizex : float
        Pixel size of the simulation space in XY [nm] (typically 1)
    Nx : int
        Number of pixels in XY dimensions in the simulation array, e.g. 1024
    z_shift : float
        Distance from the focal plane at which generate the PSF [nm] (optional)
    return_entrance_field: bool
        Returns the X and Y components of the field at the
        pupil plane in polar coordinates. They have to be
        converted into cartesian coordinate using the
        plot_in_cartesian function
    
    Returns
    -------
    exPSF : np.array(Nx x Nx)
        with the excitation PSF calculated from exPSF
    emPSF : np.array(Nx x Nx)
        with the emission PSF calculated from emPSF
    
    """
    
    if return_entrance_field == True:
        
        # Excitation PSF
        
        ex_PSF, ex_fields = singlePSF(exPar, pxsizex, Nx, z_shift = z_shift, return_entrance_field = True)
        
        # Emission PSF
        
        em_PSF, em_fields = singlePSF(emPar, pxsizex, Nx, z_shift = z_shift, return_entrance_field = True)
    
        return ex_PSF, em_PSF, ex_fields, em_fields
    
    else:
        
        # Excitation PSF
        
        ex_PSF = singlePSF(exPar, pxsizex, Nx, z_shift = z_shift)
        
        # Emission PSF
        
        em_PSF = singlePSF(emPar, pxsizex, Nx, z_shift = z_shift)
    
        return ex_PSF, em_PSF

def Pinholes(N, Nx, pxsizex, M, pxpitch, pxdim):
    """
    Simulate PSFs with PyFocus

    Parameters
    ----------
    N : int
        Number of detector elements in the array in each dimension (typically 5)
    Nx : int
        Number of pixels in each dimension in the simulation array (e.g. 1024)
    pxsize : float
        Pixel size of the simulation space [nm] (typically 1)
    M : float
        Total magnification of the optical system (typically 500)
    pxpitch : float
        Pixel pitch of the detector [nm] (real space, typically 75000)
    pxdim : float
        Detector element size [nm] (real space, typically 50000)
    
    Returns
    -------
    p : np.array(Nx x Nx x N**2)
        array with the pinholes of each detector element
    
    """
    
    p = np.zeros((Nx, Nx, N*N))
    center = Nx//2
    sizeDet = int(pxdim / M / pxsizex)
    if np.mod(sizeDet, 2) == 0:
        sizeDet -= 1 # let this be odd
    sizeDet = np.max((sizeDet, 1))
    stepDet = int(pxpitch / M / pxsizex)
    startcoord = int(np.ceil(center - np.floor(N/2) * stepDet - 0.5 * sizeDet))

    i = 0
    for dy in range(N):
        for dx in range(N):
            ymin = np.max((startcoord+dy*stepDet, 0))
            ymax = np.max((startcoord+dy*stepDet+sizeDet, 0))
            xmin = np.max((startcoord+dx*stepDet, 0))
            xmax = np.max((startcoord+dx*stepDet+sizeDet, 0))
            p[ymin:ymax, xmin:xmax, i] = 1
            i += 1    

    return p

def SPAD_PSF_2D(N, Nx, pxpitch, pxdim, pxsizex, M, exPar, emPar, stedPar = None, z_shift=0, spad = None, return_entrance_field = False):
    """
    Calculate PSFs for all pixels of the SPAD array by using FFTs

    Parameters
    ----------
    N : int
        Number of detector elements in the array in each dimension (typically 5)
    Nx : int
        Number of pixels in each dimension in the simulation array (e.g. 1024)
    pxpitch : float
        Pixel pitch of the detector [nm] (real space, typically 75000)
    pxdim : float
        Detector element size [nm] (real space, typically 50000)
    pxsize : float
        Pixel size of the simulation space [nm] (typically 1)
    M : float
        Total magnification of the optical system (typically 500)
    exPar : simSettings object
        object with excitation PSF parameters
    emPar : simSettings object
        object with emission PSF parameters
    stedPar : simSettings object
        object with STED beam parameters
    z_shift : float
        Distance from the focal plane at which generate the PSF [nm] (optional)
    spad : np.array( N**2 x Nx x Nx)
        Pinholes distribution . If none it is calculated using the input parameters
    return_entrance_field : bool
        Returns the X and Y components of the field at the
        pupil plane in polar coordinates. They have to be
        converted into cartesian coordinate using the
        plot_in_cartesian function
    
    Returns
    -------
    PSF : np.array(Nx x Nx x N**2)
        array with the overall PSFs for each detector element
    detPSF : np.array(Nx x Nx x N**2)
        array with the detection PSFs for each detector element
    exPSF : np.array(Nx x Nx)
        array with the excitation PSF
    
    """
    
    # Simulate ism psfs
    
    if return_entrance_field == True:
        exPSF, emPSF, ex_fields, em_fields = PSFs2D(exPar, emPar, pxsizex, Nx, z_shift = z_shift, return_entrance_field = True)
    else:
        exPSF, emPSF = PSFs2D(exPar, emPar, pxsizex, Nx, z_shift = z_shift)
    
    if spad is None:
        spad = Pinholes(N, Nx, pxsizex, M, pxpitch, pxdim)
    
    detPSF = np.empty( (Nx, Nx, N*N) )

    for i in range(N*N):
        detPSF[:,:,i] = sgn.convolve( emPSF, spad[:,:,i], mode ='same' )

    # Simulate donut
    
    if type(stedPar) == simSettings:
        stedPar.mask = 'VP'
        donut = singlePSF(stedPar, pxsizex, Nx, z_shift = z_shift)
        donut *= stedPar.sted_sat/np.max(donut)
        stedPSF = np.exp( - donut * stedPar.sted_pulse / stedPar.sted_tau )
        exPSF *= stedPSF
    
    # Calculate total PSF
    
    PSF = np.einsum('ijk, ij -> ijk', detPSF, exPSF)
    
    if return_entrance_field == True:
        return PSF, detPSF, exPSF, ex_fields, em_fields
    else:
        return PSF, detPSF, exPSF
    
def SPAD_PSF_3D(N, Nx, pxpitch, pxdim, pxsizex, M, exPar, emPar, Nz, pxsizez, stedPar = None, spad = None):
    """
    Calculate PSFs for all pixels of the SPAD array by using FFTs

    Parameters
    ----------
    N : int
        Number of detector elements in the array in each dimension (typically 5)
    Nx : int
        Number of pixels in each dimension in the simulation array (e.g. 1024)
    pxpitch : float
        Pixel pitch of the detector [nm] (real space, typically 75000)
    pxdim : float
        Detector element size [nm] (real space, typically 50000)
    pxsize : float
        Pixel size of the simulation space [nm] (typically 1)
    M : float
        Total magnification of the optical system (typically 500)
    exPar : simSettings object
        object with excitation PSF parameters
    emPar : simSettings object
        object with emission PSF parameters
    Nz : int
        number of axial planes (typically an odd integer)
        the planes are symmetrically calculated around the focal plane (z = 0)
    pxisez : float
        distance between axial planes [nm]
    stedPar : simSettings object
        object with STED beam parameters
    spad : np.array( N**2 x Nx x Nx)
        Pinholes distribution . If none it is calculated using the input parameters
    
    Returns
    -------
    PSF : np.array(Nz x Nx x Nx x N**2)
        array with the overall PSFs for each detector element
    detPSF : np.array(Nz x Nx x Nx x N**2)
        array with the detection PSFs for each detector element
    exPSF : np.array(Nz x Nx x Nx)
        array with the excitation PSF
    
    """
    
    zeta = ( np.arange(Nz) - Nz//2 ) * pxsizez
    
    if spad is None:
        spad = Pinholes(N, Nx, pxsizex, M, pxpitch, pxdim)
    
    PSF = np.empty( (Nz, Nx, Nx, N*N) )
    detPSF = np.empty( (Nz, Nx, Nx, N*N) )
    exPSF = np.empty( (Nz, Nx, Nx) )
    
    for i, z in enumerate(zeta):
        print( f'Calculating the PSFs at z = {z} nm')
        PSF[i, :, :, :], detPSF[i, :, :, :], exPSF[i, :, :] = SPAD_PSF_2D(N, Nx, pxpitch, pxdim, pxsizex, M, exPar, emPar, stedPar = stedPar, z_shift = z, spad = spad)
        
    return PSF, detPSF, exPSF

def Fingerprint(dset, volumetric = False):
    """
    Calculate the fingerprint of an ISM dataset.
    The last dimension has to be the spad array channel.

    Parameters
    ----------
    dset : np.array(Nz x Nx x Nx x ... x N*N)
        ISM dataset
    volumetric : bool
        if true, a fingerprint is returned for each axial plane
    
    Returns
    -------
    Fingerprint : np.array(Nz x N x N)
        Finger print
    
    """
    
    N = int( np.sqrt(dset.shape[-1]) )
    
    if volumetric == True:
        Nz = dset.shape[0]
        f = np.empty( (Nz, N*N) )
        axis = tuple( range(1, dset.ndim - 1) )
        f = np.sum( dset, axis = axis )
        f = f.reshape(Nz, N, N)
    else:
        axis = tuple( range(dset.ndim - 1) )
        f = np.sum( dset, axis = axis )
        f = f.reshape(N,N)
    return f