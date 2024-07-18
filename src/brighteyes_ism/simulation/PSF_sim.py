import numpy as np
import scipy.signal as sgn
from skimage.transform import rotate
from PyFocus.custom_mask_functions import generate_incident_field, custom_mask_focus_field_XY, plot_in_cartesian
from poppy.zernike import noll_indices, R, zern_name
import copy as cp
from tqdm import tqdm

from .detector import custom_detector

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

class GridParameters:
    """
    It calculates a z-stack of PSFs for all the elements of the SPAD array detector.

    Attributes
    ----------
    N : int
        Number of detector elements in the array in each dimension (typically 5)
    Nx : int
        Number of pixels in each dimension in the simulation array (e.g. 1024)
    pxpitch : float
        Pixel pitch of the detector [nm] (real space, typically 75000)
    pxdim : float
        Detector element size [nm] (real space, typically 50000)
    pxsizex : float
        Pixel size of the simulation space [nm] (typically 1)
    M : float
        Total magnification of the optical system (typically 500)
    Nz : int
        number of axial planes (typically an odd integer)
    """

    # __slots__ = ['pxsizex', 'pxsizez', 'Nx', 'Ny', 'Nz', 'pxpitch', 'pxdim', 'N', 'M']
    
    def __init__(self, pxsizex=40, pxsizez=50, Nx = 100, Ny = 100, Nz = 1, pxpitch = 75e3, pxdim = 50e3, N = 5, M = 450):
        self.pxsizex = pxsizex  # nm - lateral pixel size of the images
        self.pxsizez = pxsizez  # nm - distance of the axial planes
        self.Nx = Nx  # number of samples along the X and Y axis
        self.Nz = Nz  # number of axial planes
        self.pxpitch = pxpitch  # nm - spad array pixel pitch (real space)
        self.pxdim = pxdim  # nm - spad pixel size (real space) 57.3e-3 for cooled spad
        self.pinhole_shape = 'square'  # 'square', 'cirle', or 'hexagon'
        self.geometry = 'rect'  # Pinholes arrangement: 'rect' or 'hex'
        self.N = N  # number of pixels in the detector in each dimension (5x5 typically)
        self.M = M  # overall magnification of the system
        self.rotation = 0  # rototion angle of the detector array (rad)
        self.mirroring = 1  # flip of the x_d axis (+/- 1)
        self.name = None

    @property
    def rangex(self):
        return self.Nx * self.pxsizex

    @property
    def rangez(self):
        return self.Nz * self.pxsizez

    @property
    def Nch(self):
        if np.ndim(self.N) == 0:
            Ntot = self.N**2
        elif np.ndim(self.N) == 1:
            Ntot = self.N[1]*self.N[0]
        else:
            raise ValueError('N has to a be a single or a couple of positive integers.')
        return Ntot

    def spad_size(self, mode: str = 'magnified', simPar = None):
        size = (self.pxpitch * (self.N - 1) + self.pxdim)
        if simPar is not None:
            return size / self.M / simPar.airy_unit
        elif mode == 'magnified':
            return size / self.M
        elif mode == 'real':
            return size

    def copy(self):
        return cp.copy(self)

    def Print(self):
        dic = self.__dict__
        names = list(dic)
        values = list(dic.values())
        for n, name in enumerate(names):
            print(name, end = '')
            print(' ' * int(14 - len(name)), end = '')
            print("" if values[n] is None else f'{values[n]:.2f}')

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
    
    # __slots__ = ['na', 'n', 'wl', 'h', 'gamma', 'beta', 'w0', 'I0', 'field', 'mask', 'mask_sampl', 'sted_sat', 'sted_pulse', 'sted_tau', 'abe_index', 'abe_ampli']

    def __init__(self, na=1.4, n=1.5, wl=485.0, h=2.8, gamma=45.0, beta=90.0,
                 w0=100.0, I0=1, field='PlaneWave', mask=None, mask_sampl=200,
                 sted_sat=50, sted_pulse=1, sted_tau=3.5,
                 abe_index=None, abe_ampli=None):

        self.na = na  # numerical aperture
        self.n = n  # sample refractive index
        self.wl = wl  # wavelength [nm]
        self.h = h  # radius of aperture of the objective lens [mm]
        self.w0 = w0  # radius of the incident gaussian beam [mm]
        self.gamma = gamma  # parameter describing the light polarization (amplitude)
        self.beta = beta  # parameter describing the light polarization (phase)
        self.I0 = I0  # Intensity of the entrance field
        self.field = field  # entrance field at the pupil plane
        #   'PlaneWave' = Flat field
        #   'Gaussian' = Gaussian beam with waist w0
        self.mask = mask  # phase mask
        #   None = no mask
        #   'VP' = vortex phase plate
        #   'Zernike' = zernike polynomials
        self.mask_sampl = mask_sampl  # phase mask sampling
        self.sted_sat = sted_sat  # STED maximum saturation factor
        self.sted_pulse = sted_pulse  # STED pulse duration [ns]
        self.sted_tau = sted_tau  # fluorescence lifetime [ns]
        self.abe_index = abe_index  # aberration index (int or array)
        self.abe_ampli = abe_ampli  # aberration amplitude in rad (float or array)
        
    @property
    def f(self):    # focal length of the objective lens [mm]
        return self.h*self.n/self.na

    @property
    def alpha(self):    # semiangular aperture of the objective [rad]
        return np.arcsin(self.na/self.n)

    @property
    def airy_unit(self):
        au = 1.22 * self.wl / self.na
        return au

    @property
    def depth_of_field(self):
        dof = 2 * self.n * self.wl / (self.na ** 2)
        return dof

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

    def Print(self):
        dic = self.__dict__
        names = list(dic)
        values = list(dic.values())
        for n, name in enumerate(names):
            print(name, end = '')
            print(' ' * int(14 - len(name)), end = '')
            print(str(values[n]))


def singlePSF(par, pxsizex, Nx, z_shift = 0, return_entrance_field = False, verbose = True):
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
    
        EX, EY, EZ = custom_mask_focus_field_XY(ex_lens, ey_lens, par.alpha, par.h, par.wl/par.n, z_shift, Nx, par.mask_sampl, par.mask_sampl, x_range, countdown=verbose, x0=0, y0=0)
    
        # Calculation of PSF intensity
    
        PSF = np.abs(EX)**2 + np.abs(EY)**2 + np.abs(EZ)**2
        
        if return_entrance_field == True:
            return PSF, [ex_lens, ey_lens]
        else:
            return PSF


def PSFs2D(exPar, emPar, pxsizex, Nx, z_shift = 0, return_entrance_field = False, verbose = True):
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
        
        ex_PSF, ex_fields = singlePSF(exPar, pxsizex, Nx, z_shift = z_shift, return_entrance_field = True, verbose = verbose)
        
        # Emission PSF
        
        em_PSF, em_fields = singlePSF(emPar, pxsizex, Nx, z_shift = z_shift, return_entrance_field = True, verbose = verbose)
    
        return ex_PSF, em_PSF, ex_fields, em_fields
    
    else:
        
        # Excitation PSF
        
        ex_PSF = singlePSF(exPar, pxsizex, Nx, z_shift = z_shift, verbose = verbose)
        
        # Emission PSF
        
        em_PSF = singlePSF(emPar, pxsizex, Nx, z_shift = z_shift, verbose = verbose)
    
        return ex_PSF, em_PSF


def SPAD_PSF_2D(gridPar, exPar, emPar, n_photon_excitation = 1, stedPar = None, z_shift = 0, spad = None,
                return_entrance_field = False, normalize = True, verbose = True):
    """
    Calculate PSFs for all pixels of the SPAD array by using FFTs

    Parameters
    ----------
    gridPar : GridParameters object
        object with simulation space parameters
    exPar : simSettings object
        object with excitation PSF parameters
    emPar : simSettings object
        object with emission PSF parameters
    rotParam : np.ndarray
        array with the mirror and rotation angle to apply to the detection PSFs.
        The default is None.
    n_photon_excitation : int
        Order of non-linear excitation. Default is 1.
    stedPar : simSettings object
        object with STED beam parameters
    z_shift : float
        Distance from the focal plane at which generate the PSF [nm]
    spad : np.array( N**2 x Nx x Nx)
        Pinholes distribution . If none it is calculated using the input parameters
    return_entrance_field : bool
        Returns the X and Y components of the field at the
        pupil plane in polar coordinates. They have to be
        converted into cartesian coordinate using the
        plot_in_cartesian function
    normalize : bool
        If True, all the returned PSFs are divided by the total flux.
        Default is True.
    
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
        exPSF, emPSF, ex_fields, em_fields = PSFs2D(exPar, emPar, gridPar.pxsizex, gridPar.Nx, z_shift = z_shift, return_entrance_field = True, verbose = verbose)
    else:
        exPSF, emPSF = PSFs2D(exPar, emPar, gridPar.pxsizex, gridPar.Nx, z_shift = z_shift, verbose = verbose)
    
    if spad is None:
        spad = custom_detector(gridPar)
    Nch = spad.shape[-1]

    detPSF = np.empty( (gridPar.Nx, gridPar.Nx, Nch) )

    for i in range(Nch):
        detPSF[:,:,i] = sgn.convolve( emPSF, spad[:,:,i], mode ='same' )

    # Apply non-linearity to excitation

    if n_photon_excitation > 1:
        exPSF = exPSF ** n_photon_excitation

    # Simulate donut
    
    if type(stedPar) == simSettings:
        stedPar.mask = 'VP'
        donut = singlePSF(stedPar, gridPar.pxsizex, gridPar.Nx, z_shift = z_shift, verbose = verbose)
        donut *= stedPar.sted_sat/np.max(donut)
        stedPSF = np.exp( - donut * stedPar.sted_pulse / stedPar.sted_tau )
        exPSF *= stedPSF

    # Rotate and mirror detPSF

    detPSFrot = detPSF.copy()

    if gridPar.mirroring == -1:
        if np.ndim(gridPar.N) == 0:
            nx = gridPar.N
            ny = gridPar.N
        else:
            nx = gridPar.N[1]
            ny = gridPar.N[0]

        detPSFrot = detPSFrot.reshape(gridPar.Nx, gridPar.Nx, nx, ny)
        detPSFrot = np.flip(detPSFrot, axis=-1)
        detPSFrot = detPSFrot.reshape(gridPar.Nx, gridPar.Nx, Nch)

    if gridPar.rotation != 0:
        theta = np.rad2deg(gridPar.rotation)
        detPSFrot = rotate(detPSFrot, theta, resize=False, center=None, order=None, mode='constant', cval=0,
                       clip=True, preserve_range=False)

    # Calculate total PSF
    
    PSF = np.einsum('ijk, ij -> ijk', detPSFrot, exPSF)

    if normalize == True:
        PSF /= PSF.sum()
        detPSFrot /= detPSFrot.sum()
        exPSF /= exPSF.sum()

    if return_entrance_field == True:
        return PSF, detPSFrot, exPSF, ex_fields, em_fields
    else:
        return PSF, detPSFrot, exPSF
    
def SPAD_PSF_3D(gridPar, exPar, emPar, n_photon_excitation = 1, stedPar = None, spad = None, stack: str = 'symmetrical',
                normalize = True, verbose = True):
    """
    It calculates a z-stack of PSFs for all the elements of the SPAD array detector.

    Parameters
    ----------
    gridPar : GridParameters object
        object with simulation space parameters
    exPar : simSettings object
        object with excitation PSF parameters
    emPar : simSettings object
        object with emission PSF parameters
    rotParam : np.ndarray
        array with the mirror and rotation angle to apply to the detection PSFs.
        The default is None.
    n_photon_excitation : int
        Order of non-linear excitation. Default is 1.
    stedPar : simSettings object
        object with STED beam parameters
    spad : np.array( N**2 x Nx x Nx)
        Pinholes distribution . If none it is calculated using the input parameters
    stack : str
        String that defines the direction along z of the simulation.
        If "symmetrical", the stack is generated at planes around z = 0 both on the negative and positive directions.
        Other possible entries are "positive", and "negative".
        Default: "symmetrical".
    normalize : bool
        If True, the returned PSFs are divided by the total flux calculated on the focal plane (z=0).
        Default is True.
    Returns
    -------
    PSF : np.array(Nz x Nx x Nx x N**2)
        array with the overall PSFs for each detector element
    detPSF : np.array(Nz x Nx x Nx x N**2)
        array with the detection PSFs for each detector element
    exPSF : np.array(Nz x Nx x Nx)
        array with the excitation PSF
    
    """

    if stack == "symmetrical":
        zeta = (np.arange(gridPar.Nz) - gridPar.Nz//2) * gridPar.pxsizez
    elif stack == "positive":
        zeta = np.arange(gridPar.Nz) * gridPar.pxsizez
    elif stack == "negative":
        zeta = -np.arange(gridPar.Nz) * gridPar.pxsizez

    if spad is None:
        spad = custom_detector(gridPar)
    Nch = spad.shape[-1]

    PSF = np.empty( (gridPar.Nz, gridPar.Nx, gridPar.Nx, Nch) )
    detPSF = np.empty( (gridPar.Nz, gridPar.Nx, gridPar.Nx, Nch) )
    exPSF = np.empty( (gridPar.Nz, gridPar.Nx, gridPar.Nx) )

    if verbose == True:
        print(f'Calculating the PSFs stack from z = {zeta[0]} nm to z = {zeta[-1]} nm:')
        zeta_range = tqdm(zeta)
    else:
        zeta_range = zeta

    for i, z in enumerate( zeta_range ):
        PSF[i, :, :, :], detPSF[i, :, :, :], exPSF[i, :, :] = SPAD_PSF_2D(gridPar, exPar, emPar, n_photon_excitation = n_photon_excitation, stedPar = stedPar, z_shift = z, spad = spad, normalize = False, verbose = False)

    if normalize == True:
        idx = np.argwhere(zeta == 0).item()
        focal_flux = PSF[idx, :, :, :].sum()
        for i, z in enumerate(zeta):
            PSF[i, :, :, :] /= focal_flux

    return PSF, detPSF, exPSF