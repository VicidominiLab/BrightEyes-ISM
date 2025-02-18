import numpy as np
import scipy.signal as sgn
from skimage.transform import rotate

from psf_generator.propagators import VectorialCartesianPropagator
from poppy.zernike import zern_name

import copy as cp
from tqdm import tqdm

import torch

from numbers import Number

from .detector import custom_detector


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
    Nz : int
        number of axial planes (typically an odd integer)
    pinhole_shape : str
        Shape of the invidual pinhole. Valid choices are 'square', 'cirle', or 'hexagon'.
    geometry : str
        Detector geometry. Valid choices are 'rect' or 'hex'.
    name : str
        If 'airyscan', the simulated detector is the commercial 32-elements AiryScan from Zeiss.
    M : float
        Total magnification of the optical system (typically 500)
    rotation : float
        Detector rotation angle (rad)
    mirroring: int
        Flip of the horizonatal axis of the detector plane (+1 or - 1)
    """

    # __slots__ = ['pxsizex', 'pxsizez', 'Nx', 'Nz', 'pxpitch', 'pxdim', 'N', 'M']

    def __init__(self, pxsizex=40, pxsizez=50, Nx=100, Nz=1, pxpitch=75e3, pxdim=50e3, N=5, M=450):
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
        self.name = None  # None or 'airyscan'

    @property
    def rangex(self):
        return self.Nx * self.pxsizex

    @property
    def rangez(self):
        return self.Nz * self.pxsizez

    @property
    def Nch(self):
        if np.ndim(self.N) == 0:
            if self.geometry == 'rect':
                Ntot = self.N ** 2
            elif self.geometry == 'hex':
                Ntot = self.N ** 2 - self.N // 2 - (1 + (-1) ** ((self.N + 1) / 2)) * 0.5
        elif np.ndim(self.N) == 1:
            Ntot = self.N[1] * self.N[0]
        else:
            raise ValueError('N has to a be a single or a couple of positive integers.')
        return Ntot

    def spad_size(self, mode: str = 'magnified', simPar=None):
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
            print(name, end='')
            print(' ' * int(14 - len(name)), end='')
            if values[n] is None:
                print("")
            elif isinstance(values[n], Number):
                print(f'{values[n]:.2f}')
            else:
                print(values[n])


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
    def f(self):  # focal length of the objective lens [mm]
        return self.h * self.n / self.na

    @property
    def alpha(self):  # semiangular aperture of the objective [rad]
        return np.arcsin(self.na / self.n)

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

        if np.isscalar(self.abe_index):
            index = [self.abe_index]
        else:
            index = self.abe_index

        names = []

        for i in range(len(index)):
            names.append(zern_name(index[i]))

        return names

    def copy(self):
        return cp.copy(self)

    def Print(self):
        dic = self.__dict__
        names = list(dic)
        values = list(dic.values())
        for n, name in enumerate(names):
            print(name, end='')
            print(' ' * int(14 - len(name)), end='')
            print(str(values[n]))


def singlePSF(par, pxsizex, Nx, rangez, nz):
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
    
    Returns
    -------
    exPSF : np.array(Nx x Nx)
        with the excitation PSF calculated from exPSF
    emPSF : np.array(Nx x Nx)
        with the emission PSF calculated from emPSF
    
    """

    kwargs = {
        'apod_factor': True,
        'defocus_min': rangez[0],
        'defocus_max': rangez[1],
        'n_defocus': nz,
        'n_pix_psf': Nx,
        'fov': Nx * pxsizex,
        'n_pix_pupil': par.mask_sampl,
        'na': par.na,
        'n_i': par.n,
        'wavelength': par.wl / par.n,
        'e0x': np.cos(np.deg2rad(par.gamma)),
        'e0y': np.sin(np.deg2rad(par.gamma)) * np.exp(1j * np.deg2rad(par.beta))
    }

    # Amplitude envelope

    if par.field == 'Gaussian':
        kwargs.update({
            'envelope': par.wo
        })

    # Phase Mask

    if par.mask == 'VP':
        kwargs.update({
            'special_phase_mask': 'vortex'
        })

    if par.abe_index is not None:
        zernike = np.zeros(np.max(par.abe_index) + 1)
        zernike[par.abe_index] = par.abe_ampli
        kwargs.update({
            'zernike_coefficients': zernike
        })

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    propagator = VectorialCartesianPropagator(**kwargs, device=device)

    fields = propagator.compute_focus_field().cpu().detach().numpy()

    psf = np.sum(np.abs(fields) ** 2, 1)

    psf = psf * par.I0 / np.sum(psf)

    return psf, fields


def SPAD_PSF_3D(gridPar, exPar, emPar, stedPar=None, spad=None, n_photon_excitation: int = 1, stack: str = 'symmetrical',
                normalize: bool = True):
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
        zeta = (np.arange(gridPar.Nz) - gridPar.Nz // 2) * gridPar.pxsizez
    elif stack == "positive":
        zeta = np.arange(gridPar.Nz) * gridPar.pxsizez
    elif stack == "negative":
        zeta = -np.arange(gridPar.Nz) * gridPar.pxsizez
    else:
        zeta = stack

    # simulate detector array

    if spad is None:
        spad = custom_detector(gridPar)
    Nch = spad.shape[-1]

    # Simulate ism psfs

    exPSF, _ = singlePSF(exPar, gridPar.pxsizex, gridPar.Nx, [zeta[0], zeta[-1]], gridPar.Nz)
    emPSF, _ = singlePSF(emPar, gridPar.pxsizex, gridPar.Nx, [zeta[0], zeta[-1]], gridPar.Nz)

    detPSF = np.empty((gridPar.Nz, gridPar.Nx, gridPar.Nx, Nch))


    for z in range(gridPar.Nz):
        for i in range(Nch):
            detPSF[z, :, :, i] = sgn.convolve(emPSF[z], spad[:, :, i], mode='same')

    # Apply non-linearity to excitation

    if n_photon_excitation > 1:
        exPSF = exPSF ** n_photon_excitation

    # Simulate donut

    if type(stedPar) == simSettings:
        stedPar.mask = 'VP'
        donut = singlePSF(stedPar, gridPar.pxsizex, gridPar.Nx, [zeta[0], zeta[-1]], gridPar.Nz)
        donut *= stedPar.sted_sat / np.max(donut)
        stedPSF = np.exp(-donut * stedPar.sted_pulse / stedPar.sted_tau)
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

        detPSFrot = detPSFrot.reshape(gridPar.Nz, gridPar.Nx, gridPar.Nx, nx, ny)
        detPSFrot = np.flip(detPSFrot, axis=-1)
        detPSFrot = detPSFrot.reshape(gridPar.Nz, gridPar.Nx, gridPar.Nx, Nch)

    if gridPar.rotation != 0:
        theta = np.rad2deg(gridPar.rotation)
        for z in range(gridPar.Nz):
            detPSFrot[z] = rotate(detPSFrot[z], theta, resize=False, center=None, order=None, mode='constant', cval=0,
                                  clip=True, preserve_range=False)

    # Calculate total PSF

    PSF = np.einsum('zxyc, zxy -> zxyc', detPSFrot, exPSF)

    if normalize is True:
        idx = np.argwhere(zeta == 0).item()
        focal_flux = PSF[idx, :, :, :].sum()
        for i, z in enumerate(zeta):
            PSF[i, :, :, :] /= focal_flux

    return PSF, detPSF, exPSF


def SPAD_PSF_2D(gridPar, exPar, emPar, n_photon_excitation=1, stedPar=None, z_shift=0, spad=None, normalize=True):
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
    n_photon_excitation : int
        Order of non-linear excitation. Default is 1.
    stedPar : simSettings object
        object with STED beam parameters
    z_shift : float
        Distance from the focal plane at which generate the PSF [nm]
    spad : np.array( N**2 x Nx x Nx)
        Pinholes distribution . If none it is calculated using the input parameters
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

    # simulate detector array

    grid = gridPar.copy()
    grid.Nz = 1
    stack = [z_shift, z_shift]

    PSF, detPSFrot, exPSF = SPAD_PSF_3D(grid, exPar, emPar, stedPar, spad, n_photon_excitation, stack, False)

    PSF = np.squeeze(PSF)
    detPSFrot = np.squeeze(detPSFrot)
    exPSF = np.squeeze(exPSF)

    if normalize is True:
        PSF /= PSF.sum()
        detPSFrot /= detPSFrot.sum()
        exPSF /= exPSF.sum()

    return PSF, detPSFrot, exPSF