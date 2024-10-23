import numpy as np
from scipy.signal import convolve
import copy as cp


class ImageSimulator:
    """
    Object with methods to generate the forward model of a (multichannel) microscope.
    The number of dimensions of phantom and the psf should differ by 1 at most.
    In this case, the last dimension of the psf is interpreted as the channel.

    Attributes
    ----------
    image : np.ndarray
        image convolved with the psf and corrupted by shot noise
    clean_image : np.ndarray
        image convolved with the psf without noise
    phantom : np.ndarray
        stucture of the specimen
    psf : np.ndarray
        point spread function. The last dimension is the channel.
    signal : float
        brightness of the sample (units: photon counts)

    Methods
    -------
    blur :
        Generates the clean image.
    poisson_noise :
        Corrupts the clean image with shot noise.
    forward :
        Generates the blurred and noisy image.

    """
    def __init__(self, phantom=None, psf=None, signal=1, z_projection=False):
        self.image = None
        self.clean_image = None
        self.phantom = phantom
        self.psf = psf
        self.signal = np.array(signal)
        self.z_projection = z_projection

    def blur(self):
        gt = self.ground_truth
        num_ch = np.ndim(self.psf) - np.ndim(self.phantom)
        sz = self.psf.shape

        self.clean_image = np.empty(gt.shape + (self.psf.shape[-1],))

        if num_ch == 1:
            if np.ndim(self.phantom) == 3 and self.z_projection is True:
                for z in range(sz[0]):
                    for c in range(sz[-1]):
                        self.clean_image[z, ..., c] = convolve(gt[z], self.psf[z, ..., c], mode='same')
                self.clean_image = self.clean_image.sum(0)
            else:
                for c in range(sz[-1]):
                    self.clean_image[..., c] = convolve(gt, self.psf[..., c], mode='same')
        elif num_ch == 0:
            if np.ndim(self.phantom) == 3 and self.z_projection is True:
                for z in range(sz[0]):
                    self.clean_image = convolve(gt[z], self.psf[z], mode='same')
                self.clean_image = self.clean_image.sum(0)
            else:
                self.clean_image = convolve(gt, self.psf, mode='same')
        else:
            raise Exception("The PSF has less dimensions than the phantom.")
            
        self.clean_image[self.clean_image < 0] = 0

    def poisson_noise(self):
        self.image = np.random.poisson(self.clean_image)

    def forward(self):
        self.blur()
        self.poisson_noise()

    @property
    def ground_truth(self):
        if np.ndim(self.signal) == 1:
            return np.einsum('i..., i -> i...', self.phantom, self.signal)
        elif np.ndim(self.signal) == 0:
            return self.phantom * self.signal
        else:
            raise Exception("The signal should be a scalar or a 1D array.")

    def copy(self):
        return cp.copy(self)
