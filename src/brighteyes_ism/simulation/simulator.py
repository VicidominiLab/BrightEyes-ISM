import numpy as np
from scipy.signal import convolve
import copy as cp


class ImageSimulator:

    def __init__(self, phantom=None, psf=None, signal=1):
        self.image = None
        self.phantom = phantom
        self.psf = psf
        self.signal = signal

    def blur(self):
        gt = self.ground_truth
        num_ch = np.ndim(self.psf) - np.ndim(self.phantom)
        sz = self.psf.shape

        self.image = np.empty_like(self.psf)

        if num_ch == 1:
            for c in range(sz[-1]):
                self.image = convolve(self.psf[..., c], gt)
        elif num_ch == 0:
            self.image = convolve(self.psf, gt)
        else:
            raise Exception("The PSF has less dimensions than the phantom.")

    def poisson_noise(self):
        self.image = np.random.poisson(self.image)

    def forward(self):
        self.blur()
        self.poisson_noise()

    @property
    def ground_truth(self):
        return self.phantom * self.signal

    def copy(self):
        return cp.copy(self)