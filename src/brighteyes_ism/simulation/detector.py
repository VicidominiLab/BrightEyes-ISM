import numpy as np
from torchvision.transforms.functional import rotate
from skimage.draw import polygon
import torch

from .utils import partial_convolution


def circle(n, radius):
    """
    It calculates a single binary mask with the shape of a circle.

    Parameters
    ----------
    n : int
        number of pixels of the simulation space, assumed squared.
    radius : float
      Radius of the circle in pixel units.

    Returns
    -------
    circle_pinhole : np.ndarray (n x n)
        Image of the binary mask.
    """

    x = np.arange(-(n // 2), n // 2 + np.mod(n,2))

    xx, yy = np.meshgrid(x, x)
    r = np.sqrt(xx ** 2 + yy ** 2)

    circle_pinhole = np.where(r < radius, 1, 0)

    return circle_pinhole


def square(n, length):
    """
    It calculates a single binary mask with the shape of a square.

    Parameters
    ----------
    n : int
        number of pixels of the simulation space, assumed squared.
    length : float
      Length of the square in pixel units.

    Returns
    -------
    square_pinhole : np.ndarray (n x n)
        Image of the binary mask.
    """

    x = np.arange(-(n // 2), n // 2 + np.mod(n,2))

    xx, yy = np.meshgrid(x, x)

    cond = np.logical_and(np.abs(xx) < length / 2, np.abs(yy) < length / 2)

    square_pinhole = np.where(cond, 1, 0)

    return square_pinhole


def hexagon(n, radius):
    """
    It calculates a single binary mask with the shape of a regular hexagon.
    The radius is the distance from the center to a vertex.

    Parameters
    ----------
    n : int
        number of pixels of the simulation space, assumed squared.
    radius : float
      Distance from the center to a vertex of the hexagon in pixel units.

    Returns
    -------
    hex_pinhole : np.ndarray (n x n)
        Image of the binary mask.
    """

    x = np.array([-1, 0, 1, 1, 0, -1]) * radius * np.cos(np.deg2rad(30)) + n // 2
    y = np.array([1, 2, 1, -1, -2, -1]) * radius * np.sin(np.deg2rad(30)) + n // 2

    hex_pinhole = np.zeros((n, n))

    idx = polygon(x, y, shape=(n, n))
    hex_pinhole[idx] = 1

    return hex_pinhole.T


def rect_grid(n, x):
    """
    It calculates the normalized coordinates of a rectangular grid.

    Parameters
    ----------
    n : int
        number of pixels of the simulation space, assumed squared.
    x : np.ndarray (N)
      array with normalized coordinates along a single axis.

    Returns
    -------
    s : np.ndarray (2 x N**2)
        Array with rectangular grid coordinates.
    """

    s = np.empty((n, n, 2))

    for n_i, i in enumerate(x):
        for n_j, j in enumerate(x):
            s[n_i, n_j] = np.array([i, j])

    s = s.reshape(-1, 2).T

    return s


def hex_grid(n, x):
    """
    It calculates the normalized coordinates of a hexagonal grid.

    Parameters
    ----------
    n : int
        number of pixels of the simulation space, assumed squared.
    x : np.ndarray (N)
      array with normalized coordinates along a single axis.

    Returns
    -------
    s : np.ndarray (2 x N**2)
        Array with hexagonal grid coordinates.
    """

    s = np.empty((n, n, 2))

    for n_i, i in enumerate(x):
        for n_j, j in enumerate(x):
            s[n_i, n_j] = np.array([0.5 * np.sqrt(3) * i, j - 0.5 * (i % 2)])

    s = s.reshape(-1, 2).T

    condition = np.abs(s[1]) <= (n // 2)
    idx = np.argwhere(condition).flatten()
    s = s[:, idx]

    return s


def det_coords(n, geometry):
    """
    It calculates the coordinates of the pinhole centers from a given geometry.

    Parameters
    ----------
    n : int
        number of pixels of the simulation space, assumed squared.
    geometry : str
        Detector geometry. Valid choices are 'rect' or 'hex'.

    Returns
    -------
    s : np.ndarray (2 x Nch)
        Array with coordinates of the pinhole centers.
    """

    x = np.arange(-(n // 2), n // 2 + 1)

    if geometry == 'rect':
        s = rect_grid(n, x)
    elif geometry == 'hex':
        s = hex_grid(n, x)
    else:
        raise Exception("Detector geometry not valid. Select 'rect' or 'hex'.")

    return s


def pinhole_array(s, nx, mag, pxsize, pxdim, pinhole_shape, device):
    """
    It calculates Nx x Nx x Nch array of binary masks, describing a detector array.

    Parameters
    ----------
    s : np.ndarray (2 x Nch)
        Array with coordinates of the pinhole centers.
    nx : int
        number of pixels of the simulation space, assumed squared.
    mag : float
        Magnification of the microscope.
    pxsize : float
        Size of the scan pixel.
    pxdim : float
        Diameter of the individual pinhole of the detector array.
    pinhole_shape : str
        Shape of the invidual pinhole. Valid choices are 'square', 'cirle', or 'hexagon'.

    Returns
    -------
    detector : np.ndarray (Nx x Nx x Nch)
        Numpy array of binary masks. Each channel is a pinhole in a different position.
    """

    radius = pxdim / mag / pxsize / 2
    nch = s.shape[1]

    img = torch.zeros((nx, nx, nch)).to(device)
    c = s + nx // 2

    for k in range(s.shape[1]):
        img[c[0, k], c[1, k], k] = 1

    if pinhole_shape == 'square':
        pinhole = square(nx, 2 * radius)
    elif pinhole_shape == 'circle':
        pinhole = circle(nx, radius)
    elif pinhole_shape == 'hexagon':
        pinhole = hexagon(nx, radius)
    else:
        raise Exception("Pinhole shape not valid. Select 'square', 'cirle', or 'hexagon'.")

    pinhole = torch.from_numpy(pinhole).to(device)
    print(pinhole.shape)
    print(img.shape)
    detector = partial_convolution(img, pinhole, 'ijk', 'ij', 'ij')

    return detector


def transform_detector(gridPar, detector):

    Nch = detector.shape[-1]

    spad_rot = detector.clone()

    if gridPar.mirroring == -1:
        if np.ndim(gridPar.N) == 0:
            nx = ny = gridPar.N
        else:
            nx, ny = gridPar.N

        spad_rot = spad_rot.reshape(gridPar.Nx, gridPar.Nx, nx, ny)
        spad_rot = torch.flip(spad_rot, axis=-1)
        spad_rot = spad_rot.reshape(gridPar.Nx, gridPar.Nx, Nch)

    if gridPar.rotation != 0:
        theta = np.rad2deg(gridPar.rotation)
        spad_rot = torch.movedim(spad_rot, -1, 0)
        spad_rot = rotate(spad_rot, theta)
        spad_rot = torch.movedim(spad_rot, 0, -1)

    spad_rot[spad_rot < 1e-2] = 0

    return spad_rot


def custom_detector(grid, device):
    """
    It calculates Nx x Nx x Nch array of binary masks, describing a detector array.

    Parameters
    ----------
    grid : GridParameters
        Object with array detector and simulation space parameters
    Returns
    -------
    detector : np.ndarray (Nx x Nx x Nch)
        Numpy array of binary masks. Each channel is a pinhole in a different position.
    """

    if grid.name == 'airyscan':
        s = det_coords(7, 'hex')
        s = hex_to_airy(s)
    else:
        s = det_coords(grid.N, grid.geometry)

    s *= grid.pxpitch

    s = np.round(s / grid.M / grid.pxsizex).astype('int')

    detector = pinhole_array(s, grid.Nx, grid.M, grid.pxsizex, grid.pxdim, grid.pinhole_shape, device)

    detector = transform_detector(grid, detector)

    return detector


def hex_to_airy(s):
    idx_array = [22, 28, 29, 23, 16, 15, 21, 27, 34, 35, 36, 30, 24, 17, 10, 9, 8, 14, 20, 26, 33, 41, 42, 37, 31, 18,
                 11, 3, 2, 7, 13, 19]

    return s[..., idx_array]


def airy_to_hex(s):
    ss = np.ones((s.shape[:-1] + (45,))) * np.nan

    idx1 = [2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 33, 34, 35,
            36, 37, 41, 42]
    idx2 = [28, 27, 29, 16, 15, 14, 26, 30, 17, 5, 4, 13, 25, 31, 18, 6, 0, 3, 12, 19, 7, 1, 2, 11, 24, 20, 8, 9, 10,
            23, 21, 22]

    ss[..., idx1] = s[..., idx2]

    return ss
