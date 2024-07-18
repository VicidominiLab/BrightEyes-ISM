import numpy as np
from scipy.signal import convolve
from skimage.draw import polygon


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

    x = np.arange(-(n // 2), n // 2 + 1)

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

    x = np.arange(-(n // 2), n // 2 + 1)

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

    return hex_pinhole


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
            s[n_i, n_j] = np.array([i - 0.5 * (j % 2), 0.5 * np.sqrt(3) * j])

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

    s = s.reshape(-1, 2).T

    return s


def pinhole_array(s, nx, mag, pxsize, pxdim, pinhole_shape):
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

    img = np.zeros((nx, nx, nch))
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

    detector = np.zeros((nx, nx, nch))

    for k in range(s.shape[1]):
        detector[..., k] = convolve(pinhole, img[..., k], mode='same')

    return detector


def custom_detector(grid):
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

    s = det_coords(grid.N, grid.geometry)

    if grid.name == 'airyscan':
        airy_1 = np.sqrt(s[0] ** 2 + s[1] ** 2) < grid.N // 2
        airy_2 = np.logical_and(s[0] == -(grid.N // 2), s[1] == 0)
        idx = np.argwhere(np.logical_or(airy_1, airy_2)).flatten()
        s = s[:, idx]
    elif grid.geometry == 'hex':
        idx = np.argwhere(s[0] >= -(grid.N // 2)).flatten()
        s = s[:, idx]

    s *= grid.pxpitch

    s = np.round(s / grid.M / grid.pxsizex).astype('int')

    detector = pinhole_array(s, grid.Nx, grid.M, grid.pxsizex, grid.pxdim, grid.pinhole_shape)

    return detector
