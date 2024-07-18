import numpy as np
from scipy.signal import convolve
from skimage.draw import polygon


def circle(n, radius):
    x = np.arange(-(n // 2), n // 2 + 1)

    xx, yy = np.meshgrid(x, x)
    r = np.sqrt(xx ** 2 + yy ** 2)

    return np.where(r < radius, 1, 0)


def square(n, length):
    x = np.arange(-(n // 2), n // 2 + 1)

    xx, yy = np.meshgrid(x, x)

    cond = np.logical_and(np.abs(xx) < length / 2, np.abs(yy) < length / 2)

    return np.where(cond, 1, 0)


def hexagon(n, radius):
    x = np.array([-1, 0, 1, 1, 0, -1]) * radius * np.cos(np.deg2rad(30)) + n // 2
    y = np.array([1, 2, 1, -1, -2, -1]) * radius * np.sin(np.deg2rad(30)) + n // 2

    hex_array = np.zeros((n, n))

    idx = polygon(x, y, shape=(n, n))
    hex_array[idx] = 1

    return hex_array


def rect_grid(n, x):
    s = np.empty((n, n, 2))

    for n_i, i in enumerate(x):
        for n_j, j in enumerate(x):
            s[n_i, n_j] = np.array([i, j])

    return s


def hex_grid(n, x):
    s = np.empty((n, n, 2))

    for n_i, i in enumerate(x):
        for n_j, j in enumerate(x):
            s[n_i, n_j] = np.array([i - 0.5 * (j % 2), 0.5 * np.sqrt(3) * j])

    return s


def det_coords(n, geometry):
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
    radius = pxdim / mag / pxsize / 2
    nch = s.shape[1]

    img = np.zeros((nx, nx, nch))
    detector = np.zeros((nx, nx, nch))

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

    for k in range(s.shape[1]):
        detector[..., k] = convolve(pinhole, img[..., k], mode='same')

    return detector


def custom_detector(grid):
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
