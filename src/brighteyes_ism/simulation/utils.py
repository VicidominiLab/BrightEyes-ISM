from torch.fft import fftn, ifftn, ifftshift
from torch import real, einsum


def partial_convolution(psf, pinhole, dim1='ijk', dim2='jkl', axis='jk'):
    dim3 = dim1 + dim2
    dim3 = ''.join(sorted(set(dim3), key=dim3.index))

    dims = [dim1, dim2, dim3]
    axis_list = [[d.find(c) for c in axis] for d in dims]

    otf = fftn(psf, dim=axis_list[0])
    kernel = fftn(pinhole, dim=axis_list[1])

    conv = einsum(f'{dim1},{dim2}->{dim3}', otf, kernel)

    conv = ifftn(conv, dim=axis_list[2])  # inverse FFT of the product
    conv = ifftshift(conv, dim=axis_list[2])  # Rotation of 180 degrees of the phase of the FFT
    conv = real(conv)  # Clipping to zero the residual imaginary part

    return conv