from torch.fft import fftn, ifftn, ifftshift
from torch import real, einsum


def partial_convolution(volume: torch.Tensor, kernel: torch.Tensor, dim1: str = 'ijk', dim2: str = 'jkl',
                        axis: str = 'jk', fourier: tuple = (False, False)):
    dim3 = dim1 + dim2
    dim3 = ''.join(sorted(set(dim3), key=dim3.index))

    dims = [dim1, dim2, dim3]
    axis_list = [[d.find(c) for c in axis] for d in dims]

    if fourier[0] == False:
        volume_fft = fftn(volume, dim=axis_list[0])
    else:
        volume_fft = volume

    if fourier[1] == False:
        kernel_fft = fftn(kernel, dim=axis_list[1])
    else:
        kernel_fft = kernel

    conv = einsum(f'{dim1},{dim2}->{dim3}', volume_fft, kernel_fft)

    conv = ifftn(conv, dim=axis_list[2])  # inverse FFT of the product
    conv = ifftshift(conv, dim=axis_list[2])  # Rotation of 180 degrees of the phase of the FFT
    conv = real(conv)  # Clipping to zero the residual imaginary part

    return conv