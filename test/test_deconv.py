import torch
import numpy as np
import tifffile as tiff
import torch.nn as nn
from utils.data_utils import imagesc
import torch.nn.functional as F
from degrade.degrade import select_kernel
import matplotlib.pyplot as plt


def parse_kernel(blur_kernel_file, blur_kernel_type=None, blur_fwhm=None):

    if blur_kernel_file is not None:
        blur_kernel = np.load(blur_kernel_file)
    else:
        window_size = int(2 * round(blur_fwhm) + 1)
        blur_kernel = select_kernel(window_size, blur_kernel_type, fwhm=blur_fwhm)
    blur_kernel /= blur_kernel.sum()
    blur_kernel = blur_kernel.squeeze()[None, None, :, None]
    blur_kernel = torch.from_numpy(blur_kernel).float()

    return blur_kernel


volume_hr = tiff.imread('/media/ExtHDD01/Dataset/womac4HR.tif')
volume_lr = tiff.imread('/media/ExtHDD01/Dataset/womac4LR.tif')


kernel0 = parse_kernel(blur_kernel_file='test/kernel.npy')
kernel1 = parse_kernel(None, None, 8)
w0 = (kernel0.shape[2] - 1) // 2
w1 = (kernel1.shape[2] - 1) // 2

plt.plot(np.linspace(-w0, w0, kernel0.shape[2]), kernel0.squeeze().numpy())
plt.plot(np.linspace(-w1, w1, kernel1.shape[2]), kernel1.squeeze().numpy())

plt.show()

hr = volume_hr[468, :, :]
hr = torch.from_numpy(hr).unsqueeze(0).unsqueeze(0).float()

hrlr = F.conv2d(hr, kernel1, padding="same")

imagesc(hrlr.squeeze())
tiff.imwrite('temp.tif', hrlr.squeeze().numpy())