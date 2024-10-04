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
volume_raw = tiff.imread('/media/ExtHDD01/Dataset/womac4LRraw.tif')
volume_lr = tiff.imread('/media/ExtHDD01/Dataset/womac4LR.tif')


kernel0 = parse_kernel(blur_kernel_file='test/result51.npy')
#kernel0 = parse_kernel(blur_kernel_file='test/kernel.npy')
kernel1 = parse_kernel(None, None, 8)
w0 = (kernel0.shape[2] - 1) // 2
w1 = (kernel1.shape[2] - 1) // 2

plt.plot(np.linspace(-w0, w0, kernel0.shape[2]), kernel0.squeeze().numpy())
plt.plot(np.linspace(-w1, w1, kernel1.shape[2]), kernel1.squeeze().numpy())
plt.show()

hr = volume_hr[468, :, :]   # (X, Z, Y)
hr = torch.from_numpy(hr).unsqueeze(0).unsqueeze(0).float()
hrlr = F.conv2d(hr, kernel0.permute(0, 1, 3, 2), padding="same")
imagesc(hrlr.squeeze())


plt.plot(volume_lr[468,100,:]);plt.plot(volume_raw[468,100,:]);plt.plot(hr[0,0,100,:]);plt.show()

#tiff.imwrite('temp.tif', hrlr.squeeze().numpy())

if 0:
    hr = volume_hr[-384:, :, :]
    hr = torch.from_numpy(hr).unsqueeze(0).unsqueeze(0).float()
    hrlr = torch.stack([F.conv2d(hr[:, :, i, :, :], kernel1.permute(0, 1, 3, 2), padding="same") for i in range(hr.shape[2])], dim=2)

    imagesc(hr[0, 0, :, :, 100])
    imagesc(hrlr[0, 0, :, :, 100])

    #tiff.imwrite('temp.tif', hrlr.squeeze().numpy())

if 0:
    vv = torch.from_numpy(volume_hr[-384:, :, :]).unsqueeze(0).unsqueeze(0)

    #oo = F.conv3d(vv, kernel1.unsqueeze(0), padding="same")

    oo = torch.stack([F.conv2d(vv[:, :, :, :, z], kernel1.permute(0, 1, 3, 2), padding="same") for z in range(vv.shape[4])], 4)

    imagesc(vv[0,0,:,:,100])
    imagesc(oo[0,0,:,:,100])
