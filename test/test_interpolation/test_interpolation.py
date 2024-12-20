import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.data_utils import imagesc


def bicubic_3d(x):
    x = torch.nn.Upsample(size=(384, 23 * 8), mode='bicubic')(torch.from_numpy(x).float().unsqueeze(1)).squeeze().numpy()
    return x


def to_plot(x):
    for xx in x:
        plt.plot(xx[100, 100, :]);
    plt.show()


ori = tiff.imread('test/interpolation/ori.tif')
o = np.transpose(ori, (1, 2, 0)) / 800
a = tiff.imread('test/interpolation/a.tif')
c = tiff.imread('test/interpolation/c.tif')
out = tiff.imread('test/interpolation/out.tif')

[o, a, c, out] = [x - x.mean() for x in [o, a, c, out]]
[o, a, c, out] = [x / x.std() for x in [o, a, c, out]]

on = torch.nn.Upsample(size=(384, 384, 23 * 8))(torch.from_numpy(o).float().unsqueeze(0).unsqueeze(0)).squeeze().numpy()
ot = torch.nn.Upsample(size=(384, 384, 23 * 8), mode='trilinear')(torch.from_numpy(o).float().unsqueeze(0).unsqueeze(0)).squeeze().numpy()
oc = torch.nn.Upsample(size=(384, 23 * 8), mode='bicubic')(torch.from_numpy(o).float().unsqueeze(1)).squeeze().numpy()

to_plot([on, out])


