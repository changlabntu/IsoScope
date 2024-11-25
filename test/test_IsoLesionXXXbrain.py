import torch
import os, glob
import numpy as np
import tifffile as tiff
import torch.nn as nn
from utils.data_utils import imagesc


def get_aug(x0, aug, backward=False):  # X Y Z
    if not backward:
        if aug == 1:
            x0 = np.transpose(x0, (1, 0, 2))
        elif aug == 2:
            x0 = x0[:, ::-1, :]
        elif aug == 3:
            x0 = x0[:, :, ::-1]
        elif aug == 4:
            x0 = x0[:, ::-1, ::-1]
    else:
        if aug == 1:
            x0 = np.transpose(x0, (1, 0, 2))
        elif aug == 2:
            x0 = x0[:, ::-1, :]
        elif aug == 3:
            x0 = x0[:, :, ::-1]
        elif aug == 4:
            x0 = x0[:, ::-1, ::-1]
    return x0.copy()


def get_one(x0, aug, residual=False):
    x0 = get_aug(x0, aug)
    x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()#.permute(0, 1, 3, 4, 2)  # (B, C, H, W, D)
    print(x0.shape)

    # extra downsample

    x0 = x0[:, :, :, :, ::2]
    #x0 = torch.nn.Upsample(x0, size=(x0.shape[0], x0.shape[1], x0.shape[2] * 2), mode='trilinear')(x0)
    x0 = torch.nn.Upsample(size=(x0.shape[2], x0.shape[3], x0.shape[4] * 2), mode='trilinear')(x0)

    # upsample part
    x0 = torch.stack([up2d(x0[:,:,:,i,:]) for i in range(x0.shape[3])], 3)
    print(x0.shape)
    #x0 = downsample(x0)
    #x0 = torch.stack([up2d(x0[:,:,:,i,:]) for i in range(x0.shape[3])], 3)

    # padding
    if mirror_padding > 0:
        padL = torch.flip(x0[:, :, :, :, :mirror_padding], [4])
        padR = torch.flip(x0[:, :, :, :, -mirror_padding:], [4])
        #padL = x0.mean() * torch.ones(x0[:, :, :, :, :mirror_padding].shape)
        #padR = x0.mean() * torch.ones(x0[:, :, :, :, -mirror_padding:].shape)
        x0 = torch.cat([padL, x0, padR], 4)

    out = net(x0)['out0']  #(X, Y, Z )

    if residual:
        out = out + x0

    # unpadding
    if mirror_padding > 0:
        out = out[:, :, :, :, mirror_padding:-mirror_padding]
        x0 = x0[:, :, :, :, mirror_padding:-mirror_padding]
    x0 = x0.squeeze().detach().numpy()
    x0 = get_aug(x0, aug, backward=True)
    out = out.squeeze().detach().numpy()
    out = get_aug(out, aug, backward=True)
    return x0, out


def test_IsoLesion(sub):
    subject_name = sub.split('/')[-1]
    x0 = tiff.imread(sub)  # (Z, X, Y)
    print(x0.shape)
    x0 = x0[20:20+128, :, :]
    print(x0.min(), x0.max())

    #if trd[0] == None:
    #    trd[0] = np.percentile(x0, 15)

    x0 = np.transpose(x0, (1, 2, 0))  # (X, Y, Z)

    trd = [x0.min(), x0.max()]
    print(trd)

    # Normalization
    if 0:
        x0[x0 < trd[0]] = trd[0]
        x0[x0 > trd[1]] = trd[1]
        x0 = (x0 - trd[0]) / (trd[1] - trd[0])
        x0 = (x0 - 0.5) / 0.5

    x00 = 1 * x0
    print(x00.shape)
    # out: (X, Y, Z)
    xup, out = get_one(x00, aug=0, residual=residual)
    #_, out2 = get_one(x00, aug=2, residual=residual)
    #out = (out + out2) / 2

    # XY
    tiff.imwrite(root + '/out/xy/' + suffix + subject_name + '.tif', np.transpose(out, (2, 0, 1)))
    # XZ
    tiff.imwrite(root + '/out/xz/' + suffix + subject_name + '.tif', np.transpose(out, (1, 0, 2)))
    # YZ
    tiff.imwrite(root + '/out/yz/' + suffix + subject_name + '.tif', out)
    # XZ 2d
    tiff.imwrite(root + '/out/xz2d/' + suffix + subject_name + '.tif', np.transpose(xup, (1, 0, 2)))
    # YZ 2d
    tiff.imwrite(root + '/out/yz2d/' + suffix + subject_name + '.tif', xup)


# parameters
residual = False

# path
root = '/media/ExtHDD01/Dataset/paired_images/BraTSReg/train/'

# models
# (prj, epoch) = ('gd1331check3', 80)
# (prj, epoch) = ('gd2332', 60)
#(prj, epoch) = ('IsoMRIclean/gd1331', 1100)
(prj, epoch) = ('IsoMRIclean/gd1331fix/dis0B', 1900)

trd = [-1, 1]
subjects = sorted(glob.glob(root + 't1normcroptest/*'))

mirror_padding = 32

#trd = [None, 1]
#subjects = sorted(glob.glob(root + 'diffresult0921/dualE/*'))
#suffix = 'dualE/'
suffix = ''


os.makedirs(root + '/out/' + suffix, exist_ok=True)
os.makedirs(root + '/out/xy/' + suffix, exist_ok=True)
os.makedirs(root + '/out/xz/' + suffix, exist_ok=True)
os.makedirs(root + '/out/yz/' + suffix, exist_ok=True)
os.makedirs(root + '/out/xz2d/' + suffix, exist_ok=True)
os.makedirs(root + '/out/yz2d/' + suffix, exist_ok=True)


x0 = tiff.imread(subjects[0])
#upsample = torch.nn.Upsample(size=(x0.shape[1], x0.shape[2], 30 * 8), mode='trilinear')
#downsample = torch.nn.Upsample(size=(x0.shape[0], x0.shape[1], x0.shape[2]), mode='trilinear') XXXXXX
#up2d = torch.nn.Upsample(size=(x0.shape[1], x0.shape[0] * 8), mode='bicubic')
up2d = torch.nn.Upsample(scale_factor=(1, 1), mode='bicubic')

for sub in subjects[:1]:
    net = torch.load('/media/ExtHDD01/logs/BraTSReg/' + prj +
                     '/checkpoints/net_g_model_epoch_' + str(epoch) + '.pth', map_location='cpu')#.eval()  # .cuda()
    print(sub)
    test_IsoLesion(sub)
