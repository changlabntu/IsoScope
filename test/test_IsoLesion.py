import torch
import os, glob
import numpy as np
import tifffile as tiff
import torch.nn as nn
from utils.data_utils import imagesc
from tqdm import tqdm


def create_tapered_weight(S0, S1, S2, nz, nx, ny, size, edge_size: int = 64) -> np.ndarray:
    """
    Create a 3D cube with linearly tapered edges in all directions.

    Args:
        size (int): Size of the cube (size x size x size)
        edge_size (int): Size of the tapered edge section

    Returns:
        np.ndarray: 3D array with tapered weights
    """
    # Create base cube filled with ones
    weight = np.ones(size)

    # Create linear taper from 0 to 1
    # taper = np.linspace(0, 1, edge_size)
    taper_S0 = np.linspace(0, 1, S0)
    taper_S1 = np.linspace(0, 1, S1)
    taper_S2 = np.linspace(0, 1, S2)

    # Z
    if nz != 0:
        weight[:S0, :, :] *= taper_S0.reshape(-1, 1, 1)
    if nz != -1:
        weight[-S0:, :, :] *= taper_S0[::-1].reshape(-1, 1, 1)

    # X
    if nx != 0:
        weight[:, :S1, :] *= taper_S1.reshape(1, -1, 1)
    if nx != -1:
        weight[:, -S1:, :] *= taper_S1[::-1].reshape(1, -1, 1)

    # Y
    #if ny != 0:
    #    weight[:, :, :S2] *= taper_S2
    #if ny != -1:
    #    weight[:, :, -S2:] *= taper_S2[::-1]

    return weight


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
    x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float().cuda()#.permute(0, 1, 3, 4, 2)  # (B, C, H, W, D)

    # upsample part
    x0 = torch.stack([up2d(x0[:,:,:,i,:]) for i in range(x0.shape[3])], 3)
    #x0 = x0[:, :, :, :, 4::16]
    #x0 = torch.stack([up2d(x0[:,:,:,i,:]) for i in range(x0.shape[3])], 3)
    #tiff.imwrite('temp.tif', x0.squeeze().detach().numpy())

    # padding
    if mirror_padding > 0:
        randint = np.random.randint(0, 16)
        padL = torch.flip(x0[:, :, :, :, :mirror_padding+randint], [4])
        padR = torch.flip(x0[:, :, :, :, -mirror_padding+randint:], [4])
        #padL = x0.mean() * torch.ones(x0[:, :, :, :, :mirror_padding].shape)
        #padR = x0.mean() * torch.ones(x0[:, :, :, :, -mirror_padding:].shape)
        x0 = torch.cat([padL, x0, padR], 4)

    out = net(x0)['out0']  #(X, Y, Z )

    if residual:
        out = out + x0

    # unpadding
    if mirror_padding > 0:
        out = out[:, :, :, :, mirror_padding+randint:-mirror_padding+randint]
        x0 = x0[:, :, :, :, mirror_padding+randint:-mirror_padding+randint]
    x0 = x0.squeeze().detach().cpu().numpy()
    x0 = get_aug(x0, aug, backward=True)
    out = out.squeeze().detach().cpu().numpy()
    out = get_aug(out, aug, backward=True)
    return x0, out


def test_IsoLesion(sub):
    subject_name = sub.split('/')[-1]
    x0 = tiff.imread(sub)  # (Z, X, Y)
    print(x0.shape)
    print(x0.min(), x0.max())

    x0 = x0[:, :, :]

    if trd[0] == None:
        trd[0] = 0#np.percentile(x0, 15)
    if trd[1] == None:
        trd[1] = x0.max()

    x0 = np.transpose(x0, (1, 2, 0))  # (X, Y, Z)
    # Normalization
    x0[x0 < trd[0]] = trd[0]
    x0[x0 > trd[1]] = trd[1]
    x0 = (x0 - trd[0]) / (trd[1] - trd[0])
    x0 = (x0 - 0.5) / 0.5

    #  XY padding

    if padxy > 0:
        padX0 = -1 * np.ones(x0[:, :padxy, :][:, ::-1, :].shape)
        padX1 = -1 * np.ones(x0[:, -padxy:, :][:, ::-1, :].shape)
        x0 = np.concatenate([padX0, x0, padX1], 1)
        padY0 = -1 * np.ones(x0[:padxy, :, :][::-1, :, :].shape)
        padY1 = -1 * np.ones(x0[-padxy:, :, :][::-1, :, :].shape)
        x0 = np.concatenate([padY0, x0, padY1], 0)

    # separate to 128 * 128 * Z patches
    xrange = range(0, x0.shape[0] - 64, 64)
    yrange = range(0, x0.shape[1] - 64, 64)

    # (x, y, z)
    one_x = []
    for xi in range(len(xrange)):
        one_y = []
        for yi in range(len(yrange)):
            x00 = x0[xrange[xi]:xrange[xi]+128, yrange[yi]:yrange[yi]+128, :]

            # augmentations
            out_all = []
            for aug in aug_list:
                _, out = get_one(x00, aug=aug, residual=residual)
                out_all.append(out)
            out = np.array(out_all).sum(0) / len(aug_list)

            if xi == len(xrange) - 1:
                nz = -1
            else:
                nz = xi
            if yi == len(yrange) - 1:
                nx = -1
            else:
                nx = yi
            w = create_tapered_weight(S0=64, S1=64, S2=64, nz=nz, nx=nx, ny=1, size=out.shape)

            out = out * w

            if len(one_y) > 0:
                one_y[-1][:, -64:, :] = (one_y[-1][:, -64:, :] + out[:, :64, :])
                one_y.append(out[:, -64:, :])
            else:
                one_y.append(out[:, :, :])
        one_y = np.concatenate(one_y, 1)
        if len(one_x) > 0:
            one_x[-1][-64:, :, :] = (one_x[-1][-64:, :, :] + one_y[:64, :, :])
            one_x.append(one_y[-64:, :, :])
        else:
            one_x.append(one_y)
    out = np.concatenate(one_x, 0).astype(np.float32)

    if padxy > 0:
        out = out[padxy:-padxy, padxy:-padxy, :]

    xup = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float() #.permute(0, 1, 3, 4, 2)  # (B, C, H, W, D)
    xup = torch.stack([up2d(xup[:,:,:,i,:]) for i in range(xup.shape[3])], 3)
    xup = xup.squeeze().detach().cpu().numpy()

    # XY
    tiff.imwrite(root + '/out/xy/' + suffix + subject_name + '.tif', np.transpose(out, (2, 0, 1)))
    # XZ
    tiff.imwrite(root + '/out/xz/' + suffix + subject_name + '.tif', np.transpose(out, (1, 0, 2)))
    # YZ
    tiff.imwrite(root + '/out/yz/' + suffix + subject_name + '.tif', out)

    if print_2d:
        # XZ 2d
        tiff.imwrite(root + '/out/xz2d/' + suffix + subject_name + '.tif', np.transpose(xup, (1, 0, 2)))
        # YZ 2d
        tiff.imwrite(root + '/out/yz2d/' + suffix + subject_name + '.tif', xup)

# parameters
residual = False

# path
root = '/media/ghc/Ghc_data3/OAI_diffusion_final/isotropic_outs/'
suffix = ''
os.makedirs(root + '/out/' + suffix, exist_ok=True)
os.makedirs(root + '/out/xy/' + suffix, exist_ok=True)
os.makedirs(root + '/out/xz/' + suffix, exist_ok=True)
os.makedirs(root + '/out/yz/' + suffix, exist_ok=True)
os.makedirs(root + '/out/xz2d/' + suffix, exist_ok=True)
os.makedirs(root + '/out/yz2d/' + suffix, exist_ok=True)


# models
# (prj, epoch) = ('gd1331check3', 80)
# (prj, epoch) = ('gd2332', 60)
#(prj, epoch) = ('IsoMRIclean/gd1331', 180)
#(prj, epoch) = ('gd1331nocyc/1run2', 300)
#(prj, epoch) = ('IsoMRIcleanDis0/dis0', 320)
#(prj, epoch) = ('IsoLambda/B4', 300)
#(prj, epoch) = ('IsoLambda/0', 100)
#(prj, epoch) = ('IsoLambda/0run5', 260)
#(prj, epoch) = ('IsoLambda/0l1maxskip1', 180)
(prj, epoch) = ('IsoLambda/2', 60)
(prj, epoch) = ('IsoLambda/2', 100)
(prj, epoch) = ('IsoLambda/2l1maxskip12332', 300)


aug_list = [0, 1, 2, 3]
print_2d = False
mirror_padding = 64
padxy = 0

use_eval = False
trd = [None, None]  # [None, 800]

subjects = sorted(glob.glob(root + 'original/a2d/*'))

# get upsample size
x0 = tiff.imread(subjects[0])
up2d = torch.nn.Upsample(scale_factor=(1, 8), mode='bilinear')
up2dn = torch.nn.Upsample(scale_factor=(1, 8), mode='nearest')


prj_list = [prj]#['gd1331nocyc/' + x.split('/')[-1] for x in sorted(glob.glob('/media/ExtHDD01/logs/womac4/IsoMRIclean/gd1331nocyc/*'))]
epoch_list = [epoch]#list(range(100, 601, 100))
#prj_list = ['IsoLambda/' + x.split('/')[-1] for x in sorted(glob.glob('/media/ExtHDD01/logs/womac4/IsoLambda/2l1maxskip1'))]
#epoch_list = list(range(60, 821, 40))


for sub in tqdm(subjects[:1]):
    for prj in prj_list:
        for epoch in epoch_list:
            try:
                net = torch.load('/media/ExtHDD01/logs/womac4/' + prj +
                                 '/checkpoints/net_g_model_epoch_' + str(epoch) + '.pth', map_location='cpu').cuda()
                if use_eval:
                    print('using eval')
                    net = net.eval()
                print(sub)
                print(prj, epoch)
                test_IsoLesion(sub)
                data = tiff.imread('/media/ghc/Ghc_data3/OAI_diffusion_final/isotropic_outs/out/yz/9000798_00.tif.tif')
                tiff.imwrite('/media/ghc/Ghc_data3/OAI_diffusion_final/isotropic_outs/summary/' + prj.replace('/', '_') + '_' + str(epoch) + '.tif', data)
                tiff.imwrite('/media/ghc/Ghc_data3/OAI_diffusion_final/isotropic_outs/summary2/' + prj.replace('/', '_') + '_' + str(epoch) + '.tif', data[184, ::])

            except:
                print('Error:', prj, epoch)

from scipy.ndimage import uniform_filter
import numpy as np


def calculate_variance_map_fast(R, n=7):
    # Calculate local mean
    mean = uniform_filter(R, size=n)

    # Calculate local variance
    R2 = uniform_filter(R ** 2, size=n)
    var = R2 - mean ** 2

    return var


def calculate_residual():
    import tifffile as tiff
    import numpy as np
    from scipy.ndimage import uniform_filter
    root = '/media/ghc/Ghc_data3/OAI_diffusion_final/isotropic_outs/residual/'
    x = tiff.imread(root + 'x0.tif')
    y = tiff.imread(root + 'y.tif')
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    residual = ((y - x))
    var = calculate_variance_map_fast(residual, n=7)
    tiff.imwrite(root + 'residual.tif', residual)
    tiff.imwrite(root + 'var.tif', var)

