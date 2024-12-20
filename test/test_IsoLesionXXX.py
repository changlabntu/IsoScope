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

    #x0 = upsample(x0)
    #x0 = downsample(x0)
    #x0 = upsample(x0)

    # upsample part
    x0 = torch.stack([up2d(x0[:,:,:,i,:]) for i in range(x0.shape[3])], 3)
    print(x0.shape)
    #x0 = x0[:, :, :, :, 4::16]
    #x0 = torch.stack([up2d(x0[:,:,:,i,:]) for i in range(x0.shape[3])], 3)
    #tiff.imwrite('temp.tif', x0.squeeze().detach().numpy())

    # padding
    if mirror_padding > 0:

        randint = 0#np.random.randint(0, 16)

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
    x0 = x0.squeeze().detach().numpy()
    x0 = get_aug(x0, aug, backward=True)
    out = out.squeeze().detach().numpy()
    out = get_aug(out, aug, backward=True)
    return x0, out


def test_IsoLesion(sub):
    subject_name = sub.split('/')[-1]
    x0 = tiff.imread(sub)  # (Z, X, Y)
    print(x0.shape)
    print(x0.min(), x0.max())

    x0 = x0[:, :, :]

    if trd[0] == None:
        trd[0] = np.percentile(x0, 15)
    if trd[1] == None:
        trd[1] = x0.max()

    x0 = np.transpose(x0, (1, 2, 0))  # (X, Y, Z)
    # Normalization
    x0[x0 < trd[0]] = trd[0]
    x0[x0 > trd[1]] = trd[1]
    x0 = (x0 - trd[0]) / (trd[1] - trd[0])
    x0 = (x0 - 0.5) / 0.5

    # separate to two parts
    if 0:
        out_all = []
        xup_all = []
        for n in range(2):
            if n == 0:
                x00 = x0[:, :, :x0.shape[2]//2 + 4]
            elif n == 1:
                x00 = x0[:, :, x0.shape[2]//2 - 4:]

            print(x00.shape)
            # out: (X, Y, Z)
            xup, out = get_one(x00, aug=3, residual=residual)
            _, out2 = get_one(x00, aug=2, residual=residual)
            out = (out + out2) / 2
            out_all.append(out)
            xup_all.append(xup)

        # linearly overlap the weight
        overlap = np.multiply(out_all[0][:, :, -64:], np.linspace(1, 0, 64)[np.newaxis, np.newaxis,:]) + \
                    np.multiply(out_all[1][:, :, :64], np.linspace(0, 1, 64)[np.newaxis, np.newaxis, :])
        out = np.concatenate([out_all[0][:, :, :-64], overlap, out_all[1][:, :, 64:]], axis=2)

        overlapx = np.multiply(xup_all[0][:, :, -64:], np.linspace(1, 0, 64)[np.newaxis, np.newaxis,:]) + \
                   np.multiply(xup_all[1][:, :, :64], np.linspace(0, 1, 64)[np.newaxis, np.newaxis, :])
        xup = np.concatenate([xup_all[0][:, :, :-64], overlapx, xup_all[1][:, :, 64:]], axis=2)

    x00 = 1 * x0
    print(x00.shape)
    # out: (X, Y, Z)
    xup, out = get_one(x00, aug=3, residual=residual)
    _, out2 = get_one(x00, aug=2, residual=residual)
    out = (out + out2) / 2
    #_, out0 = get_one(x00, aug=0, residual=residual)
    #_, out1 = get_one(x00, aug=1, residual=residual)
    #out = (out + out0 + out1 + out2) / 4

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
root = '/media/ghc/Ghc_data3/OAI_diffusion_final/isotropic_outs/'


# models
# (prj, epoch) = ('gd1331check3', 80)
# (prj, epoch) = ('gd2332', 60)
(prj, epoch) = ('IsoMRIclean/gd1331', 180)
#(prj, epoch) = ('gd1331nocyc/1run2', 300)
#(prj, epoch) = ('IsoMRIcleanDis0/dis0', 320)
#(prj, epoch) = ('IsoMRIclean/gd1331nocyc/1nomclambda', 100)
#(prj, epoch) = ('IsoLambda/0cyc', 180)
trd = [None, None] # [None, 800]
subjects = sorted(glob.glob(root + 'original/a2d/*'))

mirror_padding = 32

suffix = ''

os.makedirs(root + '/out/' + suffix, exist_ok=True)
os.makedirs(root + '/out/xy/' + suffix, exist_ok=True)
os.makedirs(root + '/out/xz/' + suffix, exist_ok=True)
os.makedirs(root + '/out/yz/' + suffix, exist_ok=True)
os.makedirs(root + '/out/xz2d/' + suffix, exist_ok=True)
os.makedirs(root + '/out/yz2d/' + suffix, exist_ok=True)


x0 = tiff.imread(subjects[0])
up2d = torch.nn.Upsample(scale_factor=(1, 8), mode='bicubic')


prj_list = [prj]#['gd1331nocyc/' + x.split('/')[-1] for x in sorted(glob.glob('/media/ExtHDD01/logs/womac4/IsoMRIclean/gd1331nocyc/*'))]
epoch_list = [epoch]#list(range(100, 601, 100))

for sub in subjects[:1]:
    for prj in prj_list:
        for epoch in epoch_list:
            #try:
            net = torch.load('/media/ExtHDD01/logs/womac4/' + prj +
                             '/checkpoints/net_g_model_epoch_' + str(epoch) + '.pth', map_location='cpu')  # .cuda()
            print(sub)
            print(prj, epoch)
            test_IsoLesion(sub)
            #data = tiff.imread('/media/ghc/Ghc_data3/OAI_diffusion_final/isotropic_outs/out/yz/9000099_03.tif.tif')
            #tiff.imwrite('/media/ghc/Ghc_data3/OAI_diffusion_final/isotropic_outs/outimg2/' + prj.replace('/', '_') + '_' + str(epoch) + '.tif', data)
            #tiff.imwrite('/media/ghc/Ghc_data3/OAI_diffusion_final/isotropic_outs/outimg2D/' + prj.replace('/', '_') + '_' + str(
            #    epoch) + '.tif', data[184, ::])

            #except:
            #    print('Error:', prj, epoch)


import torch
import glob
import numpy as np
import skimage.io
import tifffile as tiff
from torchmetrics.image.fid import FID
from torchmetrics.image.kid import KID
from tqdm import tqdm


def load_pngs(root, irange):
    img0 = sorted(root + '/*')[irange[0]:irange[1]]
    img0 = np.stack([skimage.io.imread(x) for x in img0], 0)
    img0 = torch.from_numpy(img0).unsqueeze(1).repeat(1, 3, 1, 1)  # (Z, C, X, Y)
    return img0


irange = [0, 10]

#root = '/media/ghc/Ghc_data3/OAI_diffusion_final/isotropic_outs/outputs_all/IsoLambda2/xy/*'
root = '/media/ghc/Ghc_data3/OAI_diffusion_final/isotropic_outs/original/aval/*'

img = sorted(glob.glob(root))[irange[0]:irange[1]]

img = np.stack([tiff.imread(x) for x in img], 0)
img = torch.from_numpy(img / 1).unsqueeze(1).repeat(1, 3, 1, 1, 1)  # (B, C, Z, X, Y)

img = torch.nn.Upsample(scale_factor=(8, 1, 1), mode='nearest')(img)

img = (img - img.min()) / (img.max() - img.min()) * 255
img = img.type(torch.uint8)

img0 = img.permute(0, 2, 1, 3, 4)  # (X, Y)
img0 = img0.reshape(-1, 3, img0.shape[3], img0.shape[4])

img1 = img.permute(0, 3, 1, 4, 2)  # (Y, Z)
img1 = img1.reshape(-1, 3, img1.shape[3], img1.shape[4])


# kid by subject
kval = []
for i in tqdm(range(img.shape[0])):
    metrics = KID(subset_size=64).cuda()
    metrics.update(img0[i*184:(i+1)*184, ::].cuda(), real=True)
    metrics.update(img1[i*384:(i+1)*384, ::].cuda(), real=False)
    kval.append(metrics.compute()[0].cpu().numpy())
kval = np.array(kval)
print(kval.mean(), kval.std())


# fid by subject
fval = []
for i in tqdm(range(img.shape[0])):
    fid = FID(feature=768).cuda()
    fid.update(img0[i*184:(i+1)*184, ::].cuda(), real=True)
    fid.update(img1[i*384:(i+1)*384, ::].cuda(), real=False)
    fval.append(fid.compute().cpu().numpy())
fval = np.array(fval)
print(fval.mean(), fval.std())

prj = 'nearest'
np.save('figures/fid_' + prj + '.npy', fval)
np.save('figures/kid_' + prj + '.npy', kval)


import matplotlib.pyplot as plt
import numpy as np

datas = []
for prj in ['nearest', 'linear', 'Lambda2', 'Lambda0']:
    datas.append(np.load('figures/fid_' + prj + '.npy'))

# Create boxplot
plt.figure(figsize=(10, 6))
plt.boxplot(datas, labels=['Data 1', 'Data 2', 'Data 3', 'Data 4'])

# Customize plot
plt.title('Comparison of Four Datasets')
plt.ylabel('Values')
plt.show()

# fid
if 0:
    out = []
    for f in [768, 2048][:1]:
        fid = FID(feature=f).cuda()
        fid.update(img0[:512, ::].cuda(), real=True)
        fid.update(img1[:512, ::].cuda(), real=False)
        out.append(fid.compute())

    # kid
    import time
    tini = time.time()
    metrics = KID(subset_size=64).cuda()
    metrics.update(img0[:512, ::].cuda(), real=True)
    metrics.update(img1[:512, ::].cuda(), real=False)
    print(metrics.compute())
    print(time.time() - tini)








