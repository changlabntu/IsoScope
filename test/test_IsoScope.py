import torch
import tifffile as tiff
from utils.data_utils import imagesc
import os, glob, sys
import networks, models
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import time
import shutil
from tqdm import tqdm
import json
import glob
import tifffile as tiff
import yaml


def get_one_out(x0, model):
    x0 = [x.unsqueeze(0).unsqueeze(0).float().permute(0, 1, 3, 4, 2) for x in x0]

    #if gpu:

    if len(x0) > 1:
        out_all = model(torch.cat([x0[0].cuda(), x0[1].cuda()], 1))['out1'].cpu().detach()
    else:
        out_all = model(x0[0])['out0'].cpu().detach()

    out_all = out_all.numpy()[0, 0, :, :, :]
    return out_all


def test_IsoScope(x0, model, **kwargs):
    out_all = []
    for m in range(mc):
        tini = time.time()

        patch = [x[:, :, kwargs['patch_range']['start_dim0']:kwargs['patch_range']['end_dim0'],
            kwargs['patch_range']['start_dim1']:kwargs['patch_range']['end_dim1'],
            kwargs['patch_range']['start_dim2']:kwargs['patch_range']['end_dim2']] for x in x0]

        print(patch[0].shape)
        patch0 = 1 * patch[0]

        # extra downsampling z for brain mri
        #patch[0] = patch[0][:, :, ::4, :, :]
        #print(patch[0].shape)
        #patch[0] = nn.Upsample(scale_factor=(0.125, 1, 1), mode='trilinear')(patch[0])
        #patch[0] = nn.Upsample(scale_factor=(8, 1, 1), mode='trilinear')(patch[0])

        if gpu:
            patch = [x.cuda() for x in patch]

        patch = [upsample(x).squeeze() for x in patch]

        out = get_one_out(patch, model)
        out = np.transpose(out, (2, 0, 1))
        if gpu:
            patch = patch[0].cpu().detach()
        print('Time:', time.time() - tini)

        out_all.append(out)

    out_all = np.stack(out_all, axis=3)

    return out_all, patch[0].numpy(), patch0.numpy()


def reverse_log(x):
    return np.power(10, x)


def assemble_microscopy_volumne(kwargs, w, zrange, xrange, yrange, source):
    C = kwargs['assemble_params']['C']
    for ix in tqdm(xrange):
        one_column = []
        for iz in zrange:
            one_row = []
            for iy in yrange:
                x = tiff.imread(source + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif')
                cropped = x[C:-C, :, C:-C]
                cropped = np.multiply(cropped, w)
                if len(one_row) > 0:
                    one_row[-1][:, :, -C:] = one_row[-1][:, :, -C:] + cropped[:, :, :C]
                    one_row.append(cropped[:, :, 64:])
                else:
                    one_row.append(cropped)
            one_row = np.concatenate(one_row, axis=2)
            one_row = np.transpose(one_row, (1, 0, 2))

            if len(one_column) > 0:
                one_column[-1][:, -C:, :] = one_column[-1][:, -C:, :] + one_row[:, :C, :]
                one_column.append(one_row[:, C:, :])
            else:
                one_column.append(one_row)
        one_column = np.concatenate(one_column, axis=1).astype(np.float32)
    tiff.imwrite(source[:-1] + '.tif', one_column)


def test_over_volumne(kwargs, dx, dy, dz, zrange, xrange, yrange, destination):
    for ix in xrange:#range(0, x0.shape[2], sz)[:]:
        for iz in zrange:#[1536]:#range(0, x0.shape[3], sx):
            for iy in yrange:#range(0, x0.shape[4], sy)[:]:
                kwargs['patch_range'] = {'start_dim0': iz, 'end_dim0': iz + dz,
                                         'start_dim1': ix, 'end_dim1': ix + dx,
                                         'start_dim2': iy, 'end_dim2': iy + dy}

                out_all, patch, _ = test_IsoScope(x0, model, **kwargs)

                tiff.imwrite(destination + 'xy/' + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif', out_all.mean(axis=3).astype(np.float32))
                tiff.imwrite(destination + 'ori/' + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif', patch)

                if mc > 1:
                    tiff.imwrite(destination + 'xyvar/' + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif', out_all.std(axis=3).astype(np.float32))


def get_model(kwargs, gpu):
    dataset = kwargs['dataset']
    prj = kwargs['prj']
    epoch = kwargs['epoch']
    model_name = '/media/ExtHDD01/logs/' + dataset + prj + '/checkpoints/net_g_model_epoch_' + str(epoch) + '.pth'
    print(model_name)
    model = torch.load(model_name, map_location=torch.device('cpu'))#.eval()  # .cuda()#.eval()
    if eval:
        model = model.eval()
    upsample = torch.nn.Upsample(size=kwargs['upsample_params']['size'], mode='trilinear')
    if gpu:
        model = model.cuda()
        upsample = upsample.cuda()
    return model, upsample


def recreate_volume_folder(destination):
    # remove and recreate the folder
    if os.path.exists(destination):
        shutil.rmtree(destination)
    os.makedirs(destination)
    os.makedirs(destination + 'xy/')
    os.makedirs(destination + 'ori/')
    if mc > 1:
        os.makedirs(destination + 'xyvar/')


def view_two_other_direction(x):
    return np.concatenate([np.transpose(x, (2, 1, 0)), np.transpose(x, (1, 2, 0))], 2)


def slice_for_ganout():
    rois = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/' + kwargs["dataset"] + '/cycout/xy/*.tif'))

    for roi in tqdm(rois[:]):
        xy = tiff.imread(roi)
        ori = tiff.imread(roi.replace('/xy/', '/ori/'))

        xy = xy[64:-64, 32:-32, 64:-64]
        ori = ori[64:-64, 32:-32, 64:-64]

        if xy.mean() >= -0.5:
            for ix in range(xy.shape[1]):
                tiff.imwrite(roi.replace('/xy/', '/ganxy/')[:-4] + '_' + str(ix).zfill(3) + '.tif', xy[:, ix, :])
                tiff.imwrite(roi.replace('/xy/', '/ganori/')[:-4] + '_' + str(ix).zfill(3) + '.tif', ori[:, ix, :])


def get_weight(size, method='cross', C=64):
    # the linearly tapering weight to combine al the individual ROI
    weight = np.ones(size)
    weight[:, :, :C] = np.linspace(0, 1, C)
    weight[:, :, -C:] = np.linspace(1, 0, C)
    if method == 'row':
        return weight
    if method == 'cross':
        weight = np.multiply(np.transpose(weight, (2, 1, 0)), weight)
        return weight


def get_args(option, config_name):
    with open(config_name, 'r') as f:
        config = yaml.safe_load(f)

    kwargs = config.get(option, {})
    if not kwargs:
        raise ValueError(f"Option {option} not found in the configuration.")
    return kwargs


def get_data(kwargs):
    image_path = kwargs.get("image_path")  # if image path is a file
    image_list_path = kwargs.get("image_list_path")  # if image path is a directory

    x0 = []
    if image_path:
        for i in range(len(image_path)):
            x0.append(tiff.imread(image_path[i]))

    elif image_list_path:
        for i in range(len(image_list_path)):
            x_list = sorted(glob.glob(image_list_path[i]))
            if not x_list:
                raise ValueError(f"No images found at {image_list_path[i]}")
            x0.append(tiff.imread(x_list[kwargs.get("image_list_index")]))
    else:
        raise ValueError("No valid image path provided.")
    return x0


def norm_x0(x0, norm_method, exp_trd, exp_ftr, trd):
    if norm_method == 'exp':
        x0[x0 <= exp_trd[0]] = exp_trd[0]
        x0[x0 >= exp_trd[1]] = exp_trd[1]
        x0 = np.log10(x0 + 1)
        x0 = np.divide((x0 - x0.mean()), x0.std())
        x0[x0 <= -exp_ftr] = -exp_ftr
        x0[x0 >= exp_ftr] = exp_ftr
        x0 = x0 / exp_ftr
        x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()
    elif norm_method == '11':
        x0[x0 >= trd[1]] = trd[1]
        #x0 = x0 / x0.max()
        x0 = (x0 - x0.min()) / (x0.max() - x0.min())
        x0 = (x0 - 0.5) * 2
        x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()
    elif norm_method == '00':
        x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()
    elif norm_method == '01':
        print(trd[1])
        x0[x0 >= trd[1]] = trd[1]
        #x0 = x0 / x0.max()
        x0 = (x0 - x0.min()) / (x0.max() - x0.min())
        x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()
    return x0



if __name__ == '__main__':
    option = "Fly0B"#"aisr081424"#'Dayu1'
    kwargs = get_args(option=option, config_name='test/config.yaml')
    print(kwargs)

    destination = '/media/ExtHDD01/Dataset/paired_images/' + kwargs["dataset"]
    if option == 'womac4':
        gpu = False
    else:
        gpu = True
    gpu = False
    eval = False
    expand = False
    tilt = False
    masking = False
    mc = 1  # monte carlo inference, mean over N times

    model, upsample = get_model(kwargs, gpu)

    # Data
    x0 = get_data(kwargs)
    for i in range(len(x0)):
        x0[i] = norm_x0(x0[i], kwargs['norm_method'][i],
                        kwargs['exp_trd'][i], kwargs['exp_ftr'][i], kwargs['trd'][i])

    # FOR BRAIN
    #x0 = np.transpose(x0, (1, 2, 0))
    #x0 = x0[:, :, ::4]

    # single test
    out, patch, patch0 = test_IsoScope(x0, model, **kwargs)
    out = out.mean(axis=3)


    if masking:
        out[patch == -1] = -1

    # save single output
    if option == 'womac4':
        tiff.imite(destination + '/patch.tif', view_two_other_direction(patch))
        tiff.imwrite(destination + '/xy.tif', view_two_other_direction(out))
    else:
        tiff.imwrite(destination + '/patch.tif', np.transpose(patch, (1, 0, 2)))
        tiff.imwrite(destination + '/xy.tif', np.transpose(out, (1, 0, 2)))

    # assembly test
    if kwargs['assemble']:
        dz, dx, dy = kwargs['assemble_params']['dx_shape']
        #sz, sx, sy = kwargs['assemble_params']['sx_shape']
        w = get_weight(kwargs['assemble_params']['weight_shape'], method='cross', C=kwargs['assemble_params']['C'])
        zrange = range(*kwargs['assemble_params']['zrange'])
        xrange = range(*kwargs['assemble_params']['xrange'])
        yrange = range(*kwargs['assemble_params']['yrange'])

        recreate_volume_folder(destination + '/cycout/')  # DELETE and recreate the folder
        test_over_volumne(kwargs, dx, dy, dz, zrange=zrange, xrange=xrange, yrange=yrange,
                          destination=destination + '/cycout/')

        assemble_microscopy_volumne(kwargs, w, zrange=zrange, xrange=xrange, yrange=yrange,
                                    source=destination + '/cycout/xy/')




