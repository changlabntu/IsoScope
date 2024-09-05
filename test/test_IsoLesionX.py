import torch
import tifffile as tiff
import os, glob, sys
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import skimage.measure
import yaml


def old_function():
    if 0:
        reslice_3d_to_2d_for_visualize(destination=root + our_dir + 'expanded3d/',
                                       subjects=sorted(glob.glob(root + 'original/a2d/' + '*'))[:],
                                       suffix='a2d/', padding=False, upsample=8, trd=(None, 800))

    if 0:
        source = root + 'original/addpm2d0506/'
        IsoLesion_interpolate(destination=root + our_dir + 'addpm3d/',
                              subjects=sorted(glob.glob(source + '*'))[:],
                              net=net, to_upsample=to_upsample, padding=False, trd=(None, 800))
        reslice_3d_to_2d_for_visualize(destination=root + our_dir + 'expanded3d/',
                                       subjects=sorted(glob.glob(root + our_dir + 'addpm3d/' + '*'))[:],
                                       suffix='addpm3d/', padding=False, trd=(-1, 1))

    #####################
    # Calculate difference
    #####################
    if 0:
        x_list = sorted(glob.glob(root + 'original/a2d/*'))
        y_list = sorted(glob.glob(root + 'original/addpm2d0506/*'))
        destination = 'difference2d/'
        os.makedirs(root + our_dir + destination, exist_ok=True)
        calculate_difference(x_list, y_list, destination=destination)
    if 0:
        x_list = sorted(glob.glob(root + our_dir + 'a3d/*'))
        y_list = sorted(glob.glob(root + our_dir + 'addpm3d/*'))
        mask_list = sorted(glob.glob(root + our_dir + 'difference2d/*'))
        destination = 'difference3d/'
        os.makedirs(root + our_dir + destination, exist_ok=True)
        calculate_difference(x_list, y_list, destination=destination, mask_list=mask_list)

    if 0:
        #reslice_3d_to_2d_for_visualize(destination=root + our_dir + 'expanded3d/',
        #                               subjects=sorted(glob.glob(root + our_dir + 'difference2d/' + '*'))[:],
        #                               suffix='difference2d/', upsample=8, trd=None)
        reslice_3d_to_2d_for_visualize(destination=root + our_dir + 'expanded3d/',
                                       subjects=sorted(glob.glob(root + our_dir + 'difference3d/' + '*'))[:],
                                       suffix='difference3d/', trd=None)
        #reslice_3d_to_2d_for_visualize(destination=root + our_dir + 'expanded3d/', suffix='a2d/', upsample=8)
        #reslice_3d_to_2d_for_visualize(destination=root + our_dir + 'expanded3d/', suffix='a3d/')

        #reslice_3d_to_2d_for_visualize(destination=root + our_dir + 'expanded3d/', suffix='addpm2d/', upsample=8)
        #reslice_3d_to_2d_for_visualize(destination=root + our_dir + 'expanded3d/', suffix='addpm3d/')

    if 0:
        #seg = torch.load('/home/ghc/Dropbox/TheSource/scripts/ContrastiveDiffusion/submodels/atten_0706.pth').eval()
        #seg = torch.load('/home/ghc/Dropbox/TheSource/scripts/ContrastiveDiffusion/submodels/80.pth').eval()
        seg = torch.load('/home/ghc/Dropbox/TheSource/scripts/ContrastiveDiffusion/submodels/dual_atten_0706.pth')#.eval()
        #seg_bone(root + 'a2d/', root + 'a2d_bone/')
        #seg_bone(root + 'addpm2d/', root + 'addpm2d_bone/')
        seg_bone(root + 'test/a3d/', root + 'test/a3d_bone/')

    #y_list = sorted(glob.glob(root + 'a2d_bone/*'))
    #x_list = sorted(glob.glob(root + 'addpm2d_bone/*'))
    #os.makedirs(root + 'difference2d_bone/', exist_ok=True)
    #calculate_difference(x_list, y_list, destination='difference2d_bone/')

    if 0:
        root = '/media/ExtHDD01/oai_diffusion_interpolated/new/expanded3d/'
        xlist = sorted(glob.glob(root + 'zxdifference3d/*'))
        ylist = sorted(glob.glob(root + 'zxdifference2d/*'))

        for (x, y) in zip(xlist, ylist):

            x0 = tiff.imread(x)
            y0 = tiff.imread(y)
            print(x0.min(), x0.max(), y0.min(), y0.max())
            x0 = x0 / 255
            y0 = (y0 > 0) / 1
            mul = np.multiply(x0, y0)
            mul = (mul * 255).astype(np.uint8)
            tiff.imwrite(x.replace('zxdifference3d', 'mul'), mul)


def seg_bone(source, destination, filter_area=False):
    os.makedirs(destination, exist_ok=True)
    flist = [x.split('/')[-1] for x in sorted(glob.glob(source + '*'))]

    #x = tiff.imread('/media/ExtHDD01/oai_diffusion_interpolated/DshareZngf48mc_0504/a2d/9000099_03.tif')
    for f in flist[:1]:
        print(f)
        x = tiff.imread(source + f)
        x = (x - x.min()) / (x.max() - x.min())
        x = (x - 0.5) / 0.5
        x = torch.from_numpy(x).unsqueeze(1).float()

        # padding
        x_pad = -1 * torch.ones((4, 1, x.shape[2], x.shape[3]))
        x = torch.cat((x_pad, x, x_pad), 0)

        x = x.permute(3, 1, 0, 2)

        bone = []
        for z in range(x.shape[0]):
            out_all = []
            for mc in range(3):
                slice = x[z:z+1, :, :, :]
                #slice = slice / slice.max()
                #slice = (slice - 0.5) / 0.5
                out = seg(slice.cuda()).detach().cpu()
                out = nn.Softmax(dim=1)(out)
                out = torch.sum(out[:, 1:, :, :], 1).squeeze().numpy()
                #out = np.sum(out[:, 1:, :, :], 1)
                #out = torch.argmax(out, 1).squeeze()
                #out = (out > 0).numpy().astype(np.uint8)
                out_all.append(out)
            out = np.mean(np.stack(out_all, 0), 0)
            bone.append(out)
        bone = np.stack(bone, axis=0)

        bone = np.transpose(bone, (1, 2, 0))

        if filter_area:
            bone = filter_out_secondary_areas(bone).astype(np.uint8)
        tiff.imwrite(destination + f, bone)


def filter_out_secondary_areas(x):
    x = skimage.measure.label(bone)
    x0 = 0 * x
    areas = np.bincount(x.flatten())
    # top 3 areas:
    top2 = np.argsort(areas)[::-1][1:3]
    for t in top2:
        x0 += (x == t)
    return x0


def remove_last_after_underline(s):
    return s[:s.rfind('_')]


def IsoLesion_interpolate(destination, subjects, net, to_upsample=False, mirror_padding=False, trd=None, z_pad=False):
    os.makedirs(destination, exist_ok=True)
    for i in tqdm(range(len(subjects))):
        x0 = tiff.imread(subjects[i])
        filename = subjects[i].split('/')[-1].split('.')[0]

        x0 = norm_11(x0, trd)
        x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float().permute(0, 1, 3, 4, 2)  # (B, C, H, W, D)

        if mirror_padding > 0:
            #if to_upsample:
            padL = x0[:, :, :, :, :mirror_padding]
            padR = x0[:, :, :, :, -mirror_padding:]
            #else:
            #    padL = x0[:, :, :, :, :mirror_padding * 8]
            #    padR = x0[:, :, :, :, -mirror_padding * 8:]
            x0 = torch.cat((torch.flip(padL, [4]), x0, torch.flip(padR, [4])), 4)

        if to_upsample:
            upsample = torch.nn.Upsample(size=(x0.shape[2], x0.shape[3], x0.shape[4] * 8), mode='trilinear')
            rand_init = np.random.randint(8)
            print('rand_init', rand_init)
            x0 = upsample(x0)
            x0 = x0[:, :, :, :, rand_init::8]
            x0 = upsample(x0)

        all = []
        for _ in range(args.num_mc):
            stack_y = []

            if z_pad:
                # random shiftting
                shift_L = np.random.randint(0, 16)  # z-direction random padding
                shift_R = 16 - shift_L
                shiftL = x0.mean() * torch.ones((x0.shape[0], x0.shape[1], x0.shape[2], x0.shape[3], shift_L))
                shiftR = x0.mean() * torch.ones((x0.shape[0], x0.shape[1], x0.shape[2], x0.shape[3], shift_R))
                x0pad = torch.cat((shiftL, x0, shiftR), 4)
            else:
                x0pad = x0

            if args.stack_direction == 'y':
                for iy in [0, 128, 256]:
                    out = test_once(x0pad[:, :, :, iy:iy+128, :], net)

                    stack_y.append(out)
                combine = np.concatenate(stack_y, 2)
            elif args.stack_direction == 'x':
                for ix in [0, 128, 256]:
                    out = test_once(x0pad[:, :, ix:ix+128, :, :], net)
                    stack_y.append(out)
                combine = np.concatenate(stack_y, 1)
            else:
                out = test_once(x0pad, net)
                combine = out

            # remove shifting
            if z_pad:
                if to_upsample:
                    combine = combine[shift_L:combine.shape[0] - shift_R, :, :]
                else:
                    combine = combine[shift_L*8:combine.shape[0] - shift_R*8, :, :]
            # remove padding
            if mirror_padding > 0:
                combine = combine[mirror_padding*8:-mirror_padding*8, :, :]

            all.append(combine)
        combine = np.stack(all, 3)
        combine = np.mean(combine, 3)

        tiff.imwrite(destination + filename, combine)


def test_once(x0, net):
    if args.gpu:
        x0 = x0.cuda()
    out_all = net(x0)['out0']
    out_all = out_all.detach().cpu()
    out_all = out_all[0, 0, :, :, :].numpy()
    out_all = np.transpose(out_all, (2, 0, 1))
    return out_all


def calculate_difference(x_list, y_list, destination, mask_list=None):

    for i in range(len(x_list)):
        x = tiff.imread(x_list[i])
        y = tiff.imread(y_list[i])

        #x = x / x.max()
        #y = y / y.max()

        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())
        #for z in range(x.shape[0]):
        #    x[z, :, :] = (x[z, :, :] - x[z, :, :].min()) / (x[z, :, :].max() - x[z, :, :].min())
        #    y[z, :, :] = (y[z, :, :] - y[z, :, :].min()) / (y[z, :, :].max() - y[z, :, :].min())

        if mask_list is not None:
            m = tiff.imread(mask_list[i])
            m = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).float()
            m = torch.nn.functional.interpolate(m, scale_factor=(8, 1, 1), mode='trilinear', align_corners=False)
            m = m.squeeze().numpy()

        difference = x - y
        difference[difference < 0] = 0
        difference = difference / 1

        if mask_list is not None:
            m = (m > 60) / 1
            difference = np.multiply(difference, m)

        difference = (difference * 255).astype(np.uint8)
        tiff.imwrite(root + our_dir + destination + x_list[i].split('/')[-1], difference)


def save_a_to_2d(source, subjects, destination):
    for i in tqdm(range(len(subjects))):
        filename = subjects[i]
        tif_list = sorted(glob.glob(source + subjects[i] + '*.tif'))
        x0 = np.stack([tiff.imread(x) for x in tif_list], 0)
        tiff.imwrite(destination + filename + '.tif', x0)


def get_subjects_from_list_of_2d_tifs(source):
    l = sorted(glob.glob(source + '*'))
    subjects = sorted(list(set([remove_last_after_underline(x.split('/')[-1]) for x in l])))
    return subjects


def to_8bit(x):
    x = (x - x.min()) / (x.max() - x.min())
    x = (x * 255).astype(np.uint8)
    return x


def norm_11(x, trd=None):
    if trd is not None:
        x[x > trd[1]] = trd[1]
        if trd[0] is not None:
            x[x < trd[0]] = trd[0]
            x = (x - trd[0]) / (trd[1] - trd[0])
        else:
            x = x / trd[1]
    else:
        x = x / x.max()
    x = (x - 0.5) * 2
    return x


def reslice_3d_to_2d_for_visualize(destination, subjects, suffix, png=False, upsample=None, fill_blank=False, trd=None):
    os.makedirs(destination + 'zy' + suffix, exist_ok=True)
    os.makedirs(destination + 'zx' + suffix, exist_ok=True)
    os.makedirs(destination + 'xy' + suffix, exist_ok=True)

    for i in tqdm(range(len(subjects))):
        x0 = tiff.imread(subjects[i])  # (Z, X, Y)
        filename = subjects[i].split('/')[-1].split('.')[0]
        x0 = norm_11(x0, trd)
        x0 = x0.astype(np.float32)
        if upsample is not None:
            x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()
            x0 = torch.nn.functional.interpolate(x0, scale_factor=(upsample, 1, 1), mode='nearest')#, align_corners=False)
            x0 = x0.squeeze().numpy()

        if fill_blank:
            dz = (384 - x0.shape[0]) // 2
            pad = -1 * np.ones((dz, x0.shape[1], x0.shape[2]))
            x0 = np.concatenate((pad, x0, pad), 0).astype(np.float32)

        # reslice
        for x in range(x0.shape[1]):
            if not png:
                tiff.imwrite(destination + 'zy' + suffix + filename + '_' + str(x).zfill(3) + '.tif', np.transpose(x0[:, x, :], (1, 0)))
            else:
                out = Image.fromarray(to_8bit(np.transpose(x0[:, x, :] , (1, 0))))
                out.save(destination + 'zy' + suffix + filename + '_' + str(x).zfill(3) + '.png')
        for y in range(x0.shape[2]):
            if not png:
                tiff.imwrite(destination + 'zx' + suffix + filename + '_' + str(y).zfill(3) + '.tif', np.transpose(x0[:, :, y], (1, 0)))
            else:
                out = Image.fromarray(to_8bit(np.transpose(x0[:, :, y], (1, 0))))
                out.save(destination + 'zx' + suffix + filename + '_' + str(y).zfill(3) + '.png')
        for z in range(x0.shape[0]):
            if not png:
                tiff.imwrite(destination + 'xy' + suffix + filename + '_' + str(z).zfill(3) + '.tif', x0[z, :, :])
            else:
                out = Image.fromarray(to_8bit(x0[z, :, :]))
                out.save(destination + 'xy' + suffix + filename + '_' + str(z).zfill(3) + '.png')


def read_out_2d(ddpm_source='/media/ExtHDD01/oai_diffusion_interpolated/original/diff0506/'):
    # Copy ddpm to 2d output
    destination = root + 'original/addpm2d0506/'
    save_a_to_2d(source=ddpm_source, subjects=get_subjects_from_list_of_2d_tifs(ddpm_source), destination=destination)

    # Copy a to 2d output
    raw_source = '/media/ExtHDD01/Dataset/paired_images/womac4/full/b/'
    #source = '/media/ghc/GHc_data2/OAI_extracted/womac4min0/Processed/anorm/'
    destination = root + 'original/b2d/'
    os.makedirs(destination, exist_ok=True)
    save_a_to_2d(source=raw_source, subjects=get_subjects_from_list_of_2d_tifs(raw_source)[0:500:5], destination=destination)


def get_model():
    if args.epoch == 'last':
        last_epoch = \
        sorted(glob.glob('/media/ExtHDD01/logs/womac4' + args.prj + 'checkpoints/net_g_model_epoch_*.pth'))[-1]
        print(last_epoch)
        net = torch.load(last_epoch, map_location=torch.device('cpu'))
    else:
        model_name = '/media/ExtHDD01/logs/womac4' + args.prj + 'checkpoints/net_g_model_epoch_' + str(
            args.epoch) + '.pth'
        print(model_name)
        net = torch.load(model_name, map_location=torch.device('cpu'))
    if args.eval:
        net = net.eval()

    if args.gpu:
        net = net.cuda()

    return net




if __name__ == "__main__":

    import yaml
    import sys
    import argparse


    def load_config(yaml_file='test/config_womac4.yaml'):
        def parse_args(args):
            parser = argparse.ArgumentParser(description='Process some parameters.')
            parser.add_argument('--config', type=str, default='default',
                                help='Configuration to use from YAML file')
            parser.add_argument('--prj', type=str, default='/IsoScopeXX/cyc0lb1skip4ndf32/',
                                help='Project directory')
            parser.add_argument('--epoch', type=str, default='last',
                                help='Epoch number or "last"')
            parser.add_argument('--to_upsample', type=lambda x: (str(x).lower() == 'true'), default=True)
            parser.add_argument('--raw_trd', type=lambda x: tuple(map(int, x.split(','))),
                                default='0,800', help='Raw threshold (format: min,max)')
            parser.add_argument('--raw_source', type=str, default='a2d/',
                                help='Raw source directory')
            parser.add_argument('--stack_direction', type=str, default=None,
                                help='Stack direction')
            parser.add_argument('--gpu', action='store_true')
            parser.add_argument('--num_mc', type=int, default=1,
                                help='Number of Monte Carlo simulations')
            parser.add_argument('--eval', type=lambda x: (str(x).lower() == 'true'), default=True)
            parser.add_argument('--mirror_padding', type=int, default=1,
                                help='Mirror padding size')
            parser.add_argument('--z_pad', type=lambda x: (str(x).lower() == 'true'), default=True)
            parser.add_argument('--irange', type=lambda x: tuple(map(int, x.split(','))),
                                default='0,5', help='testing subject range')
            parser.add_argument('--suffix', type=str, default='a3d/')
            parser.add_argument('--out_dir', type=str, default='test/')

            return parser.parse_args(args)

        # Parse arguments
        args = parse_args(sys.argv[1:] if '__file__' in globals() else [])

        # Load the YAML configuration
        with open(yaml_file, 'r') as file:
            configurations = yaml.safe_load(file)

        # Get the selected configuration
        selected_config = configurations.get(args.config, {})

        # Update args with values from YAML if they exist
        vars(args).update({k: tuple(v) if k == 'raw_trd' else v
                           for k, v in selected_config.items()
                           if hasattr(args, k)})

        return args


    # Usage
    args = load_config()
    print(args)

    # Load model
    net = get_model()

    # output root
    root = '/media/ExtHDD01/oai_diffusion_interpolated/'
    our_dir = args.out_dir
    os.makedirs(root + our_dir, exist_ok=True)

    source = '/media/ExtHDD01/oai_diffusion_interpolated/original/' + args.raw_source
    IsoLesion_interpolate(destination=root + our_dir + 'a3d/',
                          subjects=sorted(glob.glob(source + '*'))[args.irange[0]:args.irange[1]],
                          net=net, to_upsample=args.to_upsample, mirror_padding=args.mirror_padding,
                          trd=args.raw_trd, z_pad=args.z_pad)
    reslice_3d_to_2d_for_visualize(destination=root + our_dir + 'expanded3d/',
                                   subjects=sorted(glob.glob(root + our_dir + 'a3d/' + '*'))[args.irange[0]:args.irange[1]],
                                   # subjects=sorted(glob.glob(root + 'redo500/' + 'a3d/' + '*'))[:],
                                   suffix=args.suffix, fill_blank=False, trd=(-1, 1))
