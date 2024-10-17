import glob, os
import subprocess
import sys
from utils.calculate_kid import load_images_from_folder, calculate_metrics
import pandas as pd
from utils.data_utils import imagesc
import tifffile as tiff
import numpy as np


if 0:
    prj_dirs = sorted(glob.glob('/media/ExtHDD01/logs/womac4/IsoScopeXX/unet/*/checkpoints/net_g_model_epoch_100.pth'))[::-1]
    prj_dirs = [x.split('womac4')[1].split('checkpoints')[0] for x in prj_dirs]

    prj_dirs = prj_dirs[-1:]

    epoch_num = ['last']
    to_upsample = True
    eval = True
    mirror_padding = 0
    z_pad = True
    num_mc = 1
elif 0:
    prj_dirs = ['/IsoScopeXX/cyc0lb1skip4ndf32nomc/']
    #prj_dirs = ['/IsoScopeXX/nomc/redo/']
    epoch_num = [300]
elif 0:
    prj_dirs = ['/IsoScopeXX/unet/redounet/']
    epoch_num = [100]
elif 1:  # BEST for now!
    prj_dirs = ['/IsoScopeXX/unet3d/unet3d/']
    epoch_num = [160]#[160]
elif 0:
    prj_dirs = ['/IsoScopeXX/unet3d/nomc/']
    epoch_num = [40]#[160]
elif 0:
    prj_dirs = ['/IsoScopeXX/unet3dres/lamb2lr05/']
    epoch_num = [120]  # [160]
elif 0:
    prj_dirs = ['/IsoMRI/unet3dres/0skip4/']
    epoch_num = [120]  # [160]
elif 0:
    prj_dirs = ['/IsoMRIclean/gd2332/']
    epoch_num = [100]  # [160]

eval = False

to_upsample = True
mirror_padding = 2
z_pad = True
num_mc = 1

root = '/media/ExtHDD01/oai_diffusion_interpolated/'
log_root = '/media/ExtHDD01/logs/'
#raw_source = "a2d/"
#suffix = "a3d/"
#raw_source = "addpm2d0506/"
#suffix = "addpm3d/"
raw_source = 'diffresult0921/vanilla/'
suffix = "vanilla/"
#['3D', 'dualE-SPADE', 'vanilla', 'dualE']

print_a2d = True
gpu = False
stack_direction = None
out_dir = None

irange = "0,10"


if eval:
    print('Using .eval()')
else:
    print('Using .train()')

#root = '/home/ubuntu/Data/oai_diffusion_interpolated/'
#log_root = '/home/ubuntu/Data/logs/'

if 0: # if test selected cases
    raw_source = 'compare/compare2/'
    mirror_padding = 4
    num_mc = 3
    stack_direction = None
    out_dir = 'selected/'
    suffix = 'a3d/'
    print_a2d = True


#prj_dirs = sorted(glob.glob('/media/ExtHDD01/logs/womac4/IsoScopeXX/unet3d/*/'))
#prj_dirs = [x.split('womac4')[1] for x in prj_dirs]
#epoch_num = [40, 60, 80, 100]

# Name of your script in the test folder
script_name = "test_IsoLesionX.py"
df = pd.DataFrame()

# Loop through each project directory
for prj in prj_dirs:
    for epoch in epoch_num:
    # Construct the command
        command = [sys.executable, "-W", "ignore", "-m", f"test.{script_name[:-3]}",
                   "--prj", prj, "--epoch", str(epoch), "--irange", irange, "--eval", str(eval),
                   "--to_upsample", str(to_upsample), "--suffix", suffix, "--mirror_padding", str(mirror_padding),
                   "--z_pad", str(z_pad), "--num_mc", str(num_mc), "--raw_source", raw_source,
                   "--root", root, "--log_root", log_root]#, "--raw_trd", str(raw_trd)]
        if print_a2d:
            command = command + ["--print_a2d", str(print_a2d)]
        if gpu:
            command = command + ["--gpu"]
        if stack_direction:
            command = command + ["--stack_direction", stack_direction]
        if out_dir:
            command = command + ["--out_dir", out_dir]

        # Run the command
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running with project directory: {prj}")
            print(f"Error: {e}")

        # load tif and slice 2d:
        if 0:
            tif3d = tiff.imread('/media/ExtHDD01/oai_diffusion_interpolated/test/a3d/9026695_00')
            slice = [np.transpose(tif3d[:, :, 136],(1, 0)), np.transpose(tif3d[:, :, 161],(1, 0)), np.transpose(tif3d[:, :, 268],(1, 0))]
            slice = np.concatenate(slice, axis=1)
            save_name = prj[1:-1].replace('/', '_') + '_' + str(epoch).zfill(4)
            tiff.imwrite(f'/media/ExtHDD01/oai_diffusion_interpolated/test/allout/{save_name}.tif', slice)


        if 0:
            folder1 = "/media/ExtHDD01/oai_diffusion_interpolated/original/expanded3d/xya2d"
            folder2 = "/media/ExtHDD01/oai_diffusion_interpolated/test/expanded3d/zxa3d"
            kid = calculate_metrics('kid', folder1, folder2, irange=(0, 1840, 512)).cpu().item()
            fid = calculate_metrics('fid', folder1, folder2, irange=(0, 1840, 512)).cpu().item()

            data = {'prj': prj, 'epoch': epoch, 'kid': kid, 'fid': fid}

            df = pd.concat([df, pd.DataFrame([data])])
            #df.to_csv('temp.csv')
            print(df)
            print("----------------------------------------")



def quick_rotate(xlist):
    for x in xlist:
        t = tiff.imread(x)
        t = np.transpose(t, (2, 1, 0))
        tiff.imwrite(x.replace('/xy/', '/zx/'), t)

quick_rotate(sorted(glob.glob('/media/ExtHDD01/oai_diffusion_interpolated/smore/xy/*')))