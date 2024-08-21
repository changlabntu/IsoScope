import glob, os
import subprocess
import sys
from utils.calculate_kid import load_images_from_folder, calculate_metrics
import pandas as pd

prj_dirs = sorted(glob.glob('/media/ExtHDD01/logs/womac4/IsoScopeXX/unet/*/checkpoints/net_g_model_epoch_100.pth'))[::-1]
prj_dirs = [x.split('womac4')[1].split('checkpoints')[0] for x in prj_dirs]

#prj_dirs = ['/IsoScopeXX/unet/netnomc/']
prj_dirs = ['/IsoScopeXXnoup/aex2/']
epoch_num = [100]#range(60, 401, 20)
irange = "0,5"
eval = False
to_upsample = False
suffix = "a3d/"


# Name of your script in the test folder
script_name = "test_IsoLesionX.py"
df = pd.DataFrame()

# Loop through each project directory
for prj in prj_dirs:
    for epoch in epoch_num:
    # Construct the command
        command = [sys.executable, "-W", "ignore", "-m", f"test.{script_name[:-3]}",
                   "--prj", prj, "--epoch", str(epoch), "--irange", irange, "--eval", str(eval),
                   "--to_upsample", str(to_upsample), "--suffix", suffix + str(epoch).zfill(3)]

        # Run the command
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running with project directory: {prj}")
            print(f"Error: {e}")

        folder1 = "/media/ExtHDD01/oai_diffusion_interpolated/original/expanded3d/xya2d"
        folder2 = "/media/ExtHDD01/oai_diffusion_interpolated/test/expanded3d/zxa3d"
        kid = calculate_metrics('kid', folder1, folder2, irange=(0, 1840, 512)).cpu().item()
        fid = calculate_metrics('fid', folder1, folder2, irange=(0, 1840, 512)).cpu().item()

        data = {'prj': prj, 'epoch': epoch, 'kid': kid, 'fid': fid}

        df = pd.concat([df, pd.DataFrame([data])])
        df.to_csv('temp.csv')
        print("----------------------------------------")


print("All runs completed.")