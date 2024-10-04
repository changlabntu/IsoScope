#python train.py --jsn cyc_imorphics --prj IsoScopeXXldm/aex2ed023e --models IsoScopeXXcyc0cutnoupE2ldm --norm group --env brcb --cropz 16 --cropsize 128  --adv 1 --direction a --rotate --ngf 32  --nm 11 --netG ldmaex2 --dataset womac4 --n_epochs 2000 --lr_policy cosine --mc --uprate 8 --lamb 1 --nocut --skipl1 4 --ndf 64

#python train.py --jsn cyc_imorphics --prj IsoScopeXX/redounetx2 --models IsoScopeXXcyc0cut --cropz 16 --cropsize 128 --env runpod --adv 1 --rotate --ngf 32 --direction ap --nm 11 --netG ed023dunetx2 --dataset womac4 --n_epochs 2000 --lr_policy cosine --mc --uprate 8 --lamb 1 --nocut --skipl1 4 --ndf 32

#python train.py --jsn cyc_imorphics --prj IsoScopeXX/redounetlsgan --models IsoScopeXXcyc0cut --cropz 16 --cropsize 128 --env runpod --adv 1 --rotate --ngf 32 --direction ap --nm 11 --netG ed023dunet --dataset womac4 --n_epochs 2000 --lr_policy cosine --mc --uprate 8 --lamb 1 --nocut --skipl1 4 --ndf 32 --gan_mode lsgan



## IsoMRI unet3D
#NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn cyc_imorphics --prj IsoMRI/unet3d/l1dsp_0_10 --models IsoMRI --l1how dsp --lamb 0 --lambB 10 --cropz 16 --cropsize 128 --env runpod --adv 1 --rotate --ngf 32 --direction ap --nm 11 --netG ed023eunet3d --dataset womac4 --n_epochs 2000 --lr_policy cosine --mc --uprate 8 --nocut --skipl1 0 --ndf 3

## womac4 ae
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn cyc_imorphics --prj ae/cyc0 --ldmyaml ldmaex2 --netG none --lamb 1 --lr 8.64e-4 --models ae0iso0cyc --nocut --cropz 16 --cropsize 128 --env brcb --adv 1 --rotate --ngf 32 --ndf 32 --direction a --nm 11 --dataset womac4 --epoch_save 20 --n_epochs 5000 --lr_policy cosine

CUDA_VISIBLE_DEVICES=1 NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn cyc_imorphics --prj IsoScopeXX/unet3d/unet3dres --models IsoScopeXXcyc0cut --cropz 16 --cropsize 128 --env t09 --adv 1 --rotate --ngf 32 --direction a --nm 11 --netG ed023eunet3dres --dataset womac4 --n_epochs 2000 --lr_policy cosine --mc --uprate 8 --lamb 1 --nocut --skipl1 4 --ndf 32
