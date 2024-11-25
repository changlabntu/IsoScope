

# a6k
# womac4 1331
#NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn cyc_imorphics --prj IsoMRIclean/gd1331fix/ --models IsoMRIclean --cropz 16 --cropsize 128 --env runpod --adv 1 --rotate --ngf 32 --direction a --netG edclean --dataset womac4 --n_epochs 2000 --lr_policy cosine --mc --uprate 8 --lamb 1 --skipl1 1 --ndf 32 --lr 0.0004 --nm 11

#NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn cyc_imorphics --prj IsoMRIclean/gd1331fix/dis0 --models IsoMRIclean --cropz 16 --cropsize 128 --env brcb --adv 1 --rotate --ngf 32 --direction ap --nm 11 --netG edclean --dataset womac4 --n_epochs 2000 --lr_policy cosine --mc --uprate 8 --lamb 1 --nocut --skipl1 4 --ndf 32 --fix_dis

# BraTSReg
#NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn cyc_imorphics --prj IsoMRIclean/gd1331dsp2/ --models IsoMRIclean --cropz 128 --cropsize 128 --env runpod --adv 1 --rotate --ngf 32 --direction t1norm2  --netG edclean --dataset BraTSReg --n_epochs 2000 --lr_policy cosine --mc --uprate 1 --dsp 2 --lamb 1 --nocut --skipl1 4 --ndf 32 --lr 0.0004 --nm 11 --epoch_save 100

#NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn cyc_imorphics --prj IsoMRIclean/gd1331fix/dis0 --models IsoMRIclean --cropz 128 --cropsize 128 --env runpod --adv 1 --rotate --ngf 32 --direction t1norm2  --netG edclean --dataset BraTSReg --n_epochs 2000 --lr_policy cosine --mc --uprate 1 --dsp 2 --lamb 1 --nocut --skipl1 4 --ndf 32 --lr 0.0004 --nm 11 --epoch_save 100 --fix_dis

# womac4 1331 dis0
#NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn cyc_imorphics --prj IsoMRIclean/gd1331fix/dis0 --models IsoMRIclean --cropz 16 --cropsize 128 --env brcb --adv 1 --rotate --ngf 32 --direction ap --nm 11 --netG edclean --dataset womac4 --n_epochs 2000 --lr_policy cosine --mc --uprate 8 --lamb 1 --nocut --skipl1 4 --ndf 32 --fix_dis

NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn cyc_imorphics --prj IsoMRIclean/gd1331fix/dis0B --models IsoMRIclean --cropsize 128 --env runpod --adv 1 --rotate --ngf 32 --direction t1norm2 --nm 11 --netG edclean --dataset BraTSReg --n_epochs 2000 --lr_policy cosine --mc --cropz 128 --dsp 2 --uprate 2 --lamb 1 --nocut --skipl1 4 --ndf 32 --fix_dis