

# a6k
# womac4 1331
#NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn cyc_imorphics --prj IsoMRIclean/gd1331fix/ --models IsoMRIclean --cropz 16 --cropsize 128 --env runpod --adv 1 --rotate --ngf 32 --direction a --netG edclean --dataset womac4 --n_epochs 2000 --lr_policy cosine --mc --uprate 8 --lamb 1 --skipl1 1 --ndf 32 --lr 0.0004 --nm 11

#NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn cyc_imorphics --prj IsoMRIclean/gd1331fix/dis0 --models IsoMRIclean --cropz 16 --cropsize 128 --env brcb --adv 1 --rotate --ngf 32 --direction ap --nm 11 --netG edclean --dataset womac4 --n_epochs 2000 --lr_policy cosine --mc --uprate 8 --lamb 1 --nocut --skipl1 4 --ndf 32 --fix_dis

# BraTSReg
#NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn cyc_imorphics --prj IsoMRIclean/gd1331dsp2/ --models IsoMRIclean --cropz 128 --cropsize 128 --env runpod --adv 1 --rotate --ngf 32 --direction t1norm2  --netG edclean --dataset BraTSReg --n_epochs 2000 --lr_policy cosine --mc --uprate 1 --dsp 2 --lamb 1 --nocut --skipl1 4 --ndf 32 --lr 0.0004 --nm 11 --epoch_save 100

#NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn cyc_imorphics --prj IsoMRIclean/gd1331fix/dis0 --models IsoMRIclean --cropz 128 --cropsize 128 --env runpod --adv 1 --rotate --ngf 32 --direction t1norm2  --netG edclean --dataset BraTSReg --n_epochs 2000 --lr_policy cosine --mc --uprate 1 --dsp 2 --lamb 1 --nocut --skipl1 4 --ndf 32 --lr 0.0004 --nm 11 --epoch_save 100 --fix_dis

# womac4 1331 dis0
#NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn cyc_imorphics --prj IsoMRIclean/gd1331fix/dis0 --models IsoMRIclean --cropz 16 --cropsize 128 --env brcb --adv 1 --rotate --ngf 32 --direction ap --nm 11 --netG edclean --dataset womac4 --n_epochs 2000 --lr_policy cosine --mc --uprate 8 --lamb 1 --nocut --skipl1 4 --ndf 32 --fix_dis

#NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn cyc_imorphics --prj IsoMRIclean/gd1331a6k/nocyc --models IsoMRIclean --cropz 16 --cropsize 128 --env a6k --adv 1 --rotate --ngf 32 --direction ap --nm 11 --netG edclean --dataset womac4 --n_epochs 2000 --lr_policy cosine --mc --uprate 8 --lamb 1 --nocut --skipl1 4 --ndf 32 --nocyc


#python train.py --jsn cyc_imorphics --prj IsoLambda/2l1maxskip1cosinelr005 --models IsoMRIcleandis2 --cropz 128 --cropsize 128 --env brcb --adv 1 --rotate --ngf 32 --direction t2norm --nm 11 --netG edclean --dataset BraTSReg --n_epochs 10000 --lr_policy cosine --uprate 1 --lamb 1 --lambB 1 --nocyc --lr 0.00005 --nocut --skipl1 1 --l1how max --ndf 32 --dataset_mode PairedSlices3D --epoch_save 100

CUDA_VISIBLE_DEVICES=0 python train.py --jsn womac3 --prj 3D/test4fixVgg1/  --models descar4fix --netG dsmc --netD bpatch_16 --dataset womac4