#python train.py --jsn cyc_imorphics --prj IsoScopeXXldm/aex2ed023e --models IsoScopeXXcyc0cutnoupE2ldm --norm group --env brcb --cropz 16 --cropsize 128  --adv 1 --direction a --rotate --ngf 32  --nm 11 --netG ldmaex2 --dataset womac4 --n_epochs 2000 --lr_policy cosine --mc --uprate 8 --lamb 1 --nocut --skipl1 4 --ndf 64

# brcb

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train.py --jsn cyc_imorphics --prj IsoScopeXXldm/aex2ed023egroupnorm --models IsoScopeXXcyc0cutnoupE2ldm --norm group --env brcb --cropz 16 --cropsize 128  --adv 1 --direction a --rotate --ngf 32  --nm 11 --netG ldmaex2 --dataset womac4 --n_epochs 2000 --lr_policy cosine --mc --uprate 8 --lamb 1 --nocut --skipl1 4 --ndf 64