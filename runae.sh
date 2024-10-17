#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train.py --jsn cyc_imorphics --prj ae/ae0discstart0 --save_d --lr 8.64e-4 --models ae0 --cropz 16 --cropsize 128 --env brcb --adv 1 --rotate --ngf 32 --direction xyori0828 --nm 00  --dataset Fly0B --epoch_save 100 --n_epochs 5000 --lr_policy cosine

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --jsn cyc_imorphics --final tanh --trd 800 --kl_weight 0.0001 --prj vae/2 --save_d --models vae --cropz 16 --cropsize 128 --env a6k --adv 1 --rotate --ngf 32 --direction b --nm 11 --netG ed0230  --dataset womac4 --epoch_save 20 --n_epochs 5000 --lr_policy cosine

# womac4
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn cyc_imorphics --prj ae/onlyae0 --ldmyaml ldmaex2 --netG none --lamb 10 --save_d --lr 8.64e-4 --models ae0iso0onlyae --cropz 0 --cropsize 128 --env brcb --adv 1 --rotate --ngf 32 --direction bclean --nm 11 --dataset womac4 --epoch_save 20 --n_epochs 5000 --lr_policy cosine

# Fly0B
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train.py --jsn cyc_imorphics --prj ae/iso0_ldmaex2_lb10 --ldmyaml ldmaex2 --netG none --lamb 10 --save_d --lr 8.64e-4 --models ae0iso0 --cropz 16 --cropsize 128 --env brcb --adv 1 --rotate --ngf 32 --direction xyori0828 --nm 00 --dataset Fly0B --epoch_save 100 --n_epochs 5000 --lr_policy cosine

# DPM4X
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train.py --jsn cyc_imorphics --prj ae/iso0_ldmaex2_lb10 --ldmyaml ldmaex2 --netG none --lamb 10 --save_d --lr 8.64e-4 --models ae0iso0 --cropz 16 --cropsize 128 --env brcb --adv 1 --rotate --ngf 32 --direction xyoriexp0 --nm 00 --dataset DPM4X --epoch_save 100 --n_epochs 5000 --lr_policy cosine

# DPM4X tc
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn cyc_imorphics --prj ae/iso0_ldmaex2_lb10_tc_oril1 --ldmyaml ldmaex2 --netG none --lamb 10 --lr 8.64e-4 --models ae0iso0tc --tc --cropz 16 --cropsize 128 --env brcb --adv 1 --rotate --ngf 32 --direction oripatch_ft0patch --nm 00 --dataset DPM4X --epoch_save 100 --n_epochs 5000 --lr_policy cosine

#CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn cyc_imorphics --prj ae/iso0_ldmaex2_lb10_tc_oril1_hbranchz --hbranch z --ldmyaml ldmaex2 --netG none --lamb 10 --lr 8.64e-4 --models ae0iso0tc --tc --cropz 16 --cropsize 128 --env brcb --adv 1 --rotate --ngf 32 --direction oripatch_ft0patch --nm 00 --dataset DPM4X --epoch_save 100 --n_epochs 5000 --lr_policy cosine

# DPM4X tc cut
#CUDA_VISIBLE_DEVICES=0,1,2,3 NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn cyc_imorphics --prj ae/cut/1 --hbranch z --ldmyaml ldmaex2 --netG none --lamb 10 --lr 0.0002 --models ae0iso0tccut --cropz 16 --cropsize 128 --env a6k --adv 1 --rotate --ngf 32 --direction xyori --nm 00 --dataset Fly0B --epoch_save 100 --n_epochs 5000 --lr_policy cosine --mc

#Fly0B CUT 1
#CUDA_VISIBLE_DEVICES=0,1,2,3 NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn cyc_imorphics --prj ae/cut/1 --hbranch z --ldmyaml ldmaex2 --netG none --lamb 10 --lr 0.0002 --models ae0iso0tccut --tc --cropz 16 --cropsize 128 --env a6k --adv 1 --rotate --ngf 32 --direction oripatch_ft0patch --nm 00 --dataset DPM4X --epoch_save 100 --n_epochs 5000 --lr_policy cosine --mc
CUDA_VISIBLE_DEVICES=0,1,2,3 NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn cyc_imorphics --prj ae/cut/lamb0 --lamb 0 --lr 0.0002 --hbranch z --ldmyaml ldmaex2 --netG none --models ae0iso0tccut --tc --cropz 16 --cropsize 128 --env a6k --adv 1 --rotate --ngf 32 --direction oripatch_ft0patch --nm 00 --dataset DPM4X --epoch_save 100 --n_epochs 5000 --lr_policy cosine --mc