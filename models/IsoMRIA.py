from models.base import BaseModel
import copy
import torch
import numpy as np
import torch.nn as nn
from models.base import VGGLoss
from networks.networks_cut import Normalize, init_net, PatchNCELoss
from models.CUT import PatchSampleF3D


class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, eval_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, eval_loader, checkpoints)

        from networks.EncoderDecoder.edclean import Generator
        self.hparams.final = 'tanh'

        print('netG:  ' + self.hparams.netG)
        if self.hparams.netG.startswith('edclean'):
            self.net_g = Generator(n_channels=self.hparams.input_nc, out_channels=self.hparams.output_nc, nf=self.hparams.ngf,
                                   norm_type=self.hparams.norm, final=self.hparams.final, mc=self.hparams.mc,
                                   encode=self.hparams.netG[-2:], decode='3d')
        else:
            self.net_g, self.net_d = self.set_networks()

        _, self.net_d = self.set_networks()
        self.hparams.final = 'tanh'

        # save model names
        self.netg_names = {'net_g': 'net_g'}
        self.netd_names = {'net_d': 'net_d'}

        # Finally, initialize the optimizers and scheduler
        self.configure_optimizers()
        self.upsample = torch.nn.Upsample(size=(hparams.cropsize, hparams.cropsize,
                                                hparams.cropz // hparams.extradsp * hparams.uprate), mode='trilinear')

        # CUT NCE
        if not self.hparams.nocut:
            netF = PatchSampleF3D(use_mlp=self.hparams.use_mlp, init_type='normal', init_gain=0.02, gpu_ids=[],
                                nc=self.hparams.c_mlp)
            self.netF = init_net(netF, init_type='normal', init_gain=0.02, gpu_ids=[])
            feature_shapes = [x * self.hparams.ngf for x in [1, 2, 4, 8]]
            self.netF.create_mlp(feature_shapes)

            if self.hparams.fWhich == None:  # which layer of the feature map to be considered in CUT
                self.hparams.fWhich = [1 for i in range(len(feature_shapes))]

            print(self.hparams.fWhich)

            self.criterionNCE = []
            for nce_layer in range(4):  # self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt=hparams))  # .to(self.device))

            self.netg_names['netF'] = 'netF'

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # coefficient for the identify loss
        parser.add_argument("--extradsp", type=int, default=1)
        parser.add_argument("--l1how", type=str, default='dsp')
        parser.add_argument("--uprate", type=int, default=4)
        parser.add_argument("--skipl1", type=int, default=1)
        parser.add_argument("--nocut", action='store_true')
        parser.add_argument("--advscheme", type=str, default='OLD')
        # CUT
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--lbNCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--use_mlp', action='store_true')
        parser.add_argument("--c_mlp", dest='c_mlp', type=int, default=256, help='channel of mlp')
        parser.add_argument('--fWhich', nargs='+', help='which layers to have NCE loss', type=int, default=None)
        return parent_parser

    def generation(self, batch):
        if self.hparams.cropz > 0:
            z_init = np.random.randint(batch['img'][0].shape[4] - self.hparams.cropz)
            batch['img'][0] = batch['img'][0][:, :, :, :, z_init:z_init + self.hparams.cropz]
            # batch['img'][1] = batch['img'][1][:, :, :, :, z_init:z_init + self.hparams.cropz]

        # extra downsample
        if self.hparams.extradsp > 1:
            batch['img'][0] = batch['img'][0][:, :, :, :, ::self.hparams.extradsp]

        self.oriX = batch['img'][0]  # (B, C, X, Y, Z) # original
        #self.oriY = batch['img'][1]  # (B, C, X, Y, Z) # original

        # X-Y permute
        if np.random.randint(2) == 1:
            self.oriX = self.oriX.permute(0, 1, 3, 2, 4)

        self.Xup = self.upsample(self.oriX)  # (B, C, X, Y, Z)

        self.XupX = self.net_g(self.Xup)['out0']

    def get_xy_plane(self, x):  # (B, C, X, Y, Z)
        return x.permute(4, 1, 2, 3, 0)[::1, :, :, :, 0]  # (Z, C, X, Y, B)

    def adv_loss_six_way(self, x, net_d, truth):
        # x (B, C, X, Y, Z)

        loss = 0

        if self.hparams.advscheme == 'NEW':
            rint = np.random.randint(3)
            if rint == 0:
                zy = x.permute(2, 1, 4, 3, 0)[:, :, :, :, 0]  # (X, C, Z, Y, B)
                yz = x.permute(2, 1, 3, 4, 0)[:, :, :, :, 0]  # (X, C, Y, Z, B)
                loss += self.add_loss_adv(a=zy, net_d=net_d, truth=truth)  # (X, C, Z, Y)
                loss += self.add_loss_adv(a=yz, net_d=net_d, truth=truth)  # (X, C, Y, Z)
                loss += self.add_loss_adv(a=torch.flip(zy, [2]), net_d=net_d, truth=truth)
                loss += self.add_loss_adv(a=torch.flip(yz, [3]), net_d=net_d, truth=truth)

            if rint == 1:
                zx = x.permute(3, 1, 4, 2, 0)[:, :, :, :, 0]  # (Y, C, Z, X, B)
                xz = x.permute(3, 1, 2, 4, 0)[:, :, :, :, 0]  # (Y, C, X, Z, B)
                loss += self.add_loss_adv(a=zx, net_d=net_d, truth=truth)  # (Y, C, Z, X)
                loss += self.add_loss_adv(a=xz, net_d=net_d, truth=truth)  # (Y, C, X, Z)
                loss += self.add_loss_adv(a=torch.flip(zx, [2]), net_d=net_d, truth=truth)
                loss += self.add_loss_adv(a=torch.flip(xz, [3]), net_d=net_d, truth=truth)

            if rint == 2:
                xy = x.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]  # (Z, C, X, Y, B)
                yx = x.permute(4, 1, 3, 2, 0)[:, :, :, :, 0]  # (Z, C, Y, X, B)
                loss += 2 * self.add_loss_adv(a=xy, net_d=net_d, truth=truth)  # (Z, C, X, Y)
                loss += 2 * self.add_loss_adv(a=yx, net_d=net_d, truth=truth)  # (Z, C, Y, X)

            loss = loss / 4

        elif self.hparams.advscheme == 'OLD':
            loss += self.add_loss_adv(a=x.permute(2, 1, 4, 3, 0)[:, :, :, :, 0],  # (X, C, Z, Y)
                                      net_d=net_d, truth=truth)
            loss += self.add_loss_adv(a=x.permute(2, 1, 3, 4, 0)[:, :, :, :, 0],  # (X, C, Y, Z)
                                      net_d=net_d, truth=truth)
            loss += self.add_loss_adv(a=x.permute(3, 1, 4, 2, 0)[:, :, :, :, 0],  # (Y, C, Z, X)
                                      net_d=net_d, truth=truth)
            loss += self.add_loss_adv(a=x.permute(3, 1, 2, 4, 0)[:, :, :, :, 0],  # (Y, C, X, Z)
                                      net_d=net_d, truth=truth)
            loss += self.add_loss_adv(a=x.permute(4, 1, 2, 3, 0)[:, :, :, :, 0],  # (Z, C, X, Y)
                                      net_d=net_d, truth=truth)
            loss += self.add_loss_adv(a=x.permute(4, 1, 3, 2, 0)[:, :, :, :, 0],  # (Z, C, Y, X)
                                      net_d=net_d, truth=truth)
            loss = loss / 6

        return loss

    def backward_g(self):
        loss_g = 0
        loss_dict = {}

        axx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=True)
        loss_l1 = self.add_loss_l1(a=self.get_projection(self.XupX, depth=self.hparams.uprate
                                                                          * self.hparams.skipl1, how=self.hparams.l1how),
                                   b=self.oriX[:, :, :, :, ::self.hparams.skipl1])

        loss_dict['axx'] = axx
        loss_g += axx
        loss_dict['l1'] = loss_l1
        loss_g += loss_l1 * self.hparams.lamb

        if not self.hparams.nocut:
            # (X, XupX)
            #self.goutz = self.net_g(self.Xup, method='encode')
            feat_q = self.goutz
            feat_k = self.net_g(self.XupX, method='encode')

            feat_k_pool, sample_ids = self.netF(feat_k, self.hparams.num_patches,
                                                None)  # get source patches by random id
            feat_q_pool, _ = self.netF(feat_q, self.hparams.num_patches, sample_ids)  # use the ids for query target

            total_nce_loss = 0.0
            for f_q, f_k, crit, f_w in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.hparams.fWhich):
                loss = crit(f_q, f_k) * f_w
                total_nce_loss += loss.mean()
            loss_nce = total_nce_loss / 4
            loss_dict['nce'] = loss_nce
            loss_g += loss_nce

        loss_dict['sum'] = loss_g

        return loss_dict

    def backward_d(self):
        loss_d = 0
        loss_dict = {}

        dxx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=False)
        # ADV(X)+
        dx = self.add_loss_adv(a=self.get_xy_plane(self.oriX), net_d=self.net_d, truth=True)

        loss_dict['dxx_x'] = dxx + dx
        loss_d += dxx + dx

        loss_dict['sum'] = loss_d

        return loss_dict

    def get_projection(self, x, depth, how='mean'):
        if how == 'dsp':
            x = x[:, :, :, :, ::self.hparams.uprate * self.hparams.skipl1]
        else:
            x = x.unfold(-1, depth, depth)
            if how == 'mean':
                x = x.mean(dim=-1)
            elif how == 'max':
                x, _ = x.max(dim=-1)
        return x


# USAGE
# python train.py --jsn cyc_imorphics --prj IsoScope0/0 --models IsoScope0 --cropz 16 --cropsize 128 --netG ed023d --env t09 --adv 1 --rotate --ngf 64 --direction xyori --nm 11 --dataset longdent