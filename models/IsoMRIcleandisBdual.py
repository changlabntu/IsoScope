from models.base import BaseModel
import copy
import torch
import numpy as np
import torch.nn as nn
from models.base import VGGLoss
from networks.networks_cut import Normalize, init_net, PatchNCELoss
#from torchmetrics.image.fid import FID
#from torchmetrics.image.kid import KID

from torchmetrics.image.kid import KernelInceptionDistance as KID
from torchmetrics.image.fid import FrechetInceptionDistance as FID


class PatchSampleF3D(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF3D, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feature_shapes):
        for mlp_id, feat in enumerate(feature_shapes):
            input_nc = feat
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            # if len(self.gpu_ids) > 0:
            # mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        # print(len(feats))
        # print([x.shape for x in feats])
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            # B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            # (B, C, H, W, Z)
            # feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)  # (B, H*W, C)
            feat = feat.permute(0, 2, 3, 4, 1)  # (B, H*W*Z, C)
            feat_reshape = feat.reshape(feat.shape[0], feat.shape[1] * feat.shape[2] * feat.shape[3], feat.shape[4])
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    # torch.randperm produces cudaErrorIllegalAddress for newer versions of PyTorch. https://github.com/taesungp/contrastive-unpaired-translation/issues/83
                    # patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = np.random.permutation(feat_reshape.shape[1])  # (random order of range(H*W))
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device) # first N patches
                    # patch_id = torch.from_numpy(patch_id).type(torch.long).to(feat.device)
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)  # Channel (1, 128, 256, 256, 256) > (256, 256, 256, 256, 256)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        # print([x.shape for x in return_feats]) # (B * num_patches, 256) * level of features
        return return_feats, return_ids


class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, eval_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, eval_loader, checkpoints)

        from networks.EncoderDecoder.edcleanB import Generator
        self.hparams.final = 'tanh'
        self.net_g = Generator(n_channels=self.hparams.input_nc, out_channels=self.hparams.output_nc, nf=self.hparams.ngf,
                               norm_type=self.hparams.norm, final=self.hparams.final, mc=self.hparams.mc, encode='1d', decode='3d')
        _, self.net_d = self.set_networks()
        self.hparams.final = 'tanh'

        print('nocyc:  ' + str(self.hparams.nocyc))
        if not self.hparams.nocyc:
            self.net_gback = Generator(n_channels=self.hparams.input_nc, out_channels=self.hparams.output_nc, nf=self.hparams.ngf,
                                   norm_type=self.hparams.norm, final=self.hparams.final, mc=self.hparams.mc, encode='3d', decode='1d')

            _, self.net_dzy = self.set_networks()
            _, self.net_dzx = self.set_networks()

        # save model names
        self.netg_names = {'net_g': 'net_g'}
        self.netd_names = {'net_d': 'net_d'}

        if not self.hparams.nocyc:
            self.netg_names['net_gback'] = 'net_gback'
            self.netd_names['net_dzy'] = 'net_dzy'
            self.netd_names['net_dzx'] = 'net_dzx'

        # Finally, initialize the optimizers and scheduler
        self.configure_optimizers()
        self.upsample = torch.nn.Upsample(size=(hparams.cropsize, hparams.cropsize, hparams.cropz // hparams.dsp * hparams.uprate), mode='trilinear')

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

        # metrics
        #self.fid = FID(feature=768, reset_real_features=True).cuda()
        #self.kid = KID(subset_size=2, reset_real_features=True).cuda()

        self.xy = []
        self.yz = []
        self.xz = []
        self.Xupbuff = []
        self.XupXbuff = []

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # coefficient for the identify loss
        parser.add_argument("--dsp", type=int, default=1)
        parser.add_argument("--lambB", type=int, default=1)
        parser.add_argument("--l1how", type=str, default='dsp')
        parser.add_argument("--uprate", type=int, default=4)
        parser.add_argument("--skipl1", type=int, default=1)
        parser.add_argument("--randl1", action='store_true')
        parser.add_argument("--nocyc", action='store_true')
        parser.add_argument("--nocut", action='store_true')
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

    def test_method(self, net_g, img):
        output = net_g(img[0])
        #output = combine(output, x[0], method='mul')
        return output[0]

    def generation(self, batch):
        if self.hparams.cropz > 0:
            z_init = np.random.randint(batch['img'][0].shape[4] - self.hparams.cropz)
            batch['img'][0] = batch['img'][0][:, :, :, :, z_init:z_init + self.hparams.cropz]
            # batch['img'][1] = batch['img'][1][:, :, :, :, z_init:z_init + self.hparams.cropz]

        # extra downsample
        if self.hparams.dsp > 1:
            batch['img'][0] = batch['img'][0][:, :, :, :, self.hparams.dsp // 2::self.hparams.dsp]

        self.oriX = batch['img'][0]  # (B, C, X, Y, Z) # original
        #self.oriY = batch['img'][1]  # (B, C, X, Y, Z) # original

        self.Xup = self.upsample(self.oriX)  # (B, C, X, Y, Z)

        self.XupP = self.Xup.permute(0, 1, 3, 2, 4)  # (B, C, Y, X, Z)

        #self.goutz = self.net_g(self.Xup, method='encode')
        self.XupX = self.net_g(self.Xup)['out0']
        self.XupXP = self.net_g(self.XupP)['out0'].permute(0, 1, 3, 2, 4)  # (B, C, Y, X, Z)

        self.XupX = (self.XupX + self.XupXP) / 2

        if not self.hparams.nocyc:
            self.XupXback = self.net_gback(self.XupX)['out0']

        self.get_metrics()

    def get_metrics(self):
        with torch.no_grad():
            # (B, C, X, Y, Z)
            Xup8 = self.to_8bit(self.Xup).repeat(1, 3, 1, 1, 1)
            XupX8 = self.to_8bit(self.XupX).repeat(1, 3, 1, 1, 1)

            self.xy.append(self.to_2d(Xup8.permute(0, 4, 1, 2, 3), dsp=True))
            self.yz.append(self.to_2d(XupX8.permute(0, 2, 1, 3, 4), dsp=True))
            self.xz.append(self.to_2d(XupX8.permute(0, 3, 1, 2, 4), dsp=True))

    def reset_metrics(self):
        with torch.no_grad():
            kid = KID(subset_size=32, reset_real_features=True).cuda()
            kid.update(torch.cat(self.xy, 0), real=True)
            kid.update(torch.cat(self.yz, 0), real=False)
            kid.update(torch.cat(self.xz, 0), real=False)
            self.log('kid', kid.compute()[0], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            fid = FID(feature=768, reset_real_features=True).cuda()
            fid.update(torch.cat(self.xy, 0), real=True)
            fid.update(torch.cat(self.yz, 0), real=False)
            fid.update(torch.cat(self.xz, 0), real=False)
            self.log('fid', fid.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            self.xy = []
            self.yz = []
            self.xz = []
            #self.fid.reset()
            #self.kid.reset()

    def to_2d(self, x, dsp=False):
        if dsp:
            x = x[:, np.random.randint(64)::64, :, :, :]
        return x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])

    def to_8bit(self, x):
        x = (x - x.min()) / (x.max() - x.min())
        x = x * 255
        x = x.type(torch.cuda.ByteTensor)
        return x

    def get_xy_plane(self, x):  # (B, C, X, Y, Z)
        x = x.permute(0, 4, 1, 2, 3)  # (B, Z, C, X, Y)
        return self.to_2d(x)  # (B*Z, C, X, Y)

    def adv_loss_six_way(self, x, net_d, truth):
        # x (B, C, X, Y, Z)
        rint = np.random.randint(3)

        loss = 0
        if rint == 0:
            zy = self.to_2d(x.permute(0, 2, 1, 4, 3))  # (B, X, C, Z, Y)
            yz = self.to_2d(x.permute(0, 2, 1, 3, 4))  # (B, X, C, Y, Z)
            loss += self.add_loss_adv(a=zy, net_d=net_d, truth=truth)  # (X, C, Z, Y)
            loss += self.add_loss_adv(a=yz, net_d=net_d, truth=truth)  # (X, C, Y, Z)
            loss += self.add_loss_adv(a=torch.flip(zy, [2]), net_d=net_d, truth=truth)
            loss += self.add_loss_adv(a=torch.flip(yz, [3]), net_d=net_d, truth=truth)

        if rint == 1:
            zx = self.to_2d(x.permute(0, 3, 1, 4, 2))  # (B, Y, C, Z, X)
            xz = self.to_2d(x.permute(0, 3, 1, 2, 4))  # (B, Y, C, X, Z)
            loss += self.add_loss_adv(a=zx, net_d=net_d, truth=truth)  # (Y, C, Z, X)
            loss += self.add_loss_adv(a=xz, net_d=net_d, truth=truth)  # (Y, C, X, Z)
            loss += self.add_loss_adv(a=torch.flip(zx, [2]), net_d=net_d, truth=truth)
            loss += self.add_loss_adv(a=torch.flip(xz, [3]), net_d=net_d, truth=truth)

        if rint == 2:
            xy = self.to_2d(x.permute(0, 4, 1, 2, 3))  # (B, Z, C, X, Y)
            yx = self.to_2d(x.permute(0, 4, 1, 3, 2))  # (B, Z, C, Y, X)
            loss += 2 * self.add_loss_adv(a=xy, net_d=net_d, truth=truth)  # (Z, C, X, Y)
            loss += 2 * self.add_loss_adv(a=yx, net_d=net_d, truth=truth)  # (Z, C, Y, X)

        loss = loss / 4
        return loss

    def backward_g(self):
        loss_g = 0
        loss_dict = {}

        axx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=True)

        if self.hparams.randl1:
            shift = np.random.randint(0, self.hparams.skipl1)
        else:
            shift = -1

        loss_l1 = self.add_loss_l1(a=self.get_projection(self.XupX, depth=8, how=self.hparams.l1how),
                                   b=self.oriX[:, :, :, :, ::self.hparams.skipl1])

        loss_dict['axx'] = axx
        loss_g += axx
        loss_dict['l1'] = loss_l1
        loss_g += loss_l1 * self.hparams.lamb

        if not self.hparams.nocyc:
            gback = self.adv_loss_six_way_y(self.XupXback, truth=True)
            loss_dict['gback'] = gback
            loss_g += gback

            loss_l1_back = self.add_loss_l1(a=self.XupXback, b=self.Xup)
            loss_dict['l1b'] = loss_l1_back
            loss_g += loss_l1_back * self.hparams.lambB

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

        # ADV dyy
        if not self.hparams.nocyc:
            dyy = self.adv_loss_six_way_y(self.XupXback, truth=False)
            dy = self.adv_loss_six_way_y(self.oriX, truth=True)

            loss_dict['dyy'] = dyy + dy
            loss_d += dyy + dy

        loss_dict['sum'] = loss_d

        return loss_dict

    def get_projection(self, x, depth, how='mean'):
        if how == 'dsp':
            x = x[:, :, :, :, (self.hparams.uprate // 2)::self.hparams.uprate * self.hparams.skipl1]
        else:
            x = x.unfold(-1, depth, depth)
            if how == 'mean':
                x = x.mean(dim=-1)
            elif how == 'max':
                x, _ = x.max(dim=-1)
        return x


# USAGE
# python train.py --jsn cyc_imorphics --prj IsoScope0/0 --models IsoScope0 --cropz 16 --cropsize 128 --netG ed023d --env t09 --adv 1 --rotate --ngf 64 --direction xyori --nm 11 --dataset longdent