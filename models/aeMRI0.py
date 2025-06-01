from models.base import BaseModel
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from ldm.modules.diffusionmodules.modelcut import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config
import yaml
from models.base import VGGLoss
from networks.networks_cut import Normalize, init_net, PatchNCELoss
from models.CUT import PatchSampleF3D
from networks.networks_cut import Normalize, init_net, PatchNCELoss


class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, eval_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, eval_loader, checkpoints)

        # Initialize encoder and decoder
        print('Reading yaml: ' + self.hparams.ldmyaml)
        with open('ldm/' + self.hparams.ldmyaml + '.yaml', "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)

        ddconfig = config['model']['params']["ddconfig"]
        if self.hparams.tc:
            ddconfig['in_channels'] = 2
            ddconfig['out_ch'] = 1
        self.hparams.netG = ddconfig['interpolator']#'ed023e'   # 128 > 128

        self.hparams.final = 'tanh'
        if self.hparams.tc:
            self.hparams.input_nc = 1  # this would not be used
            self.hparams.output_nc = 2
        self.net_g, self.net_d = self.set_networks()

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        # Initialize other components
        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * hparams.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(hparams.embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = hparams.embed_dim

        # Initialize loss
        self.loss = instantiate_from_config(config['model']['params']["lossconfig"])
        self.discriminator = self.loss.discriminator

        # Save model names
        self.netg_names = {'encoder': 'encoder', 'decoder': 'decoder',
                           'quant_conv': 'quant_conv', 'post_quant_conv': 'post_quant_conv',
                           'net_g': 'net_g'}
        self.netd_names = {'discriminator': 'discriminator', 'net_d': 'net_d'}

        # Configure optimizers
        self.configure_optimizers()

        self.upsample = torch.nn.Upsample(size=(hparams.cropsize, hparams.cropsize, hparams.cropsize), mode='trilinear')
        self.uprate = (hparams.cropsize // hparams.cropz)
        print('uprate: ' + str(self.uprate))

        # lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        # if opt.scale_lr:
        #    model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr  = 2 * 6 * 16 * 4.5e-6

        # CUT NCE
        if not self.hparams.nocut:
            netF = PatchSampleF3D(use_mlp=self.hparams.use_mlp, init_type='normal', init_gain=0.02, gpu_ids=[],
                                  nc=self.hparams.c_mlp)
            self.netF = init_net(netF, init_type='normal', init_gain=0.02, gpu_ids=[])
            feature_shapes = [64, 128, 128, 256]
            self.netF.create_mlp(feature_shapes)

            if self.hparams.fWhich == None:  # which layer of the feature map to be considered in CUT
                self.hparams.fWhich = [1 for i in range(len(feature_shapes))]

            print(self.hparams.fWhich)

            self.criterionNCE = []
            for nce_layer in range(len(feature_shapes)):  # self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt=hparams))  # .to(self.device))

            self.netg_names['netF'] = 'netF'

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("AutoencoderKL")
        parser.add_argument("--embed_dim", type=int, default=4)
        parser.add_argument("--ldmyaml", type=str, default='ldmaex2')
        parser.add_argument("--hbranch", type=str, default='mid')
        parser.add_argument("--tc", action="store_true", default=False)
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
        parser.add_argument("--downz", type=int, default=0)
        return parent_parser

    def encode(self, x):
        h, hbranch, hz = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        hz = hz[1::2]  # every other two layer  (Z, C, X, Y)
        hz = [x.permute(1, 2, 3, 0).unsqueeze(0) for x in hz]
        return posterior, hbranch, hz

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior, hbranch, _ = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior, hbranch

    def generation(self, batch):
        if self.hparams.downz > 0:
            batch['img'][0] = torch.nn.Upsample(scale_factor=(1, 1, 1 / self.hparams.downz), mode='trilinear')(batch['img'][0])
            batch['img'][0] = torch.nn.Upsample(scale_factor=(1, 1, self.hparams.downz), mode='trilinear')(batch['img'][0])

            # batch['img'][1] = batch['img'][1][:, :, :, :, ::self.hparams.down

        if self.hparams.cropz > 0:
            z_init = np.random.randint(batch['img'][0].shape[4] - self.hparams.cropz)
            batch['img'][0] = batch['img'][0][:, :, :, :, z_init:z_init + self.hparams.cropz]
            # batch['img'][1] = batch['img'][1][:, :, :, :, z_init:z_init + self.hparams.cropz]

        # extra downsample
        if self.hparams.dsp > 1:
            batch['img'][0] = batch['img'][0][:, :, :, :, ::self.hparams.dsp]

        self.oriX = batch['img'][0]  # (B, C, X, Y, Z) # original
        #self.oriY = batch['img'][1]  # (B, C, X, Y, Z) # original

        # X-Y permute
        if np.random.randint(2) == 1:
            self.oriX = self.oriX.permute(0, 1, 3, 2, 4)

        self.Xup = self.upsample(self.oriX)  # (B, C, X, Y, Z)
        #self.Yup = self.upsample(self.oriY)  # (B, C, X, Y, Z)

        # AE
        self.reconstructions, self.posterior, hbranch = self.forward(
            self.oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0])  # (Z, C, X, Y)

        if self.hparams.hbranch == 'z':
            hbranch = self.posterior.sample()
            hbranch = self.decoder.conv_in(hbranch)

        hbranch = hbranch.permute(1, 2, 3, 0).unsqueeze(0)  # (1, C, X, Y, Z)
        self.XupX = self.net_g(hbranch, method='decode')['out0']

    def get_xy_plane(self, x):  # (B, C, X, Y, Z)
        return x.permute(4, 1, 2, 3, 0)[::1, :, :, :, 0]  # (Z, C, X, Y, B)

    def adv_loss_six_way(self, x, net_d, truth):
        # x (B, C, X, Y, Z)
        rint = np.random.randint(3)

        loss = 0
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
        return loss

    def backward_g(self):
        loss_g = 0
        loss_dict = {}

        axx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=True)

        if self.hparams.randl1:
            shift = np.random.randint(0, self.hparams.skipl1)
        else:
            shift = -1

        loss_l1 = self.add_loss_l1(a=self.get_projection(self.XupX, depth=self.hparams.uprate
                                                                          * self.hparams.skipl1, how=self.hparams.l1how),
                                   b=self.oriX[:, :, :, :, ::self.hparams.skipl1])

        loss_dict['axx'] = axx
        loss_g += axx
        loss_dict['l1'] = loss_l1
        loss_g += loss_l1 * self.hparams.lamb

        # aeloss
        oriXpermute = self.oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]
        aeloss, log_dict_ae = self.loss(oriXpermute,
                                        self.reconstructions, self.posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        loss_g += aeloss

        # CUT
        if not self.hparams.nocut:
            # feat q

            posterior, hbranch, hz = self.encode(self.oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0])
            feat_q = hz

            # feat k
            posterior, hbranch, hz = self.encode(self.XupX.permute(4, 1, 2, 3, 0)[4::8, :, :, :, 0])  # (Z, C, X, Y)
            feat_k = hz

            feat_k_pool, sample_ids = self.netF(feat_k, self.hparams.num_patches,
                                                None)  # get source patches by random id
            feat_q_pool, _ = self.netF(feat_q, self.hparams.num_patches, sample_ids)  # use the ids for query target

            total_nce_loss = 0.0
            for f_q, f_k, crit, f_w in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.hparams.fWhich):
                loss = crit(f_q, f_k) * f_w
                total_nce_loss += loss.mean()
            loss_nce = total_nce_loss / 4
            loss_dict['nce'] = loss_nce
            loss_g += loss_nce * self.hparams.lbNCE

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



        # aeloss
        oriXpermute = self.oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]
        discloss, log_dict_disc = self.loss(oriXpermute,
                                            self.reconstructions, self.posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        loss_d += discloss

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

    def get_last_layer(self):
        return self.decoder.conv_out.weight

# USAGE
# python train.py --jsn cyc_imorphics --prj IsoScope0/0 --models IsoScope0 --cropz 16 --cropsize 128 --netG ed023d --env t09 --adv 1 --rotate --ngf 64 --direction xyori --nm 11 --dataset longdent