from models.base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from ldm.modules.diffusionmodules.modelcut import Encoder, Decoder
from ldm.util import instantiate_from_config
import yaml
import numpy as np
from models.CUT import PatchSampleF3D
from networks.networks_cut import Normalize, init_net, PatchNCELoss
from torch.optim.lr_scheduler import LambdaLR


class GAN(BaseModel):
    def __init__(self, hparams, train_loader, eval_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, eval_loader, checkpoints)

        # VQGAN Model

        # Initialize encoder and decoder
        print('Reading yaml: ' + self.hparams.ldmyaml)
        with open('ldm/' + self.hparams.ldmyaml + '.yaml', "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)

        ddconfig = config['model']['params']["ddconfig"]
        if self.hparams.tc:
            ddconfig['in_channels'] = 2
            ddconfig['out_ch'] = 1
        self.hparams.netG = ddconfig['interpolator']

        self.hparams.final = 'tanh'
        if self.hparams.tc:
            self.hparams.input_nc = 1
            self.hparams.output_nc = 2
        self.net_g, self.net_d = self.set_networks()

        # VQGAN components
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.embed_dim = config['model']['params']['embed_dim']
        self.n_embed =  config['model']['params']['n_embed']

        # Vector Quantizer
        self.quantize = VectorQuantizer(
            self.n_embed,
            self.embed_dim,
            beta=0.25,
            remap=getattr(hparams, 'remap', None),
            sane_index_shape=getattr(hparams, 'sane_index_shape', False)
        )

        # Quantization convolutions
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], self.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(self.embed_dim, ddconfig["z_channels"], 1)

        # Initialize loss
        self.loss = instantiate_from_config(config['model']['params']["lossconfig"])
        self.discriminator = self.loss.discriminator

        # EMA support
        self.use_ema = getattr(hparams, 'use_ema', False)
        if self.use_ema:
            try:
                from taming.modules.util import LitEma
                self.model_ema = LitEma(self)
                print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
            except ImportError:
                print("LitEma not available, disabling EMA")
                self.use_ema = False

        # Save model names
        self.netg_names = {
            'encoder': 'encoder',
            'decoder': 'decoder',
            'quantize': 'quantize',
            'quant_conv': 'quant_conv',
            'post_quant_conv': 'post_quant_conv',
            'net_g': 'net_g'
        }
        self.netd_names = {'discriminator': 'discriminator', 'net_d': 'net_d'}

        # Configure optimizers
        self.configure_optimizers()

        self.upsample = torch.nn.Upsample(
            size=(hparams.cropsize, hparams.cropsize, hparams.cropsize),
            mode='trilinear'
        )
        self.uprate = (hparams.cropsize // hparams.cropz) * hparams.dsp
        print('uprate: ' + str(self.uprate))

        # CUT NCE
        if not self.hparams.nocut:
            netF = PatchSampleF3D(
                use_mlp=self.hparams.use_mlp,
                init_type='normal',
                init_gain=0.02,
                gpu_ids=[],
                nc=self.hparams.c_mlp
            )
            self.netF = init_net(netF, init_type='normal', init_gain=0.02, gpu_ids=[])
            feature_shapes = [64, 128, 128, 256]
            self.netF.create_mlp(feature_shapes)

            if self.hparams.fWhich == None:
                self.hparams.fWhich = [1 for i in range(len(feature_shapes))]

            print(self.hparams.fWhich)

            self.criterionNCE = []
            for nce_layer in range(len(feature_shapes)):
                self.criterionNCE.append(PatchNCELoss(opt=hparams))

            self.netg_names['netF'] = 'netF'

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VQGAN")
        parser.add_argument("--embed_dim", type=int, default=256)
        parser.add_argument("--n_embed", type=int, default=1024)
        parser.add_argument("--ldmyaml", type=str, default='ldmaex2')
        parser.add_argument("--skipl1", type=int, default=4)
        parser.add_argument("--hbranch", type=str, default='mid')
        parser.add_argument("--tc", action="store_true", default=False)
        parser.add_argument("--l1how", type=str, default='dsp')
        parser.add_argument("--dsp", type=int, default=1, help='extra downsample rate')
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
        parser.add_argument("--downbranch", type=int, default=1)
        parser.add_argument("--resizebranch", type=int, default=1)
        # VQ specific arguments
        parser.add_argument("--use_ema", action='store_true', help='use exponential moving average')
        parser.add_argument("--remap", type=str, default=None, help='remap indices')
        parser.add_argument("--sane_index_shape", action='store_true', help='return indices as bhw')
        parser.add_argument("--lr_g_factor", type=float, default=1.0, help='learning rate factor for generator')
        return parent_parser

    def encode(self, x):
        """Encode input to quantized latent space"""
        h, hbranch, hz = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)

        # Process hz for CUT loss (similar to original)
        hz = hz[1::2]  # every other two layer  (Z, C, X, Y)
        hz = [x.permute(1, 2, 3, 0).unsqueeze(0) for x in hz]

        return quant, emb_loss, info, hbranch, hz

    def encode_to_prequant(self, x):
        """Encode to pre-quantization features"""
        h, _, _ = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        """Decode from quantized latent space"""
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        """Decode from codebook indices"""
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        """Forward pass through VQGAN"""
        quant, diff, (_, _, ind), hbranch, hz = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind, hbranch, hz
        return dec, diff, hbranch, hz

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def adv_loss_six_way(self, x, net_d, truth):
        loss = 0
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

    def get_xy_plane(self, x):
        return x.permute(4, 1, 2, 3, 0)[::1, :, :, :, 0]

    def generation(self, batch):
        if self.hparams.cropz > 0:
            z_init = np.random.randint(batch['img'][0].shape[4] - self.hparams.cropz)
            for b in range(len(batch['img'])):
                batch['img'][b] = batch['img'][b][:, :, :, :, z_init:z_init + self.hparams.cropz]

        # extra downsample
        if self.hparams.dsp > 1:
            z_init = np.random.randint(self.hparams.dsp)
            for b in range(len(batch['img'])):
                batch['img'][b] = batch['img'][b][:, :, :, :, z_init::self.hparams.dsp]

        if self.hparams.tc:
            self.oriX = torch.cat((batch['img'][0], batch['img'][1]), 1)
        else:
            self.oriX = batch['img'][0]  # (B, C, X, Y, Z) # original

        # VQGAN forward pass
        input_slice = self.oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]  # (Z, C, X, Y)
        self.reconstructions, self.qloss, _, hbranch, self.hz = self.forward(input_slice, return_pred_indices=False)

        # hbranch processing
        if self.hparams.hbranch == 'z':
            # Use quantized features instead of posterior sampling
            quant, _, _, _, _ = self.encode(input_slice)
            hbranch = quant

            if self.hparams.downbranch > 1:
                hbranch = hbranch.permute(1, 2, 3, 0).unsqueeze(0)  # (1, C, X, Y, Z)
                hbranch = nn.MaxPool3d((1, 1, self.hparams.downbranch))(hbranch)
                hbranch = hbranch.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]  # (Z, C, X, Y)

            if self.hparams.resizebranch != 1:
                hbranch = hbranch.permute(1, 2, 3, 0).unsqueeze(0)  # (1, C, X, Y, Z)
                hbranch = nn.Upsample(scale_factor=(1, 1, self.hparams.resizebranch), mode='trilinear')(hbranch)
                hbranch = hbranch.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]  # (Z, C, X, Y)

            hbranch = self.decoder.conv_in(hbranch)  # (16, 256, 16, 16)

        hbranch = hbranch.permute(1, 2, 3, 0).unsqueeze(0)  # (1, C, X, Y, Z)

        self.XupX = self.net_g(hbranch, method='decode')['out0']
        self.Xup = self.upsample(self.oriX)  # (B, C, X, Y, Z)

        if not self.hparams.nocut:
            self.goutz = hbranch

    def get_projection(self, x, depth, how='mean'):
        if how == 'dsp':
            x = x[:, :, :, :, (self.uprate // 2)::self.uprate * self.hparams.skipl1]
        else:
            x = x.unfold(-1, depth, depth)
            if how == 'mean':
                x = x.mean(dim=-1)
            elif how == 'max':
                x, _ = x.max(dim=-1)
        return x

    def backward_g(self):
        loss_g = 0
        loss_dict = {}

        axx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=True)

        loss_l1 = self.add_loss_l1(a=self.get_projection(self.XupX, depth=self.uprate
                                                                          * self.hparams.skipl1,
                                                         how=self.hparams.l1how),
                                   b=self.oriX[:, :, :, :, ::self.hparams.skipl1])

        loss_dict['axx'] = axx
        loss_g += axx
        loss_dict['l1'] = loss_l1
        loss_g += loss_l1 * self.hparams.lamb

        oriXpermute = self.oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]
        if self.hparams.tc:
            oriXpermute = self.oriX.permute(4, 1, 2, 3, 0)[:, :1, :, :, 0]

        # VQGAN loss (different from KL autoencoder)
        aeloss, log_dict_ae = self.loss(
            self.qloss,  # VQ loss instead of posterior
            oriXpermute,  # original input
            self.reconstructions,  # reconstructed output
            0,  # optimizer index for generator
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train"
        )
        loss_g += aeloss
        loss_dict['ae'] = aeloss

        # Add VQ commitment loss
        loss_dict['vq'] = self.qloss
        loss_g += self.qloss

        # CUT NCE Loss
        if not self.hparams.nocut:
            # feat q - use stored hz from generation
            feat_q = self.hz

            # feat k - encode generated output
            input_slice_k = self.XupX.permute(4, 1, 2, 3, 0)[4::8, :, :, :, 0]  # (Z, C, X, Y)
            _, _, _, _, feat_k = self.encode(input_slice_k)

            feat_k_pool, sample_ids = self.netF(feat_k, self.hparams.num_patches, None)
            feat_q_pool, _ = self.netF(feat_q, self.hparams.num_patches, sample_ids)

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
        dx = self.add_loss_adv(a=self.get_xy_plane(self.oriX), net_d=self.net_d, truth=True)

        loss_dict['dxx_x'] = dxx + dx
        loss_d += dxx + dx

        oriXpermute = self.oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]
        if self.hparams.tc:
            oriXpermute = self.oriX.permute(4, 1, 2, 3, 0)[:, :1, :, :, 0]

        # VQGAN discriminator loss
        discloss, log_dict_disc = self.loss(
            self.qloss,  # VQ loss
            oriXpermute,
            self.reconstructions,
            1,  # optimizer index for discriminator
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train"
        )
        loss_d += discloss
        loss_dict['disc'] = discloss

        loss_dict['sum'] = loss_d
        return loss_dict

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def configure_optimizers(self):
        lr_d = self.hparams.lr
        lr_g = getattr(self.hparams, 'lr_g_factor', 1.0) * self.hparams.lr
        print("lr_d", lr_d)
        print("lr_g", lr_g)

        # Generator optimizer includes VQ parameters
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quantize.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=lr_g, betas=(0.5, 0.9)
        )

        # Discriminator optimizer
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(),
            lr=lr_d, betas=(0.5, 0.9)
        )

        # Add scheduler support if specified
        if hasattr(self.hparams, 'scheduler_config') and self.hparams.scheduler_config is not None:
            scheduler = instantiate_from_config(self.hparams.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler

        return [opt_ae, opt_disc], []

    def save_checkpoint(self, filepath):
        # Combine all the state dicts into a single state dict
        state_dict = {}

        # Encoder
        for k, v in self.encoder.state_dict().items():
            state_dict[f'encoder.{k}'] = v

        # Decoder
        for k, v in self.decoder.state_dict().items():
            state_dict[f'decoder.{k}'] = v

        # Quantizer
        for k, v in self.quantize.state_dict().items():
            state_dict[f'quantize.{k}'] = v

        # Quant Conv
        for k, v in self.quant_conv.state_dict().items():
            state_dict[f'quant_conv.{k}'] = v

        # Post Quant Conv
        for k, v in self.post_quant_conv.state_dict().items():
            state_dict[f'post_quant_conv.{k}'] = v

        # Discriminator
        for k, v in self.discriminator.state_dict().items():
            state_dict[f'loss.discriminator.{k}'] = v

        # EMA if used
        if self.use_ema:
            for k, v in self.model_ema.state_dict().items():
                state_dict[f'model_ema.{k}'] = v

        # Create the checkpoint dictionary
        checkpoint = {
            "state_dict": state_dict,
            "global_step": self.global_step,
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
        }

        # Save additional hyperparameters if needed
        if hasattr(self, 'hparams'):
            checkpoint['hparams'] = self.hparams

        torch.save(checkpoint, filepath)
        print(f"VQGAN model saved to {filepath}")

    @classmethod
    def load_from_checkpoint(cls, filepath, train_loader=None, eval_loader=None, checkpoints=None):
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        hparams = checkpoint['hparams']

        model = cls(hparams, train_loader, eval_loader, checkpoints)

        # Load state dict
        state_dict = checkpoint['state_dict']
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")

        model.global_step = checkpoint['global_step']
        print(f"VQGAN model loaded from {filepath}")
        return model


if __name__ == '__main__':
    import argparse
    from utils.data_utils import read_json_to_args

    from argparse import Namespace
    # Test the VQGAN model
    args = read_json_to_args('env/jsn/cyc_imorphics.json')
    args = args.train
    args = Namespace(**args)

    # IsoGAN args
    args.cropz = 16
    args.cropsize = 128

    # VQGAN args
    args.ldmyaml = 'vqgan'
    args.tc = False
    args.hbranch = 'mid'
    args.use_ema = False

    ganvq = GAN(args, train_loader=None, eval_loader=None, checkpoints=None)

    # Test with sample data
    batch = dict()
    batch['img'] = [torch.randn(1, 1, 128, 128, 128)]
    ganvq.generation(batch)
    print("VQGAN model initialized and tested successfully!")

    #     "bash": "train.py --jsn cyc_imorphics --prj ae/cut/1dpmLb1downh2NCE1 --hbranch z --ldmyaml ldmaex2 --netG none --lamb 1 --lr 0.0002 --models ae0iso0tccut --cropz 32 --cropsize 128 --env brcb --adv 1 --rotate --ngf 32 --direction ori3d --nm 00 --dataset aisr122424 --epoch_save 500 --n_epochs 10000 --lr_policy cosine --mc --precrop 256 --downbranch 2 --lbNCE 1"