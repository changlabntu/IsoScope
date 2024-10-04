from models.base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config
import yaml
import numpy as np


class GAN(BaseModel):
    def __init__(self, hparams, train_loader, eval_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, eval_loader, checkpoints)

        # GAN Model

        # Initialize encoder and decoder
        print('Reading yaml: ' + self.hparams.ldmyaml)
        with open('ldm/' + self.hparams.ldmyaml + '.yaml', "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)

        ddconfig = config['model']['params']["ddconfig"]
        print(ddconfig)

        self.hparams.netG = ddconfig['interpolator']#'ed023e'   # 128 > 128
        self.hparams.final = 'tanh'
        self.net_g, self.net_d = self.set_networks()

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        # Initialize other components
        self.quant_conv = nn.Conv2d(2*ddconfig["z_channels"], 2*hparams.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(hparams.embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = hparams.embed_dim

        # Initialize loss
        lossconfig = config['model']['params']["lossconfig"]
        self.loss = instantiate_from_config()
        self.discriminator = self.loss.discriminator

        # Save model names
        self.netg_names = {'encoder': 'encoder', 'decoder': 'decoder',
                           'quant_conv': 'quant_conv', 'post_quant_conv': 'post_quant_conv'}
        self.netd_names = {'discriminator': 'discriminator', 'net_d': 'net_d'}

        # Configure optimizers
        self.configure_optimizers()

        #self.upsample = torch.nn.Upsample(size=(hparams.cropsize, hparams.cropsize, hparams.cropsize), mode='trilinear')
        #self.uprate = (hparams.cropsize // hparams.cropz)
        #print('uprate: ' + str(self.uprate))

        #lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        #if opt.scale_lr:
        #    model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr  = 2 * 6 * 16 * 4.5e-6

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("AutoencoderKL")
        parser.add_argument("--embed_dim", type=int, default=4)
        parser.add_argument("--ldmyaml", type=str, default='ldmaex2')
        #parser.add_argument("--skipl1", type=int, default=4)
        #parswr.add_argument("--ddconfig", type=str)
        return parent_parser

    def encode(self, x):
        h, hbranch = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior, hbranch

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior, hbranch = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior, hbranch

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def generation(self, batch):
        if self.hparams.cropz > 0:
            z_init = np.random.randint(batch['img'][0].shape[4] - self.hparams.cropz)
            batch['img'][0] = batch['img'][0][:, :, :, :, z_init:z_init + self.hparams.cropz]

        self.oriX = batch['img'][0]  # (B, C, X, Y, Z) # original

        #self.input = self.get_input(batch, self.hparams.image_key)
        # AE
        self.reconstructions, self.posterior, hbranch = self.forward(self.oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0])

    def backward_g(self):
        loss_g = 0
        loss_dict = {}
        # ae
        aeloss, log_dict_ae = self.loss(self.oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0],
                                        self.reconstructions, self.posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        loss_g += aeloss

        loss_dict['sum'] = loss_g
        return loss_dict

    def backward_d(self):
        loss_d = 0
        loss_dict = {}
        # ae
        discloss, log_dict_disc = self.loss(self.oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0],
                                            self.reconstructions, self.posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        loss_d += discloss

        loss_dict['sum'] = loss_d
        return loss_dict

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def save_checkpoint(self, filepath):
        # Combine all the state dicts into a single state dict
        state_dict = {}

        # Encoder
        for k, v in self.encoder.state_dict().items():
            state_dict[f'encoder.{k}'] = v

        # Decoder
        for k, v in self.decoder.state_dict().items():
            state_dict[f'decoder.{k}'] = v

        # Quant Conv
        for k, v in self.quant_conv.state_dict().items():
            state_dict[f'quant_conv.{k}'] = v

        # Post Quant Conv
        for k, v in self.post_quant_conv.state_dict().items():
            state_dict[f'post_quant_conv.{k}'] = v

        # Discriminator (if you want to include it)
        for k, v in self.discriminator.state_dict().items():
            state_dict[f'loss.discriminator.{k}'] = v

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
        print(f"Model saved to {filepath}")

    @classmethod
    def load_from_checkpoint(cls, filepath, train_loader=None, eval_loader=None, checkpoints=None):
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        hparams = checkpoint['hparams']

        model = cls(hparams, train_loader, eval_loader, checkpoints)
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        model.quant_conv.load_state_dict(checkpoint['quant_conv_state_dict'])
        model.post_quant_conv.load_state_dict(checkpoint['post_quant_conv_state_dict'])
        model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        model.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        model.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        model.global_step = checkpoint['global_step']

        print(f"Model loaded from {filepath}")
        return model


if __name__ == '__main__':
    import argparse
    from utils.data_utils import read_json_to_args
    args = read_json_to_args('/media/ExtHDD01/logs/womac4/vae/0/0.json')
    args.embed_dim = 4
    args.ldmyaml = 'ldmaex2'
    gan = GAN(args, train_loader=None, eval_loader=None, checkpoints=None)

    #z, hbranch = gan.encode(torch.randn(1, 1, 64, 64))
    batch = dict()
    batch['img'] = [torch.randn(1, 1, 128, 128, 128)]
    gan.generation(batch)