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

        # Initialize encoder and decoder
        with open('networks/ldm/ldmaex2.yaml', "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)

        ddconfig = config['model']['params']["ddconfig"]

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        # Initialize other components
        self.quant_conv = nn.Conv2d(2*ddconfig["z_channels"], 2*hparams.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(hparams.embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = hparams.embed_dim

        # Initialize loss
        self.loss = instantiate_from_config(config['model']['params']["lossconfig"])
        self.discriminator = self.loss.discriminator

        # Save model names
        self.netg_names = {'encoder': 'encoder', 'decoder': 'decoder', 'quant_conv': 'quant_conv', 'post_quant_conv': 'post_quant_conv'}
        self.netd_names = {'discriminator': 'discriminator'}

        # Configure optimizers
        self.configure_optimizers()

        #lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        #if opt.scale_lr:
        #    model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr  = 2 * 6 * 16 * 4.5e-6

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("AutoencoderKL")
        parser.add_argument("--embed_dim", type=int, default=4)
        #parswr.add_argument("--ddconfig", type=str)
        return parent_parser

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

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
        self.oriX = self.oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]  # (Z, C, X, Y)

        #self.input = self.get_input(batch, self.hparams.image_key)
        self.reconstructions, self.posterior = self.forward(self.oriX)

    def backward_g(self):
        aeloss, log_dict_ae = self.loss(self.oriX, self.reconstructions, self.posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        loss_g = aeloss
        return {'sum': loss_g, 'aeloss': aeloss}#, 'log_dict_ae': log_dict_ae}

    def backward_d(self):
        discloss, log_dict_disc = self.loss(self.oriX, self.reconstructions, self.posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        loss_d = discloss
        return {'sum': loss_d, 'discloss': discloss}#, 'log_dict_disc': log_dict_disc}

    def get_last_layer(self):
        return self.decoder.conv_out.weight



# USAGE
# python train.py --jsn autoencoder_config --prj autoencoder/test1 --models autoencoder_kl -b 16 --input_nc 3 --output_nc 3