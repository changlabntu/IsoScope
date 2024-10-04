from models.base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

class GAN(BaseModel):
    def __init__(self, hparams, train_loader, eval_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, eval_loader, checkpoints)

        self.generator, self.net_d = self.set_networks()

        # save model names
        self.netg_names = {'generator': 'generator'}
        self.netd_names = {'net_d': 'net_d'}

        self.configure_optimizers()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--kl_weight", type=float, default=0.001)
        parser.add_argument("--l1_weight", type=float, default=10)
        return parent_parser

    def encode(self, x):
        h = self.generator(x, method='encode')[-1]
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.generator(z, method='decode')['out0']

    def forward(self, x, use_posterior=True):
        mu, logvar = self.encode(x)
        if use_posterior:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        return self.decode(z), mu, logvar

    def generation(self, batch):
        self.input = batch['img'][0]  # (1, 1, 128, 128, z)
        #self.input = self.input.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]
        self.recon, self.mu, self.logvar = self(self.input)

        # tiff.imwrite self.recon and self.input side by side
        select_z = self.input.shape[-1] // 2
        i_img = self.input[..., select_z].squeeze()
        r_img = self.recon[..., select_z].squeeze()

        self.log_image['i_img'] = i_img
        self.log_image['r_img'] = r_img

    def backward_g(self):
        # L1 loss
        recon_loss = F.l1_loss(self.recon, self.input)
        kl_loss = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        loss = recon_loss * self.hparams.l1_weight + kl_loss * self.hparams.kl_weight

        return {'sum': loss, 'recon_loss': recon_loss, 'kl_loss': kl_loss}

    def backward_d(self):
        # VAE doesn't have a discriminator, so this method is not needed
        return {'sum': nn.Parameter(torch.tensor(0.0))}
