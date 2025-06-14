import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from networks.networks import get_scheduler
from networks.loss import GANLoss
import torchvision

import time, os
import pytorch_lightning as pl
from utils.data_utils import *
from pytorch_lightning.utilities import rank_zero_only
import yaml
import torchvision.transforms as transforms


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def combine(x, y, method):
    if method == 'res':
        return x + y
    elif method == 'mul':
        return torch.mul(x, y)
    elif method == 'multanh':
        return torch.mul((x + 1) / 2, y)
    elif method == 'not':
        return x


import torchvision.models as models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class BaseModel(pl.LightningModule):
    def __init__(self, hparams, train_loader, eval_loader, checkpoints):
        super().__init__()
        # adding data
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # initialize
        self.epoch = 0
        self.dir_checkpoints = checkpoints

        # save model names
        self.netg_names = {'net_g': 'netG'}
        self.netd_names = {'net_d': 'netD'}
        self.loss_g_names = ['loss_g']
        self.loss_d_names = ['loss_d']

        # Hyper-parameters
        print(hparams)
        print(hparams.not_tracking_hparams)
        hparams = {x: vars(hparams)[x] for x in vars(hparams).keys() if x not in hparams.not_tracking_hparams}
        hparams.pop('not_tracking_hparams', None)
        self.hparams.update(hparams)
        self.save_hyperparameters(self.hparams)

        # Define Loss Functions
        self.criterionL1 = nn.L1Loss()
        self.criterionL2 = nn.MSELoss()
        self.MSELoss = nn.MSELoss()
        if self.hparams.gan_mode == 'vanilla':
            self.criterionGAN = nn.BCEWithLogitsLoss()
        else:
            self.criterionGAN = GANLoss(self.hparams.gan_mode)

        # Final
        self.hparams.update(vars(self.hparams))   # updated hparams to be logged in tensorboard
        #self.train_loader.dataset.shuffle_images()  # !!! shuffle again just to make sure
        #self.train_loader.dataset.shuffle_images()  # !!! shuffle again just to make sure

        self.all_label = []
        self.all_out = []
        self.all_loss = []

        self.log_image = {}

        self.buffer = {}

    #def update_optimizer_scheduler(self):
    #    [self.optimizer_d, self.optimizer_g], [] = self.configure_optimizers()

    def save_tensor_to_png(self, tensor, path):
        # Ensure the tensor is on CPU
        tensor = tensor.detach().cpu()

        # If the tensor is 2D, convert it to 3D
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)

        # Ensure the tensor has 3 dimensions
        assert tensor.dim() == 3, "Tensor should have 3 dimensions: (C, H, W)"

        # Normalize the tensor if it's not in [0, 1] range
        if tensor.min() < 0 or tensor.max() > 1:
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

        # Convert to PIL Image
        if tensor.shape[0] == 1:  # Grayscale
            img = transforms.ToPILImage()(tensor.squeeze())
        else:  # RGB
            img = transforms.ToPILImage()(tensor)

        # Save the image
        img.save(path)

    def configure_optimizers(self):
        print('configuring optimizer being called....')
        print(self.netg_names.keys())
        print(self.netd_names.keys())
        print('configuring optimizer done')

        netg_parameters = []
        for g in self.netg_names.keys():
            netg_parameters = netg_parameters + list(getattr(self, g).parameters())
        print('Number of parameters in generator: ', sum(p.numel() for p in netg_parameters if p.requires_grad))

        netd_parameters = []
        for d in self.netd_names.keys():
            netd_parameters = netd_parameters + list(getattr(self, d).parameters())
        print('Number of parameters in discriminator: ', sum(p.numel() for p in netd_parameters if p.requires_grad))

        self.optimizer_g = optim.Adam(netg_parameters, lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        self.optimizer_d = optim.Adam(netd_parameters, lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        self.net_g_scheduler = get_scheduler(self.optimizer_g, self.hparams)
        self.net_d_scheduler = get_scheduler(self.optimizer_d, self.hparams)
        # not using pl scheduler for now....
        return [self.optimizer_d, self.optimizer_g], []

    def add_loss_adv(self, a, net_d, truth):
        disc_logits = net_d(a)[0]
        if truth:
            adv = self.criterionGAN(disc_logits, torch.ones_like(disc_logits))
        else:
            adv = self.criterionGAN(disc_logits, torch.zeros_like(disc_logits))
        return adv

    def add_loss_l1(self, a, b):
        l1 = self.criterionL1(a, b)
        return l1

    def add_loss_l2(self, a, b):
        l1 = self.criterionL2(a, b)
        return l1

    def save_auc_csv(self, auc, epoch):
        auc = auc.cpu().numpy()
        auc = np.insert(auc, 0, epoch)
        with open(os.path.join(os.environ.get('LOGS'), self.hparams.prj, 'auc.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(auc)

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            self.generation(batch)
            loss_d = self.backward_d()
            if loss_d is not None:
                for k in list(loss_d.keys()):
                    if k != 'sum':
                        self.log(k, loss_d[k], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                return loss_d['sum']
            else:
                return None

        if optimizer_idx == 1:
            if self.hparams.adv > 0:
                self.generation(batch)  # why there are two generation?
                loss_g = self.backward_g()
                for k in list(loss_g.keys()):
                    if k != 'sum':
                        self.log(k, loss_g[k], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                return loss_g['sum']
            else:
                return None

    def training_epoch_end(self, outputs):
        self.train_loader.dataset.shuffle_images()
        try:
            self.eval_loader.dataset.shuffle_images()
        except:
            pass

        # checkpoint
        if self.epoch % self.hparams.epoch_save == 0:
            for name in self.netg_names.keys():
                path_g = self.dir_checkpoints + ('/' + self.netg_names[name] + '_model_epoch_{}.pth').format(self.epoch)
                torch.save(getattr(self, name), path_g)
                print("Checkpoint saved to {}".format(path_g))

            if self.hparams.save_d:
                for name in self.netd_names.keys():
                    path_d = self.dir_checkpoints + ('/' + self.netd_names[name] + '_model_epoch_{}.pth').format(self.epoch)
                    torch.save(getattr(self, name), path_d)
                    print("Checkpoint saved to {}".format(path_d))

        self.net_g_scheduler.step()
        self.net_d_scheduler.step()

        # log saved images
        for k in self.log_image.keys():
            self.save_tensor_to_png(self.log_image[k], self.dir_checkpoints + os.path.join(str(self.epoch).zfill(4) + k + '.png'))

        self.reset_metrics()
        self.epoch += 1

    def get_metrics(self):
        pass

    def reset_metrics(self):
        pass

    def testing_step(self, batch, batch_idx):
        self.generation(batch)

    def validation_epoch_end(self, x):
        #self.log_helper.print(logger=self.logger, epoch=self.epoch)
        #self.log_helper.clear()
        return None

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        if 'loss' in tqdm_dict:
            del tqdm_dict['loss']
        return tqdm_dict

    def generation(self, batch):
        pass

    def backward_g(self):
        pass

    def backward_d(self):
        pass

    def set_networks(self, net='all'):
        # GENERATOR
        if (self.hparams.netG).startswith('de'):  # DeScarGan
            print('descargan generator: ' + self.hparams.netG)
            Generator = getattr(getattr(__import__('networks.DeScarGan.' + self.hparams.netG), 'DeScarGan'),
                                self.hparams.netG).Generator
            net_g = Generator(n_channels=self.hparams.input_nc, out_channels=self.hparams.output_nc,
                              batch_norm={'batch': True, 'none': False}[self.hparams.norm],  # only bn or none
                              final=self.hparams.final,
                              mc=self.hparams.mc)

        elif (self.hparams.netG).startswith('ed'):  # EncoderDecoder
            print('EncoderDecoder generator: ' + self.hparams.netG)
            Generator = getattr(getattr(__import__('networks.EncoderDecoder.' + self.hparams.netG), 'EncoderDecoder'),
                                self.hparams.netG).Generator
            # descargan only has options for batchnorm or none
            net_g = Generator(n_channels=self.hparams.input_nc, out_channels=self.hparams.output_nc, nf=self.hparams.ngf,
                              norm_type=self.hparams.norm, final=self.hparams.final, mc=self.hparams.mc)

        elif (self.hparams.netG).startswith('ldm'):  # ldm
            print('ldm generator: ' + self.hparams.netG)

            with open('networks/ldm/' + self.hparams.netG + '.yaml', "r") as f:
                config = yaml.load(f, Loader=yaml.Loader)

            ddconfig = config['model']['params']["ddconfig"]
            Generator = getattr(getattr(__import__('networks.ldm.' + 'ae'), 'ldm'), 'ae').AE
            net_g = Generator(ddconfig)

        else:
            from networks.networks import define_G
            net_g = define_G(input_nc=self.hparams.input_nc, output_nc=self.hparams.output_nc,
                             ngf=self.hparams.ngf, netG=self.hparams.netG,
                             norm=self.hparams.norm, use_dropout=self.hparams.mc, init_type='normal', init_gain=0.02, gpu_ids=[],
                             final=self.hparams.final)

        # DISCRIMINATOR
        if (self.hparams.netD).startswith('patch'):  # Patchgan from cyclegan (the pix2pix one is strange)
            from networks.cyclegan.models import Discriminator
            net_d = Discriminator(input_shape=(self.hparams.output_nc * 1, 256, 256), patch=int((self.hparams.netD).split('_')[-1]),
                                  ndf=self.hparams.ndf)
        else:
            from networks.networks import define_D
            net_d = define_D(input_nc=self.hparams.output_nc * 1, ndf=64, netD=self.hparams.netD)

        if net == 'all':
            return net_g, net_d
        elif net == 'g':
            return net_g
        elif net == 'd':
            return net_d

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

