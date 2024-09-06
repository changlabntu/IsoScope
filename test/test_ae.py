import torch
import tifffile as tiff
from utils.get_args import get_args
from models.ae0 import GAN
from models.autoencoder_edit import AutoencoderKL
from utils.data_utils import imagesc
import os, glob
import yaml
import argparse, json
from utils.data_utils import read_json_to_args
import numpy as np



if 0:
    parser = get_args()

    args = parser.parse_args()
    args.not_tracking_hparams = []
    args.gan_mode = 'vanilla'
    args.embed_dim = 4
    args.lr = 0.001
    args.beta1 = 0.5

    gan = GAN(args, train_loader=None, eval_loader=None, checkpoints=None)

    prj ='/ae/ae0discstart0/'
    epoch = str(2000)

    gan.encoder = torch.load('/media/ExtHDD01/logs/Fly0B' + prj + 'checkpoints/encoder_model_epoch_' + epoch + '.pth', map_location=torch.device('cpu'))
    gan.decoder = torch.load('/media/ExtHDD01/logs/Fly0B' + prj + 'checkpoints/decoder_model_epoch_' + epoch + '.pth', map_location=torch.device('cpu'))
    gan.quant_conv = torch.load('/media/ExtHDD01/logs/Fly0B' + prj + 'checkpoints/quant_conv_model_epoch_' + epoch + '.pth', map_location=torch.device('cpu'))
    gan.post_quant_conv = torch.load('/media/ExtHDD01/logs/Fly0B' + prj + 'checkpoints/post_quant_conv_model_epoch_' + epoch + '.pth', map_location=torch.device('cpu'))


    img_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/Fly0B/train/xyori0828/*'))
    x = tiff.imread(img_list[0])
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)#.cuda()

    dec, posterior = gan(x)
    y = dec.squeeze().detach().numpy()

    gan.save_checkpoint('test.ckpt')

    imagesc(x.squeeze())
    imagesc(y)

    # build ae
    # Initialize encoder and decoder
    with open('networks/ldm/ldmaex2.yaml', "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config = config['model']['params']

    ae = AutoencoderKL(config["ddconfig"],
                     config["lossconfig"],
                     embed_dim=4,
                     ckpt_path='test.ckpt',#'/media/ExtHDD01/ldmlogs/Fly0B/2024-08-13T16-08-51_yztoxy_ori_Fly0B_gan_ae3x2/checkpoints/epoch=000661.ckpt',
                     ignore_keys=[],
                     image_key="image",
                     colorize_nlabels=None,
                     monitor=None,)


def test_womac4():
    from models.vae import GAN as VAE
    args = read_json_to_args('/media/ExtHDD01/logs/womac4/vae/0/0.json')
    vae = VAE(args, train_loader=None, eval_loader=None, checkpoints=None)
    net = torch.load('/media/ExtHDD01/logs/womac4/vae/0/checkpoints/generator_model_epoch_1000.pth', map_location=torch.device('cpu'))
    vae.generator = net#.eval()

    img_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/womac4/full/a/*'))
    x = tiff.imread(img_list[41])
    x[x >= 800] = 800
    x = x / x.max()
    x = (x - 0.5) / 0.5
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).unsqueeze(4).float()
    recon, mu, logvar = vae.forward(x, use_posterior=True)
    imagesc(recon.squeeze().detach().numpy())

#test_womac4()


def test_Fly0B():
    #prj ='/Fly0B/ae/ae0discstart0/'
    #model_names = ['encoder', 'decoder', 'post_quant_conv', 'quant_conv']

    prj = '/Fly0B/ae/iso0/'
    model_names = ['encoder', 'decoder', 'net_g', 'post_quant_conv', 'quant_conv']
    epoch = str(2000)

    root = '/media/ExtHDD01/logs/' + prj
    args = read_json_to_args(root + '0.json')

    #args.ldmyaml = 'ldmaex2x2'

    GAN = getattr(__import__('models.' + args.models), args.models).GAN
    gan = GAN(args, train_loader=None, eval_loader=None, checkpoints=None)

    gan = load_pth(gan, root=root, epoch=epoch, model_names=model_names)

    img_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/Fly0B/train/xyori0828/*'))
    oriX = torch.from_numpy(np.stack([tiff.imread(y) for y in img_list[32:64]], 2)).unsqueeze(0).unsqueeze(1).float()

    xx = oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]
    reconstructions, posterior, hbranch = gan.forward(xx[:32, :, :, :], sample_posterior=False)
    # hbranch (1, 256, 8, 8)
    hbranch = hbranch.permute(1, 2, 3, 0).unsqueeze(0)
    XupX = gan.net_g(hbranch, method='decode')['out0']

    Xup = torch.nn.Upsample(size=(256, 256, 256), mode='trilinear')(xx.permute(1, 2, 3, 0).unsqueeze(0))

    Xup = Xup.permute(3, 0, 1, 4, 2).squeeze().detach().numpy()
    XupX = XupX.permute(3, 0, 1, 4, 2).squeeze().detach().numpy()

    imagesc(Xup[200, :, :])
    imagesc(XupX[200, :, :])

    tiff.imwrite('Xup.tif', Xup)
    tiff.imwrite('XupX.tif', XupX)

def load_pth(gan, root, epoch, model_names):
    for name in model_names:
        setattr(gan, name, torch.load(root + 'checkpoints/' + name + '_model_epoch_' + epoch + '.pth', map_location=torch.device('cpu')))
    return gan
    #gan.encoder = torch.load(root + 'checkpoints/encoder_model_epoch_' + epoch + '.pth', map_location=torch.device('cpu'))
    #gan.decoder = torch.load(root + 'checkpoints/decoder_model_epoch_' + epoch + '.pth', map_location=torch.device('cpu'))
    #gan.quant_conv = torch.load(root + 'checkpoints/quant_conv_model_epoch_' + epoch + '.pth', map_location=torch.device('cpu'))
    #gan.post_quant_conv = torch.load(root + 'checkpoints/post_quant_conv_model_epoch_' + epoch + '.pth', map_location=torch.device('cpu'))


