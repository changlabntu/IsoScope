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


def test_womac4ae():
    #prj ='/Fly0B/ae/ae0discstart0/'
    #model_names = ['encoder', 'decoder', 'post_quant_conv', 'quant_conv']

    prj = '/womac4/ae/cyc0/'
    model_names = ['encoder', 'decoder', 'net_g', 'post_quant_conv', 'quant_conv']
    epoch = str(60)

    root = '/media/ExtHDD01/logs/' + prj
    args = read_json_to_args(root + '0.json')

    #args.ldmyaml = 'ldmaex2x2'

    GAN = getattr(__import__('models.' + args.models), args.models).GAN
    gan = GAN(args, train_loader=None, eval_loader=None, checkpoints=None)

    gan = load_pth(gan, root=root, epoch=epoch, model_names=model_names)

    #img_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/Fly0B/train/xyori0828/*'))
    #oriX = torch.from_numpy(np.stack([tiff.imread(y) for y in img_list[32:64]], 2)).unsqueeze(0).unsqueeze(1).float()

    oriX = tiff.imread('/media/ExtHDD01/oai_diffusion_interpolated/original/a2d/9074878_01.tif')
    #oriX[oriX>= 800] = 800
    oriX = oriX / oriX.max()
    oriX = (oriX - 0.5) / 0.5
    oriX = torch.from_numpy(oriX).unsqueeze(1).float()

    xx = oriX
    reconstructions, posterior, hbranch = gan.forward(xx[:32, :, :, :], sample_posterior=False)
    #imagesc(oriX[0, 0, :, :, 10])
    #imagesc(reconstructions[10, 0, :, :].detach())
    # hbranch (1, 256, 8, 8)
    hbranch = hbranch.permute(1, 2, 3, 0).unsqueeze(0)
    XupX = gan.net_g(hbranch, method='decode')['out0']

    Xup = torch.nn.Upsample(size=(384, 384, 184), mode='trilinear')(xx[:32, :,:,:].permute(1, 2, 3, 0).unsqueeze(0))

    Xup = Xup.permute(3, 0, 1, 4, 2).squeeze().detach().numpy()
    XupX = XupX.permute(3, 0, 1, 4, 2).squeeze().detach().numpy()

    imagesc(Xup[200, :, :])
    imagesc(XupX[200, :, :])

    tiff.imwrite('Xup.tif', Xup)
    tiff.imwrite('XupX.tif', XupX)


def test_womac4_vae():
    from models.vae import GAN as VAE
    args = read_json_to_args('/media/ExtHDD01/logs/womac4/vae/0/0.json')
    vae = VAE(args, train_loader=None, eval_loader=None, checkpoints=None)
    net = torch.load('/media/ExtHDD01/logs/womac4/vae/0/checkpoints/generator_model_epoch_500.pth', map_location=torch.device('cpu'))
    vae.generator = net.eval()
    img_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/womac4/full/a/*'))
    x = tiff.imread(img_list[41])
    x[x >= 800] = 800
    x = x / x.max()
    x = (x - 0.5) / 0.5
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).unsqueeze(4).float()
    recon, mu, logvar = vae(x)
    imagesc(recon.squeeze().detach().numpy())



def test_Fly0B_gan():
    #gan = torch.load('/media/ExtHDD01/logs/Fly0B/IsoScopeXXcut/ngf32lb10/checkpoints/net_g_model_epoch_2000.pth',
    #                 map_location=torch.device('cpu'))

    gan = torch.load('/media/ExtHDD01/logs/Fly0B/IsoScopeXY/ngf32lb10skip4/checkpoints/net_g_model_epoch_2800.pth',
                     map_location=torch.device('cpu'))

    img_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/Fly0B/train/xyori0828/*'))
    oriX = torch.from_numpy(np.stack([tiff.imread(y) for y in img_list[32:64]], 2)).unsqueeze(0).unsqueeze(1).float()

    xx = oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]

    xx = torch.nn.Upsample(size=(256, 256, 256), mode='trilinear')(xx.permute(1, 2, 3, 0).unsqueeze(0))

    out = gan(xx)['out0']

    out = out.permute(3, 0, 1, 4, 2).squeeze().detach().numpy()
    tiff.imwrite('XupXgan.tif', out)


def test_DPM4X(prj, epoch):
    # BEST
    #prj = '/Fly0B/ae/iso0_ldmaex2_lb10/'
    epoch = str(epoch)

    model_names = ['encoder', 'decoder', 'net_g', 'post_quant_conv', 'quant_conv']
    root = '/media/ExtHDD01/logs/' + prj
    args = read_json_to_args(root + '0.json')

    GAN = getattr(__import__('models.' + args.models), args.models).GAN
    gan = GAN(args, train_loader=None, eval_loader=None, checkpoints=None)

    gan = load_pth(gan, root=root, epoch=epoch, model_names=model_names)

    #img_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/DPM4X/xyoriexp0/*'))
    #oriX = torch.from_numpy(np.stack([tiff.imread(y) for y in img_list[800:832]], 2)).unsqueeze(0).unsqueeze(1).float()

    img_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/DPM4X/oripatch/*'))
    oriX = torch.from_numpy(np.stack([tiff.imread(y) for y in img_list[64:96]], 2)).unsqueeze(0).unsqueeze(1).float()

    xx = oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]
    reconstructions, posterior, hbranch = gan.forward(xx[:32, :, :, :], sample_posterior=False)
    imagesc(oriX[0, 0, :, :, 10])
    imagesc(reconstructions[10, 0, :, :].detach())
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


def test_Fly0B(prj, epoch):

    model_names = ['encoder', 'decoder', 'net_g', 'post_quant_conv', 'quant_conv']
    root = '/media/ExtHDD01/logs/' + prj
    args = read_json_to_args(root + '0.json')

    GAN = getattr(__import__('models.' + args.models), args.models).GAN
    gan = GAN(args, train_loader=None, eval_loader=None, checkpoints=None)

    gan = load_pth(gan, root=root, epoch=epoch, model_names=model_names)

    img_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/Fly0B/train/xyori0828/*'))
    oriX = torch.from_numpy(np.stack([tiff.imread(y) for y in img_list[64:96]], 2)).unsqueeze(0).unsqueeze(1).float()

    xx = oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]
    reconstructions, posterior, hbranch = gan.forward(xx[:32, :, :, :], sample_posterior=False)
    imagesc(oriX[0, 0, :, :, 10])
    imagesc(reconstructions[10, 0, :, :].detach())
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


def test_DPM4Xtc(prj, epoch, irange, hbranchz=False, tc=True, masking=False):

    model_names = ['encoder', 'decoder', 'net_g', 'post_quant_conv', 'quant_conv']
    root = '/media/ExtHDD01/logs/' + prj
    args = read_json_to_args(root + '0.json')

    GAN = getattr(__import__('models.' + args.models), args.models).GAN
    gan = GAN(args, train_loader=None, eval_loader=None, checkpoints=None)
    gan = load_pth(gan, root=root, epoch=str(epoch), model_names=model_names)

    img_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/DPM4X/oripatch/*'))
    oriX = torch.from_numpy(np.stack([tiff.imread(y) for y in img_list[irange[0]:irange[1]]], 2)).unsqueeze(0).unsqueeze(1).float()
    img_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/DPM4X/ft0patch/*'))
    oriF = torch.from_numpy(np.stack([tiff.imread(y) for y in img_list[irange[0]:irange[1]]], 2)).unsqueeze(0).unsqueeze(1).float()

    if tc:
        oriX = torch.cat([oriX, oriF], 1)

    xx = oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]
    ff = oriF.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]
    reconstructions, posterior, hbranch = gan.forward(xx[:32, :, :, :], sample_posterior=False)
    imagesc(oriX[0, 0, :, :, 10])
    imagesc(reconstructions[10, 0, :, :].detach())
    # hbranch (1, 256, 8, 8)
    hbranch = hbranch.permute(1, 2, 3, 0).unsqueeze(0)

    if hbranchz:
        hbranch = posterior.sample()
        hbranch = gan.decoder.conv_in(hbranch)
        hbranch = hbranch.permute(1, 2, 3, 0).unsqueeze(0)

    XupX = gan.net_g(hbranch, method='decode')['out0']

    Xup = torch.nn.Upsample(size=(256, 256, 256), mode='trilinear')(xx.permute(1, 2, 3, 0).unsqueeze(0))
    fup = torch.nn.Upsample(size=(256, 256, 256), mode='trilinear')(ff.permute(1, 2, 3, 0).unsqueeze(0))

    Xup = Xup.permute(3, 0, 1, 4, 2).squeeze(1).detach().numpy()
    XupX = XupX.permute(3, 0, 1, 4, 2).squeeze(1).detach().numpy()
    fup = fup.permute(3, 0, 1, 4, 2).squeeze(1).detach().numpy()

    imagesc(Xup[200, 0, :, :])
    imagesc(XupX[200, 0, :, :])
    #tiff.imwrite('Xup.tif', Xup[:, 0, :, :])
    #tiff.imwrite('XupX.tif', XupX[:, 0, :, :])

    if tc:
        to_print = [Xup[:, 0, :, :], XupX[:, 0, :, :], Xup[:, 1, :, :], XupX[:, 1, :, :]]
    else:
        to_print = [Xup[:, 0, :, :], XupX[:, 0, :, :], fup[:, 0, :, :]]

    # normalize to 0-1
    to_print = [(x - x.min()) for x in to_print]
    to_print = [(x / x.max()) for x in to_print]

    if masking:
        to_print[1] = np.multiply(to_print[1], to_print[2])

    # normalize by mean and std
    to_print = [(x - x.mean()) for x in to_print]
    to_print = [(x / x.std()) for x in to_print]

    tiff.imwrite('out.tif', np.concatenate(to_print, 2))


####
def womac4_onlyae0():
    prj = '/womac4/ae/onlyae0/'
    model_names = ['encoder', 'decoder', 'post_quant_conv', 'quant_conv']
    epoch = str(160)

    root = '/media/ExtHDD01/logs/' + prj
    args = read_json_to_args(root + '0.json')

    GAN = getattr(__import__('models.' + args.models), args.models).GAN
    gan = GAN(args, train_loader=None, eval_loader=None, checkpoints=None)

    gan = load_pth(gan, root=root, epoch=epoch, model_names=model_names)

    # img_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/Fly0B/train/xyori0828/*'))
    # oriX = torch.from_numpy(np.stack([tiff.imread(y) for y in img_list[32:64]], 2)).unsqueeze(0).unsqueeze(1).float()

    oriX = tiff.imread('/media/ExtHDD01/oai_diffusion_interpolated/original/a2d/9047539_08.tif')
    # oriX[oriX>= 800] = 800
    oriX = oriX / oriX.max()
    oriX = (oriX - 0.5) / 0.5
    oriX = torch.from_numpy(oriX).unsqueeze(1).float()

    xx = oriX
    reconstructions, posterior, hbranch = gan.forward(xx[:32, :, :, :], sample_posterior=True)
    imagesc(oriX[15, 0, :, :].detach())
    imagesc(reconstructions[15, 0, :, :].detach())

    tiff.imwrite('oriX.tif', oriX.squeeze().detach().numpy())
    tiff.imwrite('reconstructions.tif', reconstructions.squeeze().detach().numpy())
    tiff.imwrite('diff.tif', (oriX - reconstructions).squeeze().detach().numpy())


def load_pth(gan, root, epoch, model_names):
    for name in model_names:
        setattr(gan, name, torch.load(root + 'checkpoints/' + name + '_model_epoch_' + epoch + '.pth', map_location=torch.device('cpu')))
    return gan
    #gan.encoder = torch.load(root + 'checkpoints/encoder_model_epoch_' + epoch + '.pth', map_location=torch.device('cpu'))
    #gan.decoder = torch.load(root + 'checkpoints/decoder_model_epoch_' + epoch + '.pth', map_location=torch.device('cpu'))
    #gan.quant_conv = torch.load(root + 'checkpoints/quant_conv_model_epoch_' + epoch + '.pth', map_location=torch.device('cpu'))
    #gan.post_quant_conv = torch.load(root + 'checkpoints/post_quant_conv_model_epoch_' + epoch + '.pth', map_location=torch.device('cpu'))


#test_womac4ae()
#test_Fly0B(prj='/Fly0B/ae/iso0_ldmaex2x2_lb10/', epoch=500)

destination = '/media/ExtHDD01/logs/DPM4X/ae/iso0_ldmaex2_lb10/'
#test_DPM4Xtc(prj='/DPM4X/ae/iso0_ldmaex2_lb10/', epoch=1000, irange=(3968, 3968+32), tc=False, masking=False)
test_DPM4Xtc(prj='/DPM4X/ae/iso0_ldmaex2_lb10_tc/', epoch=1500, irange=(3968, 3968+32))
#test_DPM4Xtc(prj='/DPM4X/ae/iso0_ldmaex2_lb10_tc_oril1/', epoch=1000, irange=(3968, 3968+32))
#test_DPM4Xtc(prj='/DPM4X/ae/iso0_ldmaex2_lb10_tc_oril1_hbranchz/', epoch=1500, irange=(3968, 3968+32), hbranchz=True)
#test_DPM4Xtc(prj='/DPM4X/ae/2d/3_l1max_advmax/', epoch=500, irange=(3968, 3968+32), hbranchz=True, tc=False)


#womac4_onlyae0()
