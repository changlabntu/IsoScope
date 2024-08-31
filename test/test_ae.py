import torch
import tifffile as tiff
from utils.get_args import get_args
from models.ae0 import GAN
from models.autoencoder_edit import AutoencoderKL
from utils.data_utils import imagesc
import os, glob
import yaml

parser = get_args()

args = parser.parse_args()
args.not_tracking_hparams = []
args.gan_mode = 'vanilla'
args.embed_dim = 4
args.lr = 0.001
args.beta1 = 0.5

gan = GAN(args, train_loader=None, eval_loader=None, checkpoints=None)


gan.encoder = torch.load('/media/ExtHDD01/logs/Fly0B/ae/ae0g6/checkpoints/encoder_model_epoch_1900.pth', map_location=torch.device('cpu'))
gan.decoder = torch.load('/media/ExtHDD01/logs/Fly0B/ae/ae0g6/checkpoints/decoder_model_epoch_1900.pth', map_location=torch.device('cpu'))
gan.quant_conv = torch.load('/media/ExtHDD01/logs/Fly0B/ae/ae0g6/checkpoints/quant_conv_model_epoch_1900.pth', map_location=torch.device('cpu'))
gan.post_quant_conv = torch.load('/media/ExtHDD01/logs/Fly0B/ae/ae0g6/checkpoints/post_quant_conv_model_epoch_1900.pth', map_location=torch.device('cpu'))


img_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/Fly0B/train/xyori0828/*'))
x = tiff.imread(img_list[0])
x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)#.cuda()

dec, posterior = gan(x)
y = dec.squeeze().detach().numpy()

gan.save_checkpoint('test.ckpt')

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

# Assuming 'model' is an instance of your GAN class
#
#loaded_model = GAN.load_from_checkpoint('test.ckpt', train_loader, eval_loader, checkpoints)
#loaded_model = GAN.load_from_checkpoint('/media/ExtHDD01/ldmlogs/Fly0B/2024-08-13T16-08-51_yztoxy_ori_Fly0B_gan_ae3x2/checkpoints/epoch=000661.ckpt')#, train_loader, eval_loader, checkpoints)