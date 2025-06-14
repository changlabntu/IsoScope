import argparse


def get_args():
    # Arguments
    parser = argparse.ArgumentParser()  # add_help=False)
    # Env
    parser.add_argument('--jsn', type=str, default='default', help='name of ini file')
    parser.add_argument('--env', type=str, default=None, help='environment_to_use')
    # Project name
    parser.add_argument('--prj', type=str, help='name of the project')
    parser.add_argument('-m', dest='message', type=str, help='message')
    parser.add_argument('--models', dest='models', type=str, help='use which models')
    # Data
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--preload', action='store_true')
    parser.add_argument('--split', type=str, help='split of data')

    parser.add_argument('--dataset_mode', type=str, help='chooses how datasets are loaded')
    parser.add_argument('--load3d', action='store_true', dest='load3d', default=False, help='do 3D')

    parser.add_argument('--flip', action='store_true', dest='flip', help='image flip left right')
    parser.add_argument('--resize', type=int, help='size for resizing before cropping, 0 for no resizing')
    parser.add_argument('--cropsize', type=int, help='size for xy cropping, 0 for no crop')
    parser.add_argument('--cropz', type=int, help='size for z cropping, 0 for no crop')
    parser.add_argument('--rotate', action='store_true')

    parser.add_argument('--direction', type=str, help='paired: a_b, unpaired a%b ex:(a_b%c_d)')
    parser.add_argument('--nm', type=str, help='way to normalize itensity value')
    parser.add_argument('--precrop', type=int, help='size for precrop', default=0)
    parser.add_argument('--rgb', action='store_true', dest='rgb', default=False, help='copy img to 3 channel')
    parser.add_argument('--spd', action='store_true', dest='spd', default=False, help='USE SPADE?')
    # Model
    parser.add_argument('--gan_mode', type=str, help='gan mode')
    parser.add_argument('--netG', type=str, help='netG model')
    parser.add_argument('--norm', type=str, help='normalization in generator')
    parser.add_argument('--mc', action='store_true', dest='mc', default=False,
                        help='monte carlo dropout for some of the generators')
    parser.add_argument('--netD', type=str, help='netD model')
    parser.add_argument('--input_nc', type=int, help='input image channels')
    parser.add_argument('--output_nc', type=int, help='output image channels')
    parser.add_argument('--ngf', type=int, help='generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, help='discriminator filters in first conv layer')
    # parser.add_argument("--n_attrs", type=int, default=1)   # NOT IN USE?
    parser.add_argument('--final', type=str, dest='final', help='activation of final layer')
    parser.add_argument('--cmb', dest='cmb', help='method to combine the outputs to the original')
    parser.add_argument('--trd', type=float, dest='trd', help='threshold of images')
    # Training
    parser.add_argument('--adv', type=float, default=1, help='weight for adversarial loss, 0 for no generator training')
    parser.add_argument('-b', dest='batch_size', type=int, help='training batch size')
    parser.add_argument('--bt', dest='test_batch_size', type=int, help='testing batch size')  # NOT IN USE?
    parser.add_argument('--n_epochs', type=int, help='# of iter at starting learning rate')
    parser.add_argument('--lr', type=float, help='initial learning rate f -or adam')
    parser.add_argument('--beta1', type=float, help='beta1 for adam. default=0.5')
    parser.add_argument('--threads', type=int, help='number of threads for data loader to use')
    # parser.add_argument('--epoch_count', type=int, help='the starting epoch count')
    # parser.add_argument('--epoch_load', type=int, help='to load checkpoint form the epoch count')
    parser.add_argument('--epoch_save', type=int, default=20, help='to save checkpoint every epoch')
    parser.add_argument('--n_epochs_decay', type=int, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--lr_policy', type=str, help='learning rate policy: lambda|step|plateau|cosine')
    parser.add_argument('--lr_decay_iters', type=int, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--save_d', action='store_true', dest='save_d', default=False,
                        help='save checkpoints of discriminators')

    # Loss
    parser.add_argument('--lamb', type=float, help='weight on L1 term in objective')
    # Misc
    parser.add_argument('--seed', type=int, help='random seed to use. Default=123')
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--port', type=str, default='dummy')
    parser.add_argument('--host', type=str, default='dummy')
    return parser