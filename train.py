from __future__ import print_function
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import os, shutil, time, sys
from tqdm import tqdm
from dotenv import load_dotenv
from utils.make_config import load_json, save_json
import json
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from dataloader.data_multi import MultiData as Dataset
from utils.get_args import get_args

os.environ['OPENBLAS_NUM_THREADS'] = '1'


def prepare_log(args):
    """
    finalize arguments, creat a folder for logging, save argument in json
    """
    args.not_tracking_hparams = []  # 'mode', 'port', 'epoch_load', 'legacy', 'threads', 'test_batch_size']
    os.makedirs(os.environ.get('LOGS') + args.dataset + '/', exist_ok=True)
    os.makedirs(os.environ.get('LOGS') + args.dataset + '/' + args.prj + '/', exist_ok=True)
    save_json(args, os.environ.get('LOGS') + args.dataset + '/' + args.prj + '/' + '0.json')
    shutil.copy('models/' + args.models + '.py',
                os.environ.get('LOGS') + args.dataset + '/' + args.prj + '/' + args.models + '.py')
    return args


if __name__ == '__main__':
    parser = get_args()

    # Model-specific Arguments
    models = parser.parse_known_args()[0].models
    GAN = getattr(__import__('models.' + models), models).GAN
    parser = GAN.add_model_specific_args(parser)

    # Read json file and update it
    with open('env/jsn/' + parser.parse_args().jsn + '.json', 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f)['train'])
        args = parser.parse_args(namespace=t_args)

    # environment file
    if args.env is not None:
        load_dotenv('env/.' + args.env)
    else:
        load_dotenv('env/.t09')

    # Finalize Arguments and create files for logging
    args.bash = ' '.join(sys.argv)
    args = prepare_log(args)
    print(args)

    # Load Dataset and DataLoader
    train_set = Dataset(root=os.environ.get('DATASET') + args.dataset + '/train/',
                        path=args.direction,
                        opt=args, mode='train', index=None, filenames=True)

    train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, drop_last=True)

    try:
        eval_set = Dataset(root=os.environ.get('DATASET') + args.dataset + '/val/',
                           path=args.direction,
                           opt=args, mode='test', index=None, filenames=True)
        eval_loader = DataLoader(dataset=eval_set, num_workers=1, batch_size=args.test_batch_size, shuffle=False,
                                 pin_memory=True)
    except:
        eval_loader = None
        print('No validation set')

    # preload
    if args.preload:
        tini = time.time()
        print('Preloading...')
        for i, x in enumerate(tqdm(train_loader)):
            pass
        if eval_loader is not None:
            for i, x in enumerate(tqdm(eval_loader)):
                pass
        print('Preloading time: ' + str(time.time() - tini))

    # Logger
    os.makedirs(os.path.join(os.environ.get('LOGS'), args.dataset, args.prj, 'logs'), exist_ok=True)
    logger = pl_loggers.TensorBoardLogger(os.path.join(os.environ.get('LOGS'), args.dataset, args.prj, 'logs'))

    # Trainer
    checkpoints = os.path.join(os.environ.get('LOGS'), args.dataset, args.prj, 'checkpoints')
    os.makedirs(checkpoints, exist_ok=True)
    #if eval_loader is not None:
    net = GAN(hparams=args, train_loader=train_loader, eval_loader=eval_loader, checkpoints=checkpoints)
    #else:
    #net = GAN(hparams=args, train_loader=train_loader, checkpoints=checkpoints)
    trainer = pl.Trainer(gpus=-1, strategy='ddp_spawn',
                         max_epochs=args.n_epochs,  # progress_bar_refresh_rate=20,
                         logger=logger,
                         enable_checkpointing=True, log_every_n_steps=100,
                         check_val_every_n_epoch=1, accumulate_grad_batches=2)
    if eval_loader is not None:
        trainer.fit(net, train_loader, eval_loader)
    else:
        trainer.fit(net, train_loader)

    #train(net, args, train_set, eval_set, loss_function, metrics)

    #print(len(train_set.subset[0].images))
    #print(len(test_set.subset[0].images))
    #print(len(set(train_set.subset[0].images + test_set.subset[0].images)))

    # Examples of  Usage
    # CUDA_VISIBLE_DEVICES=3 python train.py --prj test --models lesion_cutGB_xbm --jsn intersubject --lbX 1 --cropsize 256 -b 1 --xbm --start_iter 0 --xbm_size 1000

def quick_copy():
    import glob
    import tifffile as tiff
    import numpy as np
    root = '/media/ExtHDD01/oai_diffusion_interpolated/original/'
    a2d = sorted(glob.glob(root + 'a2d/*.tif'))
    add = sorted(glob.glob(root + 'addpm2d0504/*.tif'))
    for i in range(len(a2d)):
        a = tiff.imread(a2d[i])
        b = tiff.imread(add[i])
        (a, b) = (x - x.mean() for x in (a, b))
        (a, b) = (x / x.std() for x in (a, b))
        diff = a - b
        diff[diff < 0] = 0
        #diff = diff.astype(np.float32)
        rgb = np.stack([b, b+diff, b], 3)
        # turn to unit8
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255
        rgb = rgb.astype(np.uint8)
        tiff.imsave(root + 'diff0504/' + a2d[i].split('/')[-1], rgb)
