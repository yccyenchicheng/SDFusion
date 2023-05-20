import os
import argparse

from termcolor import colored
from omegaconf import OmegaConf

import torch
from torch.utils.tensorboard import SummaryWriter

import utils

from utils.distributed import (
    get_rank,
    synchronize,
)

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # hyper parameters
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        # log stuff
        self.parser.add_argument('--logs_dir', type=str, default='./logs', help='the root of the logs dir. All training logs are saved here')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')

        # dataset stuff
        self.parser.add_argument('--dataroot', type=str, default=None, help='root dir for data. if None, specify by `hostname` in configs/paths.py')
        self.parser.add_argument('--dataset_mode', type=str, default='snet', help='chooses how datasets are loaded. [mnist, snet, abc, snet-abc]')
        self.parser.add_argument('--res', type=int, default=64, help='dataset resolution')
        self.parser.add_argument('--cat', type=str, default='chair', help='category for shapenet')
        self.parser.add_argument('--trunc_thres', type=float, default=0.2, help='threshold for truncated sdf.')
        
        self.parser.add_argument('--ratio', type=float, default=1., help='ratio of the dataset to use. for debugging and overfitting')
        self.parser.add_argument('--max_dataset_size', default=2147483648, type=int, help='chooses the maximum dataset size.')
        self.parser.add_argument('--nThreads', default=9, type=int, help='# threads for loading data')        
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')

        ############## START: model related options ################
        self.parser.add_argument(
                            '--model', type=str, default='sdfusion',
                            choices=['vqvae', 'sdfusion', 'sdfusion-img2shape', 'sdfusion-txt2shape','sdfusion-mm2shape'],
                            help='chooses which model to use.'
                        )
        self.parser.add_argument('--ckpt', type=str, default=None, help='ckpt to load.')

        # diffusion stuff
        self.parser.add_argument('--df_cfg', type=str, default='configs/sdfusion_snet.yaml', help="diffusion model's config file")
        self.parser.add_argument('--ddim_steps', type=int, default=100, help='steps for ddim sampler')
        self.parser.add_argument('--ddim_eta', type=float, default=0.0)
        self.parser.add_argument('--uc_scale', type=float, default=1.0, help='scale for un guidance')
        
        # vqvae stuff
        self.parser.add_argument('--vq_model', type=str, default='vqvae', help='for choosing the vqvae model to use.')
        self.parser.add_argument('--vq_cfg', type=str, default='configs/vqvae_snet.yaml', help='vqvae model config file')
        self.parser.add_argument('--vq_dset', type=str, default=None, help='dataset vqvae originally trained on')
        self.parser.add_argument('--vq_cat', type=str, default=None, help='dataset category vqvae originally trained on')
        self.parser.add_argument('--vq_ckpt', type=str, default=None, help='vqvae ckpt to load.')
        ############## END: model related options ################

        # misc
        self.parser.add_argument('--debug', default='0', type=str, choices=['0', '1'], help='if true, debug mode')
        self.parser.add_argument('--seed', default=111, type=int, help='seed')

        # multi-gpu stuff
        self.parser.add_argument("--backend", type=str, default="gloo", help="which backend to use")
        self.parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")

        self.initialized = True

    def parse_and_setup(self):
        import sys
        cmd = ' '.join(sys.argv)
        print(f'python {cmd}')

        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        if self.opt.isTrain:
            self.opt.phase = 'train'
        else:
            self.opt.phase = 'test'

        # setup multi-gpu stuffs here
        # basically from stylegan2-pytorch, train.py by rosinality
        self.opt.device = 'cuda'
        n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        self.opt.distributed = n_gpu > 1

        if self.opt.distributed:
            torch.cuda.set_device(self.opt.local_rank)
            torch.distributed.init_process_group(backend=self.opt.backend, init_method="env://")
            synchronize()

        name = self.opt.name
        if self.opt.isTrain and self.opt.ckpt is not None:
            name = f'continue-{name}'

        self.opt.name = name
        
        self.opt.gpu_ids_str = self.opt.gpu_ids

        # NOTE: seed or not?
        # seed = opt.seed
        # util.seed_everything(seed)

        self.opt.rank = get_rank()

        if get_rank() == 0:
            # print args
            args = vars(self.opt)

            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

            # make experiment dir
            if self.opt.isTrain:
                expr_dir = os.path.join(self.opt.logs_dir, self.opt.name)
                utils.util.mkdirs(expr_dir)
                
                ckpt_dir = os.path.join(self.opt.logs_dir, self.opt.name, 'ckpt')
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                self.opt.ckpt_dir = ckpt_dir
                    
                file_name = os.path.join(expr_dir, 'opt.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k), str(v)))
                    opt_file.write('-------------- End ----------------\n')
                
                # tensorboard writer
                tb_dir = '%s/tboard' % expr_dir
                if not os.path.exists(tb_dir):
                    os.makedirs(tb_dir)
                self.opt.tb_dir = tb_dir
                writer = SummaryWriter(log_dir=tb_dir)
                self.opt.writer = writer

        return self.opt
