"""
    adopted from: https://github.com/shubhtuls/PixelTransformer/blob/03b65b8612fe583b3e35fc82b446b5503dd7b6bd/data/shapenet.py
"""

import h5py
import numpy as np
from PIL import Image
from termcolor import colored, cprint

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms

from datasets.base_dataset import BaseDataset


# from https://github.com/laughtervv/DISN/blob/master/preprocessing/info.json
class BuildingNetDataset(BaseDataset):

    def initialize(self, opt, phase='train', cat='all', res=64):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size
        self.res = res

        dataroot = opt.dataroot
        file_list = f'{dataroot}/BuildingNet_dataset_v0_1/splits/{phase}_split.txt'

        SDF_dir = f'{dataroot}/BuildingNet_dataset_v0_1/SDF/resolution_{res}'

        self.model_list = []
        self.z_list = []
        with open(file_list) as f:
            model_list_s = []
            z_list_s = []
            for l in f.readlines():
                model_id = l.rstrip('\n')
                
                path = f'{SDF_dir}/{model_id}/ori_sample_grid.h5'
                model_list_s.append(path)
            
            self.model_list += model_list_s
            self.z_list += z_list_s

        np.random.default_rng(seed=0).shuffle(self.model_list)
        np.random.default_rng(seed=0).shuffle(self.z_list)

        self.model_list = self.model_list[:self.max_dataset_size]
        self.z_list = self.z_list[:self.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.model_list)), 'yellow')

        self.N = len(self.model_list)

        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):

        sdf_h5_file = self.model_list[index]
        
        h5_f = h5py.File(sdf_h5_file, 'r')
        sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
        sdf = torch.Tensor(sdf).view(1, self.res, self.res, self.res)

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)

        ret = {
            'sdf': sdf,
            'path': sdf_h5_file,
        }

        if self.load_from_cached:
            z_path = self.z_list[index]
            z = torch.from_numpy(np.load(z_path))
            ret['z'] = z
            ret['z_path'] = z_path

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return f'BuildingNetDataset-{self.res}'