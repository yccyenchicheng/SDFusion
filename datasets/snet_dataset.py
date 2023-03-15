"""
    adopted from: https://github.com/shubhtuls/PixelTransformer/blob/03b65b8612fe583b3e35fc82b446b5503dd7b6bd/data/shapenet.py
"""
import os.path
import json

import h5py
import numpy as np
from termcolor import colored, cprint

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from datasets.base_dataset import BaseDataset

# from https://github.com/laughtervv/DISN/blob/master/preprocessing/info.json
class ShapeNetDataset(BaseDataset):

    def initialize(self, opt, phase='train', cat='all', res=64):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size
        self.res = res

        dataroot = opt.dataroot
        # with open(f'{dataroot}/ShapeNet/info.json') as f:
        with open(f'dataset_info_files/info-shapenet.json') as f:
            self.info = json.load(f)
            
        self.cat_to_id = self.info['cats']
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}
        
        if cat == 'all':
            all_cats = self.info['all_cats']
        else:
            all_cats = [cat]

        self.model_list = []
        self.cats_list = []
        for c in all_cats:
            synset = self.info['cats'][c]
            # with open(f'{dataroot}/ShapeNet/filelists/{synset}_{phase}.lst') as f:
            with open(f'dataset_info_files/ShapeNet_filelists/{synset}_{phase}.lst') as f:
                model_list_s = []
                for l in f.readlines():
                    model_id = l.rstrip('\n')
                    
                    # path = f'{dataroot}/ShapeNet/SDF_v1_64/{synset}/{model_id}/ori_sample_grid.h5'
                    path = f'{dataroot}/ShapeNet/SDF_v1/resolution_{self.res}/{synset}/{model_id}/ori_sample_grid.h5'

                    
                    if os.path.exists(path):
                        model_list_s.append(path)
                
                self.model_list += model_list_s
                self.cats_list += [synset] * len(model_list_s)
                print('[*] %d samples for %s (%s).' % (len(model_list_s), self.id_to_cat[synset], synset))

        np.random.default_rng(seed=0).shuffle(self.model_list)
        np.random.default_rng(seed=0).shuffle(self.cats_list)

        self.model_list = self.model_list[:self.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.model_list)), 'yellow')

        self.N = len(self.model_list)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __getitem__(self, index):

        synset = self.cats_list[index]
        sdf_h5_file = self.model_list[index]
        
        h5_f = h5py.File(sdf_h5_file, 'r')
        sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
        sdf = torch.Tensor(sdf).view(1, self.res, self.res, self.res)
        # print(sdf.shape)
        # sdf = sdf[:, :64, :64, :64]

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)

        ret = {
            'sdf': sdf,
            'cat_id': synset,
            'cat_str': self.id_to_cat[synset],
            'path': sdf_h5_file,
        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'ShapeNetSDFDataset'