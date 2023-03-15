"""
    adopted from: https://github.com/shubhtuls/PixelTransformer/blob/03b65b8612fe583b3e35fc82b446b5503dd7b6bd/data/shapenet.py
"""
import os.path
import json
import csv
import collections

import h5py
import numpy as np
from PIL import Image
from termcolor import colored, cprint
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from datasets.base_dataset import BaseDataset


# from https://github.com/laughtervv/DISN/blob/master/preprocessing/info.json
class Text2ShapeDataset(BaseDataset):

    def initialize(self, opt, phase='train', cat='all', res=64):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size
        self.res = res

        dataroot = opt.dataroot

        self.text_csv = f'{dataroot}/ShapeNet/text2shape/captions.tablechair_{phase}.csv'

        with open(self.text_csv) as f:
            reader = csv.reader(f, delimiter=',')
            self.header = next(reader, None)

            self.data = [row for row in reader]

        with open(f'{dataroot}/ShapeNet/info.json') as f:
            self.info = json.load(f)

        self.cat_to_id = self.info['cats']
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}
        
        assert cat.lower() in ['all', 'chair', 'table']
        if cat == 'all':
            valid_cats = ['chair', 'table']
        else:
            valid_cats = [cat]
        
        self.model_list = []
        self.cats_list = []
        self.text_list = []

        for d in tqdm(self.data, total=len(self.data), desc=f'readinging text data from {self.text_csv}'):
            id, model_id, text, cat_i, synset, subSynsetId = d
            
            if cat_i.lower() not in valid_cats:
                continue
            
            sdf_path = f'{dataroot}/ShapeNet/SDF_v1_{res}/{synset}/{model_id}/ori_sample_grid.h5'

            if not os.path.exists(sdf_path):
                continue
                # {'Chair': 26523, 'Table': 33765} vs {'Chair': 26471, 'Table': 33517}
                # not sure why there are some missing files
                
            self.model_list.append(sdf_path)
            self.text_list.append(text)
            self.cats_list.append(synset)
            
        self.model_list = self.model_list[:self.max_dataset_size]
        self.text_list = self.text_list[:self.max_dataset_size]
        self.cats_list = self.cats_list[:self.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.model_list)), 'yellow')

        self.N = len(self.model_list)

        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):

        synset = self.cats_list[index]
        sdf_h5_file = self.model_list[index]
        text = self.text_list[index]
        
        h5_f = h5py.File(sdf_h5_file, 'r')
        sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
        sdf = torch.Tensor(sdf).view(1, self.res, self.res, self.res)

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)

        ret = {
            'sdf': sdf,
            'text': text,
            'cat_id': synset,
            'cat_str': self.id_to_cat[synset],
            'path': sdf_h5_file,
        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'Text2ShapeDataset'

class ShapeGlotDataset(BaseDataset):

    def initialize(self, opt, phase='train', cat='chair', res=64):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size
        self.phase = phase
        self.res = opt.res

        dataroot = opt.dataroot
        with open(f'{dataroot}/ShapeNet/info.json') as f:
            self.info = json.load(f)
        
        self.cat_to_id = self.info['cats']
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}


        self.shapenet_lang = collections.defaultdict(list)
        with open(f'{dataroot}/ShapeNet/shapeglot/data/main_data_for_chairs/language/shapenet_chairs.csv') as f:
            shapeglot_info = csv.reader(f)
            next(shapeglot_info)
            for r in shapeglot_info:
                target = int(r[7])
                shapenet_name = r[target]
                text = r[8]
                self.shapenet_lang[shapenet_name].append(text)

        cats = ['chair']
        self.lang_list = []
        self.model_list = []
        self.cats_list = []
        for c in cats:
            synset = self.info['cats'][c]

            with open(f'{dataroot}/ShapeNet/filelists/{synset}_{phase}.lst') as f:
                lang_list_s = []
                model_list_s = []
                for l in f.readlines():
                    model_id = l.rstrip('\n')
                    
                    model_texts = self.shapenet_lang[model_id]
                    for model_text in model_texts:
                        lang_list_s.append(model_text)
                    n_model_texts = len(model_texts)
                    
                    sdf_path = f'{dataroot}/ShapeNet/SDF_v1/resolution_{self.res}/{synset}/{model_id}/ori_sample_grid.h5'

                    for _ in range(n_model_texts):
                        model_list_s.append(sdf_path)

                    if not os.path.exists(sdf_path):
                        # import pdb; pdb.set_trace()
                        continue
                

                self.lang_list += lang_list_s
                self.model_list += model_list_s
                self.cats_list += [synset] * len(lang_list_s)
                print('[*] %d samples for %s (%s).' % (len(lang_list_s), self.id_to_cat[synset], synset))

        idx = np.arange(len(self.lang_list))
        np.random.default_rng(seed=0).shuffle(idx)
        
        self.lang_list = np.array(self.lang_list)[idx]
        self.model_list = np.array(self.model_list)[idx]
        self.cats_list = np.array(self.cats_list)[idx]
        
        self.lang_list = self.lang_list[:self.max_dataset_size]
        self.model_list = self.model_list[:self.max_dataset_size]
        self.cats_list = self.cats_list[:self.max_dataset_size]
        cprint('[*] %d lang_list loaded.' % (len(self.lang_list)), 'yellow')
        cprint('[*] %d code loaded.' % (len(self.model_list)), 'yellow')
        
        self.N = len(self.lang_list)

        
    def __getitem__(self, index):
        
        synset = self.cats_list[index]
        sdf_h5_file = self.model_list[index]
        text = self.lang_list[index]

        h5_f = h5py.File(sdf_h5_file, 'r')
        sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
        sdf = torch.Tensor(sdf).view(1, self.res, self.res, self.res)

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)

        ret = {

            'sdf': sdf,
            'text': text,
            'cat_id': synset,
            'cat_str': self.id_to_cat[synset],
            'path': sdf_h5_file,
        }

        return ret
    def __len__(self):
        return self.N

    def name(self):
        return 'ShapeGlotDataset'
