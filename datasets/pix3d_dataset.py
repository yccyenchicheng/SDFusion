"""
    adopted from
        - https://github.com/shubhtuls/PixelTransformer/blob/03b65b8612fe583b3e35fc82b446b5503dd7b6bd/data/shapenet.py
        - https://github.com/hzxie/Pix2Vox
    
"""

import os
import glob
import json
import socket

import scipy.io
import h5py
import numpy as np
from PIL import Image
from termcolor import colored, cprint

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from datasets.base_dataset import BaseDataset
from utils.pix3d_util import downsample_voxel

hostname = socket.gethostname()

class RandomNoise(object):
    def __init__(self,
                 noise_std,
                 eigvals=(0.2175, 0.0188, 0.0045),
                 eigvecs=((-0.5675, 0.7192, 0.4009), (-0.5808, -0.0045, -0.8140), (-0.5836, -0.6948, 0.4203))):
        self.noise_std = noise_std
        self.eigvals = np.array(eigvals)
        self.eigvecs = np.array(eigvecs)

    def __call__(self, rendering_images):
        alpha = np.random.normal(loc=0, scale=self.noise_std, size=3)
        noise_rgb = \
            np.sum(
                np.multiply(
                    np.multiply(
                        self.eigvecs,
                        np.tile(alpha, (3, 1))
                    ),
                    np.tile(self.eigvals, (3, 1))
                ),
                axis=1
            )

        # Allocate new space for storing processed images
        c, h, w = rendering_images.shape
        processed_images = torch.zeros_like(rendering_images)
        for i in range(c):
            processed_images[i, :, :] = rendering_images[i, :, :]
            processed_images[i, :, :] += noise_rgb[i]
        
        return processed_images

class RandomPermuteRGB(object):
    def __call__(self, rendering_images):
        # assert (isinstance(rendering_images, np.ndarray))

        random_permutation = np.random.permutation(3)
        rendering_images = rendering_images[random_permutation, ...] 

        return rendering_images

class Pix3DImg2ShapeDataset(BaseDataset):
    def initialize(self, opt, phase='train', cat='chair', input_txt=None, by_imgs=True):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size
        self.phase = phase
        self.ratio = opt.ratio

        info_path = f'dataset_info_files/info-pix3d.json'

        with open(info_path) as f:
            self.info = json.load(f)
        
        self.img_list, self.model_list, self.cats_list, self.gt_voxel_list, self.z_list = \
            load_custom_splits_for_img2shape_dset(self, cat, phase, opt)

        np.random.default_rng(seed=0).shuffle(self.img_list)
        np.random.default_rng(seed=0).shuffle(self.model_list)
        np.random.default_rng(seed=0).shuffle(self.cats_list)
        np.random.default_rng(seed=0).shuffle(self.gt_voxel_list)
        np.random.default_rng(seed=0).shuffle(self.z_list)

        self.img_list = self.img_list[:self.max_dataset_size]
        self.model_list = self.model_list[:self.max_dataset_size]
        self.cats_list = self.cats_list[:self.max_dataset_size]
        self.gt_voxel_list = self.gt_voxel_list[:self.max_dataset_size]
        self.z_list = self.z_list[:self.max_dataset_size]
        cprint('[*] %d img_list loaded.' % (len(self.img_list)), 'yellow')
        cprint('[*] %d code loaded.' % (len(self.model_list)), 'yellow')
        cprint(f'[*] ratio: {self.ratio}')
        
        self.N = len(self.img_list)
        self.to_tensor = transforms.ToTensor()

        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        if phase == 'train':
            self.transforms = transforms.Compose([
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
                RandomNoise(0.1),
                transforms.RandomAffine(0, scale=(0.7, 1.25), interpolation=InterpolationMode.BILINEAR),
                transforms.Normalize(mean, std),
                transforms.RandomHorizontalFlip(),
                RandomPermuteRGB(),
                transforms.Resize((256, 256)),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Normalize(mean, std),
                transforms.Resize((256, 256)),
            ])

        self.n_view = 1
    
    def process_img(self, img):
        img_t = self.to_tensor(img)

        _, oh, ow = img_t.shape

        ls = max(oh, ow)

        pad_h1, pad_h2 = (ls - oh) // 2, (ls - oh) - (ls - oh) // 2
        pad_w1, pad_w2 = (ls - ow) // 2, (ls - ow) - (ls - ow) // 2
        img_t = F.pad(img_t[None, ...], (pad_w1, pad_w2, pad_h1, pad_h2), mode='constant', value=0)

        img_t = self.transforms(img_t[0])

        return img_t

    def read_vox(self, f):
        gt_size = 32

        voxel_p = f
        voxel = scipy.io.loadmat(voxel_p)['voxel']

        # downsample
        voxel = downsample_voxel(voxel, 0.5, (gt_size, gt_size, gt_size))
        voxel = torch.from_numpy(voxel)
        voxel = voxel.float()
        return voxel

    def __getitem__(self, index):
        
        cat_name = self.cats_list[index]
        sdf_h5_file = self.model_list[index]

        h5_f = h5py.File(sdf_h5_file, 'r')
        sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
        sdf = torch.Tensor(sdf).view(1, 64, 64, 64)

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)

        # load img; randomly sample 1
        imgs = []
        img_paths = []
        imgs_all_view = self.img_list[index]
        sample_ixs = np.random.choice(len(imgs_all_view), self.n_view)
        for ix in sample_ixs:
            p = imgs_all_view[ix]
            im = Image.open(p).convert('RGB')
            im = self.process_img(im)
            imgs.append(im)
            img_paths.append(p)

        imgs = torch.stack(imgs).clamp(-1., 1.)
        img = imgs[0]
        img_path = img_paths[0]

        gt_vox_path = self.gt_voxel_list[index]
        gt_vox = self.read_vox(gt_vox_path) # already downsample

        ret = {
            'sdf': sdf, 'sdf_path': sdf_h5_file,
            'img': img, 'img_path': img_path,
            'imgs': imgs, 'img_paths': img_paths,
            'gt_vox': gt_vox, 'gt_vox_path': gt_vox_path,
            'cat_str': cat_name,
        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'Pix3DImage2ShapeDataset'


####################################################################################################

def load_pix3d_info(opt):

    dataroot = opt.dataroot

    with open(f'{dataroot}/pix3d/pix3d.json', 'r') as f:
        pix3d_info = json.load(f)

    # map 
    map_input_to_pix3dinfo = {}
    map_obj_to_imgs = {}

    for d in pix3d_info:
        cat = d['category']
        img_name = d['img']
        obj_file = d['model']
        model_id = obj_file.split('/')[-2]
        # img_name = os.path.splitext(img_name)[0]
        map_input_to_pix3dinfo[img_name] = d

        img_basename = os.path.basename(img_name)
        img_name_by_model = f"{dataroot}/pix3d/imgs_by_model_noBG/{cat}/{model_id}/{img_basename}"

        if obj_file not in map_obj_to_imgs:
            map_obj_to_imgs[obj_file] = [img_name_by_model]
        else:
            map_obj_to_imgs[obj_file].append(img_name_by_model)

    return pix3d_info, map_input_to_pix3dinfo, map_obj_to_imgs

def load_custom_splits_for_img2shape_dset(dset_instance, cat, phase, opt): 
    dataroot = opt.dataroot

    pix3d_info, map_input_to_pix3dinfo, map_obj_to_imgs = load_pix3d_info(opt)
    
    # check chair "test images" from official split
    with open(f'{dataroot}/pix3d/input.txt', 'r') as f:
        lines = [l.rstrip('\n') for l in f.readlines()]
        official_test_imgs = [l.replace('../', '') for l in lines]

    if cat == 'all':
        cats = dset_instance.info['all_cats']
    else:
        cats = [cat]

    img_list = []
    model_list = []
    z_list = []
    cats_list = []
    gt_voxel_list = []

    for c in cats:
        
        lst_f = f'./dataset_info_files/pix3d_filelists/{c}_{phase}.lst'     
        with open(lst_f) as f:
            lines = f.readlines()
            lines = [l.rstrip('\n') for l in lines]
            
        img_list_s = []
        model_list_s = []
        z_list_s = []
        gt_voxel_list_s = []

        all_cat_imgs = glob.glob(f'{dataroot}/pix3d/img/{c}/*')

        for i, img_path in enumerate(all_cat_imgs):
            img_basename = os.path.basename(img_path)
            img_name_as_key = f'img/{c}/{img_basename}'
            info_i = map_input_to_pix3dinfo[img_name_as_key]

            obj_f = info_i['model']
            model_id = obj_f.split('/')[-2]
            
            # get our img_path
            img_name_by_model = f"{dataroot}/pix3d/imgs_by_model_noBG/{c}/{model_id}/{img_basename}"

            # check whether in lines or not
            is_in_split = False
            if c == 'chair':
                if img_name_by_model.replace(dataroot, '') in lines:
                    is_in_split = True
            else:
                if model_id in lines:
                    is_in_split = True
            
            if is_in_split:
                img_list_s.append([img_name_by_model])

                obj_name = os.path.basename(obj_f)
                obj_name = obj_name.replace('.obj', '')

                # find sdf
                if obj_name != 'model':
                    # NOTE: some obj will have: "model_XXXXX_XXXXX.obj"
                    #       exclude the "model" and ".obj" to get the sdf file name
                    obj_name = obj_name.replace('model', '')
                    sdf_name_noext = f'ori_sample_grid{obj_name}'
                    z_name_noext = f'{obj_name}'
                else:
                    sdf_name_noext = f'ori_sample_grid'
                    z_name_noext = f''

                sdf_path = f'{dataroot}/pix3d/SDF_v1_64/{c}/{model_id}/{sdf_name_noext}.h5'
                if not os.path.exists(sdf_path): import pdb; pdb.set_trace()
                model_list_s.append(sdf_path)

                # find gt voxel
                if obj_name != 'model':
                    gt_vox_name = f'voxel{obj_name}'
                else:
                    gt_vox_name = obj_name.replace('model', 'voxel')
                gt_vox_path = f'{dataroot}/pix3d/model/{c}/{model_id}/{gt_vox_name}.mat'
                if not os.path.exists(gt_vox_path):
                    import pdb; pdb.set_trace()

                gt_voxel_list_s.append(gt_vox_path)


        # sanity check
        if c == 'chair':
            all_chair_imgs = []
            for s in img_list_s:
                all_chair_imgs += s

        nimgs_img_list_s = len(img_list_s)
        nimgs_to_take = int(nimgs_img_list_s * opt.ratio)

        img_list += img_list_s[:nimgs_to_take]
        model_list += model_list_s[:nimgs_to_take]
        z_list += z_list_s[:nimgs_to_take]
        gt_voxel_list += gt_voxel_list_s[:nimgs_to_take]
        cats_list += [c] * len(img_list_s[:nimgs_to_take])

        try:
            assert len(img_list) == len(model_list) == len(gt_voxel_list) == len(cats_list)
        except:
            import pdb; pdb.set_trace()

        print('[*] %d samples for %s.' % (len(model_list_s), c))

    cprint(f'[*] ({dset_instance.name()}) there are {len(cats)} categories.', 'yellow')

    """ sanity check """
    with open(f'{dataroot}/pix3d/input.txt', 'r') as f:
        official_test_imgs = [l.rstrip('\n') for l in f.readlines()]

    bnames1 = [os.path.basename(f) for f in official_test_imgs]
    bnames2 = [os.path.basename(f) for f in all_chair_imgs]

    if phase == 'test':
        assert set(bnames1).intersection(set(bnames2)) == set(bnames1)
    else:
        assert len(set(bnames1).intersection(set(bnames2))) == 0

    return img_list, model_list, cats_list, gt_voxel_list, z_list