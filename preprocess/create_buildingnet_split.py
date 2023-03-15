
import os
import argparse
import random

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='data', help='path to dataset root')
parser.add_argument('--thread_num', type=int, default='9', help='how many objs are creating at the same time')
parser.add_argument('--cat', type=str, default="all", help='Which single class to generate on [default: all, can '
                                                                'be chair or plane, etc.]')

opt = parser.parse_args()


if __name__ == "__main__":

    seed = 111
    np.random.seed(seed)
    random.seed(seed)

    dataroot = opt.dataroot
    buildingnet_root = os.path.join(dataroot, 'BuildingNet_dataset_v0_1')

    filelists_out = f'{buildingnet_root}/filelists'
    
    if not os.path.exists(filelists_out): os.makedirs(filelists_out)

    # create train/test for each cat
    # the images will also be copied since each model might contain multiple images
    
    obj_root = f'{buildingnet_root}/OBJ_MODELS'
    all_objs = [f.replace('.obj', '') for f in os.listdir(obj_root) if '.obj' in f] # 1938

    # make train/test split
    train_ratio = 0.8

    N = len(all_objs)

    train_N = int(N * train_ratio)

    np.random.shuffle(all_objs)

    train_objs = sorted(all_objs[:train_N])
    test_objs = sorted(all_objs[train_N:])

    # sanity check
    inter = set(train_objs).intersection(set(test_objs)) 
    union = set(train_objs).union(set(test_objs)) 

    assert len(inter) == 0
    assert set(all_objs) == union


    for phase in ['train', 'test']:
        if phase == 'train':
            model_list = train_objs
        else:
            model_list = test_objs

        f_out = f'{filelists_out}/{phase}.lst'

        with open(f_out, 'w') as f:
            for model_id in model_list:
                # model_id = m.split('/')[-2]
                # f.write(f'{model_id}\n')
                f.write(f'{model_id}\n')