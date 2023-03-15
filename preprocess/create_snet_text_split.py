import os
import csv
import numpy as np

import sys
sys.path.append('..')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='data', help='path to dataset')

opt = parser.parse_args()

dataroot = opt.dataroot
csv_file = f'{dataroot}/ShapeNet/text2shape/captions.tablechair.csv'

assert os.path.exists(csv_file)

seed = 777
np.random.seed(seed)

with open(csv_file) as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader, None)
    data = [row for row in reader]

# split train test
train_ratio = 0.8
N = len(data)
N_train = int(N * train_ratio)

np.random.shuffle(data)

train_data = data[:N_train]
test_data = data[N_train:]


# sanity check
train_data_as_str = ['-'.join(d) for d in train_data]
test_data_as_str = ['-'.join(d) for d in test_data]
assert len(set(train_data_as_str).intersection(set(test_data_as_str))) == 0

for phase in ['train', 'test']:
    if phase == 'train':
        data_phase = train_data
    else:
        data_phase = test_data

    out_csv = f'{dataroot}/ShapeNet/text2shape/captions.tablechair_{phase}.csv'
    with open(out_csv, 'wt') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header)
        writer.writerows(data_phase)