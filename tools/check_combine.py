from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import h5py
from skimage.metrics import peak_signal_noise_ratio as PSNR
import random
import sys
import importlib
from tqdm import tqdm
importlib.reload(sys)
from data.dataset import my_dataset, CombineDataset

class CombineDataset(data.Dataset):
    def __init__(self, gf_dataset, qb_dataset, wv_dataset):
        self.gf_dataset, self.qb_dataset, self.wv_dataset = gf_dataset, qb_dataset, wv_dataset
        self.WV_NUM, QB_NUM, GF_NUM  = 9714, 17139, 19809  
        self.WV_indices, self.QB_indices, self.GF_indices = list(range(self.WV_NUM)), list(range(QB_NUM)), list(range(GF_NUM))

    def __len__(self):
        return self.WV_NUM
    
    def shuffle(self):
        random.shuffle(self.QB_indices)
        random.shuffle(self.GF_indices)

    def __getitem__(self, index):
        wv_list = list(self.wv_dataset[self.WV_indices[index]])
        qb_list = list(self.qb_dataset[self.QB_indices[index]])
        gf_list = list(self.gf_dataset[self.GF_indices[index]])
        return wv_list, qb_list, gf_list
    

dataset_folder = '/data/datasets/pansharpening/training'
qb_path = os.path.join(dataset_folder,'train_qb.h5')
wv_path = os.path.join(dataset_folder,'train_wv3.h5')
gf_path = os.path.join(dataset_folder,'train_gf2.h5')


qb_data, gf_data, wv_data = h5py.File(qb_path, 'r'), h5py.File(gf_path, 'r'), h5py.File(wv_path, 'r')
qb_dataset, gf_dataset, wv_dataset = my_dataset(qb_data), my_dataset(gf_data), my_dataset(wv_data)
del qb_data, gf_data, wv_data   ## 删除以节约内存

combine_dataset = CombineDataset(gf_dataset, qb_dataset, wv_dataset)
del qb_dataset, gf_dataset, wv_dataset

combine_loader = data.DataLoader(combine_dataset, batch_size = 16, shuffle=True)

for epoch in range(2):
    count = 0
    for i,(wv_list, qb_list, gf_list) in enumerate(combine_loader):
        count += wv_list[0].shape[0]
    combine_dataset.shuffle()   #每个epoch结束shuffle另外两个数据集
    print(count)


