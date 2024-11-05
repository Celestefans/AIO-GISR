#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES']='5'
import matplotlib.pyplot as plt
import numpy as np
import h5py
from model.Model import Spatial_channel_gate
from model.DIRFL import DIRFL
from data.dataset import MatDataset
from torch.autograd import Variable
import torch.utils.data as data
import torch
import os
import sys
import importlib
importlib.reload(sys)
import scipy.io 

 
dtype = torch.cuda.FloatTensor
if __name__ == "__main__":
    ##### read dataset #####
    # exp_name = 'Spatial_channel_gate_128size_2'
    exp_name = 'DIRFL_single_QB'

    ckpt_dir = '/home/wtt/Pansharpening/All-In-One-Medical-Image-Restoration-via-Task-Adaptive-Routing/Checkpoint/NBU_dataset'
    weight_path = os.path.join(ckpt_dir, exp_name, 'epoch_100.pth')
   
    dataset_folder = '/data/datasets/pansharpening/NBU_dataset0730'
    ikonos_select_path = os.path.join(dataset_folder,'IKONOS/test_select')
    wv3_select_path = os.path.join(dataset_folder,'WV3/test_select')
    test_path = ikonos_select_path
    # test_path = wv3_select_path
    SaveDataPath = os.path.join(test_path, exp_name)
    if not os.path.exists(SaveDataPath):
        os.makedirs(SaveDataPath)
    
    with torch.no_grad():
        # CNN = Spatial_channel_gate(inp_channels=9, out_channels=8, dim = 16, num_blocks=[2, 2, 2, 3],promptsize=128)
        CNN = DIRFL()
        CNN.cuda()  
        CNN.load_state_dict(torch.load(weight_path))
        CNN.eval()
        count = 0    
 
        for img_name in os.listdir(os.path.join(test_path,'GT_128')):
            
            ms = scipy.io.loadmat(os.path.join(test_path,'MS_32',img_name))['ms0'][...]
            pan = scipy.io.loadmat(os.path.join(test_path,'PAN_128',img_name))['pan0'][...]
            gt = scipy.io.loadmat(os.path.join(test_path,'GT_128',img_name))['gt0'][...]
            
            ms = torch.from_numpy(ms).float().cuda()
            pan = torch.from_numpy(pan).float().cuda()
            gt = torch.from_numpy(gt).float().cuda()
            ms = ms.unsqueeze(0).permute(0,3,1,2)
            pan = pan.unsqueeze(0).unsqueeze(1)
            gt = gt.unsqueeze(0).permute(0,3,1,2)

            out = CNN(ms, pan)
            out = out.cpu().data.numpy()
            save = os.path.join(SaveDataPath,img_name)
            scipy.io.savemat(os.path.join(SaveDataPath,img_name),{'sr':out})


            





    