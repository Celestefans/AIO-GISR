import os 
os.environ['CUDA_VISIBLE_DEVICES']='0'  
from model.Model import Restormer
from loss.losses import CharbonnierLoss
from data.dataset import *
import numpy as np
import torch
from torch import nn
import torch.utils.data as data
from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import CosineAnnealingLR 
from tqdm import tqdm
import random
import pandas as pd
import h5py
from tools.evaluation_metric import Fpsnr, RMSE, SSIM, SAM, ERGAS
from skimage.metrics import peak_signal_noise_ratio as PSNR


test_name = 'channel_prompt_size_512_Fpsnr'
save_test_path = os.path.join('./Checkpoint/test_metric','{}.txt'.format(test_name))
os.makedirs(os.path.dirname(save_test_path), exist_ok=True)

depth_NYU_root = '/data/datasets/NYU/NYUDepthv2'
mri_root = '/data/wtt/WJ_Code_DURE_HF/MRI_dataset/BT'
pan_root = '/data/datasets/pansharpening/NBU_dataset0730/WV4'

# validation_dataset
data_transform = transforms.Compose([transforms.ToTensor()])
val_depth_NYU_dataset = NYU_v2_datset(root_dir=depth_root, scale=8, transform=data_transform, train=False)
val_mri_dataset = MRI_pre_dataset(os.path.join(mri_root,'t2_test'), os.path.join(mri_root,'T2_test'), os.path.join(mri_root,'T1_test'))
val_pan_dataset = Pansharpening_mat_Dataset(os.path.join(pan_root,'test'))
list_val_dataset = [val_pan_dataset, val_depth_dataset, val_mri_dataset]

# weight_path = './Checkpoint/NBU_dataset/Spatial_channel_gate_128size_2/epoch_175.pth'
weight_path = './Checkpoint/All-in-one/Restormer_2/epoch_195.pth'
# weight_path = './Checkpoint/NBU_dataset/Restormer/epoch_140.pth'

with torch.no_grad():
    Generator = Restormer(inp_channels=9, out_channels=8, dim = 16, num_blocks=[2, 2, 2, 3]) 
    Generator.cuda()
    Generator.load_state_dict(torch.load(weight_path)) 
    test_pan_dataloader = data.DataLoader(val_pan_dataset, batch_size=1, shuffle=False)
    test_depth_dataloader = data.DataLoader(val_depth_dataset, batch_size=1, shuffle=False)
    test_mri_dataloader = data.DataLoader(val_mri_dataset, batch_size=1, shuffle=False)
    PSNR_pan, PSNR_mri, RMSE_depth = 0,0,0
    
    ### 测试Pansharpening
    sum_psnr, count = 0, 0
    for index, datas in enumerate(test_pan_dataloader):
        count += 1

        inp_ms, inp_gt, inp_pan = datas[0], datas[1], datas[2]
        inp_ms = inp_ms.type(torch.FloatTensor).cuda().permute(0,3,1,2)
        inp_pan = inp_pan.type(torch.FloatTensor).cuda().unsqueeze(1)
        inp_gt = inp_gt.type(torch.FloatTensor).permute(0,3,1,2)

        output = Generator(inp_ms, inp_pan)
        netOutput_np = output.cpu().numpy()[0]
        gtLabel_np = inp_gt.numpy()[0]
        sum_psnr += PSNR(gtLabel_np, netOutput_np)
    PSNR_pan = sum_psnr / count

    ### 测试MRI
    sum_psnr, count = 0, 0
    for index, datas in enumerate(test_mri_dataloader):
        count += 1

        inp_ms, inp_gt, inp_pan = datas[0], datas[1], datas[2]
        inp_ms = inp_ms.type(torch.FloatTensor).cuda()
        inp_pan = inp_pan.type(torch.FloatTensor).cuda()
        inp_gt = inp_gt.type(torch.FloatTensor)

        output = Generator(inp_ms, inp_pan)
        netOutput_np = output.cpu().numpy() * 255.
        gtLabel_np = inp_gt.numpy() * 255.
        sum_psnr += Fpsnr(gtLabel_np, netOutput_np)
    PSNR_mri = sum_psnr / count

    ### 测试Depth
    sum_rmse, count = 0, 0
    for index, datas in enumerate(test_depth_dataloader):
        count += 1

        inp_ms, inp_pan, inp_gt = datas[0], datas[1], datas[2]
        inp_ms = inp_ms.type(torch.FloatTensor).cuda()
        inp_pan = inp_pan.type(torch.FloatTensor).cuda()
        inp_gt = inp_gt.type(torch.FloatTensor)

        output = Generator(inp_ms, inp_pan)
        netOutput_np = output.cpu().numpy()[0]
        gtLabel_np = inp_gt.numpy()[0]
        sum_rmse += RMSE(gtLabel_np, netOutput_np)
    RMSE_depth = sum_rmse / count

    with open(save_test_path, 'w') as file:
        file.write(f"PSNR_Pan: {PSNR_pan:.4f}\n")
        file.write(f"PSNR_MRI: {PSNR_mri:.4f}\n")
        file.write(f"RMSE_Depth: {RMSE_depth:.4f}\n")

    print('done')








# with torch.no_grad(): 
#     Generator = Restormer(inp_channels=9, out_channels=8, dim = 16, num_blocks=[2, 2, 2, 3]) 
#     Generator.cuda()
#     Generator.load_state_dict(torch.load(weight_path))
#     psnr,rmse,ssim,sam,ergas = [],[],[],[],[]
    
#     for dataset in list_val_dataset:
#         test_dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
#         count = 0
#         sum_psnr, sum_rmse, sum_ssim, sum_sam, sum_ergas = 0,0,0,0,0
#         for index, datas in enumerate(test_dataloader):
#             count+=1
#             inp_ms, inp_gt, inp_pan = datas[0], datas[1], datas[2]
#             inp_ms = inp_ms.type(torch.FloatTensor).cuda()
#             inp_pan = inp_pan.type(torch.FloatTensor).cuda()
#             inp_gt = inp_gt.type(torch.FloatTensor)
#             if inp_ms.shape[1] > 5:
#                 inp_ms = inp_ms.permute(0,3,1,2)
#                 inp_pan = inp_pan.unsqueeze(1)
#                 inp_gt = inp_gt.permute(0,3,1,2)

#             output = Generator(inp_ms, inp_pan)

#             netOutput_np = output.cpu().numpy()[0]
#             gtLabel_np = inp_gt.numpy()[0]
#             psnrValue, rmseValue, ssimValue, samValue, eragsValue = PSNR(gtLabel_np, netOutput_np),RMSE(gtLabel_np, netOutput_np), SSIM(gtLabel_np, netOutput_np), SAM(gtLabel_np, netOutput_np), ERGAS(gtLabel_np, netOutput_np)
#             sum_psnr += psnrValue
#             sum_rmse += rmseValue
#             sum_ssim += ssimValue
#             sum_sam += samValue
#             sum_ergas += eragsValue
                                  
#         avg_psnr = sum_psnr / count
#         avg_rmse = sum_rmse / count
#         avg_ssim = sum_ssim / count
#         avg_sam = sum_sam / count
#         avg_erags = sum_ergas / count
#         psnr.append(avg_psnr)
#         rmse.append(avg_rmse)
#         ssim.append(avg_ssim)
#         sam.append(avg_sam)
#         ergas.append(avg_erags)
