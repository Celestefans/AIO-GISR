import os 
os.environ['CUDA_VISIBLE_DEVICES']='6'  
from model.Model import Restormer
from loss.losses import CharbonnierLoss
from data.dataset import Pansharpening_mat_Dataset, MRI_pre_dataset,Depth_dataset,MultiTaskDataset,MRI_dataset
import numpy as np
import torch
from torch import nn
import torch.utils.data as data
from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import CosineAnnealingLR 
from tqdm import tqdm
import random 
import logging 
import pandas as pd
import h5py
from skimage.metrics import peak_signal_noise_ratio as PSNR


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

##############################################################
######################### Settings ############################
##############################################################

exp_name = 'Restormer_mri'
batch_size = 8
# batch_size = 12
val_batch = 1
eps=1e-8
lr=2e-4
psnr_max=0
save_dir = os.path.join('./Checkpoint/All-in-one',exp_name)
os.makedirs(save_dir,exist_ok=True)

##############################################################
######################### Dataset ############################
##############################################################

# train
depth_root = '/data/datasets/NYU/NYUDepthv2'
mri_root = '/data/wtt/WJ_Code_DURE_HF/MRI_dataset/BT'
pan_root = '/data/datasets/pansharpening/NBU_dataset0730/WV4'

depth_dataset = Depth_dataset(root=depth_root, split='train', scale=4, downsample='bicubic', augment=True, input_size=256)
mri_dataset = MRI_pre_dataset(os.path.join(mri_root,'t2_train'), os.path.join(mri_root,'T2_train'), os.path.join(mri_root,'T1_train'))
# mri_dataset = MRI_dataset(os.path.join(mri_root,'T1_train'), os.path.join(mri_root,'T2_train'))
pan_dataset = Pansharpening_mat_Dataset(os.path.join(pan_root,'train'))

# mix_dataset = MultiTaskDataset(depth_dataset, mri_dataset, pan_dataset)
mix_dataset = MultiTaskDataset(mri_dataset)

# validation
val_depth_dataset = Depth_dataset(root=depth_root, split='test', scale=4, downsample='bicubic', augment=True, input_size=256)
val_mri_dataset = MRI_pre_dataset(os.path.join(mri_root,'t2_test'), os.path.join(mri_root,'T2_test'), os.path.join(mri_root,'T1_test'))
val_pan_dataset = Pansharpening_mat_Dataset(os.path.join(pan_root,'test'))
# list_val_dataset = [val_pan_dataset]
list_val_dataset = [val_mri_dataset]

##############################################################
######################### Model ##############################
##############################################################

Generator = Restormer(inp_channels=9, out_channels=8, dim = 16, num_blocks=[2, 2, 2, 3])
Generator.cuda() 
num_epoch = 200
total_iteration = 3.5e4
optimizer_G = torch.optim.Adam(Generator.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08) 
lr_scheduler_G = CosineAnnealingLR(optimizer_G, num_epoch, eta_min=1.0e-6)
L1 = nn.L1Loss().cuda() 
best_psnr = 0
best_index = 0
logger = get_logger(os.path.join(save_dir,'run.log'))

##############################################################
######################### Train ##############################
##############################################################

for epoch in range(1,num_epoch):
    # 每个epoch开始shuffle所有dataset
    mix_dataset.shuffle()
    train_dataloader = DataLoader(mix_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    pbar = tqdm(train_dataloader)
    Generator.train()
    for index, train_data_lists in enumerate(pbar):
        ### 四个数据集分别输进网络训练
        for list in train_data_lists:
            optimizer_G.zero_grad() 
            inp_ms, inp_gt, inp_pan = list[0], list[1], list[2]
            inp_ms = inp_ms.type(torch.FloatTensor).cuda()
            inp_pan = inp_pan.type(torch.FloatTensor).cuda()
            inp_gt = inp_gt.type(torch.FloatTensor).cuda()
            if inp_ms.shape[1] > 5:
                inp_ms = inp_ms.permute(0,3,1,2)
                inp_pan = inp_pan.unsqueeze(1)
                inp_gt = inp_gt.permute(0,3,1,2)   
            restored  = Generator(inp_ms, inp_pan)
            loss_l1 = L1(restored, inp_gt)
            loss_G = loss_l1 
            loss_G.backward()
            optimizer_G.step()

        torch.cuda.empty_cache() 
    lr_scheduler_G.step()
    current_lr = optimizer_G.param_groups[0]['lr']
    pbar.set_description("Epoch:{}      loss_G:{:6}    lr:{:.6f}".format(epoch, loss_G.item(), current_lr))
    
##############################################################
######################### Validation #########################
##############################################################
    if epoch % 5== 0:
        Generator.eval() 
        with torch.no_grad():
            psnr = []
            for dataset in list_val_dataset:
                val_dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
                count = 0
                sum_psnr = 0
                for index, datas in enumerate(val_dataloader):
                    count += 1
                    inp_ms, inp_gt, inp_pan = datas[0], datas[1], datas[2]
                    inp_ms = inp_ms.type(torch.FloatTensor).cuda()
                    inp_pan = inp_pan.type(torch.FloatTensor).cuda()
                    inp_gt = inp_gt.type(torch.FloatTensor)
                    if inp_ms.shape[1] > 5:
                        inp_ms = inp_ms.permute(0,3,1,2)
                        inp_pan = inp_pan.unsqueeze(1)
                        inp_gt = inp_gt.permute(0,3,1,2)
                    output = Generator(inp_ms, inp_pan)
                    netOutput_np = output.cpu().numpy()[0]
                    gtLabel_np = inp_gt.numpy()[0]
                    psnrValue = PSNR(gtLabel_np, netOutput_np)
                    sum_psnr += psnrValue                         
                avg_psnr = sum_psnr / count
                psnr.append(avg_psnr)
                torch.cuda.empty_cache()   
            if psnr[0] > best_psnr: ## 以Pan为主
                best_psnr = psnr[0]
                best_index = epoch
                torch.save(Generator.state_dict(), os.path.join(save_dir,'epoch_{}.pth'.format(epoch)))
            # if psnr[1] > best_psnr_depth: best_psnr_depth = psnr[1] 
            # if psnr[2] > best_psnr_mri: best_psnr_mri = psnr[2]
            
            ## record
            logger.info('Epoch:[{}]\t PSNR_Pan = {:.4f}\t BEST_Pan_PSNR = {:.4f}\t BEST_epoch = {}'.format(
                        epoch,psnr[0],best_psnr, best_index))
            # logger.info('Epoch:[{}]\t PSNR_Pan = {:.4f}\t  PSNR_Depth = {:.4f}\t PSNR_Mri = {:.4f}\t BEST_Pan_PSNR = {:.4f}\t BEST_epoch = {}'.format(
            #             epoch,psnr[0],psnr[1],psnr[2],best_psnr_pan, best_index))


