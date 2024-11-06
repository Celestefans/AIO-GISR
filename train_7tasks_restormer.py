import os
 
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from tools.evaluation_metric import calc_rmse
from tools.logger import *
from skimage.metrics import peak_signal_noise_ratio as PSNR
from model.Model import Restormer
from model.Model_AMIR import AMIR
from data.dataset import Pansharpening_mat_Dataset, MRI_pre_dataset, NYU_v2_datset, MultiTaskDataset
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='4'

# Settings
exp_name = 'Restormer_larger'
batch_size = 4
num_epoch = 500
lr = 2e-4
save_dir = os.path.join('./Checkpoint/All-in-one', exp_name)
os.makedirs(save_dir, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard_logs'))

# Datasets and DataLoader
data_transform = transforms.Compose([transforms.ToTensor()])
depth_root = '/data/cjj/dataset/NYU_V2'
mri_root = '/data/wtt/WJ_Code_DURE_HF/MRI_dataset/BT'
pan_root = '/data/datasets/pansharpening/NBU_dataset0730/WV4'

depth_train_dataset = NYU_v2_datset(root_dir=depth_root, scale=4, transform=data_transform, train=True)
mri_dataset = MRI_pre_dataset(os.path.join(mri_root, 't2_train'), os.path.join(mri_root, 'T2_train'), os.path.join(mri_root, 'T1_train'))
pan_dataset = Pansharpening_mat_Dataset(os.path.join(pan_root, 'train'))
mix_dataset = MultiTaskDataset(depth_train_dataset, mri_dataset, pan_dataset)

# Validation Datasets
test_minmax = np.load(f'{depth_root}/test_minmax.npy')
val_depth_dataset = NYU_v2_datset(root_dir=depth_root, scale=4, transform=data_transform, train=False)
val_mri_dataset = MRI_pre_dataset(os.path.join(mri_root, 't2_test'), os.path.join(mri_root, 'T2_test'), os.path.join(mri_root, 'T1_test'))
val_pan_dataset = Pansharpening_mat_Dataset(os.path.join(pan_root, 'test'))
list_val_dataset = [val_pan_dataset, val_mri_dataset, val_depth_dataset]

# Model, Optimizer and Scheduler
# Generator = AMIR(inp_channels=9, out_channels=8, dim=16, num_blocks=[2, 2, 2, 3]).cuda() # 1.7M
# Generator = AMIR(dim = 22, num_blocks=[3, 4, 4, 5]).cuda() # 4.3M
Generator = Restormer(inp_channels=9, out_channels=8, dim = 22, num_blocks=[3, 4, 4, 6]).cuda() # 4.3M

optimizer_G = torch.optim.Adam(Generator.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
lr_scheduler_G = CosineAnnealingLR(optimizer_G, num_epoch, eta_min=1.0e-6)
L1 = nn.L1Loss().cuda()

best_psnr_pan, best_psnr_mri, best_depth_rmse, best_index = 0, 0, 100, 0
logger = get_logger(os.path.join(save_dir, 'run.log'))

# Training Function
def train_one_epoch(epoch, dataloader, model, optimizer, loss_fn):
    
    model.train()
    epoch_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for train_data_lists in pbar:
        for list in train_data_lists:
            inp_lr, inp_gt, inp_guide = (item.type(torch.FloatTensor).cuda() for item in list)
            optimizer.zero_grad()
            restored = model(inp_lr, inp_guide)
            loss_l1 = loss_fn(restored, inp_gt)
            loss_G = loss_l1
            loss_G.backward()
            optimizer.step()
            epoch_loss += loss_G.item()
        pbar.set_postfix(loss_G=loss_G.item(), lr=optimizer.param_groups[0]['lr'])
    avg_epoch_loss = epoch_loss / len(dataloader)
    writer.add_scalar('Train/Loss', avg_epoch_loss, epoch)
    
# Validation Function
def validate_one_epoch(model, datasets, test_minmax, logger, epoch):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        psnr = []
        rmse = 0
        for dataset_id, dataset in enumerate(datasets):
            val_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            if dataset_id == 2:
                # RMSE calculation for depth
                rmse_list = [
                    calc_rmse(model(inp_lr.cuda(), inp_guide.cuda())[0, 0], 
                            inp_gt.cuda()[0,0], torch.from_numpy(test_minmax[:, index]).cuda()) 
                    for index, (inp_lr, inp_gt, inp_guide) in enumerate(val_dataloader)
                ]
                rmse = torch.mean(torch.stack(rmse_list)).item()
            else:
                # PSNR calculation for pan and mri
                avg_psnr = np.mean([
                    PSNR(inp_gt.numpy()[0], 
                         model(inp_lr.cuda(), inp_guide.cuda()).cpu().numpy()[0]) 
                    for inp_lr, inp_gt, inp_guide in val_dataloader
                ])
                psnr.append(avg_psnr)
        
        writer.add_scalar('Validation/PSNR_Pan', psnr[0], epoch)
        writer.add_scalar('Validation/PSNR_MRI', psnr[1], epoch)
        writer.add_scalar('Validation/RMSE_Depth', rmse, epoch)
        # Save the model if performance improves
        if psnr[0] > best_psnr_pan:
            torch.save(model.state_dict(), os.path.join(save_dir, f'Best_Pan.pth'))
        if psnr[1] > best_psnr_mri:
            torch.save(model.state_dict(), os.path.join(save_dir, 'Best_MRI.pth'))
        if rmse < best_depth_rmse:
            torch.save(model.state_dict(), os.path.join(save_dir, 'Best_Depth.pth'))
        
        # Logging the results
        logger.info(f'Epoch {epoch} - PSNR Pan: {psnr[0]:.4f}, PSNR MRI: {psnr[1]:.4f}, RMSE Depth: {rmse:.4f}')


# Main Loop
for epoch in range(1, num_epoch + 1):
    mix_dataset.shuffle()
    train_dataloader = DataLoader(mix_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    train_one_epoch(epoch, train_dataloader, Generator, optimizer_G, L1)
    lr_scheduler_G.step()
    if (epoch < 250 and epoch % 20 == 0) or ((epoch > 250 and epoch % 5 == 0)):
    # if epoch % 1 == 0:
        validate_one_epoch(Generator, list_val_dataset, test_minmax, logger, epoch)
    writer.close()

