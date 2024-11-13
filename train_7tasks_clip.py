import os
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from torch.nn import L1Loss
from tools.evaluation_metric import calc_rmse
from tools.logger import *
from skimage.metrics import peak_signal_noise_ratio as PSNR
from model.Model_AMIR import AMIR
# from model.Ablation.moe_IFM_clip import MOE_IFM_Clip
from model.Ablation.moe_IFM_clip_test import MOE_IFM_Clip_test
from data.dataset import Pansharpening_mat_Dataset, MRI_pre_dataset, NYU_v2_datset, MultiTaskDataset
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='4'

############ Settings ############
exp_name = 'MOE_IFM_Clip_test_7tasks'
batch_size = 4
num_epoch = 500
lr = 2e-4
save_dir = os.path.join('./Checkpoint/All-in-one', exp_name)
os.makedirs(save_dir, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard_logs'))

# 只创建一次logger
logger = get_logger(os.path.join(save_dir, 'run.log'))
logger.info(f"{'='*20} Experiment Settings {'='*20}")
logger.info(f"Experiment: {exp_name}")
logger.info(f"Batch size: {batch_size}")
logger.info(f"Learning rate: {lr}")
logger.info(f"Number of epochs: {num_epoch}")
logger.info(f"{'='*60}")

############ Datasets and DataLoader ############
data_transform = transforms.Compose([transforms.ToTensor()])
depth_root = '/data/cjj/dataset/NYU_V2'
mri_root = '/data/wtt/WJ_Code_DURE_HF/MRI_dataset/BT'
pan_root = '/data/datasets/pansharpening/NBU_dataset0730'

depth_dataset_4 = NYU_v2_datset(root_dir=depth_root, scale=4, transform=data_transform, train=True)
depth_dataset_8 = NYU_v2_datset(root_dir=depth_root, scale=8, transform=data_transform, train=True)
depth_dataset_16 = NYU_v2_datset(root_dir=depth_root, scale=16, transform=data_transform, train=True)
mri_dataset = MRI_pre_dataset(os.path.join(mri_root, 't2_train'), os.path.join(mri_root, 'T2_train'), os.path.join(mri_root, 'T1_train'))
pan_dataset_WV4 = Pansharpening_mat_Dataset(os.path.join(pan_root, 'WV4', 'train'))
pan_dataset_QB = Pansharpening_mat_Dataset(os.path.join(pan_root, 'QB', 'train'))
pan_dataset_GF1 = Pansharpening_mat_Dataset(os.path.join(pan_root, 'GF1', 'train'))
mix_dataset = MultiTaskDataset(depth_dataset_4, depth_dataset_8, depth_dataset_16, mri_dataset, pan_dataset_WV4, pan_dataset_QB, pan_dataset_GF1)
# del depth_dataset_4, depth_dataset_8, depth_dataset_16, mri_dataset, pan_dataset_WV4, pan_dataset_QB, pan_dataset_GF1

############ Validation Datasets ############
test_minmax = np.load(f'{depth_root}/test_minmax.npy')
val_depth_dataset_4 = NYU_v2_datset(root_dir=depth_root, scale=4, transform=data_transform, train=False)
val_depth_dataset_8 = NYU_v2_datset(root_dir=depth_root, scale=8, transform=data_transform, train=False)
val_depth_dataset_16 = NYU_v2_datset(root_dir=depth_root, scale=16, transform=data_transform, train=False)
val_mri_dataset = MRI_pre_dataset(os.path.join(mri_root, 't2_test'), os.path.join(mri_root, 'T2_test'), os.path.join(mri_root, 'T1_test'))
val_pan_dataset_WV4 = Pansharpening_mat_Dataset(os.path.join(pan_root,'WV4', 'test'))
val_pan_dataset_QB = Pansharpening_mat_Dataset(os.path.join(pan_root,'QB', 'test'))
val_pan_dataset_GF1 = Pansharpening_mat_Dataset(os.path.join(pan_root,'GF1', 'test'))
list_val_dataset = [val_depth_dataset_4, val_depth_dataset_8, val_depth_dataset_16, val_mri_dataset, val_pan_dataset_WV4, val_pan_dataset_QB, val_pan_dataset_GF1]  # 0,1,2,3,4,5,6


############ Model, Optimizer and Scheduler ############
Generator = MOE_IFM_Clip_test(dim = 22, num_blocks=[3, 4, 4, 5]).cuda()
optimizer_G = torch.optim.Adam(Generator.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
lr_scheduler_G = CosineAnnealingLR(optimizer_G, num_epoch, eta_min=1.0e-6)
l1_loss = L1Loss().cuda()
clip_feat = torch.load('./data/SR_text_feature_GF2_WV4_QB_MRI_Dep4_Dep8_Dep16.th').cuda()
# 定义任务ID到clip_feat索引的映射
task_to_clip = {
    0: 4,
    1: 5,
    2: 6,
    3: 3,
    4: 1,
    5: 2,
    6: 0
}

best_psnr_pan_WV4, best_psnr_pan_QB, best_psnr_pan_GF1 ,best_psnr_mri= -float('inf'), -float('inf') ,-float('inf'), -float('inf')
best_rmse_4, best_rmse_8, best_rmse_16 = float('inf'), float('inf'), float('inf')

# 删除这行重复的logger创建
# logger = get_logger(os.path.join(save_dir, 'run.log'))  # 删除这行

############ Training Function ############
def train_one_epoch(epoch, dataloader, model, optimizer, loss_fn, clip_feat):
    model.train()
    epoch_losses = {'total': 0, 'task_losses': [[] for _ in range(7)]}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for train_data_lists in pbar:
        for task_id, data_list in enumerate(train_data_lists):
            inp_lr, inp_gt, inp_guide = (item.type(torch.FloatTensor).cuda() for item in data_list)
            
            optimizer.zero_grad()


            # 获取对应的clip_feat
            inp_clip = clip_feat[task_to_clip[task_id]]
            restored, loss_moe = model(inp_lr, inp_guide, inp_clip)
            
            # 使用L1 loss
            loss_task = loss_fn(restored, inp_gt)
            
            # 总损失
            loss_G = loss_task + 0.001 * loss_moe
            
            loss_G.backward()
            optimizer.step()
            
            # 更新损失记录
            epoch_losses['total'] += loss_G.item()
            epoch_losses['task_losses'][task_id].append(loss_task.item())
            
            # 记录到tensorboard
            writer.add_scalar(f'Train/Task_{task_id}_Loss', loss_task.item(), epoch)
            
            pbar.set_postfix(
                loss=loss_G.item(),
                task=task_id,
                lr=optimizer.param_groups[0]['lr']
            )
    
    # 计算并记录平均损失
    avg_total_loss = epoch_losses['total'] / len(dataloader)
    avg_task_losses = [np.mean(losses) if losses else 0 for losses in epoch_losses['task_losses']]
    
    # 记录到tensorboard
    writer.add_scalar('Train/Total_Loss', avg_total_loss, epoch)
    for task_id, avg_loss in enumerate(avg_task_losses):
        writer.add_scalar(f'Train/Task_{task_id}_Loss', avg_loss, epoch)
    
    return avg_total_loss, avg_task_losses

############ Validation Function ############
def validate_one_epoch(model, datasets, test_minmax, logger, epoch, clip_feat):
    global best_psnr_pan_WV4, best_psnr_pan_QB, best_psnr_pan_GF1, best_psnr_mri
    global best_rmse_4, best_rmse_8, best_rmse_16
    model.eval()
    with torch.no_grad():
        psnr = []
        rmse = []
        # 使用字典存储最佳结果
        best_metrics = {
            'psnr_pan_wv4': best_psnr_pan_WV4,
            'psnr_pan_qb': best_psnr_pan_QB,
            'psnr_pan_gf1': best_psnr_pan_GF1,
            'psnr_mri': best_psnr_mri,
            'rmse_4': best_rmse_4,
            'rmse_8': best_rmse_8,
            'rmse_16': best_rmse_16
        }
        
        for dataset_id, dataset in enumerate(datasets):
            val_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            metric_values = []
            inp_clip = clip_feat[task_to_clip[dataset_id]]
            if dataset_id <= 2:  # depth estimation tasks
                for index, (inp_lr, inp_gt, inp_guide) in enumerate(tqdm(val_dataloader, desc=f'Validating Depth_{4*(2**dataset_id)}')):
                    output = model(inp_lr.cuda(), inp_guide.cuda(), inp_clip)[0]
                    metric_values.append(
                        calc_rmse(output[0], 
                                inp_gt.cuda()[0,0], 
                                torch.from_numpy(test_minmax[:, index]).cuda())
                    )
                avg_metric = torch.mean(torch.stack(metric_values)).item()
                rmse.append(avg_metric)
           
            else:  # pansharpening and MRI tasks
                for inp_lr, inp_gt, inp_guide in tqdm(val_dataloader, desc=f'Validating Task_{dataset_id}'):
                    output = model(inp_lr.cuda(), inp_guide.cuda(), inp_clip)
                    metric_values.append(
                        PSNR(inp_gt.numpy()[0], output.cpu().numpy()[0])
                    )
                avg_metric = np.mean(metric_values)
                psnr.append(avg_metric)

        # 记录验证指标
        metrics = {
            'psnr_pan_wv4': psnr[1],
            'psnr_pan_qb': psnr[2],
            'psnr_pan_gf1': psnr[3],
            'psnr_mri': psnr[0],
            'rmse_4': rmse[0],
            'rmse_8': rmse[1],
            'rmse_16': rmse[2]
        }
        
        # 记录到tensorboard时转换为大写
        for name, value in metrics.items():
            writer.add_scalar(f'Validation/{name.upper()}', value, epoch)
        
        # 检查并保存最佳模型
        improved = False
        for metric_name, metric_value in metrics.items():
            is_better = metric_value > best_metrics[metric_name] if 'psnr' in metric_name else metric_value < best_metrics[metric_name]
            if is_better:
                improved = True
                best_metrics[metric_name] = metric_value
                model_path = os.path.join(save_dir, f'Best_{metric_name.upper()}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metric_value': metric_value,
                    'optimizer_state_dict': optimizer_G.state_dict(),
                }, model_path)
                # logger.info(f'New best {metric_name}: {metric_value:.4f}, saved to {model_path}')
        
        # 记录所有指标
        log_message = f'Epoch {epoch} Validation:\n'
        for name, value in metrics.items():
            log_message += f'{name}: {value:.4f}, '
        logger.info(log_message)
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        return metrics, improved


############ Main Loop ############
# 在主循环外部初始化

for epoch in range(1, num_epoch + 1):
    mix_dataset.shuffle()
    train_dataloader = DataLoader(
        mix_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    avg_total_loss, avg_task_losses = train_one_epoch(
        epoch, train_dataloader, Generator, optimizer_G, l1_loss, clip_feat  # 使用L1Loss
    )
    lr_scheduler_G.step()
    
    ## validation
    if (epoch < 250 and epoch % 20 == 0) or (epoch > 250 and epoch % 5 == 0):
    # if epoch % 1 == 0 :
        metrics, improved = validate_one_epoch(Generator, list_val_dataset, test_minmax, logger, epoch, clip_feat)
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': Generator.state_dict(),
            'optimizer_state_dict': optimizer_G.state_dict(),
            'scheduler_state_dict': lr_scheduler_G.state_dict(),
            'metrics': metrics
        }
        # 保存最新的检查点
        torch.save(checkpoint, os.path.join(save_dir, 'latest_checkpoint.pth'))
        
        # 如果性能有提升，额外保存一个检查点
        if improved:
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    # 在每个epoch结束时清理缓存
    torch.cuda.empty_cache()
    
writer.close()

