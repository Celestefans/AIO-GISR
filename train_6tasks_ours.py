import os
import time
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
# from model.Ablation.moe_IFM_clip_test import MOE_IFM_Clip_test
from model.MOE_Clip_AIO_GISR import MOE_IFM_Clip
from data.dataset import Pansharpening_mat_Dataset, MRI_pre_dataset, NYU_v2_datset, MultiTaskDataset
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training 6 Tasks with CLIP')
    parser.add_argument('--exp_name', type=str, default='MOE_Clip_AIO_GISR_6tasks', help='experiment name')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--num_epoch', type=int, default=500, help='number of epochs')
    parser.add_argument('--lr', type=float, default=4e-4, help='learning rate')
    parser.add_argument('--gpu', type=str, default='4', help='GPU ID')
    parser.add_argument('--depth_root', type=str, default='/data/cjj/dataset/NYU_V2', help='depth dataset root')
    parser.add_argument('--mri_root', type=str, default='/data/wtt/MRI_align/BT', help='MRI dataset root')
    parser.add_argument('--pan_root', type=str, default='/data/datasets/pansharpening/NBU_dataset0730', help='pansharpening dataset root')
    parser.add_argument('--clip_path', type=str, default='./data/SR_text_feature_Dep4_8_16_MRI2_4_8_WV4_QB_GF1.th', help='CLIP feature path')
    parser.add_argument('--validate_first', action='store_true', help='run validation before training')
    return parser.parse_args()

def train_one_epoch(epoch, dataloader, model, optimizer, loss_fn, clip_feat, writer):
    try:
        model.train()
        epoch_losses = {'total': 0, 'task_losses': [[] for _ in range(6)]}
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for train_data_lists in pbar:
            for task_id, data_list in enumerate(train_data_lists):
                inp_lr, inp_gt, inp_guide = (item.type(torch.FloatTensor).cuda() for item in data_list)
                
                optimizer.zero_grad()
                
                inp_clip = clip_feat[task_id]
                restored, loss_moe = model(inp_lr, inp_guide, inp_clip)
                
                loss_task = loss_fn(restored, inp_gt)
                loss_G = loss_task + 0.001 * loss_moe
                
                loss_G.backward()
                optimizer.step()
                
                epoch_losses['total'] += loss_G.item()
                epoch_losses['task_losses'][task_id].append(loss_task.item())
                
                writer.add_scalar(f'Train/Task_{task_id}_Loss', loss_task.item(), epoch)
                
                pbar.set_postfix(
                    loss=loss_G.item(),
                    task=task_id,
                    lr=optimizer.param_groups[0]['lr']
                )
        
        avg_total_loss = epoch_losses['total'] / len(dataloader)
        avg_task_losses = [np.mean(losses) if losses else 0 for losses in epoch_losses['task_losses']]
        
        return avg_total_loss, avg_task_losses
    finally:
        # 确保数据加载器被正确清理
        if hasattr(dataloader, '_iterator'):
            del dataloader._iterator

def validate_one_epoch(model, datasets, test_minmax, logger, epoch, clip_feat, writer, save_dir, optimizer, scheduler):
    global best_psnr_pan_WV4, best_psnr_pan_QB
    global best_psnr_mri_2x, best_psnr_mri_4x
    global best_rmse_4, best_rmse_8
    
    model.eval()
    with torch.no_grad():
        psnr = []
        rmse = []
        
        best_metrics = {
            'rmse_4': best_rmse_4,
            'rmse_8': best_rmse_8,
            'psnr_mri_2x': best_psnr_mri_2x,
            'psnr_mri_4x': best_psnr_mri_4x,
            'psnr_pan_wv4': best_psnr_pan_WV4,
            'psnr_pan_qb': best_psnr_pan_QB
        }
        
        for dataset_id, dataset in enumerate(datasets):
            val_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            metric_values = []
            inp_clip = clip_feat[dataset_id]
            if dataset_id < 2:  # depth estimation tasks
                for index, (inp_lr, inp_gt, inp_guide) in enumerate(tqdm(val_dataloader, desc=f'Validating Depth_{4*(2**dataset_id)}')):
                    output = model(inp_lr.cuda(), inp_guide.cuda(), inp_clip)[0]
                    metric_values.append(
                        calc_rmse(output[0], 
                                inp_gt.cuda()[0,0], 
                                torch.from_numpy(test_minmax[:, index]).cuda())
                    )
                avg_metric = torch.mean(torch.stack(metric_values)).item()
                rmse.append(avg_metric)
            
            else:  # MRI and pansharpening tasks
                for inp_lr, inp_gt, inp_guide in tqdm(val_dataloader, desc=f'Validating Task_{dataset_id}'):
                    output = model(inp_lr.cuda(), inp_guide.cuda(), inp_clip)
                    metric_values.append(
                        PSNR(inp_gt.numpy()[0], output.cpu().numpy()[0])
                    )
                avg_metric = np.mean(metric_values)
                psnr.append(avg_metric)
        
        # 记录验证指标
        metrics = {
            'rmse_4': rmse[0],
            'rmse_8': rmse[1],
            'psnr_mri_2x': psnr[0],
            'psnr_mri_4x': psnr[1],
            'psnr_pan_wv4': psnr[2],
            'psnr_pan_qb': psnr[3]
        }
        
        # 记录到tensorboard
        for name, value in metrics.items():
            writer.add_scalar(f'Validation/{name.upper()}', value, epoch)
        
        # 检查并保存最佳模型
        improved = False
        for metric_name, metric_value in metrics.items():
            is_better = metric_value > best_metrics[metric_name] if 'psnr' in metric_name else metric_value < best_metrics[metric_name]
            if is_better:
                improved = True
                best_metrics[metric_name] = metric_value
                globals()[f'best_{metric_name}'] = metric_value
                model_path = os.path.join(save_dir, f'Best_{metric_name.upper()}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metric_value': metric_value,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, model_path)
        
        # 记录所有指标
        log_message = f'Epoch {epoch} Validation:\n'
        for name, value in metrics.items():
            log_message += f'{name}: {value:.4f}, '
        logger.info(log_message)
        
        return metrics, improved

def main():
    # Parse arguments
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Create save directory
    save_dir = os.path.join('./Checkpoint/All-in-one', args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard_logs'))
    
    # Initialize logger
# Initialize logger
    logger = get_logger(os.path.join(save_dir, 'run.log'), mode='a')  # 改为'a'模式进行追加
    logger.info(f"\n{'='*20} New Training Session {'='*20}")  # 添加分隔符标识新的训练会话
    logger.info(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")  # 记录开始时间
    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Number of epochs: {args.num_epoch}")
    logger.info(f"{'='*60}")
    
    # 在这里定义全局变量
    global best_psnr_pan_WV4, best_psnr_pan_QB
    global best_psnr_mri_2x, best_psnr_mri_4x
    global best_rmse_4, best_rmse_8
    
    # Initialize best metrics
    best_psnr_pan_WV4, best_psnr_pan_QB = -float('inf'), -float('inf')
    best_psnr_mri_2x, best_psnr_mri_4x = -float('inf'), -float('inf')
    best_rmse_4, best_rmse_8 = float('inf'), float('inf')
    
    # Initialize datasets
    data_transform = transforms.Compose([transforms.ToTensor()])
    
    # Training datasets
    depth_dataset_4 = NYU_v2_datset(root_dir=args.depth_root, scale=4, transform=data_transform, train=True)
    depth_dataset_8 = NYU_v2_datset(root_dir=args.depth_root, scale=8, transform=data_transform, train=True)
    mri_dataset_2 = MRI_pre_dataset(os.path.join(args.mri_root, 'x2_t2_train'), os.path.join(args.mri_root, 'T2_train'), os.path.join(args.mri_root, 'T1_train'))
    mri_dataset_4 = MRI_pre_dataset(os.path.join(args.mri_root, 'x4_t2_train'), os.path.join(args.mri_root, 'T2_train'), os.path.join(args.mri_root, 'T1_train'))
    pan_dataset_WV4 = Pansharpening_mat_Dataset(os.path.join(args.pan_root, 'WV4', 'train'))
    pan_dataset_QB = Pansharpening_mat_Dataset(os.path.join(args.pan_root, 'QB', 'train'))
    
    mix_dataset = MultiTaskDataset(depth_dataset_4, depth_dataset_8,
                                  mri_dataset_2, mri_dataset_4,
                                  pan_dataset_WV4, pan_dataset_QB)
    
    # Validation datasets
    test_minmax = np.load(f'{args.depth_root}/test_minmax.npy')
    val_depth_dataset_4 = NYU_v2_datset(root_dir=args.depth_root, scale=4, transform=data_transform, train=False)
    val_depth_dataset_8 = NYU_v2_datset(root_dir=args.depth_root, scale=8, transform=data_transform, train=False)
    val_mri_dataset_2 = MRI_pre_dataset(os.path.join(args.mri_root, 'x2_t2_test'), os.path.join(args.mri_root, 'T2_test'), os.path.join(args.mri_root, 'T1_test'))
    val_mri_dataset_4 = MRI_pre_dataset(os.path.join(args.mri_root, 'x4_t2_test'), os.path.join(args.mri_root, 'T2_test'), os.path.join(args.mri_root, 'T1_test'))
    val_pan_dataset_WV4 = Pansharpening_mat_Dataset(os.path.join(args.pan_root,'WV4', 'test'))
    val_pan_dataset_QB = Pansharpening_mat_Dataset(os.path.join(args.pan_root,'QB', 'test'))
    
    list_val_dataset = [val_depth_dataset_4, val_depth_dataset_8,
                        val_mri_dataset_2, val_mri_dataset_4,
                        val_pan_dataset_WV4, val_pan_dataset_QB]
    
    # Initialize model, optimizer, scheduler and loss
    Generator = MOE_IFM_Clip(dim=22, num_blocks=[3, 4, 4, 5], prompt_num=32).cuda()
    optimizer_G = torch.optim.Adam(Generator.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    lr_scheduler_G = CosineAnnealingLR(optimizer_G, args.num_epoch, eta_min=1.0e-6)
    l1_loss = L1Loss().cuda()
    clip_feat = torch.load(args.clip_path).cuda()
    
    # 添加从checkpoint加载的逻辑
    start_epoch = 1
    checkpoint_path = os.path.join(save_dir, 'latest_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        Generator.load_state_dict(checkpoint['model_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler_G.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
        # 恢复最佳指标
        metrics = checkpoint.get('metrics', {})
        
        best_rmse_4 = metrics.get('rmse_4', float('inf'))
        best_rmse_8 = metrics.get('rmse_8', float('inf'))
        best_psnr_mri_2x = metrics.get('psnr_mri_2x', -float('inf'))
        best_psnr_mri_4x = metrics.get('psnr_mri_4x', -float('inf'))
        best_psnr_pan_WV4 = metrics.get('psnr_pan_wv4', -float('inf'))
        best_psnr_pan_QB = metrics.get('psnr_pan_qb', -float('inf'))
        
        logger.info(f"Resuming from epoch {start_epoch}")
        logger.info("Loaded best metrics:")
        for name, value in metrics.items():
            logger.info(f"{name}: {value:.4f}")
    
    # 在训练循环之前测试验证代码
    if args.validate_first:
        logger.info("Running initial validation...")
        metrics, improved = validate_one_epoch(
            Generator, list_val_dataset, test_minmax, logger, 0, clip_feat,
            writer, save_dir, optimizer_G, lr_scheduler_G
        )
        logger.info("Initial validation completed.")
        
    # 修改训练循环的起始epoch
    for epoch in range(start_epoch, args.num_epoch + 1):
        mix_dataset.shuffle()
        train_dataloader = DataLoader(
            mix_dataset, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2
        )
        
        try:
            avg_total_loss, avg_task_losses = train_one_epoch(
                epoch, train_dataloader, Generator, optimizer_G, l1_loss, clip_feat, writer
            )
            lr_scheduler_G.step()
            
            # 验证条件
            if (epoch < 250 and epoch % 20 == 0) or (epoch > 250 and epoch % 5 == 0):
                metrics, improved = validate_one_epoch(
                    Generator, list_val_dataset, test_minmax, logger, epoch, clip_feat,
                    writer, save_dir, optimizer_G, lr_scheduler_G
                )
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': Generator.state_dict(),
                    'optimizer_state_dict': optimizer_G.state_dict(),
                    'scheduler_state_dict': lr_scheduler_G.state_dict(),
                    'metrics': metrics
                }
                torch.save(checkpoint, os.path.join(save_dir, 'latest_checkpoint.pth'))
                if improved:
                    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
        finally:
            # 确保在每个epoch结束时清理数据加载器
            del train_dataloader
            torch.cuda.empty_cache()
    writer.close()
    
if __name__ == '__main__':
    main()