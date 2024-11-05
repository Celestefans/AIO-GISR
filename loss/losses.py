import torch.nn.functional as F
from torch import nn
import torch
import numpy as np

class Rebalance_L1(nn.Module):
    def __init__(self, num_tasks=7, momentum=0.9, T=10):
        super(Rebalance_L1, self).__init__()
        self.num_tasks = num_tasks
        self.l1_loss = nn.L1Loss()
        self.momentum = momentum
        self.T = T
        
        # 使用tensor存储EMA值，便于计算
        self.register_buffer('loss_grads_ema', torch.zeros(num_tasks))
        self.register_buffer('running_std', torch.ones(num_tasks))  # 用于标准化
        self.register_buffer('running_mean', torch.zeros(num_tasks))  # 用于标准化
        self.register_buffer('update_counts', torch.zeros(num_tasks))  # 添加这行
        
    def update_statistics(self, grad, task_id):
        # 更新计数
        with torch.no_grad():
            self.update_counts[task_id] += 1
            
            # 更新均值和标准差
            self.running_mean[task_id] = (
                self.momentum * self.running_mean[task_id] + 
                (1 - self.momentum) * grad
            )
            self.running_std[task_id] = (
                self.momentum * self.running_std[task_id] + 
                (1 - self.momentum) * torch.abs(grad - self.running_mean[task_id])
            )
            
            # 标准化后的梯度
            normalized_grad = (grad - self.running_mean[task_id]) / (self.running_std[task_id] + 1e-8)
            
            # 更新EMA
            self.loss_grads_ema[task_id] = (
                self.momentum * self.loss_grads_ema[task_id] + 
                (1 - self.momentum) * normalized_grad
            )
    
    def forward(self, restored, target, task_id, T=None, loss_list_last=None, loss_list_min=None):
        if loss_list_last is None:
            loss_list_last = [1.0] * self.num_tasks
        if loss_list_min is None:
            loss_list_min = [0.1] * self.num_tasks
        if T is None:
            T = self.T

        # 计算基础损失
        task_loss = self.l1_loss(restored, target)
        
        # 计算当前任务的loss gradient
        curr_loss_grad = (
            task_loss * torch.log10(torch.tensor(loss_list_min[task_id]).to(restored.device)) / 
            loss_list_last[task_id]
        ) / (torch.log10(task_loss + 1e-8) * T)
        
        # 更新统计信息
        self.update_statistics(curr_loss_grad, task_id)
        
        # 应用温度缩放并计算权重
        scaled_grads = self.loss_grads_ema / T
        weights = F.softmax(scaled_grads, dim=0)
        
        # 记录权重用于监控
        weight_list_dict = torch.zeros_like(weights)
        weight_list_dict[task_id] = weights[task_id]
        
        # 计算加权损失
        weighted_loss = task_loss * weights[task_id]
        
        return weighted_loss, task_loss.item(), self.loss_grads_ema.detach(), weight_list_dict
    
    def get_task_weights(self):
        """返回当前所有任务的权重分布"""
        with torch.no_grad():
            return F.softmax(self.loss_grads_ema / self.T, dim=0)
