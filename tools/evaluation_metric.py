import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR
import math

def Fpsnr(pred, gt, data_range = 1):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    bs = gt.shape[0]
    ans = 0
    for i in range(bs):  # * std[i]
        ans += calculate_psnr(gt[i, 0, ...], pred[i, 0, ...])
    return ans / bs

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def Fssim(pred, gt, data_range = 1):
    """ Compute Structural Similarity Index Metric (SSIM). """
    bs = gt.shape[0]
    ans = 0
    for i in range(bs):
        ans += compare_ssim(gt[i, 0, ...], pred[i, 0, ...], data_range=255.)
    return ans / bs

def calculate_rmse(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    rmse = np.sqrt(np.mean((img1 - img2)**2))
    return rmse

def Frmse(pred, gt, data_range = 1):
    """ Compute Normalized Mean Squared Error (NMSE) """
    bs = gt.shape[0]
    ans = 0
    for i in range(bs):  # * std[i]
        ans += calculate_rmse(gt[i, 0, ...], pred[i, 0, ...])
    return ans / bs
def SAM(I1,I2):
    p1 = np.sum(I1*I2,0)
    p2 = np.sum(I1*I1,0)
    p3 = np.sum(I2*I2,0)
    p4 = np.sqrt(p2*p3)
    p5 = p4.copy()
    p5[p5==0]=1e-15

    sam = np.arccos(p1/p5)
    p1 = p1.ravel()
    p4 = p4.ravel()
    s1 = p1[p4!=0]
    s2 = p4[p4!=0]
    x = (s1/s2)
    x[x>1] = 1
    angolo = np.mean(np.arccos(x))
    sam = np.real(angolo)*180/np.pi
    
    return sam

def ERGAS(I1,I2,c=4):
    s = 0
    R = I1-I2
    for i in range(c):
        res = R[i]
        s += np.mean(res*res)/(np.mean(I1[i])*np.mean(I1[i]))
    s = s/c
    ergas = (100/4) * np.sqrt(s)
    
    return ergas

def SSIM(pred, gt):
    """ Compute Structural Similarity Index Metric (SSIM). """
    bs = gt.shape[0]
    ans = 0
    for i in range(bs):
        ans += compare_ssim(gt[i, 0, ...], pred[i, 0, ...], data_range=1.)
    return ans / bs

def calc_rmse(a, b, minmax):
    """
    NYU_Depth_V2_Dataset
    参考:https://github.com/yanzq95/SGNet/blob/main/utils.py
    """
    a = a[6:-6, 6:-6]
    b = b[6:-6, 6:-6]
    
    a = a*(minmax[0]-minmax[1]) + minmax[1]
    b = b*(minmax[0]-minmax[1]) + minmax[1]
    a = a * 100
    b = b * 100
    
    return torch.sqrt(torch.mean(torch.pow(a-b,2)))

def midd_calc_rmse(gt, out):
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]
    gt = gt * 255.0
    out = out * 255.0
    
    return torch.sqrt(torch.mean(torch.pow(gt-out,2)))




if __name__ == "__main__":
    # x, y = torch.rand(2,3,16,16),torch.rand(2,3,16,16)
    x, y = np.random.rand(2,3,16,16), np.random.rand(2,3,16,16)
    # metric1 = PSNR(x,y)
    # metric2 = Fpsnr(x,y)
    rmse = RMSE(x,y)
    # print(metric1)
    # print(metric2)
    # x = x * 255.
    # y = y * 255.
    # metric1 = PSNR(x,y,data_range=255)
    # metric2 = Fpsnr(x,y)
    # print(metric1)
    # print(metric2)
    print(rmse)
    rmse = RMSE(x*255., y*255.)
    print(rmse)