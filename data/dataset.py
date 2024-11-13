import re
import torch
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms
import scipy.io
import os
import random
import numpy as np
import glob
import h5py
import cv2
from PIL import Image
import random

########################################################
###################### 需要用到的工具 ####################
########################################################
def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors.
    Args:
        x (torch.Tensor): A PyTorch tensor.
        shift (int): Amount to roll.
        dim (int): Which dimension to roll.
    Returns:
        torch.Tensor: Rolled version of x.
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=(-2,-1)):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    Args:
        x (torch.Tensor): A PyTorch tensor.
        dim (int): Which dimension to fftshift.
    Returns:
        torch.Tensor: fftshifted version of x.
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]

    return roll(x, shift, dim)

def real_to_complex(img):
    if len(img.shape)==3:
        data = img.unsqueeze(0)
    else:
        data = img
    y = torch.fft.fft2(data)
    y = fftshift(y, dim=(-2,-1))  ## (1,1,h,w)
    y_complex = torch.cat([y.real, y.imag], 1)  ## (1,2,h,w)
    if len(img.shape)==3:
        y_complex = y_complex[0]
    return y_complex

def complex_to_real(data):
    if len(data.shape)==3:
        data1 = data.unsqueeze(0)
    else:
        data1 = data
    h, w = data.shape[-2], data.shape[-1]
    y_real, y_imag = torch.chunk(data1, 2, dim=1)
    y = torch.complex(y_real, y_imag)
    y = fftshift(y, dim=(-2,-1))  ## (1,1,h,w)
    y = torch.fft.irfft2(y, s=(h, w))
    # y = torch.fft.ifft2(y,s=(h,w)).abs()
    if len(data.shape)==3:
        y = y[0]
    return y

def crop_k_data(data, scale):
    _,h,w = data.shape
    lr_h = h//scale
    lr_w = w//scale
    top_left_h = h//2-lr_h//2
    top_left_w = w//2-lr_w//2
    croped_data = data[:, top_left_h:(top_left_h+lr_h), top_left_w:(top_left_w+lr_w)]
    return croped_data


########################################################
################# Pansharpening相关数据集 ################
########################################################
class Pansharpening_H5_dataset(Dataset):
    def __init__(self, mat_data):
        gt_set = mat_data['gt'][...]
        pan_set = mat_data['pan'][...]
        ms_set = mat_data['ms'][...]

        self.gt_set = np.array(gt_set, dtype=np.float32) / 2047.
        self.pan_set = np.array(pan_set, dtype=np.float32) / 2047.
        self.ms_set = np.array(ms_set, dtype=np.float32) / 2047.

    def __getitem__(self, index):
        gt = self.gt_set[index, :, :, :]
        pan = self.pan_set[index, :, :, :]
        ms = self.ms_set[index, :, :, :]
        return ms, gt, pan 

    def __len__(self):
        return self.gt_set.shape[0]

class Pansharpening_full_dataset(Dataset):
    def __init__(self, mat_data):
        pan_set = mat_data['pan'][...]
        ms_set = mat_data['ms'][...]

        self.pan_set = np.array(pan_set, dtype=np.float32) / 2047.
        self.ms_set = np.array(ms_set, dtype=np.float32) / 2047.

    def __getitem__(self, index):
        pan = self.pan_set[index, :, :, :]
        ms = self.ms_set[index, :, :, :]
        return ms, pan

    def __len__(self):
        return self.pan_set.shape[0]
    

class Pansharpening_mat_Dataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_list = os.listdir(os.path.join(img_dir, 'MS_32/'))
        self.img_list.sort()

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        pan = scipy.io.loadmat(os.path.join(self.img_dir, 'PAN_128', self.img_list[idx]))['pan0'][...]
        ms = scipy.io.loadmat(os.path.join(self.img_dir, 'MS_32', self.img_list[idx]))['ms0'][...]
        gt = scipy.io.loadmat(os.path.join(self.img_dir, 'GT_128', self.img_list[idx]))['gt0'][...]

        pan = np.array(pan, dtype=np.float32)
        pan = np.expand_dims(pan, axis=0)
        ms = np.array(ms, dtype=np.float32)
        ms = np.transpose(ms, (2,0,1))
        gt = np.array(gt, dtype=np.float32)
        gt = np.transpose(gt, (2,0,1))
        
        return ms, gt, pan

class Combine_mat_Dataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.min_num = min(len(dataset) for dataset in self.datasets)
        self.dataset_indices = [list(range(len(dataset))) for dataset in self.datasets]

    def __len__(self):
        return self.min_num
    
    def shuffle(self):
        for indices in self.dataset_indices:
            random.shuffle(indices)
    
    def __getitem__(self, index):
        data_lists = [list(dataset[self.dataset_indices[i][index]]) for i,dataset in enumerate(self.datasets)]
        return tuple(data_lists)
        

########################################################
###################### MRI超分数据集 ####################
########################################################
class MRI_dataset(Dataset):
    def __init__(self, PD_dir, T2_dir):
        self.PD_dir = PD_dir
        self.T2_dir = T2_dir
        self.filesname = os.listdir(PD_dir)

    def __len__(self):
        return len(self.filesname)

    def __getitem__(self, index):

        PD_path = os.path.join(self.PD_dir, self.filesname[index])
        T2_path = os.path.join(self.T2_dir, self.filesname[index])

        PD_img = cv2.imread(PD_path, cv2.IMREAD_UNCHANGED)
        T2_img = cv2.imread(T2_path, cv2.IMREAD_UNCHANGED)

        PD_img = torch.tensor(PD_img).unsqueeze(0).float()/255.
        T2_img = torch.tensor(T2_img).unsqueeze(0).float()/255.

        # FFT
        T2_complex = real_to_complex(T2_img)

        # center_crop
        T2_lr_complex = crop_k_data(T2_complex, 4)

        # IFFT
        T2_lr_img = complex_to_real(T2_lr_complex)
        T2_lr_img = (T2_lr_img - T2_lr_img.min()) / (T2_lr_img.max()-T2_lr_img.min())

        return T2_lr_img, T2_img, PD_img

## 加载预先降采样好的数据
class MRI_pre_dataset(Dataset):

    def __init__(self, t2_dir, T2_dir, T1_dir):
        def numerical_sort_key(s):
            number_part = re.findall(r'\d+', s)[0]
            return int(number_part)
        self.T1_dir = T1_dir
        self.T2_dir = T2_dir
        self.t2_dir = t2_dir

        self.files = os.listdir(self.T1_dir)
        self.filesname = sorted(self.files, key=numerical_sort_key)

    def __getitem__(self, index):

        T1_path = os.path.join(self.T1_dir, self.filesname[index])
        T2_path = os.path.join(self.T2_dir, self.filesname[index])
        t2_path = os.path.join(self.t2_dir, self.filesname[index])

        t2_img = cv2.imread(t2_path, cv2.IMREAD_UNCHANGED)
        T2_img = cv2.imread(T2_path, cv2.IMREAD_UNCHANGED)
        T1_img = cv2.imread(T1_path, cv2.IMREAD_UNCHANGED)

        t2 = torch.tensor(t2_img).unsqueeze(0).float()/255.
        T2 = torch.tensor(T2_img).unsqueeze(0).float()/255.
        T1 = torch.tensor(T1_img).unsqueeze(0).float()/255.


        return t2, T2, T1

    def __len__(self):
        return len(self.filesname)
########################################################
###################### 深度图超分数据集 ###################
########################################################
def augment(img,gt, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5

    if hflip: 
        img = img[:, ::-1, :].copy()
        gt = gt[:, ::-1, :].copy()
    if vflip: 
        img = img[::-1, :, :].copy()
        gt = gt[::-1, :, :].copy()

    return img, gt


def get_patch(img, gt, patch_size=16):
    th, tw = img.shape[:2]

    tp = round(patch_size)

    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))

    return img[ty:ty + tp, tx:tx + tp, :], gt[ty:ty + tp, tx:tx + tp, :]
class NYU_v2_datset(Dataset):
    """NYUDataset."""

    def __init__(self, root_dir, down="bicubic", scale=4, train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            train (bool): train or test
            transform (callable, optional): Optional transform to be applied on a sample.
            
        """
        self.down = down
        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale
        self.train = train
        
        if train:
            self.depths = np.load('%s/train_depth_split.npy'%root_dir)
            # print(f"depths.shape:{self.depths.shape}")
            self.images = np.load('%s/train_images_split.npy'%root_dir)
            # print(f"image.shape:{self.images.shape}")
        else:
            self.depths = np.load('%s/test_depth.npy'%root_dir)
            # print(f"depths.shape:{self.depths.shape}")
            self.images = np.load('%s/test_images_v2.npy'%root_dir)
            # print(f"image.shape:{self.images.shape}")
    def __len__(self):
        return self.depths.shape[0]

    def __getitem__(self, idx):
        depth = self.depths[idx]
        image = self.images[idx]
        if self.train:
            image, depth = get_patch(img=image, gt=np.expand_dims(depth,2), patch_size=256)
            image, depth = augment(img=image, gt=depth)
        h, w = depth.shape[:2]
        s = self.scale

        if self.down == "bicubic":
            # bicubic down-sampling
            lr = np.array(Image.fromarray(depth.squeeze()).resize((w//s,h//s), Image.BICUBIC))
        if self.down == "direct":
            lr = np.array(Image.fromarray(depth.squeeze()).resize((w//s,h//s)))

        if self.transform:
            image = self.transform(image).float()
            depth = self.transform(depth).float()
            lr = self.transform(np.expand_dims(lr, 2)).float()
        
        return lr, depth, image
    
class MultiTaskDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.min_num = min(len(dataset) for dataset in self.datasets)
        self.dataset_indices = [list(range(len(dataset))) for dataset in self.datasets]

    def __len__(self):
        return self.min_num

    def shuffle(self):
        for indices in self.dataset_indices:
            random.shuffle(indices)

    def __getitem__(self, index):
        data_lists = [list(dataset[self.dataset_indices[i][index]]) for i,dataset in  enumerate(self.datasets)]    
        return tuple(data_lists)


if __name__ == "__main__": 

    depth_root = '/data/cjj/dataset/NYU_V2'
    mri_root = '/data/wtt/WJ_Code_DURE_HF/MRI_dataset/BT'
    pan_root = '/data/datasets/pansharpening/NBU_dataset0730/WV4/train'

    data_transform = transforms.Compose([transforms.ToTensor()])
    depth_dataset = NYU_v2_datset(root_dir=depth_root, scale=4, transform=data_transform, train=True)
    # depth_dataset = NYU_v2_datset(root_dir=depth_root, scale=4, train=True)
    mri_dataset = MRI_pre_dataset(os.path.join(mri_root,'t2_train'), os.path.join(mri_root,'T2_train'), os.path.join(mri_root,'T1_train'))
    # mri_dataset = MRI_dataset(os.path.join(mri_root,'T1_train'), os.path.join(mri_root,'T2_train'))
    pan_dataset = Pansharpening_mat_Dataset(pan_root)
    mix_dataset = MultiTaskDataset(depth_dataset, mri_dataset, pan_dataset)

    dataloader = DataLoader(mix_dataset, batch_size=1, shuffle=False, num_workers=0)
    count = 0
    for i,data_list in enumerate(dataloader):
        for data in data_list:
            lr, gt, gi = data[0], data[1], data[2]
            lr = lr.type(torch.FloatTensor)
            gt = gt.type(torch.FloatTensor)
            gi = gi.type(torch.FloatTensor)
            print( lr.shape, gt.shape, gi.shape)
            print(lr.max(), gt.max(), gi.max())
        count += 1
        if(count == 2): 
            break    
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")  
        
