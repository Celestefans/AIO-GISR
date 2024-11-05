import scipy.io as sio
import numpy as np



def load_and_inspect_mat_file(file_path):
    # 加载 .mat 文件
    mat_data = sio.loadmat(file_path)
    
    # 打印所有变量的名字
    print("Variables in the .mat file:")
    for key in mat_data.keys():
        # 跳过元数据键
        if '__' not in key:
            print(f"Key: {key}")
            value = mat_data[key]
            
            # 如果是 numpy 数组，打印形状
            if isinstance(value, np.ndarray):
                print(f"Shape of {key}: {value.shape}")
            else:
                print(f"Type of {key}: {type(value)}")
                
            # 打印数组内容（可选）
            # print(f"Content of {key}: {value}\n")
        else:
            continue

# 加载 .mat 文件
# mat_file_path = '/data/datasets/pansharpening/NBU_dataset_FR/IKONOS/PAN_128/56.mat'
# mat_data = scipy.io.loadmat(mat_file_path)
# file_path = '/data/datasets/pansharpening/NBU_dataset_FR/IKONOS/PAN_128/56.mat'
file_path = '/data/datasets/pansharpening/NBU_dataset0730/QB/test/GT_128/1901.mat'
load_and_inspect_mat_file(file_path)