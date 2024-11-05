import os
from scipy.io import loadmat
import numpy as np

def find_max_in_mat_files(directory):
    max_value = -np.inf  # 初始化最大值为负无穷大

    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.mat'):  # 检查文件是否为 .mat 文件
            file_path = os.path.join(directory, filename)
            
            # 加载 .mat 文件
            mat_data = loadmat(file_path)

            # 遍历所有变量，查找最大值
            for key in mat_data:
                # 跳过 __header__、__version__ 和 __globals__ 特殊键
                if not key.startswith('__'):
                    data = mat_data[key]
                    # 检查数据是否为数组
                    if isinstance(data, np.ndarray):
                        # 更新最大值
                        current_max = np.max(data)
                        if current_max > max_value:
                            max_value = current_max
                            max_file = filename  # 记录最大值所在的文件

    return max_value, max_file

# 指定要查找的目录路径
directory_path = '/data/datasets/pansharpening/NBU_dataset0730/QB/train/GT_128'

# 查找目录下所有 .mat 文件中的最大值
max_value, max_file = find_max_in_mat_files(directory_path)
print(max_value)