import os
import shutil
import glob
import re

def natural_sort_key(s):
    """自然排序的关键字生成器，用于字符串列表排序"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def move_last_100_files(source_folder, destination_folder):
    # 创建目标文件夹
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 获取所有的 .mat 文件并排序
    mat_files = sorted(glob.glob(os.path.join(source_folder, '*.mat')), key=natural_sort_key)
    
    # 如果有足够的文件，则只取最后100个
    if len(mat_files) >= 100:
        files_to_move = mat_files[-100:]
    else:
        files_to_move = mat_files
    
    for file in files_to_move:
        shutil.move(file, destination_folder)
        # shutil.copy(file, destination_folder)

# 主函数
def main():
    # 假设你的文件夹结构如下
    base_folder = '/data/datasets/pansharpening/NBU_dataset0730/WV3/test'
    subfolders = [os.path.join(base_folder, folder_name) for folder_name in ['PAN_128', 'MS_32', 'GT_128']]
    destination_base = '/data/datasets/pansharpening/NBU_dataset0730/WV3/test_select'

    # 对于每个子文件夹
    for source_folder, folder_name in zip(subfolders, ['PAN_128', 'MS_32', 'GT_128']):
        destination_folder = os.path.join(destination_base, folder_name)
        move_last_100_files(source_folder, destination_folder)
        print(f"Moved files from {source_folder} to {destination_folder}")

if __name__ == "__main__":
    main()