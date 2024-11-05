import os
import shutil
import glob
import random

def copy_common_files(source_folders, destination_folder, percentage=10):
    # 获取第一个文件夹中的所有.mat文件
    mat_files = glob.glob(os.path.join(source_folders[0], '*.mat'))
    
    # 计算要选择的文件数量
    num_files_to_select = max(1, int(len(mat_files) * percentage / 100))
    
    # 随机选择文件
    selected_files = random.sample(mat_files, num_files_to_select)
    
    # 创建目标文件夹
    for folder_name in ['PAN_128', 'MS_32', 'GT_128']:
        dest_subfolder = os.path.join(destination_folder, folder_name)
        if not os.path.exists(dest_subfolder):
            os.makedirs(dest_subfolder)
    
    # 复制文件
    for source_folder, folder_name in zip(source_folders, ['PAN_128', 'MS_32', 'GT_128']):
        dest_subfolder = os.path.join(destination_folder, folder_name)
        for file_path in selected_files:
            # 获取文件名
            filename = os.path.basename(file_path)
            # 构造源文件和目标文件的完整路径
            src_file = os.path.join(source_folder, filename)
            dest_file = os.path.join(dest_subfolder, filename)
            # 复制文件
            shutil.move(src_file, dest_file)

# 主函数
def main():
    # 指定原始文件夹和目标文件夹的路径
    base_folder = '/data/datasets/pansharpening/NBU_dataset0730/WV4/train'
    source_folders = [os.path.join(base_folder, folder_name) for folder_name in ['PAN_128', 'MS_32', 'GT_128']]
    destination_folder = '/data/datasets/pansharpening/NBU_dataset0730/WV4/validation'

    # 复制文件
    copy_common_files(source_folders, destination_folder)
    
    print("Files moved successfully.")

if __name__ == "__main__":
    main()