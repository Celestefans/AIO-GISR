import h5py

def print_structure(h5_file, indent=''):
    """
    Recursively prints the structure of the HDF5 file.

    :param h5_file: HDF5 file object or group within the file.
    :param indent: A string of spaces for indentation.
    """
    for key in h5_file.keys():
        print(indent + 'Key:', key)
        if isinstance(h5_file[key], h5py.Dataset):
            # It's a dataset, print its shape and dtype
            print(indent + '  Shape:', h5_file[key].shape, ', Type:', h5_file[key].dtype)
        elif isinstance(h5_file[key], h5py.Group):
            # It's a group, recurse into it
            print_structure(h5_file[key], indent + '  ')

# Path to your HDF5 file
h5_path = '/data/datasets/pansharpening/testing/test_gf2_OrigScale_multiExm1.h5'
# Open the file and print its structure
with h5py.File(h5_path, 'r') as file:
    print_structure(file)
with h5py.File(h5_path, 'r') as f:
    # 读取名为 'gt' 的数据集
    mat_data = f['pan'][...]
    # 打印数据类型
    print(type(mat_data))