import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
import time

# 创建一个占用约10GB显存的张量
num_elements = int(10 * 1024 * 1024 * 1024 / 4)  # 基于单精度浮点数计算元素数量
tensor = torch.randn(num_elements, device='cuda')
print(f"Tensor of size {tensor.size()} created on GPU, occupying approximately 10GB.")

# 添加一个无限循环来保持程序运行
try:
    print("Tensor is now occupying memory. Press Ctrl+C to exit.")
    while True:
        # 可以添加一些操作，比如打印时间，以观察程序仍在运行
        print(".", end="", flush=True)
        time.sleep(1)  # 暂停一秒，防止CPU占用过高
except KeyboardInterrupt:
    print("\nExiting program. Tensor will be garbage collected and memory released.")