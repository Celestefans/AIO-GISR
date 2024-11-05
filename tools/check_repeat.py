import torch

# 创建一个形状为 (B, C, H, W) 的示例张量，B=2, C=4, H=3, W=3
B, C, H, W = 2, 4, 3, 3
input_tensor = torch.tensor([
    [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
        [[28, 29, 30], [31, 32, 33], [34, 35, 36]]
    ],
    [
        [[37, 38, 39], [40, 41, 42], [43, 44, 45]],
        [[46, 47, 48], [49, 50, 51], [52, 53, 54]],
        [[55, 56, 57], [58, 59, 60], [61, 62, 63]],
        [[64, 65, 66], [67, 68, 69], [70, 71, 72]]
    ]
])

# 使用 repeat_interleave 在通道维度上进行重复
output_tensor = input_tensor.repeat_interleave(2, dim=1)
restored_tensor = output_tensor.float().view(B, C, 2, H, W).mean(dim=2)
# 打印输入张量和输出张量的形状
print("Input shape:", input_tensor.shape)  # 输出: torch.Size([2, 4, 3, 3])
print("Output shape:", output_tensor.shape)  # 输出: torch.Size([2, 8, 3, 3])

# 打印输出张量
# print("Output tensor:\n", output_tensor)
print('restored:\n', restored_tensor, restored_tensor.shape)