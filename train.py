# ... existing code ...

# 将rebalance loss替换为简单的L1 loss
loss = torch.abs(pred - target).mean()  # 直接使用L1 loss

# ... existing code ... 