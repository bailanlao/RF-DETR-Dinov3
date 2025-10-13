from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.models.backbone.dinov3 import DinoV3
from rfdetr import RFDETRNano, RFDETRBase, RFDETRMedium, RFDETRLarge, RFDETRMediumV3, RFDETRNanoV3,RFDETRMediumV3Plus
import torch
from rfdetr.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)
import torch
import torch.nn as nn
import sys
import numpy as np

class NestedTensor:
    def __init__(self, tensors, mask):
        self.tensors = tensors  # 图像张量 (B, C, H, W)
        self.mask = mask        # 掩码 (B, H, W)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
plus=''
if plus == 'plus':
    model = RFDETRMediumV3Plus(position_embedding='sine')
else:
    model = RFDETRMediumV3(position_embedding='sine',select_mode=2)
model = model.model.model.to(device)
backbone=model.backbone

# 1. 生成随机输入（320x320，3通道，批次大小为1）
batch_size = 1
H, W = 480, 480
channels = 3
# 随机图像张量：(B, C, H, W)
random_img = torch.randn(batch_size, channels, H, W).to(device)
# 生成掩码（全False，表示无遮挡区域）
mask = torch.zeros(batch_size, H, W, dtype=torch.bool).to(device)
# 构造NestedTensor（包含图像和掩码）
input_tensor = NestedTensor(random_img, mask)

# 2. 前向传播
with torch.no_grad():  # 关闭梯度计算，加速测试
    output = backbone[0](input_tensor)

# 3. 打印输出特征信息
print(f"\n输入图像形状: {random_img.shape}")
print(f"输出特征数量: {len(output)}")

for i, feat in enumerate(output):
    # feat是NestedTensor，包含.tensors（特征图）和.mask（掩码）
    print(f"特征{i+1} - 特征图形状: {feat.tensors.shape}, 掩码形状: {feat.mask.shape}")

result = model(input_tensor)