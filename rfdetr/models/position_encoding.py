# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from rfdetr.util.misc import NestedTensor
from dinov3.layers import RopePositionEmbedding

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self._export = False
    
    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export

    def forward(self, tensor_list: NestedTensor, align_dim_orders = True):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        if align_dim_orders:
            pos = torch.cat((pos_y, pos_x), dim=3).permute(1, 2, 0, 3)
            # return: (H, W, bs, C)
        else:
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
            # return: (bs, C, H, W)
        return pos
    
    def forward_export(self, mask:torch.Tensor, align_dim_orders = True):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        if align_dim_orders:
            pos = torch.cat((pos_y, pos_x), dim=3).permute(1, 2, 0, 3)
            # return: (H, W, bs, C)
        else:
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
            # return: (bs, C, H, W)
        return pos


class PositionEmbeddingLearned(nn.Module):
    def __init__(self, hidden_dim, max_len=60, normalize=True, scale=None):
        super().__init__()
        self.hidden_dim = hidden_dim  # 位置编码总维度
        self.max_len = max_len        # 最大行列长度（用于初始化嵌入）
        self.normalize = normalize    # 是否归一化有效区域
        self.scale = scale if scale is not None else 2.0  # 缩放因子
        
        # 行/列嵌入各占一半维度
        self.row_embed = nn.Embedding(max_len, hidden_dim // 2)
        self.col_embed = nn.Embedding(max_len, hidden_dim // 2)
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化嵌入参数
        nn.init.uniform_(self.row_embed.weight, -0.02, 0.02)
        nn.init.uniform_(self.col_embed.weight, -0.02, 0.02)

    def forward(self, tensor_list: NestedTensor, align_dim_orders=False):
        """从mask中获取维度，避免依赖x的形状"""
        mask = tensor_list.mask  # (bs, H, W)
        assert mask is not None, "mask不能为空"
        # 从mask获取批次大小和特征图尺寸（不再使用x）
        bs, h, w = mask.shape
        device = mask.device
        return self._compute_pos_emb(mask, h, w, device, align_dim_orders)

    def forward_export(self, mask: torch.Tensor, align_dim_orders=False):
        """导出专用接口，仅用mask计算位置编码"""
        assert mask is not None, "mask不能为空"
        bs, h, w = mask.shape  # 直接从mask获取维度
        device = mask.device
        return self._compute_pos_emb(mask, h, w, device, align_dim_orders)

    def export(self):
        """模型导出配置：冻结参数并切换到推理模式"""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        return self

    def _compute_pos_emb(self, mask, h, w, device, align_dim_orders):
        """核心计算逻辑：基于mask生成位置编码"""
        # 1. 生成基础行/列索引并获取可学习嵌入
        row_idx = torch.arange(self.max_len, device=device)  # (max_len,)
        col_idx = torch.arange(self.max_len, device=device)  # (max_len,)
        row_emb = self.row_embed(row_idx)  # (max_len, hidden_dim//2)
        col_emb = self.col_embed(col_idx)  # (max_len, hidden_dim//2)

        # 2. 插值到目标尺寸（h, w）
        row_emb = row_emb.permute(1, 0).unsqueeze(0)  # (1, C//2, max_len)
        col_emb = col_emb.permute(1, 0).unsqueeze(0)  # (1, C//2, max_len)
        row_emb = F.interpolate(row_emb, size=h, mode="linear", align_corners=False)  # (1, C//2, h)
        col_emb = F.interpolate(col_emb, size=w, mode="linear", align_corners=False)  # (1, C//2, w)

        # 3. 扩展到批次维度并广播为特征图尺寸
        row_emb = row_emb.repeat(bs, 1, 1).unsqueeze(3)  # (bs, C//2, h, 1)
        col_emb = col_emb.repeat(bs, 1, 1).unsqueeze(2)  # (bs, C//2, 1, w)
        row_emb = row_emb.expand(-1, -1, -1, w)  # (bs, C//2, h, w)
        col_emb = col_emb.expand(-1, -1, h, -1)  # (bs, C//2, h, w)

        # 4. 拼接行/列嵌入并处理mask
        pos_emb = torch.cat([row_emb, col_emb], dim=1)  # (bs, hidden_dim, h, w)
        not_mask = ~mask  # (bs, h, w)：有效区域为True
        pos_emb = pos_emb * not_mask.unsqueeze(1).float()  # padding区域归零

        # 5. 归一化有效区域（与原始正弦编码逻辑一致）
        if self.normalize:
            eps = 1e-6
            h_valid = not_mask.sum(dim=1, keepdim=True).float()  # (bs, 1, w)：每行有效像素数
            w_valid = not_mask.sum(dim=2, keepdim=True).float()  # (bs, h, 1)：每列有效像素数
            h_valid = torch.clamp(h_valid, min=eps)
            w_valid = torch.clamp(w_valid, min=eps)
            # 对行/列嵌入分别缩放
            pos_emb[:, :self.hidden_dim//2, :, :] *= (self.scale / h_valid).unsqueeze(1)
            pos_emb[:, self.hidden_dim//2:, :, :] *= (self.scale / w_valid).unsqueeze(1)

        # 6. 调整维度顺序（默认返回 (bs, C, H, W)）
        if align_dim_orders:
            pos_emb = pos_emb.permute(2, 3, 0, 1)  # (H, W, bs, C)
        # else: 保持 (bs, C, H, W)

        return pos_emb


def build_position_encoding(hidden_dim, position_embedding):
    N_steps = hidden_dim // 2
    if position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif position_embedding in ('learned'):
        position_embedding = PositionEmbeddingLearned(hidden_dim)
        # print("position_embedding",position_embedding)
    elif position_embedding in ('rope', 'v3'):
        position_embedding = RopePositionEmbedding(embed_dim=hidden_dim)
    else:
        raise ValueError(f"not supported {position_embedding}")

    return position_embedding
