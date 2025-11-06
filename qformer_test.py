# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import List, Tuple, Union, Optional

import torch
import torch.nn.functional as F
from dinov3.utils import cat_keep_shapes, uncat_with_shapes
from torch import Tensor, nn
from einops import rearrange
from timm.models.vision_transformer import Mlp as MLP
from timm.models.layers import trunc_normal_
from rfdetr.models.backbone.rope_position_encoding import RopePositionEmbedding
from rfdetr import RFDETRMediumV3

# Import LayerScale from dinov3
from dinov3.layers.layer_scale import LayerScale

# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class RectifyCoordsGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, coords_lambda=20):
        ctx.in1 = coords_lambda
        ctx.save_for_backward(coords)
        return coords

    @staticmethod
    def backward(ctx, grad_output):
        coords_lambda = ctx.in1
        coords, = ctx.saved_tensors
        grad_output[coords < -1.001] += -coords_lambda * 10
        grad_output[coords > 1.001] += coords_lambda * 10
        # print(f'coords shape: {coords.shape}')
        # print(f'grad_output shape: {grad_output.shape}')
        # print(f'grad sum for OOB locations: {grad_output[coords<-1.5].sum()}')
        # print(f'OOB location num: {(coords<-1.5).sum()}')

        return grad_output, None

class Q_SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        window_size: int = 7,
        device=None,
    ) -> None:
        super().__init__()
        self.dim=dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.window_size = window_size
        self.window_num=1
        self.coords_lambda=5e-1
        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 添加窗口变换模块，用于学习窗口变换参数
        self.transform = nn.Sequential(
            nn.AvgPool2d(kernel_size=window_size, stride=window_size), 
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(dim, self.num_heads*9, kernel_size=1, stride=1)
        )
        
    def apply_rope(self, x: Tensor, rope_sincos: Tuple[Tensor, Tensor], prefix: int = 0) -> Tensor:

        x_pre=x[:, :, :prefix,:]
        x=x[:, :, prefix:,:]
        sin, cos = rope_sincos
        x_dtype = x.dtype
        rope_dtype = sin.dtype

        x = x.to(dtype=rope_dtype)
        sin = sin.to(dtype=rope_dtype)
        cos = cos.to(dtype=rope_dtype)

        x_rot = (x * cos) + (rope_rotate_half(x) * sin)
        x_rot=torch.cat([x_pre, x_rot], dim=-2)
        return x_rot.to(dtype=x_dtype)
    
    
    def compute_attention(self, x: Tensor, attn_bias=None, rope=None, h: int = None, w: int = None) -> Tuple[Tensor, Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x)
        prefix=N-h*w
        qkv_pre=qkv[:, :prefix, :] # B, prefix, 3*C
        qkv_img=qkv[:, prefix:, :] # B, h*w, 3*C
        x_pre=x[:, :prefix, :]
        x_img=x[:,prefix:,:]
        shortcut=x_img.reshape(B, h, w, C).permute(0, 3, 1, 2)
        # B, 3*C, H, W
        qkv_shortcut = qkv_img.reshape(B, h, w, 3*C).permute(0, 3, 1, 2)
        ws = self.window_size
        padding_t = 0
        padding_d = (ws - h % ws) % ws
        padding_l = 0
        padding_r = (ws - w % ws) % ws
        expand_h, expand_w = h+padding_t+padding_d, w+padding_l+padding_r
        # 窗口数目
        window_num_h = expand_h // ws
        window_num_w = expand_w // ws
        assert expand_h % ws == 0
        assert expand_w % ws == 0
        # 创建一个标准化的坐标系统，其中图像的坐标范围被映射到[-1, 1]，适配模型对坐标的要求
        image_reference_h = torch.linspace(-1, 1, expand_h).to(x.device)
        image_reference_w = torch.linspace(-1, 1, expand_w).to(x.device)
        image_reference = torch.stack(torch.meshgrid(image_reference_w, image_reference_h), 0).permute(0, 2, 1).unsqueeze(0) # 2, h, w
        # 用平均池化求窗口中心
        window_reference = nn.functional.avg_pool2d(image_reference, kernel_size=ws)
        # 把全局坐标拆分成窗口坐标 (batch, 坐标通道(x/y), 高度方向窗口数, 窗口高度, 宽度方向窗口数, 窗口宽度)
        # 核心目的：后续可快速索引每个窗口内的所有像素坐标（如第 i 个高度窗口、第 j 个宽度窗口的坐标：
        # image_reference[0, :, i, :, j, :]）
        image_reference = image_reference.reshape(1, 2, window_num_h, ws, window_num_w, ws)
        window_center_coords = window_reference.reshape(1, 2, window_num_h, 1, window_num_w, 1)

        base_coords_h = torch.arange(ws).to(x.device) * 2 / (expand_h-1)
        base_coords_h = (base_coords_h - base_coords_h.mean())
        base_coords_w = torch.arange(ws).to(x.device) * 2 / (expand_w-1)
        base_coords_w = (base_coords_w - base_coords_w.mean())


        expanded_base_coords_h = base_coords_h.unsqueeze(dim=0).repeat(window_num_h, 1)
        assert expanded_base_coords_h.shape[0] == window_num_h
        assert expanded_base_coords_h.shape[1] == ws
        expanded_base_coords_w = base_coords_w.unsqueeze(dim=0).repeat(window_num_w, 1)
        assert expanded_base_coords_w.shape[0] == window_num_w
        assert expanded_base_coords_w.shape[1] == ws
        expanded_base_coords_h = expanded_base_coords_h.reshape(-1)
        expanded_base_coords_w = expanded_base_coords_w.reshape(-1)
        window_coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0).permute(0, 2, 1).reshape(1, 2, window_num_h, ws, window_num_w, ws).permute(0, 2, 4, 1, 3, 5)
        # base_coords = image_reference

        qkv = qkv_shortcut
        qkv = torch.nn.functional.pad(qkv, (padding_l, padding_r, padding_t, padding_d))
        # 3, num_heads//window_num, dim//num_heads, hh, ww
        qkv = rearrange(qkv, 'b (num h dim) hh ww -> num (b h) dim hh ww', h=self.num_heads//self.window_num, num=3, dim=self.dim//self.num_heads, b=B, hh=expand_h, ww=expand_w)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        if h > ws or w > ws:
            # getting the learned params for the varied windows and the coordinates of each pixel
            x = torch.nn.functional.pad(shortcut, (padding_l, padding_r, padding_t, padding_d))
            # self.transform目的是为每个注意力头的每个窗口学习9个变换参数，用于构建可变形的四边形窗口
            sampling_ = self.transform(x).reshape(B*self.num_heads//self.window_num, 9, window_num_h, window_num_w).permute(0, 2, 3, 1)
            sampling_offsets = sampling_[..., :2,]
            sampling_offsets[..., 0] = sampling_offsets[..., 0] / (expand_w // ws)
            sampling_offsets[..., 1] = sampling_offsets[..., 1] / (expand_h // ws)
            # sampling_offsets = sampling_offsets.permute(0, 3, 1, 2)
            sampling_offsets = sampling_offsets.reshape(-1, window_num_h, window_num_w, 2, 1)
            sampling_scales = sampling_[..., 2:4] + 1
            sampling_shear = sampling_[..., 4:6]
            sampling_projc = sampling_[..., 6:8]
            sampling_rotation = sampling_[..., -1]
            zero_vector = torch.zeros(B*self.num_heads//self.window_num, window_num_h, window_num_w).cuda()
            sampling_projc = torch.cat([
                sampling_projc.reshape(-1, window_num_h, window_num_w, 1, 2),
                torch.ones_like(zero_vector).cuda().reshape(-1, window_num_h, window_num_w, 1, 1)
                ], dim=-1)

            shear_matrix = torch.stack([
                torch.ones_like(zero_vector).cuda(),
                sampling_shear[..., 0],
                sampling_shear[..., 1],
                torch.ones_like(zero_vector).cuda()], dim=-1).reshape(-1, window_num_h, window_num_w, 2, 2)
            scales_matrix = torch.stack([
                sampling_scales[..., 0],
                torch.zeros_like(zero_vector).cuda(),
                torch.zeros_like(zero_vector).cuda(),
                sampling_scales[..., 1],
            ], dim=-1).reshape(-1, window_num_h, window_num_w, 2, 2)
            rotation_matrix = torch.stack([
                sampling_rotation.cos(),
                sampling_rotation.sin(),
                -sampling_rotation.sin(),
                sampling_rotation.cos()
            ], dim=-1).reshape(-1, window_num_h, window_num_w, 2, 2)
            basic_transform_matrix = rotation_matrix @ shear_matrix @ scales_matrix
            affine_matrix = torch.cat(
                (torch.cat((basic_transform_matrix, sampling_offsets), dim=-1), sampling_projc), dim=-2)
            window_coords_pers = torch.cat([
                window_coords.flatten(-2, -1), torch.ones(1, window_num_h, window_num_w, 1, ws*ws).cuda()
            ], dim=-2)
            transform_window_coords = affine_matrix @ window_coords_pers
            # transform_window_coords = rotation_matrix @ shear_matrix @ scales_matrix @ window_coords.flatten(-2, -1)
            _transform_window_coords3 = transform_window_coords[..., -1, :]
            _transform_window_coords3[_transform_window_coords3==0] = 1e-6
            transform_window_coords = transform_window_coords[..., :2, :] / _transform_window_coords3.unsqueeze(dim=-2)
            # _transform_window_coords0 = transform_window_coords[..., 0, :] / _transform_window_coords3
            # _transform_window_coords1 = transform_window_coords[..., 1, :] / _transform_window_coords3
            # transform_window_coords = torch.stack((_transform_window_coords0, _transform_window_coords1), dim=-2)
            # transform_window_coords = transform_window_coords[..., :2, :]
            transform_window_coords_distance = transform_window_coords.reshape(-1, window_num_h, window_num_w, 2, ws*ws, 1)
            transform_window_coords_distance = transform_window_coords_distance - window_coords.reshape(-1, window_num_h, window_num_w, 2, 1, ws*ws)
            transform_window_coords_distance = torch.sqrt((transform_window_coords_distance[..., 0, :, :]*(expand_w-1)/2) ** 2 + (transform_window_coords_distance[..., 1, :, :]*(expand_h-1)/2) ** 2)
            transform_window_coords_distance = rearrange(transform_window_coords_distance, '(b h) hh ww n1 n2 -> (b hh ww) h n1 n2', b=B, h=self.num_heads, hh=window_num_h, ww=window_num_w, n1=ws*ws, n2=ws*ws)
            transform_window_coords = transform_window_coords.reshape(-1, window_num_h, window_num_w, 2, ws, ws).permute(0, 3, 1, 4, 2, 5)
            #TODO: adjust the order of transformation

            coords = window_center_coords.repeat(B*self.num_heads, 1, 1, 1, 1, 1) + transform_window_coords

            # coords = base_coords.repeat(B*self.num_heads//self.window_num, 1, 1, 1, 1, 1) + window_coords * sampling_scales[:, :, :, None, :, None] + sampling_offsets[:, :, :, None, :, None]
            sample_coords = coords.permute(0, 2, 3, 4, 5, 1).reshape(B*self.num_heads, ws*window_num_h, ws*window_num_w, 2)
            sample_coords = RectifyCoordsGradient.apply(sample_coords, self.coords_lambda)

            k_selected = F.grid_sample(k, grid=sample_coords, padding_mode='zeros', align_corners=True)
            v_selected = F.grid_sample(v, grid=sample_coords, padding_mode='zeros', align_corners=True)


            q = rearrange(q, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=B, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)
            # k = k_selected.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.attn_ws, window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(B*window_num_h*window_num_w, self.num_heads, self.attn_ws*self.attn_ws, self.dim//self.num_heads)
            k = rearrange(k_selected, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=B, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)
            # v = v_selected.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.attn_ws, window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(B*window_num_h*window_num_w, self.num_heads, self.attn_ws*self.attn_ws, self.dim//self.num_heads)
            v = rearrange(v_selected, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=B, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)

        else:
            transform_window_coords_distance = None
            q = rearrange(q, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=B, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)
            # k = k_selected.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.attn_ws, window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(B*window_num_h*window_num_w, self.num_heads, self.attn_ws*self.attn_ws, self.dim//self.num_heads)
            k = rearrange(k, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=B, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)
            # v = v_selected.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.attn_ws, window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(B*window_num_h*window_num_w, self.num_heads, self.attn_ws*self.attn_ws, self.dim//self.num_heads)
            v = rearrange(v, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=B, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)

            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = rearrange(x, '(b hh ww) h (ws1 ws2) dim -> b (h dim) (hh ws1) (ww ws2)', h=self.num_heads, b=B, hh=window_num_h, ww=window_num_w, ws1=ws, ws2=ws)
            
            # 移除padding
            if padding_t + padding_d + padding_l + padding_r > 0:
                x = x[:, :, padding_t:height+padding_t, padding_l:width+padding_l]
            
            x = x.reshape(B, C, -1).permute(0, 2, 1)
            return x
        # ========== RoPE内部处理（修正后） ==========
        rope_q = None
        rope_k = None
        if rope is not None:
            sin_expanded, cos_expanded = rope  # [b, HW, D]
            D = sin_expanded.shape[-1]
            head_dim = self.dim // self.num_heads
            # assert head_dim == 2 * D, f"RoPE维度D={D}必须是head_dim={head_dim}的一半"
            device = x_img.device  # 用图像部分的device，避免x被覆盖影响
            
            # 1. RoPE还原为2D + 补边（保持原逻辑）
            sin_2d = sin_expanded.reshape(B, h, w, D)
            cos_2d = cos_expanded.reshape(B, h, w, D)
            sin_padded = torch.nn.functional.pad(
                sin_2d, (0, 0, padding_l, padding_r, padding_t, padding_d), mode='replicate'
            ).permute(0, 3, 1, 2)  # [b, D, expand_h, expand_w]
            cos_padded = torch.nn.functional.pad(
                cos_2d, (0, 0, padding_l, padding_r, padding_t, padding_d), mode='replicate'
            ).permute(0, 3, 1, 2)
            
            # 2. Q的RoPE处理（保持原逻辑）
            sin_q = sin_padded.permute(0, 2, 3, 1).reshape(B, window_num_h, ws, window_num_w, ws, D).permute(0, 1, 3, 2, 4, 5)
            cos_q = cos_padded.permute(0, 2, 3, 1).reshape(B, window_num_h, ws, window_num_w, ws, D).permute(0, 1, 3, 2, 4, 5)
            sin_q = sin_q.reshape(-1, ws*ws, D).unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # [(b×hh×ww), num_heads, ws², D]
            cos_q = cos_q.reshape(-1, ws*ws, D).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            rope_q = (sin_q, cos_q)
            
            # 3. K的RoPE处理（修正：补充小图像逻辑）
            if h > ws or w > ws:
                # 大图像：双线性插值（保持原逻辑）
                B_k = sample_coords.shape[0]
                sin_k = sin_padded.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1).reshape(B_k, D, expand_h, expand_w)
                cos_k = cos_padded.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1).reshape(B_k, D, expand_h, expand_w)
                
                sin_k_selected = F.grid_sample(sin_k, sample_coords, padding_mode='zeros', align_corners=True).permute(0, 2, 3, 1)
                cos_k_selected = F.grid_sample(cos_k, sample_coords, padding_mode='zeros', align_corners=True).permute(0, 2, 3, 1)
                
                sin_k_selected = sin_k_selected.reshape(B, self.num_heads, window_num_h, ws, window_num_w, ws, D).permute(0, 2, 4, 1, 3, 5, 6).reshape(-1, self.num_heads, ws*ws, D)
                cos_k_selected = cos_k_selected.reshape(B, self.num_heads, window_num_h, ws, window_num_w, ws, D).permute(0, 2, 4, 1, 3, 5, 6).reshape(-1, self.num_heads, ws*ws, D)
            else:
                # 小图像：K的RoPE与Q完全一致（修正核心）
                sin_k_selected = sin_q.clone()
                cos_k_selected = cos_q.clone()
            rope_k = (sin_k_selected, cos_k_selected)
        qkv_pre = rearrange(
            qkv_pre, 
            'b p (num_qkv h dim) -> num_qkv b h p dim',  # 通道维度拆分为3×num_heads×head_dim
            num_qkv=3,  # 对应Q/K/V
            h=self.num_heads,  # 多头数
            dim=self.head_dim  # 每个头的维度（self.dim//self.num_heads）
        )
        # q,k,v: B*num_windows, num_heads, prefix_len+ws², head_dim
        q_pre, k_pre, v_pre = qkv_pre.unbind(0)
        num_windows = window_num_h * window_num_w
        q_pre_repeat = q_pre.repeat(num_windows, 1, 1, 1)  # [b*num_windows, h, p, d]
        k_pre_repeat = k_pre.repeat(num_windows, 1, 1, 1)  # [b*num_windows, h, p, d]
        v_pre_repeat = v_pre.repeat(num_windows, 1, 1, 1)  # [b*num_windows, h, p, d]
        q = torch.cat([q_pre_repeat, q], dim=-2)  # [B×num_windows, num_heads, prefix_len+ws², head_dim]
        k = torch.cat([k_pre_repeat, k], dim=-2)
        v = torch.cat([v_pre_repeat, v], dim=-2)
        q=self.apply_rope(q,rope_q,prefix)
        k=self.apply_rope(k,rope_k,prefix)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x_pre=x[:, :, :prefix, :] # B*window_num_h*window_num_w, num_heads, prefix_len, head_dim
        x_img=x[:, :, prefix:, :]
        x_img = rearrange(x_img, '(b hh ww) h (ws1 ws2) dim -> b (h dim) (hh ws1) (ww ws2)', h=self.num_heads//self.window_num, b=B, hh=window_num_h, ww=window_num_w, ws1=ws, ws2=ws)
        # x_pre = rearrange(x_pre, '(b hh ww) h prefix dim -> b (h dim) prefix (hh ww)', h=self.num_heads//self.window_num, b=B, hh=window_num_h, ww=window_num_w, prefix=prefix)
        x_pre = x_pre.reshape(B, window_num_h*window_num_w, self.num_heads//self.window_num, prefix, -1)
        # TODO: mean
        x_pre = x_pre.mean(dim=1)
        x_pre = rearrange(x_pre, 'b h prefix dim -> b (h dim) prefix', h=self.num_heads//self.window_num, b=B, prefix=prefix).transpose(1, 2)
        if padding_t + padding_d + padding_l + padding_r > 0:
            x_img = x_img[:, :, padding_t:h+padding_t, padding_l:w+padding_l]
        x_img = x_img.reshape(B, C, N-prefix).transpose(1, 2)  # [B, N, C]
        x=torch.cat([x_pre, x_img], dim=-2)
        return x


    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None, height: int = None, width: int = None, output_attentions=False) -> Tuple[Tensor, Tensor]:
        attn_v = self.compute_attention(x=x, attn_bias=attn_bias, rope=rope, h=height, w=width)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        outputs = (x,)  # TODO: no output_attentions
        return outputs



class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=nn.GELU, drop=0.,
                bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class WindowedDinov3WithRegistersLayer(nn.Module):
    """Simplified version for basic forward testing."""

    def __init__(self) -> None:
        super().__init__()
        # Fixed parameters for simple testing
        self.hidden_size = 384
        self.num_heads = 8
        self.num_register_tokens = 0  # Fixed value instead of getting from config
        
        # Simplified layers with fixed parameters
        self.norm1 = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.attn = Q_SelfAttention(
            self.hidden_size,
            num_heads=self.num_heads,
            qkv_bias=True,
            proj_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
            mask_k_bias=False,
        )

        self.ls1 = LayerScale(self.hidden_size)
        self.rope_embed = RopePositionEmbedding(
            embed_dim=384,
            num_heads=8,
            base=100,
        )
        self.norm2 = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.mlp = Mlp(
            self.hidden_size,
            hidden_features=self.hidden_size * 4,
            act_layer=nn.GELU,
            drop=0.0,
            bias=True,
        )
        self.ls2 = LayerScale(self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        height: Optional[int] = None,
        width: Optional[int] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        """
        Simplified forward pass for basic testing.
        
        Args:
            hidden_states: Input tensor of shape [B, N, C]
            height: Height of the input feature map
            width: Width of the input feature map
            head_mask: Not used in this simplified version
            output_attentions: Not used in this simplified version
            
        Returns:
            Output tensor of the same shape as input
        """
        # Store shortcut for residual connection
        shortcut = hidden_states
        
        # Self-attention block
        # Apply layer norm before attention
        normed_hidden_states = self.norm1(hidden_states)
        
        # Calculate H and W from the sequence length if not provided
        B, N, C = hidden_states.shape
        if height is None or width is None:
            # Assuming square arrangement if not provided
            height = width = int(N**0.5)
        
        # Generate RoPE embeddings based on H and W
        sin, cos = self.rope_embed(H=height, W=width)
        # Expand RoPE embeddings to match the expected shape [bs, 1, HW, D]
        B = hidden_states.shape[0]
        sin_expanded = sin.unsqueeze(0).repeat(B, 1, 1)  # [bs, HW, D]
        cos_expanded = cos.unsqueeze(0).repeat(B, 1, 1)  # [bs, HW, D]
        rope_sincos = (sin_expanded, cos_expanded)

        # Attention with RoPE and window coordinates
        self_attention_outputs = self.attn(
            normed_hidden_states,
            rope=rope_sincos,
            height=height,
            width=width,
            output_attentions=False,
        )
        attention_output = self_attention_outputs[0]
        # Apply LayerScale and first residual connection
        attention_output = self.ls1(attention_output)
        hidden_states = attention_output + shortcut

        # MLP block
        # Apply layer norm before MLP
        normed_hidden_states = self.norm2(hidden_states)
        
        # Apply MLP
        mlp_output = self.mlp(normed_hidden_states)
        
        # Apply LayerScale and second residual connection
        mlp_output = self.ls2(mlp_output)
        output = mlp_output + hidden_states
        
        return output



# 测试代码
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建一个12层的模型
    num_layers = 12
    layers = nn.ModuleList([
        WindowedDinov3WithRegistersLayer().to(device) for _ in range(num_layers)
    ])
    layers.to(device)

    # 创建测试输入张量 [B, N, C] = [1, 400, 384] (20*20=400)
    batch_size = 3
    height = 20
    width = 20
    channels = 384
    seq_length = height * width
    patch_tokens = torch.randn(batch_size, seq_length, channels).to(device)
    class_token = torch.randn(batch_size, 1, channels).to(device)
    input_tensor = torch.cat([class_token, patch_tokens], dim=1)
    
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # 运行100次前向传播测试以计算平均时间
    import time
    warmup_runs = 10  # 预热运行次数
    test_runs = 100   # 实际测试次数
    
    print(f"Running {warmup_runs} warmup iterations...")
    # 预热运行
    for _ in range(warmup_runs):
        x = input_tensor.clone()
        for layer in layers:
            x = layer(x, height=height, width=width)
    
    print(f"Running {test_runs} test iterations...")
    # 实际测试运行
    start_time = time.time()
    for run in range(test_runs):
        x = input_tensor.clone()
        for layer in layers:
            x = layer(x, height=height, width=width)
        if (run + 1) % 20 == 0:  # 每20次迭代打印一次进度
            print(f"Completed {run + 1}/{test_runs} iterations")
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_forward = total_time / test_runs
    avg_time_per_layer = avg_time_per_forward / num_layers
    
    print(f"Total time for {test_runs} forward passes: {total_time:.4f} seconds")
    print(f"Average time per forward pass: {avg_time_per_forward*1000:.4f} ms")
    print(f"Average time per layer: {avg_time_per_layer*1000:.4f} ms")
    
    print("测试完成!")

    batch_size = 3
    height = 320
    width = 320
    channels = 3
    seq_length = height * width
    x = torch.randn(batch_size, seq_length, channels).to(device)
    model = RFDETRMediumV3(position_embedding='sine')
    core_model=model.model.model.to(device)
    dinov3_backbone = core_model.backbone[0].encoder
