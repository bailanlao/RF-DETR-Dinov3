# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from dinov3.utils import cat_keep_shapes, uncat_with_shapes
from torch import Tensor, nn
from timm.models.vision_transformer import Mlp as MLP
from timm.models.layers import trunc_normal_

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


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        device=None,
        use_fdam: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_fdam = use_fdam
        # if use_fdam:
        #     self.dy_freq = nn.Linear(dim, self.num_heads, bias=True)
        #     self.hf_gamma= nn.Parameter(1e-5 * torch.ones((dim)),requires_grad=True)
        #     self.dy_freq_2 = nn.Linear(dim, self.num_heads, bias=True)
        #     self.lf_gamma= nn.Parameter(1e-5 * torch.ones((dim)),requires_grad=True)
        # if use_fdam:
        #     self.star_relu = StarReLU()
        # else:
        #     self.star_relu=nn.Identity()
        
    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        sin = sin.unsqueeze(1)  # [bs, 1, HW, D]
        cos = cos.unsqueeze(1)  # [bs, 1, HW, D]
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype) # B,num_heads,N,head_dim
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k
    

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None, output_attentions=False) -> Tuple[Tensor, Tensor]:
        attn_v = self.compute_attention(x=x, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        outputs = (x,)  # TODO: no output_attentions
        return outputs

    def compute_attention(self, x: Tensor, attn_bias=None, rope=None) -> Tuple[Tensor, Tensor]:
        assert attn_bias is None
        B, N, C = x.shape
        prefix = N - rope[0].shape[-2] # ignore the prefix
        # if self.use_fdam:
        #     x_pre=x[:, :prefix]
        #     dy_freq_feat=self.star_relu(x[:,prefix:])
        #     dy_freq_lf = self.dy_freq_2(dy_freq_feat).tanh_() # B,N,num_heads
        #     dy_freq_lf = dy_freq_lf.reshape(B, N - prefix, self.num_heads, 1).repeat(1, 1, 1, C // self.num_heads)
        #     dy_freq_lf = dy_freq_lf.reshape(B, N - prefix, C)
        #
        #     dy_freq = F.softplus(self.dy_freq(dy_freq_feat))
        #     dy_freq2 = dy_freq ** 2
        #     dy_freq = 2 * dy_freq2 / (dy_freq2 + 0.3678)
        #     dy_freq = dy_freq.reshape(B, N - prefix, self.num_heads, 1).repeat(1, 1, 1, C // self.num_heads)
        #     dy_freq = dy_freq.reshape(B, N - prefix, C)
        #     if prefix>0: # zero to ignore prefix
        #         dy_freq = torch.cat([torch.zeros([B, prefix, C], device=dy_freq.device), dy_freq], dim=1)
        #     x = torch.cat((x_pre, dy_freq_feat), dim=-2)

        qkv = self.qkv(x) # B,N,3C
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # B,N,3,num_heads,head_dim
        q, k, v = torch.unbind(qkv, 2) # B,N,num_heads,head_dim
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]] # B,num_heads,N,head_dim
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        # if self.use_fdam:
        #     v = v.transpose(1, 2).reshape(B, N, C)
        #     v_hf = (v - x)
        #     x_patch = x[:, prefix:]
        #     x_pre = x[:, :prefix]
        #     x_patch = x_patch + x_patch * dy_freq_lf * self.lf_gamma.view(1, 1, -1)
        #     x = torch.cat([x_pre, x_patch], dim=1)
        #     x = x + dy_freq * v_hf * self.hf_gamma.view(1, 1, -1)
        return x

class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
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

class GroupDynamicScale(nn.Module):
    def __init__(self, dim, expansion_ratio=1, reweight_expansion_ratio=.125,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14, weight_resize=True, group=32, init_scale=1e-5,
                 **kwargs):
        super().__init__()
        
        self.size = size
        self.filter_size = size // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize
        # self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        # self.act1 = act1_layer()
        self.reweight = Mlp(dim, reweight_expansion_ratio, group * num_filters, bias=False)
        self.complex_weights = nn.Parameter(
            torch.randn(num_filters, dim//group, self.size, self.filter_size,dtype=torch.float32) * init_scale)
        trunc_normal_(self.complex_weights, std=init_scale)
        self.act2 = act2_layer()
        # self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias) 
        # self.init_reweight_bias(group, num_filters)

    def init_reweight_bias(self, group, num_filters):
            # 创建一个 (group, num_filters) 的矩阵，对角线部分为单位矩阵，其余为 0
            bias_matrix = torch.zeros(group, num_filters)
            min_dim = min(group, num_filters)
            for i in range(min_dim):
                bias_matrix[i][i] = 1.0
            
            # 展开为一维向量
            bias_vector = bias_matrix.view(-1)
            bias_vector = bias_vector.repeat(group * num_filters // len(bias_vector))
            
            # 设置 fc2 的 bias
            self.reweight.fc2.bias.data = bias_vector

    def forward(self, x):
        B, C, H, W, = x.shape
        x_rfft = torch.fft.rfft2(x.to(torch.float32), dim=(2, 3), norm='ortho')
        B, C, RH, RW, = x_rfft.shape
        x = x.permute(0, 2, 3, 1)

        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, -1, self.num_filters).tanh_() # b, num_filters, group

        weight = self.complex_weights
        if not weight.shape[2:4] == x_rfft.shape[2:4]:
            weight = F.interpolate(weight, size=x_rfft.shape[2:4], mode='bicubic', align_corners=True)

        weight = torch.einsum('bgf,fchw->bgchw', routeing, weight)
        weight = weight.reshape(B, C, RH, RW)
        x_rfft = torch.view_as_complex(torch.stack([x_rfft.real * weight, x_rfft.imag * weight], dim=-1))
        x = torch.fft.irfft2(x_rfft, s=(H, W), dim=(2, 3), norm='ortho')
        return x


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(
        self, init_attn_std: float | None = None, init_proj_std: float | None = None, factor: float = 1.0
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        nn.init.normal_(self.qkv.weight, std=init_attn_std)
        nn.init.normal_(self.proj.weight, std=init_proj_std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x
