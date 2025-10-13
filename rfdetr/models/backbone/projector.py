# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from ViTDet (https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Projector
"""
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        """
        LayerNorm forward
        TODO: this is a hack to avoid overflow when using fp16
        """
        #if x.dtype == torch.half:
        #    x = x / (x.max() + self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "LN": lambda channels: LayerNorm(channels),
        }[norm]
    return norm(out_channels)


def get_activation(name, inplace=False):
    """ get activation """
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name in ["LeakyReLU", 'leakyrelu', 'lrelu']:
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name is None:
        module = nn.Identity()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class ConvX(nn.Module):
    """ Conv-bn module"""
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, groups=1, dilation=1, act='relu', layer_norm=False, rms_norm=False):
        super(ConvX, self).__init__()
        if not isinstance(kernel, tuple):
            kernel = (kernel, kernel)
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel,
                              stride=stride, padding=padding, groups=groups,
                              dilation=dilation, bias=False)
        if rms_norm:
            self.bn = nn.RMSNorm(out_planes)
        else:
            self.bn = get_norm('LN', out_planes) if layer_norm else nn.BatchNorm2d(out_planes)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        """ forward """
        out = self.act(self.bn(self.conv(x)))
        return out


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, act='silu', layer_norm=False, rms_norm=False):
        """ ch_in, ch_out, shortcut, groups, kernels, expand """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvX(c1, c_, k[0], 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX(c_, c2, k[1], 1, groups=g, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, act='silu', layer_norm=False, rms_norm=False):
        """ ch_in, ch_out, number, shortcut, groups, expansion """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvX(c1, 2 * self.c, 1, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX((2 + n) * self.c, c2, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, act=act, layer_norm=layer_norm, rms_norm=rms_norm) for _ in range(n))

    def forward(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class MultiScaleProjector(nn.Module):
    """
    This module implements MultiScaleProjector in :paper:`lwdetr`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factors,
        num_blocks=3,
        layer_norm=False,
        rms_norm=False,
        survival_prob=1.0,
        force_drop_last_n_features=0,
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
        """
        super(MultiScaleProjector, self).__init__()

        self.scale_factors = scale_factors
        self.survival_prob = survival_prob
        self.force_drop_last_n_features = force_drop_last_n_features

        stages_sampling = []
        stages = []
        # use_bias = norm == ""
        use_bias = False
        self.use_extra_pool = False
        for scale in scale_factors:
            stages_sampling.append([])
            for in_dim in in_channels:
                out_dim = in_dim
                layers = []

                # if in_dim > 512:
                #     layers.append(ConvX(in_dim, in_dim // 2, kernel=1))
                #     in_dim = in_dim // 2

                if scale == 4.0:
                    layers.extend([
                        nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                        get_norm('LN', in_dim // 2),
                        nn.GELU(),
                        nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
                    ])
                    out_dim = in_dim // 4
                elif scale == 2.0:
                    # a hack to reduce the FLOPs and Params when the dimention of output feature is too large
                    # if in_dim > 512:
                    #     layers = [
                    #         ConvX(in_dim, in_dim // 2, kernel=1),
                    #         nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
                    #     ]
                    #     out_dim = in_dim // 4
                    # else:
                    layers.extend([
                        nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                    ])
                    out_dim = in_dim // 2
                elif scale == 1.0:
                    pass
                elif scale == 0.5:
                    layers.extend([
                        ConvX(in_dim, in_dim, 3, 2, layer_norm=layer_norm),
                    ])
                elif scale == 0.25:
                    self.use_extra_pool = True
                    continue
                else:
                    raise NotImplementedError("Unsupported scale_factor:{}".format(scale))
                layers = nn.Sequential(*layers)
                stages_sampling[-1].append(layers)
            stages_sampling[-1] = nn.ModuleList(stages_sampling[-1])

            in_dim = int(sum(in_channel // max(1, scale) for in_channel in in_channels))
            layers = [
                C2f(in_dim, out_channels, num_blocks, layer_norm=layer_norm),
                get_norm('LN', out_channels),
            ]
            layers = nn.Sequential(*layers)
            stages.append(layers)

        self.stages_sampling = nn.ModuleList(stages_sampling)
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        num_features = len(x)
        print("\n=== MultiScaleProjector 前向调试 ===")
        print(f"输入特征总数: {num_features}")
        for i, feat in enumerate(x):
            print(f"  输入特征{i+1} 形状: {feat.shape} (B={feat.shape[0]}, C={feat.shape[1]}, H={feat.shape[2]}, W={feat.shape[3]})")

        if self.survival_prob < 1.0 and self.training:
            final_drop_prob = 1 - self.survival_prob
            drop_p = np.random.uniform()
            for i in range(1, num_features):
                critical_drop_prob = i * (final_drop_prob / (num_features - 1))
                if drop_p < critical_drop_prob:
                    x[i][:] = 0
        elif self.force_drop_last_n_features > 0:
            for i in range(self.force_drop_last_n_features):
                # don't do it inplace to ensure the compiler can optimize out the backbone layers
                x[-(i+1)] = torch.zeros_like(x[-(i+1)])
                
        results = []
        # x list of len(out_features_indexes)
        for i, stage in enumerate(self.stages):
            feat_fuse = []
            for j, stage_sampling in enumerate(self.stages_sampling[i]):
                feat_fuse.append(stage_sampling(x[j]))
            if len(feat_fuse) > 1:
                feat_fuse = torch.cat(feat_fuse, dim=1)
            else:
                feat_fuse = feat_fuse[0]
            results.append(stage(feat_fuse))

        if self.use_extra_pool:
            results.append(
                F.max_pool2d(results[-1], kernel_size=1, stride=2, padding=0)
            )

        print("\nMultiScaleProjector 输出特征总数: {len(results)}")
        for i, res in enumerate(results):
            print(f"  输出特征{i+1} 形状: {res.shape} (B={res.shape[0]}, C={res.shape[1]}, H={res.shape[2]}, W={res.shape[3]})")

        return results


class SimpleProjector(nn.Module):
    def __init__(self, in_dim, out_dim, factor_kernel=False):
        super(SimpleProjector, self).__init__()
        if not factor_kernel:
            self.convx1 = ConvX(in_dim, in_dim*2, layer_norm=True, act='silu')
            self.convx2 = ConvX(in_dim*2, out_dim, layer_norm=True, act='silu')
        else:
            self.convx1 = ConvX(in_dim, out_dim, kernel=(3, 1), layer_norm=True, act='silu')
            self.convx2 = ConvX(out_dim, out_dim, kernel=(1, 3), layer_norm=True, act='silu')
        self.ln = get_norm('LN', out_dim)

    def forward(self, x):
        """ forward """
        out = self.ln(self.convx2(self.convx1(x[0])))
        return [out]

class SpatialTuningAdapter(nn.Module):
    def __init__(self, in_channels=3, num_out_scales=4, init_channels=64, device=None):
        super().__init__()
        self.stages = nn.ModuleList()
        self.stage_channels = []
        self.stage_strides = []
        current_channels = init_channels
        self.device = device or torch.device("cpu")

        target_strides = [8, 16, 32, 64]

        for idx in range(num_out_scales):
            stride = target_strides[idx]
            prev_stride = self.stage_strides[-1] if idx > 0 else 1
            current_required_stride = stride // prev_stride

            layers = []
            remaining_stride = current_required_stride

            while remaining_stride > 1:
                if remaining_stride >= 2:
                    if len(layers) == 0:
                        if idx == 0:
                            conv_in = in_channels 
                            conv_out = current_channels 
                            current_channels = conv_out
                        else:
                            conv_in = current_channels 
                            conv_out = current_channels * 2 
                            current_channels *= 2
                    else:
                        conv_in = current_channels
                        conv_out = current_channels

                    conv_layer = nn.Conv2d(
                        in_channels=conv_in,
                        out_channels=conv_out,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False
                    )
                    layers.extend([
                        conv_layer,
                        nn.SyncBatchNorm(conv_out),
                        nn.GELU()
                    ])
                    remaining_stride //= 2
                else:
                    break

            self.stages.append(nn.Sequential(*layers))
            self.stage_channels.append(current_channels)
            self.stage_strides.append(stride)
        self.init_weights()
        self._export = False
        
    def init_weights(self):
        gain_relu = math.sqrt(2)
        gain_gelu = math.sqrt(2 / math.pi) 
        gelu_ratio = gain_gelu / gain_relu
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                with torch.no_grad():
                    m.weight.data *= gelu_ratio
                if m.bias is not None:
                    constant_(m.bias, 0.0)
            elif isinstance(m, nn.SyncBatchNorm):
                constant_(m.weight, 1.0)
                constant_(m.bias, 0.0)
    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export

    def forward(self, x):
        features = []
        out = x.to(self.device)
        for stage in self.stages:
            out = stage(out)
            features.append(out)
        return features
    
    def forward_export(self, x):
        features = []
        out = x
        for stage in self.stages:
            out = stage(out)
            features.append(out)
        return features


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, use_sync_bn=True):
                 # a:in_channels, b:out_channels,ks:kernel_size
        super().__init__()
        BN = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm2d
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', BN(b))
        constant_(self.bn.weight, bn_weight_init)
        constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps)**0.5
        m = nn.Conv2d(
            w.size(1) * self.c.groups, w.size(0), w.shape[2:],
            stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
            groups=self.c.groups, device=c.weight.device
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            mask = torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
            return x + self.m(x) * mask
        else:
            return x + self.m(x)


class FFN(nn.Module):
    def __init__(self, ed, h, use_sync_bn=True):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h, use_sync_bn=use_sync_bn)
        self.act = nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0, use_sync_bn=use_sync_bn)

    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))


class SqueezeExcite(nn.Module):
    def __init__(self, dim, rd_ratio=0.25):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, int(dim * rd_ratio), kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(int(dim * rd_ratio), dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class RepVGGDW(nn.Module):
    def __init__(self, ed, use_sync_bn=True):
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, use_sync_bn=use_sync_bn)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed, use_sync_bn=use_sync_bn)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()
        conv1_w = nn.functional.pad(conv1.weight, [1, 1, 1, 1])
        identity_w = nn.functional.pad(torch.ones_like(conv1.weight[:, :, :1, :1]), [1, 1, 1, 1])
        final_w = conv.weight + conv1_w + identity_w
        final_b = conv.bias + conv1.bias
        fused_conv = nn.Conv2d(
            final_w.size(1) * conv.groups, final_w.size(0), final_w.shape[2:],
            stride=conv.stride, padding=conv.padding, groups=conv.groups, device=conv.weight.device
        )
        fused_conv.weight.data.copy_(final_w)
        fused_conv.bias.data.copy_(final_b)
        return fused_conv


class LKP(nn.Module):
    def __init__(self, dim, lks=7, sks=3, groups=8, use_sync_bn=True):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2, use_sync_bn=use_sync_bn)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(
            dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2, use_sync_bn=use_sync_bn
        )
        self.cv3 = Conv2d_BN(dim // 2, dim // 2, use_sync_bn=use_sync_bn)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)

        self.sks = sks
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, wd = w.size()
        return w.view(b, self.dim // self.groups, self.sks ** 2, h, wd) 


class SKA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w):
        B, C, H, W = x.shape
        G = w.shape[1]
        Ks = int(math.sqrt(w.shape[2]))
        group_ch = C // G 

        x_grouped = x.view(B, G, group_ch, H, W)
        x_unfold = nn.functional.unfold(
            x_grouped.view(B * G, group_ch, H, W),
            kernel_size=Ks, padding=(Ks - 1) // 2, stride=1
        ).view(B, G, group_ch, Ks ** 2, H, W) 
        w = w.unsqueeze(2)  
        x_agg = (x_unfold * w).sum(dim=3)
        return x_agg.view(B, C, H, W)


class LSConv(nn.Module):
    def __init__(self, dim, use_sync_bn=True):
        super().__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8, use_sync_bn=use_sync_bn)
        self.ska = SKA()
        self.bn = nn.SyncBatchNorm(dim) if use_sync_bn else nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x))) + x


class LSBlock(nn.Module):
    def __init__(self, ed, depth_idx, stage_idx, use_sync_bn=True):
        super().__init__()
        if depth_idx % 2 == 0:
            self.mixer = RepVGGDW(ed, use_sync_bn=use_sync_bn)
            self.se = SqueezeExcite(ed)  # SE层增强通道注意力
        else:
            self.mixer = LSConv(ed, use_sync_bn=use_sync_bn)  # 核心：LSConv
            self.se = nn.Identity()  # LSConv已含全局信息，暂不额外加SE

        self.ffn = Residual(FFN(ed, h=int(ed * 2), use_sync_bn=use_sync_bn))

    def forward(self, x):
        return self.ffn(self.se(self.mixer(x)))

class LSBasedSpatialTuningAdapter(nn.Module):
    def __init__(self, in_channels=3, num_out_scales=4, init_channels=64, 
                 use_sync_bn=True, device=None):
        super().__init__()
        self.stages = nn.ModuleList()
        self.stage_channels = []
        self.stage_strides = []
        self.use_sync_bn = use_sync_bn
        self.device = device or torch.device("cpu")

        # 1. 目标步长与通道配置（严格对齐文档LSNet-T：[8,16,32,64]步长）
        target_strides = [8, 16, 32, 64][:num_out_scales]
        self.embed_dims = [init_channels * (2 ** i) for i in range(num_out_scales)]  # 通道翻倍
        self.block_nums_per_stage = [0, 2, 8, 10][:num_out_scales]  # LSNet-T的Block数量配置

        # 2. 初始1/2下采样（全局层，符合文档轻量化下采样逻辑）
        self.initial_downsample = nn.Sequential()
        if in_channels > 0:
            initial_dw = Conv2d_BN(
                in_channels, in_channels,
                ks=3, stride=2, pad=1, groups=in_channels,  # 深度可分离，stride=2
                use_sync_bn=use_sync_bn
            )
            initial_pw = Conv2d_BN(
                in_channels, init_channels,
                ks=1, stride=1, pad=0, use_sync_bn=use_sync_bn  # 通道→init_channels
            )
            self.initial_downsample = nn.Sequential(initial_dw, nn.ReLU(), initial_pw, nn.ReLU())
        current_channels = init_channels  # 初始下采样后通道为init_channels

        # 3. 构建每个Stage（核心：循环下采样满足required_stride）
        for idx in range(num_out_scales):
            curr_embed_dim = self.embed_dims[idx]
            curr_total_stride = target_strides[idx]
            # 前一步长：初始下采样步长（2）或上一Stage总步长
            prev_total_stride = self.stage_strides[-1] if idx > 0 else 2
            required_stride = curr_total_stride // prev_total_stride  # 当前Stage需累积的步长

            stage_layers = nn.ModuleList()

            if required_stride > 1:
                current_down_accum = 1  # 记录当前Stage内已累积的下采样步长
                while current_down_accum < required_stride:
                    # 通道策略：首次下采样→curr_embed_dim，后续保持通道不变
                    down_out_ch = curr_embed_dim if current_down_accum == 1 else current_channels
                    # 深度可分离卷积（stride=2，下采样）
                    down_dw = Conv2d_BN(
                        current_channels, current_channels,
                        ks=3, stride=2, pad=1, groups=current_channels,
                        use_sync_bn=use_sync_bn
                    )
                    # 点卷积（调整通道）
                    down_pw = Conv2d_BN(
                        current_channels, down_out_ch,
                        ks=1, stride=1, pad=0, use_sync_bn=use_sync_bn
                    )
                    stage_layers.extend([down_dw, nn.ReLU(), down_pw, nn.ReLU()])
                    # 更新累积步长和当前通道
                    current_down_accum *= 2
                    current_channels = down_out_ch

            for depth_idx in range(self.block_nums_per_stage[idx]):
                stage_layers.append(
                    LSBlock(
                        ed=current_channels,
                        depth_idx=depth_idx,
                        stage_idx=idx,
                        use_sync_bn=use_sync_bn
                    )
                )

            self.stages.append(nn.Sequential(*stage_layers))
            self.stage_channels.append(current_channels)
            self.stage_strides.append(curr_total_stride)

        self.init_weights()
        self._export = False

    def forward(self, x):
        features = []
        out = x.to(self.device) if not self._export else x
        out = self.initial_downsample(out)
        for stage in self.stages:
            out = stage(out)
            features.append(out)
        return features

    def init_weights(self):
        gain_relu = math.sqrt(2)
        gain_gelu = math.sqrt(2 / math.pi)
        gelu_ratio = gain_gelu / gain_relu

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                with torch.no_grad():
                    m.weight.data *= gelu_ratio
                if m.bias is not None:
                    constant_(m.bias, 0.0)
            elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
                constant_(m.weight, 1.0)
                constant_(m.bias, 0.0)
            elif isinstance(m, nn.GroupNorm):
                constant_(m.weight, 1.0)
                constant_(m.bias, 0.0)

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export

    def forward_export(self, x):
        return self.forward(x) 


class BiFusion(nn.Module):
    def __init__(self, context_channels, detail_channels, out_channels):
        super().__init__()
        self.concat_conv = nn.Conv2d(
            context_channels + detail_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.norm = nn.SyncBatchNorm(out_channels)
        self.gelu = nn.GELU()
        self.init_weights()

    def init_weights(self):
        gain_relu = math.sqrt(2)
        gain_gelu = math.sqrt(2 / math.pi)
        gelu_ratio = gain_gelu / gain_relu
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                with torch.no_grad():
                    m.weight.data *= gelu_ratio
                if m.bias is not None:
                    constant_(m.bias, 0.0)
            elif isinstance(m, nn.SyncBatchNorm):
                constant_(m.weight, 1.0)
                constant_(m.bias, 0.0)

    def forward(self, context_scaled, detail):
        combined = torch.cat([context_scaled, detail], dim=1)
        out = self.concat_conv(combined)
        out = self.norm(out)
        out = self.gelu(out)
        return out

class MultiScaleBiFusion(nn.Module):
    def __init__(self, scale_factors, context_channels_list, detail_channels_list, out_channels, enc_stride=16):
        super().__init__()
        self.scale_factors = scale_factors
        self.num_scales = len(scale_factors) 
        self.enc_stride = enc_stride

        self.scale2sta_stride = {2.0:8, 1.0:16, 0.5:32, 0.25:64}
        self.sta_strides = [8,16,32,64]

        self.enc_indices = list(range(len(context_channels_list)))[-self.num_scales:]
        self.sta_indices = []
        sta_indices_rev = list(range(len(self.sta_strides)))[::-1]
        for i, scale in enumerate(scale_factors):
            target_sta_stride = self.scale2sta_stride[scale]
            for sta_idx in sta_indices_rev:
                if self.sta_strides[sta_idx] == target_sta_stride:
                    self.sta_indices.append(sta_idx)
                    break
            else:
                raise ValueError(f"STA scale={scale} no feature with stride {target_sta_stride}")
        self.fusion_layers = nn.ModuleList()
        for enc_idx, sta_idx in zip(self.enc_indices, self.sta_indices):
            ctx_ch = context_channels_list[enc_idx]
            det_ch = detail_channels_list[sta_idx]   
            self.fusion_layers.append(BiFusion(ctx_ch, det_ch, out_channels))
        self._export = False
        
    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export

    def forward(self, context_feats, detail_feats):
        fused_feats = []
        for i in range(self.num_scales):
            scale = self.scale_factors[i]
            enc_idx = self.enc_indices[i]
            sta_idx = self.sta_indices[i]
            vit_feat = context_feats[enc_idx]
            sta_feat = detail_feats[sta_idx]
            vit_h, vit_w = vit_feat.shape[2:]
            target_h, target_w = sta_feat.shape[2:]

            try:
                if isinstance(scale, torch.Tensor):
                    expected_target_h = int(torch.round(vit_feat.shape[2] * scale).item())
                    expected_target_w = int(torch.round(vit_feat.shape[3] * scale).item())
                else:
                    expected_target_h = int(round(vit_feat.shape[2] * scale))
                    expected_target_w = int(round(vit_feat.shape[3] * scale))
            except Exception as e:
                print(f"Warning: Exception in scale calculation: {e}")
                expected_target_h = target_h
                expected_target_w = target_w
            
            if (expected_target_h != target_h) or (expected_target_w != target_w):
                print(f"scale={scale} expected size:({expected_target_h}×{expected_target_w}) target size:({target_h}×{target_w}) use target size")

            vit_scaled = F.interpolate(
                vit_feat,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False
            )

            fused = self.fusion_layers[i](vit_scaled, sta_feat)
            fused_feats.append(fused)

        return fused_feats
    
    def forward_export(self, context_feats, detail_feats):
        fused_feats = []
        for i in range(self.num_scales):
            enc_idx = self.enc_indices[i]
            sta_idx = self.sta_indices[i]
            vit_feat = context_feats[enc_idx]
            sta_feat = detail_feats[sta_idx]
            target_h, target_w = sta_feat.shape[2:]
            vit_scaled = F.interpolate(
                vit_feat,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False
            )
            fused = self.fusion_layers[i](vit_scaled, sta_feat)
            fused_feats.append(fused)
        
        return fused_feats