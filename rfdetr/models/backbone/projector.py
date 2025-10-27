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
import torch.cuda.amp as amp

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


    def init_weights(self):
        """
        Initialize weights for all layers in the MultiScaleProjector.
        Uses appropriate initialization methods based on layer type and activation function.
        """
        def _init_weights(module):
            """Internal function to handle initialization recursively."""
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                # Use Kaiming initialization for convolutional layers
                nn.init.kaiming_uniform_(module.weight, mode='fan_out')
                gain_relu = math.sqrt(2)
                gain_gelu = math.sqrt(2 / math.pi)
                gelu_ratio = gain_gelu / gain_relu
                with torch.no_grad():
                    module.weight.data *= gelu_ratio
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, LayerNorm):
                # Initialize LayerNorm weights to 1 and biases to 0
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                # Standard BatchNorm initialization
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # Kaiming initialization for linear layers
                nn.init.kaiming_uniform_(module.weight, mode='fan_in')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Apply initialization to all submodules
        self.apply(_init_weights)

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
        
        for i in range(len(features) - 2, -1, -1):
            current_feat = features[i]
            next_feat = features[i + 1]
            
            next_h, next_w = next_feat.shape[2:]
            expected_current_h = next_h * 2
            expected_current_w = next_w * 2
            
            current_h, current_w = current_feat.shape[2:]
            if current_h != expected_current_h or current_w != expected_current_w:
                features[i] = F.interpolate(
                    current_feat,
                    size=(expected_current_h, expected_current_w),
                    mode="bilinear",
                    align_corners=False
                )
        
        return features
    
    def forward_export(self, x):
        features = []
        out = x
        
        for stage in self.stages:
            out = stage(out)
            features.append(out)
        
        for i in range(len(features) - 2, -1, -1):
            current_feat = features[i]
            next_feat = features[i + 1]
            
            next_h, next_w = next_feat.shape[2:]
            expected_current_h = next_h * 2
            expected_current_w = next_w * 2
            
            current_h, current_w = current_feat.shape[2:]
            if current_h != expected_current_h or current_w != expected_current_w:
                features[i] = F.interpolate(
                    current_feat,
                    size=(expected_current_h, expected_current_w),
                    mode="bilinear",
                    align_corners=False
                )
        
        return features


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
        self.vit_channels = context_channels_list
        self.sta_channels = detail_channels_list
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
        self.init_weights()
        
    def init_weights(self):
        for fusion_layer in self.fusion_layers:
            if hasattr(fusion_layer, 'init_weights'):
                fusion_layer.init_weights()
            for m in fusion_layer.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                            
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