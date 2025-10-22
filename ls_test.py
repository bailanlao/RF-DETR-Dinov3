import torch
import torch.nn as nn
import math
from itertools import product
from torch.nn.init import kaiming_normal_, constant_
from thop import profile  # 导入thop库计算FLOPs

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
    def __init__(self, dim, lks=7, sks=3, use_sync_bn=True):
        super().__init__()
        self.lkp = LKP(dim, lks=lks, sks=sks, groups=8, use_sync_bn=use_sync_bn)
        self.ska = SKA()
        self.bn = nn.SyncBatchNorm(dim) if use_sync_bn else nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x))) + x


class LSBlock(nn.Module):
    def __init__(self, ed, depth_idx, stage_idx,lsk=7, use_sync_bn=True):
        super().__init__()
        if depth_idx % 2 == 0:
            self.mixer = RepVGGDW(ed, use_sync_bn=use_sync_bn)
            self.se = SqueezeExcite(ed)
        else:
            self.mixer = LSConv(ed,lks=lsk, use_sync_bn=use_sync_bn)
            self.se = nn.Identity()
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

        self.patch_embed = nn.Sequential(
            Conv2d_BN(in_channels, init_channels // 2, 3, 2, 1, use_sync_bn=use_sync_bn), 
            nn.ReLU(),
            Conv2d_BN(init_channels // 2, init_channels, 3, 2, 1, use_sync_bn=use_sync_bn), 
            nn.ReLU(),
            Conv2d_BN(init_channels, init_channels * 2, 3, 2, 1, use_sync_bn=use_sync_bn)
        )
        
        target_strides = [8, 16, 32, 64][:num_out_scales]
        self.embed_dims = [init_channels * (2 ** i) for i in range(num_out_scales)]
        self.block_nums_per_stage = [1, 2, 2, 4][:num_out_scales]

        current_channels = init_channels * 2

        for idx in range(num_out_scales):
            curr_embed_dim = self.embed_dims[idx]
            curr_total_stride = target_strides[idx]
            prev_total_stride = self.stage_strides[-1] if idx > 0 else 8
            required_stride = curr_total_stride // prev_total_stride

            stage_layers = []

            if required_stride > 1:
                down_dw = Conv2d_BN(
                    current_channels, current_channels,
                    ks=3, stride=2, pad=1, groups=current_channels,
                    use_sync_bn=use_sync_bn
                )
                down_pw = Conv2d_BN(
                    current_channels, curr_embed_dim,
                    ks=1, stride=1, pad=0, use_sync_bn=use_sync_bn
                )
                stage_layers.extend([down_dw, nn.ReLU(), down_pw, nn.ReLU()])
                current_channels = curr_embed_dim

            for depth_idx in range(self.block_nums_per_stage[idx]):
                lsk_value = 21 if idx < 2 else 7
                stage_layers.append(
                    LSBlock(
                        ed=current_channels,
                        depth_idx=depth_idx,
                        stage_idx=idx,
                        lsk=lsk_value,
                        use_sync_bn=use_sync_bn
                    )
                )

            if stage_layers:
                self.stages.append(nn.Sequential(*stage_layers))
            else:
                self.stages.append(nn.Identity())
            self.stage_channels.append(current_channels)
            self.stage_strides.append(curr_total_stride)

        self.init_weights()
        self._export = False

    def forward(self, x):
        features = []
        out = x.to(self.device) if not self._export else x
        out = self.patch_embed(out)
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


if __name__ == "__main__":
    # model = SpatialTuningAdapter(
    #     in_channels=3, num_out_scales=4, init_channels=64
    # ).eval()  # 切换到评估模式，避免dropout等影响计算量

    model = LSBasedSpatialTuningAdapter(
        in_channels=3, num_out_scales=4, init_channels=64
    ).eval()  # 切换到评估模式，避免dropout等影响计算量
    
    # 使用自定义的模型表示方法
    print("Model Architecture:")
    print(model)
    print("\nFeature shapes:")
    x = torch.randn(1, 3, 320, 320)
    print(f"Input shape: {x.shape}")
    features = model(x)
    for i, feat in enumerate(features):
        print(f"Stage {i} Feature Shape: {feat.shape} (Stride: {model.stage_strides[i]})")
    
    # 统计LSBlock数量
    lsblock_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LSBlock):
            lsblock_count += 1
    
    # 计算每个stage的参数量
    print("\nDetailed Stage Parameter Information:")
    total_params = 0
    
    # 计算每个stage的参数量
    for i, stage in enumerate(model.stages):
        stage_params = sum(p.numel() for p in stage.parameters())
        total_params += stage_params
        print(f"Stage {i} 参数量: {stage_params/1e6:.2f} M")
    
    # 打印总体参数量和计算量
    total_params_M = total_params / 1e6
    try:
        flops, _ = profile(model, inputs=(x,), verbose=False)
        total_flops_G = flops / 1e9
        print(f"\n模型总参数量: {total_params_M:.2f} M")
        print(f"输入320×320时的总计算量: {total_flops_G:.2f} GFlops")
    except Exception as e:
        print(f"\n计算FLOPs时出错: {e}")
        print(f"模型总参数量: {total_params_M:.2f} M")
    
    print(f"LSBlock数量: {lsblock_count}")