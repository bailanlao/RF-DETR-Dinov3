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
    """LSNet中的大核感知模块（Large-Kernel Perception）"""
    def __init__(self, dim, lks=7, sks=3, groups=8, use_sync_bn=True):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2, use_sync_bn=use_sync_bn)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(
            dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2, use_sync_bn=use_sync_bn
        )  # 大核深度可分离卷积
        self.cv3 = Conv2d_BN(dim // 2, dim // 2, use_sync_bn=use_sync_bn)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)  # 生成SKA的动态权重
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)

        self.sks = sks
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        # 大核感知流程：通道降维 → 大核DW → 通道映射 → 生成动态权重 → GroupNorm
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, wd = w.size()
        return w.view(b, self.dim // self.groups, self.sks ** 2, h, wd)  # 权重形状：(B, G, Ks², H, W)


class SKA(nn.Module):
    """LSNet中的小核聚合模块（Small-Kernel Aggregation），需与LKP配合使用"""
    def __init__(self):
        super().__init__()

    def forward(self, x, w):
        """
        Args:
            x: 输入特征图，shape=(B, C, H, W)
            w: LKP生成的动态权重，shape=(B, G, Ks², H, W)，G=C//groups
        Returns:
            聚合后的特征图，shape=(B, C, H, W)
        """
        B, C, H, W = x.shape
        G = w.shape[1]
        Ks = int(math.sqrt(w.shape[2]))
        group_ch = C // G  # 每组通道数

        # 1. 通道分组：(B, C, H, W) → (B, G, group_ch, H, W)
        x_grouped = x.view(B, G, group_ch, H, W)
        # 2. 展开小核邻域：每个位置的Ks×Ks邻域展开为向量（使用unfold）
        x_unfold = nn.functional.unfold(
            x_grouped.view(B * G, group_ch, H, W),  # 合并G到B维度，方便unfold
            kernel_size=Ks, padding=(Ks - 1) // 2, stride=1
        ).view(B, G, group_ch, Ks ** 2, H, W)  # 展开后形状：(B, G, group_ch, Ks², H, W)
        # 3. 动态权重聚合：权重与邻域向量点积 → (B, G, group_ch, H, W)
        w = w.unsqueeze(2)  # 权重扩维：(B, G, 1, Ks², H, W)
        x_agg = (x_unfold * w).sum(dim=3)  # 按Ks²维度求和，完成聚合
        # 4. 重组通道：(B, G, group_ch, H, W) → (B, C, H, W)
        return x_agg.view(B, C, H, W)


class LSConv(nn.Module):
    """LSNet中的核心卷积模块（Large-Small Conv），含残差连接"""
    def __init__(self, dim, use_sync_bn=True):
        super().__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8, use_sync_bn=use_sync_bn)  # 固定LSNet参数
        self.ska = SKA()
        self.bn = nn.SyncBatchNorm(dim) if use_sync_bn else nn.BatchNorm2d(dim)

    def forward(self, x):
        # 流程：LKP生成权重 → SKA聚合 → BN → 残差连接
        return self.bn(self.ska(x, self.lkp(x))) + x


class LSBlock(nn.Module):
    """LSNet中的基础块（LS Block）：mixer（LSConv/RepVGGDW）+ SE + 残差FFN"""
    def __init__(self, ed, depth_idx, stage_idx, use_sync_bn=True):
        super().__init__()
        # 1. Mixer选择：偶数depth用RepVGGDW（局部增强），奇数用LSConv（全局-局部融合）
        if depth_idx % 2 == 0:
            self.mixer = RepVGGDW(ed, use_sync_bn=use_sync_bn)
            self.se = SqueezeExcite(ed)  # SE层增强通道注意力
        else:
            self.mixer = LSConv(ed, use_sync_bn=use_sync_bn)  # 核心：LSConv
            self.se = nn.Identity()  # LSConv已含全局信息，暂不额外加SE

        # 2. 残差FFN：通道混合，LSNet中FFN隐藏层维度为2×ed
        self.ffn = Residual(FFN(ed, h=int(ed * 2), use_sync_bn=use_sync_bn))

    def forward(self, x):
        # 流程：mixer → SE → FFN（残差）
        return self.ffn(self.se(self.mixer(x)))


# -------------------------- 改造后的SpatialTuningAdapter（基于LS Block） --------------------------
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
    # 初始化模型
    # model = SpatialTuningAdapter(
    #     in_channels=3, num_out_scales=4, init_channels=64
    # ).eval()  # 切换到评估模式，避免dropout等影响计算量

    model = LSBasedSpatialTuningAdapter(
        in_channels=3, num_out_scales=4, init_channels=64
    ).eval()  # 切换到评估模式，避免dropout等影响计算量
    # 测试输入
    x = torch.randn(1, 3, 320, 320)  # 用batch=1计算FLOPs（更准确）
    
    # 1. 计算参数量（Parameters）
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()  # 累加每个参数的元素数量
    total_params_M = total_params / 1e6  # 转换为百万（M）单位
    
    # 2. 计算计算量（FLOPs，使用thop库）
    # 注意：thop统计的是"乘加操作数"，通常视为1 FLOP = 1次乘加
    flops, _ = profile(model, inputs=(x,), verbose=False)
    total_flops_G = flops / 1e9  # 转换为十亿（G）单位
    
    # 3. 前向传播并打印特征形状
    features = model(x)
    for i, feat in enumerate(features):
        print(f"Stage {i} Feature Shape: {feat.shape} (Stride: {model.stage_strides[i]})")
    
    # 4. 打印参数量和计算量
    print(f"\n模型总参数量: {total_params_M:.2f} M")  # 保留2位小数
    print(f"输入320×320时的总计算量: {total_flops_G:.2f} GFlops")
