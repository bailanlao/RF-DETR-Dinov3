# models/feataug.py
import random
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align


def _flip_tensor(t: torch.Tensor) -> torch.Tensor:
    # 支持 BCHW / BHW 两种
    if t.dim() == 4:
        return t.flip(-1)
    elif t.dim() == 3:
        return t.flip(-1)
    return t


def feataug_flip(
    srcs: List[torch.Tensor],
    masks: Optional[List[torch.Tensor]],
    poss: Optional[List[torch.Tensor]],
):
    srcs_f = [_flip_tensor(s) for s in srcs]
    masks_f = [_flip_tensor(m) for m in masks] if masks is not None else None
    poss_f = [_flip_tensor(p) for p in poss] if poss is not None else None
    meta = {"type": "flip"}  # 仅需标注类型，后续按此变换 GT
    return srcs_f, masks_f, poss_f, meta


def _roi_align_level(x: torch.Tensor, boxes_norm: torch.Tensor) -> torch.Tensor:
    """ x: (B,C,H,W); boxes_norm: (B,4) in [0,1] (x1,y1,x2,y2) """
    B, C, H, W = x.shape
    # 构造 ROIAlign 的像素级 boxes: (idx, x1, y1, x2, y2)
    rois = torch.stack([
        torch.arange(B, device=x.device, dtype=torch.float32),
        boxes_norm[:, 0] * W, boxes_norm[:, 1] * H,
        boxes_norm[:, 2] * W, boxes_norm[:, 3] * H
    ], dim=1)
    return roi_align(x, rois, output_size=(H, W), aligned=True)


def feataug_crop(
    srcs: List[torch.Tensor],
    masks: Optional[List[torch.Tensor]],
    poss: Optional[List[torch.Tensor]],
    min_scale: float = 0.6,
    max_scale: float = 1.0,
):

    device = srcs[0].device
    B = srcs[0].shape[0]
    # 为每张图采样一个正方形裁剪框（可以很容易扩展为宽高不同）
    boxes = []
    for _ in range(B):
        s = random.uniform(min_scale, max_scale)   # 切出比例 s*s
        w = h = s
        cx = random.uniform(w / 2, 1 - w / 2)
        cy = random.uniform(h / 2, 1 - h / 2)
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        boxes.append([x1, y1, x2, y2])
    boxes = torch.tensor(boxes, dtype=torch.float32, device=device)  # (B,4)

    srcs_c = [_roi_align_level(s, boxes) for s in srcs]
    masks_c = ([_roi_align_level(m.unsqueeze(1).float(), boxes).squeeze(1).to(torch.bool)
                for m in masks] if masks is not None else None)
    poss_c = ([_roi_align_level(p, boxes) for p in poss] if poss is not None else None)

    meta = {"type": "crop", "boxes": boxes}  # 保存归一化裁剪框，后续变换 GT
    return srcs_c, masks_c, poss_c, meta


def feature_augment(
    srcs: List[torch.Tensor],
    masks: Optional[List[torch.Tensor]],
    poss: Optional[List[torch.Tensor]],
    types: Tuple[str, ...],
    prob: float = 1.0,
    crop_min_scale: float = 0.6,
    crop_max_scale: float = 1.0,
):
    """返回增强后的 (srcs, masks, poss, meta)。若未采样中增强，则返回 None。"""
    if random.random() >= prob or len(types) == 0:
        return None

    t = random.choice(types)
    if t == "flip":
        return feataug_flip(srcs, masks, poss)
    elif t == "crop":
        return feataug_crop(srcs, masks, poss, crop_min_scale, crop_max_scale)
    elif t in ("flip_or_crop", "fc"):
        # 等概率 Flip/Crop
        if random.random() < 0.5:
            return feataug_flip(srcs, masks, poss)
        return feataug_crop(srcs, masks, poss, crop_min_scale, crop_max_scale)
    else:
        raise ValueError(f"Unknown feataug type: {t}")

def feature_augment_multi(
    srcs, masks, poss,
    aug_type: str,                # 'flip' | 'crop' | 'flip_or_crop' | 'fc'
    prob: float = 1.0,
    crop_min_scale: float = 0.6,
    crop_max_scale: float = 1.0,
):
    """返回若干增强分支：List[(srcs_aug, masks_aug, poss_aug, meta)]。
       未采样或不可用则返回 []。"""
    if random.random() >= prob or aug_type == 'none':
        return []

    branches = []
    def _add_flip(): branches.append(feataug_flip(srcs, masks, poss))
    def _add_crop(): branches.append(feataug_crop(srcs, masks, poss, crop_min_scale, crop_max_scale))

    if aug_type == 'flip':
        _add_flip()
    elif aug_type == 'crop':
        _add_crop()
    elif aug_type in ('flip_or_crop',):
        if random.random() < 0.5:
            _add_crop()
        else:
            _add_flip()
    elif aug_type in ('fc',):
        # 两者都做（K=2）
        _add_flip()
        _add_crop()
    else:
        raise ValueError(f'Unknown feataug type: {aug_type}')
    return branches

@torch.no_grad()
def transform_targets_by_meta(targets: list, meta: Dict) -> list:
    """把 GT（cx,cy,w,h 归一化）同步到增强坐标系下."""
    new_targets = []
    for i, t in enumerate(targets):
        nt = {k: v.clone() if torch.is_tensor(v) else v for k, v in t.items()}
        if "boxes" in nt:
            boxes = nt["boxes"].clone()
            if meta["type"] == "flip":
                boxes[:, 0] = 1.0 - boxes[:, 0]           # cx -> 1-cx
                # cy, w, h 不变
            elif meta["type"] == "crop":
                x1, y1, x2, y2 = meta["boxes"][i].tolist()
                sx, sy = (x2 - x1), (y2 - y1)
                boxes[:, 0] = (boxes[:, 0] - x1) / sx
                boxes[:, 1] = (boxes[:, 1] - y1) / sy
                boxes[:, 2] = boxes[:, 2] / sx
                boxes[:, 3] = boxes[:, 3] / sy
                boxes = boxes.clamp(0, 1)
            nt["boxes"] = boxes
        new_targets.append(nt)
    return new_targets

import torch.nn as nn

class FeatAugProjHead(nn.Module):
    """仅用于裁剪特征的域间隙缩小头；每个尺度一套 1x1 Conv + GN + SiLU。"""
    def __init__(self, num_levels: int, channels: int, gn_groups: int = 32):
        super().__init__()
        gn = max(1, min(gn_groups, channels))   # 兜底
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=min(gn, channels), num_channels=channels),
                nn.SiLU(inplace=True),
            )
            for _ in range(num_levels)
        ])
        
    def init_weights(self):
        """初始化权重"""
        for block in self.blocks:
            nn.init.kaiming_normal_(block[0].weight, mode='fan_out', nonlinearity='relu')
            if hasattr(block[1], 'weight'):
                nn.init.constant_(block[1].weight, 1.0)
            if hasattr(block[1], 'bias'):
                nn.init.constant_(block[1].bias, 0.0)

    def forward(self, srcs: list):
        assert len(srcs) == len(self.blocks), f"{len(srcs)} vs {len(self.blocks)}"
        return [blk(x) for blk, x in zip(self.blocks, srcs)]
