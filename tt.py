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

def initialize_weights(model):
    for name, module in model.named_modules():
        if hasattr(module, 'init_weights') and callable(getattr(module, 'init_weights')):
            # print(f"Initializing {name} with init_weights()")
            module.init_weights()
        elif hasattr(module, '_reset_parameters') and callable(getattr(module, '_reset_parameters')):
            module._reset_parameters()
            # print(f"Initializing {name} with _reset_parameters()")
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

def transfer_dinov3_compatible(dinov3, dinov3_backbone,strict=True):
    device = next(dinov3.parameters()).device
    dinov3 = dinov3.to(device).eval()  # 源模型移到目标设备
    dinov3_backbone = dinov3_backbone.to(device).eval()  # 目标backbone移到目标设备
    dinov3_encoder=dinov3_backbone.encoder.encoder
    print(f"✅ 源/目标模型已同步到设备：{device}")
    try:
        src_patch = dinov3.patch_embed
        tgt_patch = dinov3_backbone.encoder.embeddings.patch_embeddings

        src_proj_state = src_patch.proj.state_dict()
        tgt_patch.projection.load_state_dict(src_proj_state, strict=strict)
        print("✅ 卷积投影层（projection）迁移完成")
    except Exception as e:
        raise RuntimeError(f"❌ 卷积投影层迁移失败：{str(e)}") from e
    
    try:
        for i, layer in enumerate(dinov3_encoder.layer):
            v3_block = dinov3.blocks[i]
            layer.norm1.load_state_dict(v3_block.norm1.state_dict(), strict=strict)
            layer.attn.qkv.load_state_dict(v3_block.attn.qkv.state_dict(), strict=strict)
            layer.attn.proj.load_state_dict(v3_block.attn.proj.state_dict(), strict=strict)
            layer.ls1.load_state_dict(v3_block.ls1.state_dict(), strict=strict)
            layer.norm2.load_state_dict(v3_block.norm2.state_dict(), strict=strict)
            layer.ls2.load_state_dict(v3_block.ls2.state_dict(), strict=strict)
            layer.mlp.load_state_dict(v3_block.mlp.state_dict(), strict=strict)
        print("✅ 编码器（encoder）迁移完成")
    except Exception as e:
        raise RuntimeError(f"❌ 编码器迁移失败：{str(e)}") from e
    
    try:
        dinov3_backbone.encoder.layernorm.load_state_dict(dinov3.norm.state_dict(), strict=strict)
        print("✅ 编码器层归一化（layernorm）迁移完成")
    except Exception as e:
        raise RuntimeError(f"❌ 编码器层归一化迁移失败：{str(e)}") from e
    try:
        dinov3_backbone.encoder.rope_embed.load_state_dict(dinov3.rope_embed.state_dict(), strict=strict)
        print("✅ 编码器旋转位置编码（rope_embed）迁移完成")
    except Exception as e:
        raise RuntimeError(f"❌ 编码器旋转位置编码迁移失败：{str(e)}") from e

def print_trainable_parameters(model):
    """打印模型中需要训练的参数信息"""
    print("\n" + "="*50)
    print("模型中需要训练的参数:")
    print("="*50)
    
    trainable_params = []
    total_params = 0
    trainable_params_count = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params.append((name, param.shape))
            trainable_params_count += param.numel()
            print(f"{name}: {param.shape}")
    
    print("="*50)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params_count:,}")
    print(f"可训练参数占比: {100 * trainable_params_count / total_params:.2f}%")
    print("="*50)
    
    return trainable_params

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    plus=''
    dinov3=torch.hub.load(
        'D:/__easyHelper__/dinov3-main', 
        f'dinov3_vits16{plus}', 
        source='local', 
        weights=f'D:/__easyHelper__/dinov3-main/checkpoint/dinov3_vits16{plus}.pth'
    )
    dinov3=dinov3.to(device)
    print(dinov3)
    if plus == 'plus':
        model = RFDETRMediumV3Plus(position_embedding='sine')
    else:
        model = RFDETRMediumV3(position_embedding='sine',use_fdam=True,pretrain_weights="medium-dinov3-randomdecoder-fdam.pth")
    core_model=model.model.model.to(device)
    # print(core_model)
    dinov3_backbone = core_model.backbone[0].encoder
    # print(dinov3_backbone)
    