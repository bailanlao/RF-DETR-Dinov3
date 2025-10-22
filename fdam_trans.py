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
        model = RFDETRMediumV3(position_embedding='sine',use_fdam=True,freeze_encoder=True)
    core_model=model.model.model.to(device)
    print(core_model)
    initialize_weights(core_model)
    dinov3_backbone = core_model.backbone[0].encoder
    print(dinov3_backbone)
    print_trainable_parameters(dinov3_backbone)
    transfer_dinov3_compatible(dinov3, dinov3_backbone)
    
    def test_encoder_forward(encoder_module, device):
        import traceback
        print("\n" + "="*60)
        print("=== 开始 encoder 前向传播测试（32×32 随机输入） ===")
        print("=== （直接传入普通张量，不使用 NestedTensor） ===")
        print("="*60)

        # 1. 构造普通输入张量（无需 NestedTensor，直接用图像张量）
        batch_size = 1
        input_channels = 3
        input_h = 32  # 16×2，确保能被 patch size=16 整除
        input_w = 32
        dummy_input = torch.randn(batch_size, input_channels, input_h, input_w).to(device)
        print(f"输入张量信息：")
        print(f"  形状：{dummy_input.shape}（batch, channel, h, w）")
        print(f"  设备：{dummy_input.device}")
        print(f"  数值范围：[{dummy_input.min():.4f}, {dummy_input.max():.4f}]")

        # 2. 确保 encoder 所有参数在目标设备上
        encoder_module = encoder_module.to(device)
        first_param_device = next(encoder_module.parameters()).device
        print(f"\nencoder 权重设备：{first_param_device}（应与输入设备一致）")

        # 3. 前向传播（评估模式，关闭梯度）
        encoder_module.eval()
        with torch.no_grad():
            try:
                # 直接传入普通张量（无需 NestedTensor）
                output_features = encoder_module(dummy_input)
                print(f"\n✅ 前向传播成功！")
            except Exception as e:
                print(f"\n❌ 前向传播失败！完整错误信息：")
                print(f"错误类型：{type(e).__name__}")
                print(f"错误描述：{str(e)}")
                print(f"完整调用堆栈：")
                traceback.print_exc()
                return

        # 4. 验证输出特征（根据 encoder 逻辑解析）
        # 4.1 确定输出格式（通常是特征图张量或包含特征的字典）
        if isinstance(output_features, torch.Tensor):
            feat = output_features
        elif isinstance(output_features, (list, tuple)):
            feat = output_features[0]  # 取主特征图
        elif hasattr(output_features, "last_hidden_state"):  # 若输出是类似 transformers 的结构
            feat = output_features.last_hidden_state
        else:
            print(f"❌ 输出格式未知：{type(output_features)}")
            return

        # 4.2 检查输出有效性
        has_nan = torch.isnan(feat).any().item()
        has_inf = torch.isinf(feat).any().item()

        # 4.3 检查特征尺寸合理性（32×32 输入，16×16 patch → 特征尺寸应为 2×2）
        expected_h = input_h // 16  # 32//16=2
        expected_w = input_w // 16  # 32//16=2
        # 特征图通常形状为 [batch, feat_dim, h, w] 或 [batch, seq_len, feat_dim]（seq_len = h×w）
        if len(feat.shape) == 4:  # [B, C, H, W]
            actual_h, actual_w = feat.shape[2], feat.shape[3]
        elif len(feat.shape) == 3:  # [B, seq_len, C] → seq_len = H×W
            seq_len = feat.shape[1]
            actual_h, actual_w = int(seq_len**0.5), int(seq_len**0.5)  # 假设 h=w
        else:
            actual_h, actual_w = -1, -1  # 无法解析

        print(f"\n输出特征验证：")
        print(f"  输出形状：{feat.shape}")
        print(f"  包含 NaN：{'❌ 是' if has_nan else '✅ 否'}")
        print(f"  包含 Inf：{'❌ 是' if has_inf else '✅ 否'}")
        if actual_h != -1:
            print(f"  尺寸合理性：{'✅ 符合' if (actual_h==expected_h and actual_w==expected_w) else f'⚠️ 不符（预期{expected_h}×{expected_w}，实际{actual_h}×{actual_w}）'}")
        else:
            print(f"  尺寸合理性：⚠️ 无法验证（输出形状维度异常）")

        print("\n" + "="*60)
        print("=== encoder 测试完成 ===")
        print("="*60)

    test_encoder_forward(encoder_module=core_model.backbone[0].encoder, device=device)

    # torch.save({
    #     'model': core_model.state_dict(),
    # }, f'medium-dinov3{plus}-randomdecoder-fdam.pth')
