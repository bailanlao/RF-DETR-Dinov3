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

def print_instance_attributes(obj, indent=0, max_depth=3):
    
    indent_str = "  " * indent
    print(f"{indent_str}{type(obj).__name__}(")
    
    # 获取对象的属性
    if hasattr(obj, '__dict__'):
        attributes = obj.__dict__
    else:
        attributes = {}
    
    # 处理PyTorch模块的子模块
    if hasattr(obj, 'named_children'):
        for name, child in obj.named_children():
            if indent < max_depth:
                print(f"{indent_str}  ({name}): ", end="")
                print_instance_attributes(child, indent + 1, max_depth)
            else:
                print(f"{indent_str}  ({name}): {type(child).__name__}(...)")
    
    # 打印其他属性
    for name, value in attributes.items():
        # 跳过已经通过named_children显示的子模块
        if hasattr(obj, 'named_children') and name in [n for n, _ in obj.named_children()]:
            continue
        
        # 简单值直接打印，复杂对象只打印类型
        if isinstance(value, (int, float, str, bool, NoneType)):
            print(f"{indent_str}  {name}: {value}")
        else:
            print(f"{indent_str}  {name}: {type(value).__name__}")
    
    print(f"{indent_str})")

class NestedTensor:
    def __init__(self, tensors, mask=None):
        self.tensors = tensors  # 普通图像张量（[B, C, H, W]）
        self.mask = mask        # 注意力掩码（[B, 1, H, W] 或 None）

    def to(self, device):
        # 提供设备迁移方法（backbone 会调用）
        cast_tensor = self.tensors.to(device)
        mask = self.mask.to(device) if self.mask is not None else None
        return NestedTensor(cast_tensor, mask)

def transfer_dinov2_to_core(dinov2, dino_encoder, proj):
    # 1. 迁移卷积投影层（不变，保留中心补0逻辑）
    src_proj_weight = dinov2.patch_embed.proj.weight  # 源：[384, 3, 14, 14]
    src_proj_bias = dinov2.patch_embed.proj.bias      # 源：[384]
    
    tgt_proj_weight = torch.zeros_like(proj.weight)   # 目标：[384, 3, 16, 16]
    offset = (16 - 14) // 2  # 计算补0偏移（16x16中心放14x14的源权重）
    tgt_proj_weight[:, :, offset:offset+14, offset:offset+14] = src_proj_weight
    proj.weight.data = tgt_proj_weight  # 赋值权重
    proj.bias.data = src_proj_bias      # 偏置直接迁移
    print("卷积投影层参数迁移完成")


    # 2. 迁移编码器层（12层一一对应，核心适配 lambda1 参数名）
    src_blocks = dinov2.blocks  # 源模型的12层 NestedTensorBlock
    tgt_layers = dino_encoder.layer  # 目标模型的12层 WindowedDinov2WithRegistersLayer

    for layer_idx in range(12):
        src_block = src_blocks[layer_idx]
        tgt_layer = tgt_layers[layer_idx]

        # 2.1 迁移归一化层（norm1、norm2，参数名完全匹配）
        tgt_layer.norm1.weight.data = src_block.norm1.weight.data.clone()
        tgt_layer.norm1.bias.data = src_block.norm1.bias.data.clone()
        tgt_layer.norm2.weight.data = src_block.norm2.weight.data.clone()
        tgt_layer.norm2.bias.data = src_block.norm2.bias.data.clone()


        # 2.2 迁移注意力层（拆分源模型的 qkv 合并参数）
        # 源：qkv 合并 Linear（384→1152）→ 目标：query/key/value 分离 Linear（各384→384）
        src_qkv_weight = src_block.attn.qkv.weight  # 源：[1152, 384]
        src_qkv_bias = src_block.attn.qkv.bias      # 源：[1152]
        
        # 拆分 qkv 为 3 个独立参数（按输出维度切分）
        q_weight, k_weight, v_weight = src_qkv_weight.chunk(3, dim=0)  # 各[384, 384]
        q_bias, k_bias, v_bias = src_qkv_bias.chunk(3, dim=0)          # 各[384]
        
        # 赋值到目标注意力层的 query/key/value
        tgt_layer.attention.attention.query.weight.data = q_weight.clone()
        tgt_layer.attention.attention.query.bias.data = q_bias.clone()
        tgt_layer.attention.attention.key.weight.data = k_weight.clone()
        tgt_layer.attention.attention.key.bias.data = k_bias.clone()
        tgt_layer.attention.attention.value.weight.data = v_weight.clone()
        tgt_layer.attention.attention.value.bias.data = v_bias.clone()
        
        # 迁移注意力输出投影层（源 proj → 目标 dense）
        tgt_layer.attention.output.dense.weight.data = src_block.attn.proj.weight.data.clone()
        tgt_layer.attention.output.dense.bias.data = src_block.attn.proj.bias.data.clone()


        # 2.3 迁移层缩放参数（关键修改：目标参数名改为 lambda1）
        # 源：ls1.gamma / ls2.gamma → 目标：layer_scale1.lambda1 / layer_scale2.lambda1
        tgt_layer.layer_scale1.lambda1.data = src_block.ls1.gamma.data.clone()
        tgt_layer.layer_scale2.lambda1.data = src_block.ls2.gamma.data.clone()


        # 2.4 迁移 MLP 层（参数名完全匹配，直接克隆）
        tgt_layer.mlp.fc1.weight.data = src_block.mlp.fc1.weight.data.clone()
        tgt_layer.mlp.fc1.bias.data = src_block.mlp.fc1.bias.data.clone()
        tgt_layer.mlp.fc2.weight.data = src_block.mlp.fc2.weight.data.clone()
        tgt_layer.mlp.fc2.bias.data = src_block.mlp.fc2.bias.data.clone()

    print(f"编码器12层参数迁移完成（已适配 lambda1 参数名）")

def initialize_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
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

def verify_transfer(dinov2, dino_encoder, proj):
    # 1. 验证卷积投影层（检查源权重是否已写入目标）
    src_proj_center = dinov2.patch_embed.proj.weight  # 源14x14权重
    tgt_proj_center = proj.weight[:, :, 1:15, 1:15]   # 目标16x16的中心14x14区域
    proj_match = torch.allclose(src_proj_center, tgt_proj_center, atol=1e-6)
    print(f"卷积投影层权重匹配：{'✅ 是' if proj_match else '❌ 否'}")

    # 2. 验证编码器第0层（检查norm1、layer_scale1、mlp.fc1的参数）
    src_block0 = dinov2.blocks[0]
    tgt_layer0 = dino_encoder.layer[0]
    
    # 验证norm1权重
    norm1_match = torch.allclose(src_block0.norm1.weight, tgt_layer0.norm1.weight, atol=1e-6)
    # 验证layer_scale1（lambda1）
    ls1_match = torch.allclose(src_block0.ls1.gamma, tgt_layer0.layer_scale1.lambda1, atol=1e-6)
    # 验证mlp.fc1权重
    mlp_fc1_match = torch.allclose(src_block0.mlp.fc1.weight, tgt_layer0.mlp.fc1.weight, atol=1e-6)
    
    print(f"编码器第0层 norm1 匹配：{'✅ 是' if norm1_match else '❌ 否'}")
    print(f"编码器第0层 layer_scale1 匹配：{'✅ 是' if ls1_match else '❌ 否'}")
    print(f"编码器第0层 MLP.fc1 匹配：{'✅ 是' if mlp_fc1_match else '❌ 否'}")

    # 3. 验证设备（确保所有参数都在cuda上）
    tgt_device = next(dino_encoder.parameters()).device
    proj_device = proj.weight.device
    print(f"编码器设备：{tgt_device}（应显示cuda）")
    print(f"投影层设备：{proj_device}（应显示cuda）")

def transfer_dinov3_compatible(dinov3, dinov3_backbone,strict=True):
    device = next(dinov3.parameters()).device
    dinov3 = dinov3.to(device).eval()  # 源模型移到目标设备
    dinov3_backbone = dinov3_backbone.to(device).eval()  # 目标backbone移到目标设备
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
        src_rope = dinov3.rope_embed
        tgt_rope = dinov3_backbone.encoder.rope_embed
        tgt_rope.load_state_dict(src_rope.state_dict(), strict=strict)
        print("✅ RoPE位置编码（含buffer）迁移完成")
    except Exception as e:
        if not strict:
            print(f"⚠️ RoPE迁移存在非致命差异：{str(e)}，已继续执行（strict=False）")
            tgt_rope.load_state_dict(src_rope.state_dict(), strict=False)
        else:
            raise RuntimeError(f"❌ RoPE位置编码迁移失败（strict=True）：{str(e)}") from e

    try:
        src_blocks = dinov3.blocks  # 源12层（SelfAttentionBlock）
        tgt_layers = dinov3_backbone.encoder.encoder.layer  # 目标12层（WindowedDinov3WithRegistersLayer）
        
        assert len(src_blocks) == len(tgt_layers) == 12, \
            f"❌ 层数量不匹配：源{len(src_blocks)}层，目标{len(tgt_layers)}层（需均为12层）"

        for layer_idx in range(12):
            src_block = src_blocks[layer_idx]  # 源第i层
            tgt_layer = tgt_layers[layer_idx]  # 目标第i层
            print(f"\n=== 开始迁移第{layer_idx}层 ===")

            # 4.1 迁移norm1（LayerNorm）
            try:
                tgt_layer.norm1.load_state_dict(src_block.norm1.state_dict(), strict=strict)
                print(f"  ✅ norm1 迁移完成")
            except Exception as e:
                raise RuntimeError(f"  ❌ 第{layer_idx}层norm1迁移失败：{str(e)}") from e

            # 4.2 迁移注意力层（attn：qkv+proj+dropout）
            try:
                tgt_layer.attn.load_state_dict(src_block.attn.state_dict(), strict=strict)
                print(f"  ✅ 注意力层（attn）迁移完成")
            except Exception as e:
                raise RuntimeError(f"  ❌ 第{layer_idx}层attn迁移失败：{str(e)}") from e

            # 4.3 迁移LayerScale（ls1/ls2，含gamma）
            try:
                tgt_layer.ls1.load_state_dict(src_block.ls1.state_dict(), strict=strict)
                tgt_layer.ls2.load_state_dict(src_block.ls2.state_dict(), strict=strict)
                print(f"  ✅ LayerScale（ls1/ls2）迁移完成")
            except Exception as e:
                raise RuntimeError(f"  ❌ 第{layer_idx}层LayerScale迁移失败：{str(e)}") from e

            # 4.4 迁移norm2（LayerNorm）
            try:
                tgt_layer.norm2.load_state_dict(src_block.norm2.state_dict(), strict=strict)
                print(f"  ✅ norm2 迁移完成")
            except Exception as e:
                raise RuntimeError(f"  ❌ 第{layer_idx}层norm2迁移失败：{str(e)}") from e

            # 4.5 迁移MLP层（兼容SwiGLUFFN/标准MLP）
            try:
                src_mlp = src_block.mlp
                tgt_mlp = tgt_layer.mlp

                # 判断MLP类型（基于属性，避免依赖类名）
                if hasattr(src_mlp, "w1") and hasattr(src_mlp, "w3"):
                    # 类型1：SwiGLUFFN（dinov3_vits16plus）
                    tgt_mlp.load_state_dict(src_mlp.state_dict(), strict=strict)
                    print(f"  ✅ MLP（SwiGLUFFN）迁移完成")
                elif hasattr(src_mlp, "fc1") and hasattr(src_mlp, "fc2"):
                    # 类型2：标准MLP（dinov3_vits16）
                    tgt_mlp.load_state_dict(src_mlp.state_dict(), strict=strict)
                    print(f"  ✅ MLP（标准fc1/fc2）迁移完成")
                else:
                    raise ValueError(f"  ❌ 未知MLP类型：{type(src_mlp)}（需含w1/w3或fc1/fc2）")
            except Exception as e:
                raise RuntimeError(f"  ❌ 第{layer_idx}层MLP迁移失败：{str(e)}") from e

            print(f"=== 第{layer_idx}层迁移完成 ===")

        print("\n✅ 12层编码器全量迁移完成")
    except Exception as e:
        raise RuntimeError(f"❌ 编码器层迁移失败：{str(e)}") from e

    try:
        src_top_norm = dinov3.norm
        tgt_top_norm = dinov3_backbone.encoder.layernorm

        tgt_top_norm.load_state_dict(src_top_norm.state_dict(), strict=strict)
        print("✅ 顶层归一化层（layernorm）迁移完成")
    except Exception as e:
        raise RuntimeError(f"❌ 顶层归一化层迁移失败：{str(e)}") from e

    print("\n" + "="*80)
    print("✅ 全量迁移完成！目标backbone已包含dinov3预训练权重")
    print("="*80)
    return dinov3_backbone


def verify_transfer_compatible(dinov3, dinov3_backbone):
    """
    兼容版验证函数：支持 SwiGLUFFN（vits16plus）和标准MLP（vits16）
    """
    print("\n" + "="*60)
    print("迁移验证结果（兼容版）")
    print("="*60)

    # 1. 验证卷积投影层（不变）
    src_proj = dinov3.patch_embed.proj
    tgt_proj = dinov3_backbone.encoder.embeddings.patch_embeddings.projection
    proj_match = torch.allclose(src_proj.weight, tgt_proj.weight, atol=1e-6)
    print(f"1. 卷积投影层参数匹配：{'✅ 成功' if proj_match else '❌ 失败'}")

    # 2. 验证第0层注意力层（不变）
    src_attn_qkv = dinov3.blocks[0].attn.qkv.weight
    tgt_attn_qkv = dinov3_backbone.encoder.encoder.layer[0].attn.qkv.weight
    attn_match = torch.allclose(src_attn_qkv, tgt_attn_qkv, atol=1e-6)
    print(f"2. 第0层注意力层QKV参数匹配：{'✅ 成功' if attn_match else '❌ 失败'}")

    # 3. 验证第0层LayerScale（不变）
    src_ls1 = dinov3.blocks[0].ls1.gamma
    tgt_ls1 = dinov3_backbone.encoder.encoder.layer[0].ls1.gamma
    ls_match = torch.allclose(src_ls1, tgt_ls1, atol=1e-6)
    print(f"3. 第0层LayerScale参数匹配：{'✅ 成功' if ls_match else '❌ 失败'}")

    # 4. 验证第0层MLP（核心修改：兼容两种类型）
    src_mlp = dinov3.blocks[0].mlp
    tgt_mlp = dinov3_backbone.encoder.encoder.layer[0].mlp
    mlp_match = False

    if hasattr(src_mlp, "w1") and hasattr(tgt_mlp, "w1"):
        # 验证 SwiGLUFFN（vits16plus）的 w1 参数
        src_mlp_w1 = src_mlp.w1.weight
        tgt_mlp_w1 = tgt_mlp.w1.weight
        mlp_match = torch.allclose(src_mlp_w1, tgt_mlp_w1, atol=1e-6)
        print(f"4. 第0层MLP（SwiGLUFFN）w1参数匹配：{'✅ 成功' if mlp_match else '❌ 失败'}")
    elif hasattr(src_mlp, "fc1") and hasattr(tgt_mlp, "fc1"):
        # 验证 标准MLP（vits16）的 fc1 参数
        src_mlp_fc1 = src_mlp.fc1.weight
        tgt_mlp_fc1 = tgt_mlp.fc1.weight
        mlp_match = torch.allclose(src_mlp_fc1, tgt_mlp_fc1, atol=1e-6)
        print(f"4. 第0层MLP（标准）fc1参数匹配：{'✅ 成功' if mlp_match else '❌ 失败'}")
    else:
        print(f"4. 第0层MLP验证失败：未知的MLP类型")

    # 5. 验证顶层归一化（不变）
    src_top_norm = dinov3.norm
    tgt_top_norm = dinov3_backbone.encoder.layernorm
    top_norm_match = torch.allclose(src_top_norm.weight, tgt_top_norm.weight, atol=1e-6)
    print(f"5. 顶层归一化参数匹配：{'✅ 成功' if top_norm_match else '❌ 失败'}")

    print("="*60)
    # 总体验证结果
    all_match = proj_match and attn_match and ls_match and mlp_match and top_norm_match
    print(f"总体迁移验证：{'✅ 全部成功' if all_match else '❌ 部分失败'}")
    print("="*60)

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
        model = RFDETRMediumV3(position_embedding='sine',use_fdam=False)
        model1=RFDETRMediumV3(position_embedding='sine',use_fdam=True)
    core_model=model.model.model.to(device)
    initialize_weights(core_model)
    pre_transfer_hash = torch.sum(core_model.backbone[0].encoder.encoder.embeddings.patch_embeddings.projection.weight).item()
    print(f"迁移前 core_model 投影层权重总和：{pre_transfer_hash:.4f}")

    dinov3_backbone = core_model.backbone[0].encoder
    dinov3_backbone_1=model1.model.model.backbone[0].encoder
    print("没有使用FDAM的dinov3_backbone")
    print(dinov3_backbone)
    print("使用FDAM的dinov3_backbone")
    print(dinov3_backbone_1)
    tgt_layer0 = dinov3_backbone.encoder.encoder.layer[0]
    print("目标LayerScale参数名：")
    for name, param in tgt_layer0.ls1.named_parameters():
        print(f"ls1参数名：{name}，形状：{param.shape}")
    try:
        dinov3_backbone = transfer_dinov3_compatible(
            dinov3=dinov3,
            dinov3_backbone=dinov3_backbone
        )
    except RuntimeError as e:
        print(f"迁移中断：{e}")
    verify_transfer_compatible(dinov3, dinov3_backbone)

    def retransfer_layerscale_gamma(target_encoder, dinov3_model, device):
        """强制迁移原始DINOv3的gamma参数，确保数值完全一致"""
        print("\n" + "="*60)
        print("=== 强制迁移LayerScale的gamma参数 ===")
        print("="*60)
        
        for layer_idx in range(12):
            # 定位源和目标的LayerScale模块
            src_block = dinov3_model.blocks[layer_idx]
            tgt_block = target_encoder.encoder.layer[layer_idx]
            
            # 强制复制ls1.gamma和ls2.gamma
            tgt_block.ls1.gamma.data.copy_(src_block.ls1.gamma.to(device))
            tgt_block.ls2.gamma.data.copy_(src_block.ls2.gamma.to(device))
            
            # 验证迁移结果 - 使用detach()来避免RuntimeError
            tgt_ls1 = tgt_block.ls1.gamma.data.cpu().detach().numpy()
            src_ls1 = src_block.ls1.gamma.cpu().detach().numpy()
            tgt_ls2 = tgt_block.ls2.gamma.data.cpu().detach().numpy()
            src_ls2 = src_block.ls2.gamma.cpu().detach().numpy()
            
            if not (np.allclose(tgt_ls1, src_ls1, atol=1e-6) and 
                    np.allclose(tgt_ls2, src_ls2, atol=1e-6)):
                print(f"❌ 第{layer_idx}层gamma迁移失败，强制重新赋值")
                tgt_block.ls1.gamma.data.copy_(src_block.ls1.gamma.to(device))
                tgt_block.ls2.gamma.data.copy_(src_block.ls2.gamma.to(device))
            else:
                print(f"✅ 第{layer_idx}层gamma迁移成功")
        
        # 最终验证 - 同样使用detach()
        print("\n=== 迁移后gamma参数范围检查 ===")
        for layer_idx in range(12):
            tgt_block = target_encoder.encoder.layer[layer_idx]
            ls1 = tgt_block.ls1.gamma.data.cpu().detach().numpy()
            ls2 = tgt_block.ls2.gamma.data.cpu().detach().numpy()
            print(f"第{layer_idx}层 - ls1: [{ls1.min():.6f}, {ls1.max():.6f}] | ls2: [{ls2.min():.6f}, {ls2.max():.6f}]")
    retransfer_layerscale_gamma(target_encoder=core_model.backbone[0].encoder.encoder, dinov3_model=dinov3, device=device)

    post_transfer_hash = torch.sum(core_model.backbone[0].encoder.encoder.embeddings.patch_embeddings.projection.weight).item()
    print(f"迁移后 core_model 投影层权重总和：{post_transfer_hash:.4f}")
    print(f"core_model 是否被修改：{'✅ 是' if pre_transfer_hash != post_transfer_hash else '❌ 否'}")
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
    # }, f'medium-dinov3{plus}-randomdecoder.pth')
