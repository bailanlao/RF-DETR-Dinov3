import torch
from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.models.backbone.dinov3 import DinoV3
from rfdetr import RFDETRNano, RFDETRBase, RFDETRMedium, RFDETRLarge, RFDETRMediumV3,RFDETRNanoV3
from torch.nn import Module
from typing import Dict, Tuple

import torch
from torch.nn import Module
from typing import Tuple, List, Dict


def count_module_params(module: Module) -> int:
    return sum(param.numel() for param in module.parameters())


def transfer_rfdetr_to_dinov3_weights(
    rfdetr_core_model: Module,
    dinov3_core_model: Module,
    device: torch.device = None,
    transfer_top_heads: bool = True
) -> Tuple[Module, List[str], int]:
    """
    核心函数：将rfdetr_core_model的可迁移模块权重迁移到dinov3_core_model
    可迁移模块：Decoder、TransformerEncoder相关模块、Backbone Projector、（可选）顶层任务头
    
    Args:
        rfdetr_core_model: 权重来源模型（rfdetr的核心模型，即rfdetr.model.model）
        dinov3_core_model: 权重目标模型（dinov3的核心模型，即model.model.model）
        device: 模型运行设备（默认自动选择GPU/CPU）
        transfer_top_heads: 是否迁移顶层任务头（class_embed/bbox_embed/refpoint_embed/query_feat），默认True
    
    Returns:
        Tuple[Module, List[str], int]:
            - 迁移后的dinov3_core_model（已加载权重）
            - 迁移日志列表（记录每个模块的迁移结果和参数量）
            - 总迁移参数量（元素总数）
    
    Raises:
        AssertionError: 模型结构不匹配时抛出（如关键模块缺失）
        RuntimeError: 权重加载失败时抛出（如参数形状不兼容）
    """
    # 1. 初始化设备（默认自动选择）
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[权重迁移] 使用设备：{device}")

    # 2. 确保两个模型都在指定设备上
    rfdetr_core = rfdetr_core_model.to(device).eval()  # eval模式避免BatchNorm等层干扰
    dinov3_core = dinov3_core_model.to(device).eval()
    total_transferred = 0
    transfer_log: List[str] = []

    # -------------------------- 3. 迁移 Transformer Decoder（必迁模块） --------------------------
    try:
        src_decoder = rfdetr_core.transformer.decoder
        dst_decoder = dinov3_core.transformer.decoder
        dst_decoder.load_state_dict(src_decoder.state_dict(), strict=True)
        params = count_module_params(src_decoder)
        total_transferred += params
        log = f"✅ [Transformer Decoder] 迁移完成 | 参数量：{params:,}"
        transfer_log.append(log)
        print(log)
    except Exception as e:
        raise RuntimeError(f"[Transformer Decoder] 迁移失败：{str(e)}") from e

    # -------------------------- 4. 迁移 Transformer Encoder 相关模块（必迁模块） --------------------------
    encoder_related_modules: Dict[str, str] = {
        "enc_output": "transformer.enc_output",
        "enc_output_norm": "transformer.enc_output_norm",
        "enc_out_bbox_embed": "transformer.enc_out_bbox_embed",
        "enc_out_class_embed": "transformer.enc_out_class_embed"
    }
    for module_name, module_path in encoder_related_modules.items():
        try:
            # 通过字符串路径获取模块（避免硬编码层级）
            src_module = eval(f"rfdetr_core.{module_path}")
            dst_module = eval(f"dinov3_core.{module_path}")
            dst_module.load_state_dict(src_module.state_dict(), strict=True)
            params = count_module_params(src_module)
            total_transferred += params
            log = f"✅ [{module_name}] 迁移完成 | 参数量：{params:,}"
            transfer_log.append(log)
            print(log)
        except Exception as e:
            raise RuntimeError(f"[{module_name}] 迁移失败：{str(e)}") from e

    # -------------------------- 5. 迁移 Backbone Projector（必迁模块） --------------------------
    try:
        # Backbone是Joiner类型，第一个元素是实际Backbone实例（含projector）
        src_projector = rfdetr_core.backbone[0].projector
        dst_projector = dinov3_core.backbone[0].projector
        dst_projector.load_state_dict(src_projector.state_dict(), strict=True)
        params = count_module_params(src_projector)
        total_transferred += params
        log = f"✅ [Backbone Projector] 迁移完成 | 参数量：{params:,}"
        transfer_log.append(log)
        print(log)
    except Exception as e:
        raise RuntimeError(f"[Backbone Projector] 迁移失败：{str(e)}") from e

    # -------------------------- 6. 可选迁移：顶层任务头 --------------------------
    if transfer_top_heads:
        top_head_modules: Dict[str, str] = {
            "class_embed": "class_embed",
            "bbox_embed": "bbox_embed",
            "refpoint_embed": "refpoint_embed",
            "query_feat": "query_feat"
        }
        for module_name, module_path in top_head_modules.items():
            try:
                src_module = eval(f"rfdetr_core.{module_path}")
                dst_module = eval(f"dinov3_core.{module_path}")
                dst_module.load_state_dict(src_module.state_dict(), strict=True)
                params = count_module_params(src_module)
                total_transferred += params
                log = f"✅ [顶层任务头-{module_name}] 迁移完成 | 参数量：{params:,}"
                transfer_log.append(log)
                print(log)
            except Exception as e:
                raise RuntimeError(f"[顶层任务头-{module_name}] 迁移失败：{str(e)}") from e
    else:
        log = "ℹ️ [可选配置] 未迁移顶层任务头（transfer_top_heads=False）"
        transfer_log.append(log)
        print(log)

    # -------------------------- 7. 输出迁移总结 --------------------------
    summary_log = f"\n[迁移总结] 总迁移参数量：{total_transferred:,} 个（{total_transferred / 1e6:.2f} M）"
    transfer_log.append(summary_log)
    print("="*80)
    print(summary_log)
    print("⚠️  不可迁移模块：backbone.encoder（DinoV2 vs DinoV3 结构不兼容）")
    print("="*80)

    return dinov3_core, transfer_log, total_transferred

def transfer_dinov3_to_core_model(
    dinov3_core_model: Module,
    pretrained_dinov3: Module,
    device: torch.device = None
) -> Tuple[Module, List[str], int]:
    """
    将预训练dinov3模型的权重迁移到dinov3_core_model的编码器中
    自动定位编码器路径：dinov3_core_model.backbone[0].encoder.encoder.encoder
    
    Args:
        dinov3_core_model: 目标核心模型（含待迁移的编码器）
        pretrained_dinov3: 预训练的dinov3模型（DinoVisionTransformer）
        device: 设备（默认自动选择）
    
    Returns:
        迁移后的dinov3_core_model、迁移日志、总迁移参数量
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 将模型移至指定设备
    dinov3_core_model = dinov3_core_model.to(device).eval()
    pretrained_dinov3 = pretrained_dinov3.to(device).eval()
    
    # 2. 定位目标编码器（dinov3_core_model中的编码器）
    try:
        dinov3_encoder = dinov3_core_model.backbone[0].encoder.encoder.encoder
        print(f"✅ 成功定位编码器：{type(dinov3_encoder).__name__}")
    except AttributeError as e:
        raise RuntimeError(f"无法定位编码器路径，请检查模型结构：{str(e)}") from e
    
    # 3. 提取预训练模型的核心层（12个SelfAttentionBlock）
    src_blocks = pretrained_dinov3.blocks
    dst_layers = dinov3_encoder.layer  # 目标编码器的12个层
    
    # 验证层数量是否匹配（均为12层）
    assert len(src_blocks) == len(dst_layers), \
        f"核心层数量不匹配：预训练模型有{len(src_blocks)}层，目标编码器有{len(dst_layers)}层"
    
    total_params = 0
    transfer_log = []
    
    # 4. 逐层迁移参数（12层一一对应）
    for i in range(len(src_blocks)):
        src_block = src_blocks[i]
        dst_layer = dst_layers[i]
        layer_log = []
        layer_total = 0
        
        # 迁移各组件（结构完全匹配）
        components = [
            ("norm1", src_block.norm1, dst_layer.norm1),
            ("attn", src_block.attn, dst_layer.attn),
            ("ls1", src_block.ls1, dst_layer.ls1),
            ("norm2", src_block.norm2, dst_layer.norm2),
            ("mlp", src_block.mlp, dst_layer.mlp),
            ("ls2", src_block.ls2, dst_layer.ls2)
        ]
        
        for comp_name, src_comp, dst_comp in components:
            try:
                dst_comp.load_state_dict(src_comp.state_dict(), strict=True)
                params = count_module_params(src_comp)
                layer_total += params
                layer_log.append(f"{comp_name}（{params:,}）")
            except Exception as e:
                raise RuntimeError(f"第{i}层{comp_name}迁移失败：{str(e)}")
        
        # 记录当前层迁移结果
        total_params += layer_total
        log = f"✅ 第{i}层迁移完成 | 组件：{', '.join(layer_log)} | 总参数量：{layer_total:,}"
        transfer_log.append(log)
        print(log)
    
    # 5. 迁移总结（编码器已嵌入回dinov3_core_model）
    summary = f"\n迁移完成 | 总参数量：{total_params:,}（{total_params/1e6:.2f} M）"
    transfer_log.append(summary)
    print("="*80)
    print(summary)
    print(f"编码器已更新至 dinov3_core_model.backbone[0].encoder.encoder.encoder")
    print("="*80)
    
    return dinov3_core_model, transfer_log, total_params

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    # model = DinoV2(out_feature_indexes=[2, 4, 5, 9],load_dinov2_weights=False)
    # model.export()
    # x = torch.randn(1, 3, 840, 840)
    # print(model(x))
    # for j in model(x):
    #     print(j.shape)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # rfdetr = RFDETRNano(pretrain_weights='D:/__easyHelper__/RF-DETR/rfdetr/checkpoint/nano-coco.pth')
    # core_model = rfdetr.model.model.to(device)
    # dino_encoder = core_model.backbone[0].encoder.encoder.encoder
    # print(dino_encoder)
    # dinov3=torch.hub.load(
    #     'D:/__easyHelper__/RF-DETR/dinov3-main', 
    #     'dinov3_vits16', 
    #     source='local', 
    #     weights='D:/__easyHelper__/RF-DETR/dinov3-main/checkpoint/dinov3_vits16.pth'
    # )
    # print("dinov3")
    # print(dinov3)
    # print(dinov3.blocks[0].ls1.inplace)
    # print(dinov3.blocks[0].ls1.init_values)
    # print(dinov3.blocks[0].ls2.inplace)
    # print(dinov3.blocks[0].ls2.init_values)
    save_path='rf-detr-nano-dinov3.pth'
    model=RFDETRNanoV3()
    rfdetr = RFDETRNano(pretrain_weights='D:/__easyHelper__/RF-DETR/rfdetr/checkpoint/nano-coco.pth')
    rfdetr_core_model=rfdetr.model.model.to(device)
    dinov3_core_model=model.model.model.to(device)
    print("dinov3",dinov3_core_model)
    print("rfdetr",rfdetr_core_model)
    dinov3_core_model, transfer_log, total_params = transfer_rfdetr_to_dinov3_weights(
        rfdetr_core_model=rfdetr_core_model,
        dinov3_core_model=dinov3_core_model,
        device=None,  # 自动选择GPU/CPU
        transfer_top_heads=False  # 迁移顶层任务头
    ) # 能用
    dinov3_encoder = dinov3_core_model.backbone[0].encoder.encoder.encoder
    dinov3=torch.hub.load(
        'D:/__easyHelper__/RF-DETR/dinov3-main', 
        'dinov3_vits16', 
        source='local', 
        weights='D:/__easyHelper__/RF-DETR/dinov3-main/checkpoint/dinov3_vits16.pth'
    )
    print("dinov3参数量：",count_module_params(dinov3))

    updated_core_model, transfer_log, total_params = transfer_dinov3_to_core_model(
        dinov3_core_model=dinov3_core_model,  # 目标核心模型
        pretrained_dinov3=dinov3,            # 预训练dinov3模型
        device=None  # 自动选择GPU/CPU
    )
    torch.save({
            'model': dinov3_core_model.state_dict()
        }, save_path)
