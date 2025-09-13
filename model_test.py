import torch
from rfdetr import RFDETRNano

# 1. 自定义NestedTensor类（保持属性名与模型一致）
class SimpleNestedTensor:
    def __init__(self, tensors: torch.Tensor, mask: torch.Tensor = None):
        self.tensors = tensors  # 复数形式，匹配模型内部NestedTensor
        if mask is None:
            self.mask = torch.zeros(
                tensors.shape[0], tensors.shape[2], tensors.shape[3],
                dtype=torch.bool, device=tensors.device
            )
        else:
            self.mask = mask.to(tensors.device)

    def to(self, device: torch.device):
        """移动到指定设备"""
        return SimpleNestedTensor(
            tensors=self.tensors.to(device),
            mask=self.mask.to(device)
        )

def test_dinov2_encoder():
    # 1. 自动检测设备（GPU/CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")
    print("="*60)

    # 2. 初始化模型并提取DinoV2编码器
    rfdetr_wrapper = RFDETRNano(pretrain_weights='D:/__easyHelper__/RF-DETR/rfdetr/checkpoint/nano-coco.pth')
    core_model = rfdetr_wrapper.model.model  # 获取核心模型
    core_model = core_model.to(device)       # 移动核心模型到目标设备
    backbone = core_model.backbone           # 获取Joiner实例
    encoder = backbone[0].encoder            # 提取DinoV2编码器（需测试的部分）
    encoder.eval()                           # 切换到评估模式

    # 3. 创建模拟输入（直接使用图像张量，无需NestedTensor包装）
    # 注意：DinoV2编码器的输入是原始图像张量，而非NestedTensor
    batch_size = 2
    channels = 3
    height = 640  # 匹配模型target_shape
    width = 640
    dummy_images = torch.randn(batch_size, channels, height, width, device=device)

    # 4. 前向传播（直接输入图像张量）
    with torch.no_grad():
        # 参考Backbone类的forward逻辑：feats = self.encoder(tensor_list.tensors)
        encoder_outputs = encoder(dummy_images)

    # 5. 解析输出并打印信息
    print("1. 输入信息")
    print(f"   - 图像张量形状: {dummy_images.shape} | 设备: {dummy_images.device}")
    print("="*60)

    print("\n2. DinoV2编码器输出信息")
    # 处理输出（可能是单特征层或多特征层列表）
    if isinstance(encoder_outputs, (list, tuple)):
        print(f"   - 输出特征层总数: {len(encoder_outputs)}")
        for idx, feat in enumerate(encoder_outputs):
            print(f"\n   特征层 {idx}:")
            print(f"   - 形状: {feat.shape} | 设备: {feat.device}")
            print(f"   - 通道数: {feat.shape[1]} | 空间尺寸: {feat.shape[2:4]}")
    else:
        # 单特征层情况
        print(f"   - 特征形状: {encoder_outputs.shape} | 设备: {encoder_outputs.device}")
        print(f"   - 通道数: {encoder_outputs.shape[1]} | 空间尺寸: {encoder_outputs.shape[2:4]}")

    print("\n" + "="*60)
    print("3. 关键信息总结")
    print("   - 编码器输入: 图像张量 [B, 3, H, W]")
    print("   - 编码器输出: 特征张量 [B, C, H', W'] (下采样后的特征)")
    print("="*60)


if __name__ == "__main__":
    test_dinov2_encoder()