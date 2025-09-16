from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.models.backbone.dinov3 import DinoV3
from rfdetr import RFDETRNano, RFDETRBase, RFDETRMedium, RFDETRLarge, RFDETRMediumV3, RFDETRNanoV3,RFDETRMediumV3Plus
import torch

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
    #     'D:/__easyHelper__/dinov3-main', 
    #     'dinov3_vits16plus', 
    #     source='local', 
    #     weights='D:/__easyHelper__/dinov3-main/checkpoint/dinov3_vits16plus.pth'
    # )
    dinov3sp=torch.hub.load(
        'D:/__easyHelper__/dinov3-main', 
        'dinov3_vits16plus', 
        source='local', 
        weights='D:/__easyHelper__/dinov3-main/checkpoint/dinov3_vits16plus.pth'
    )
    print("dinov3smallplus")
    print(dinov3sp)
    model=RFDETRMediumV3Plus(position_embedding='learned')
    dinov3_core_model=model.model.model.to(device)
    dinov3_encoder = dinov3_core_model.backbone[0].encoder
    print(dinov3_encoder)
    for param in dinov3_core_model.parameters():
        param.requires_grad = False
    # print(dinov3_encoder)
    x = torch.randn(5, 3, 640, 640).to(device)
    for j in dinov3_encoder(x):
        print(j.shape)
    dinov3_core_model(x)