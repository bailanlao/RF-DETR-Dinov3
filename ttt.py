from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.models.backbone.dinov3 import DinoV3
from rfdetr import RFDETRNano, RFDETRBase, RFDETRMedium, RFDETRLarge, RFDETRMediumV3,RFDETRNanoV3
import torch

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
    model=RFDETRNanoV3(pretrain_weights='rf-detr-nano-dinov3.pth')
    dinov3_core_model=model.model.model.to(device)
    # print(dinov3_core_model)
    dinov3_encoder = dinov3_core_model.backbone[0].encoder
    # print(dinov3_encoder)
    x = torch.randn(1, 3, 640, 640).to(device)
    print(dinov3_encoder(x)) # 前向传播测试
    for j in dinov3_encoder(x):
        print(j.shape)