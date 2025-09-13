import torch.distributed as dist
import argparse
from rfdetr import RFDETRNano, RFDETRBase, RFDETRMedium, RFDETRLarge

# 解析命令行参数
parser = argparse.ArgumentParser(description="RF-DETR 训练脚本")
parser.add_argument("--model_size", type=str, required=True, 
                    choices=["nano", "base", "medium", "large"],
                    help="模型大小：nano/base/medium/large")
parser.add_argument("--batch_size", type=int, required=False,default=32,
                    help="训练批次大小")
parser.add_argument("--img_size", type=int, required=False,default=320,
                    help="img大小")
parser.add_argument("--dataset_dir", type=str, required=True, help="数据集路径")
args = parser.parse_args()

# 根据 model_size 选择对应的模型
model_classes = {
    "nano": RFDETRNano,
    "base": RFDETRBase,
    "medium": RFDETRMedium,
    "large": RFDETRLarge
}
ModelClass = model_classes[args.model_size]
dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))

# 初始化模型（使用预训练权重）
model = ModelClass(pretrain_weights=f'/home/cobot/github_code/RF-DETR/rfdetr/checkpoint/{args.model_size}_coco.pth')

# 启动训练（使用传入的 batch_size）
model.train(
    dataset_dir= args.dataset_dir,
    epochs=100,
    batch_size=args.batch_size,  # 使用传入的批次大小
    grad_accum_steps=2,
    lr=1e-4,
    resolution=args.img_size,
    project='rf-detr-traffic',
    early_stopping=True,
    early_stopping_patience=30,
    output_dir=f'/home/cobot/github_code/RF-DETR/output-{args.model_size}-{dataset_name}',
)

# 清理分布式进程组
if dist.is_initialized():
    dist.destroy_process_group()

