import torch.distributed as dist
import argparse
from rfdetr import RFDETRNano, RFDETRBase, RFDETRMedium, RFDETRLarge, RFDETRMediumV3,RFDETRNanoV3,RFDETRMediumV3Plus
import os
import torch
import torch.nn as nn
from rfdetr.util.misc import init_distributed_mode

# 解析命令行参数
ALLOWED_SCALES = {"P3", "P4", "P5", "P6"}
parser = argparse.ArgumentParser(description="RF-DETR 训练脚本")
parser.add_argument("--model_size", type=str, required=True,
                    help="模型大小：nano/base/medium/large")
parser.add_argument("--batch_size", type=int, required=False, default=32,
                    help="训练批次大小")
parser.add_argument("--grad_accum_steps", type=int, required=False, default=2,
                    help="")
parser.add_argument("--img_size", type=int, required=False,default=320,
                    help="img大小")
parser.add_argument("--weight_path", type=str, required=False,default="/home/cobot/github_code/RF-DETR/rfdetr/mae_checkpoint/40w_iter_size224_bs48_public_medium.pth",
                    help="weight path")
parser.add_argument("--out_dir", type=str, required=False,default='public',
                    help="输出路径")
parser.add_argument("--freeze_encoder", type=int, required=False,default=0,choices=[0, 1],
                    help="freeze encoder")
parser.add_argument("--dataset_dir", type=str, required=True, help="数据集路径")
parser.add_argument("--decoder_mode", type=int, required=True, help="1排除encoder，2所有随机初始化")
parser.add_argument("--decoder_sa_type", type=str, required=False, help="normal or diff")
parser.add_argument("--lr", type=float, required=True, help="学习率")
parser.add_argument("--lr_encoder", type=float, required=True, help="学习率")
parser.add_argument("--dataset_file", type=str,default="roboflow",  required=False, help="数据集模式")
parser.add_argument("--decoder_pos", type=str, default="sine",  required=False, help="decoder pos")
parser.add_argument("--early_stopping_patience", type=int, default=50,  required=False, help="early_stopping_patience")
parser.add_argument("--select_mode", type=int, required=True, help="1 rf，2 deim")
parser.add_argument("--use_fdam",type=int,required=False)
parser.add_argument("--use_featAug",type=int,required=False,default=0)
parser.add_argument(
    "--projector_scale",
    type=str,
    nargs='+',
    default=["P4", "P5", "P6"],
    help=f"Projector scales, must be one of {ALLOWED_SCALES}, e.g., --projector_scale P3 P4 P5"
)
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
args = parser.parse_args()
if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
    init_distributed_mode(args)

model_classes = {
    "nano": RFDETRNanoV3,
    "base": RFDETRBase,
    "medium": RFDETRMediumV3,
    "mediumplus":RFDETRMediumV3Plus,
    "large": RFDETRLarge
}

ModelClass = model_classes[args.model_size]
dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))

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
        
freeze_encoder=True
if args.freeze_encoder==0:
    freeze_encoder=False
# 初始化模型（使用预训练权重）
use_fdam=False
if args.use_fdam==1:
    use_fdam=True
feataug_enable = True

if args.use_featAug==0:
    feataug_enable=False
    feataug_types=tuple()
elif args.use_featAug==1:
    feataug_types=('flip',)
elif args.use_featAug==2:
    feataug_types=('fc',)

print(feataug_enable)
print(feataug_types)
model = ModelClass(
    pretrain_weights=args.weight_path,
    freeze_encoder=freeze_encoder,
    decoder_sa_type=args.decoder_sa_type,
    select_mode=args.select_mode,
    projector_scale=args.projector_scale,
    use_fdam=use_fdam,
    feataug_enable=feataug_enable,
    feataug_types=feataug_types,
    use_checkpoint=True,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
core_model = model.model.model.to(device)

# --------------------------
# 1. 单独统计 dino_encoder 的参数数量
# --------------------------
dino_encoder = core_model.backbone[0]
encoder_total_params = 0
for param in dino_encoder.parameters():
    encoder_total_params += param.numel()
print("="*50)
print(f"dino_encoder 类型: {type(dino_encoder)}")
print(f"dino_encoder 子模块数量: {len(list(dino_encoder.modules()))}")
print(f"dino_encoder 自身参数总数（百万）: {encoder_total_params / 1e6:.2f}M")  # 关键：查看encoder是否有参数
print("="*50)

# 2. 收集 encoder 及其所有子模块的 ID（用于排除）
encoder_module_ids = set()
for m in dino_encoder.modules():
    encoder_module_ids.add(id(m))

# 3. 统计 core_model 总参数（用于验证）
total_core_params = 0
for param in core_model.parameters():
    total_core_params += param.numel()
print(f"core_model 总参数（百万）: {total_core_params / 1e6:.2f}M")
print("="*50)

total_initialized_params = 0
seen_params = set()

if args.decoder_mode == 1:
    # mode==1：排除 encoder 及其参数
    for name, component in core_model.backbone[0].__dict__.items():
        if name == 'encoder':
            continue
        if isinstance(component, torch.nn.Module):
            print(f"初始化组件: {name}")
            initialize_weights(component)
        elif isinstance(component, torch.nn.ModuleList):
            print(f"初始化模块列表: {name}")
            for i, sub_module in enumerate(component):
                print(f"  初始化子模块 {i}")
                initialize_weights(sub_module)
    initialize_weights(core_model.transformer)
    
elif args.decoder_mode == 2:
    # mode==2：初始化所有参数
    initialize_weights(core_model)
    for module in core_model.modules():
        for param in module.parameters():
            param_id = id(param)
            if param_id not in seen_params:
                seen_params.add(param_id)
                total_initialized_params += param.numel()

# 打印结果（增加对比参考）
print(f"初始化模式: {args.decoder_mode}")
print(f"已初始化的参数总数（百万）: {total_initialized_params / 1e6:.2f}M")
print(f"理论上模式1应初始化的参数（总参数 - encoder参数）: {(total_core_params - encoder_total_params)/1e6:.2f}M")

# 启动训练（使用传入的 batch_size）
model.train(
    dataset_dir= args.dataset_dir,
    dataset_file=args.dataset_file,
    epochs=500,
    num_workers=16,
    batch_size=args.batch_size,  # 使用传入的批次大小
    grad_accum_steps=args.grad_accum_steps,
    lr=args.lr,
    lr_encoder=args.lr_encoder,
    resolution=args.img_size,
    project='rf-detr-traffic',
    early_stopping=True,
    early_stopping_patience=args.early_stopping_patience,
    output_dir=args.out_dir,
    multi_scale=True,
    use_amp=True,

)

# 清理分布式进程组
if dist.is_initialized():
    dist.destroy_process_group()

