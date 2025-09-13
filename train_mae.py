import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
import torchvision.transforms as transforms

from rfdetr import RFDETRNano, RFDETRMedium

# assert timm.__version__ == "0.3.2"  # version check
from timm.optim import optim_factory

import mae.util.misc as misc
from mae.util.misc import NativeScalerWithGradNormCount as NativeScaler

from rfdetr.models.backbone.dinov2 import get_config
from types import SimpleNamespace
from mae_model import WindowedDinov2WithRegistersEncoder,MaskedAutoencoderViT

from engine_pretrain import train_one_epoch
class RecursiveImageDataset(Dataset):
    """传入数据集根路径，自动递归遍历所有子文件夹收集图像"""
    def __init__(self, root_dir, transform=None, img_extensions=('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG')):
        self.root_dir = root_dir
        self.transform = transform
        self.img_extensions = tuple(ext.lower() for ext in img_extensions)
        self.img_paths = self._collect_all_images()

        if len(self.img_paths) == 0:
            raise ValueError(
                f"在 {root_dir} 及其子文件夹中未找到图像\n支持格式：{img_extensions}"
            )

    def _collect_all_images(self):
        img_paths = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.lower().endswith(self.img_extensions):
                    img_paths.append(os.path.join(dirpath, filename))
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ 加载失败：{img_path}\n错误：{e}\n用第一张图像替代")
            img = Image.open(self.img_paths[0]).convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img, 0  # 标签占位（自监督训练无需真实标签）



def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training (RFDETR DinoV2)', add_help=False)
    # 1. 训练批次参数
    parser.add_argument('--batch_size', default=16, type=int,
                        help='单GPU批次大小（有效批次=batch_size × accum_iter × GPU数）')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='梯度累积次数（显存不足时增大）')
    parser.add_argument('--epochs', default=400, type=int,
                        help='自动计算（由total_iter决定）')

    # 2. 核心控制：固定总迭代次数（测试时可设小，如10；正式训练改回100000）
    parser.add_argument('--total_iter', default=10, type=int,
                        help='所有数据集统一的总训练迭代次数')

    # 3. MAE模型参数（固定用DinoV2编码器）
    parser.add_argument('--input_size', default=320, type=int,
                        help='图像输入尺寸（需被16整除，如320/16=20）')
    parser.add_argument('--model_size', default='medium', type=str,
                        help='模型的尺寸')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='MAE掩码比例（默认75%）')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='是否用归一化像素计算损失')
    parser.set_defaults(norm_pix_loss=False)

    # 4. 优化器参数
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='权重衰减（默认0.05）')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='绝对学习率（建议用blr自动计算）')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='基础学习率（实际LR=blr × 有效批次/256）')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='学习率下限')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='学习率热身epoch数（测试时可设小，如1）')

    # 5. 数据集与RFDETR参数
    parser.add_argument('--data_path', default='/opt/public/huangtao/data/US_datasets/ALL_IMG_SUB', type=str,
                        help='数据集根路径（递归加载图像）')
    parser.add_argument('--rfdetr_pretrain_path', 
                        default='/home/huangtao/env/RF-DETR/rfdetr/checkpoint/medium-coco.pth', 
                        type=str, help='RFDETR预训练权重路径')

    # 6. 输出与设备
    parser.add_argument('--output_dir', default='./output_rfdetr_mae',
                        help='保存core_model的根目录')
    parser.add_argument('--device', default='cuda',
                        help='训练设备（cuda/cpu）')
    parser.add_argument('--seed', default=0, type=int,
                        help='随机种子')
    parser.add_argument('--resume', default='',
                        help='断点续训：从core_model checkpoint加载')

    # 7. 数据加载
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='起始epoch')
    parser.add_argument('--num_workers', default=10, type=int,
                        help='数据加载线程数（测试时可设0，避免多线程问题）')
    parser.add_argument('--pin_mem', action='store_true',
                        help='锁定CPU内存（默认开启）')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # 8. 分布式训练
    parser.add_argument('--world_size', default=1, type=int,
                        help='GPU数量')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='本地进程排名（自动分配）')
    parser.add_argument('--dist_on_itp', action='store_true',
                        help='ITP环境分布式训练')
    parser.add_argument('--dist_url', default='env://',
                        help='分布式通信地址')

    return parser

def sync_mae_encoder_to_core_model(mae_model, core_model, is_distributed):
    if is_distributed:
        mae_encoder = mae_model.module.encoder
    else:
        mae_encoder = mae_model.encoder
    core_model.backbone[0].encoder.encoder.encoder.load_state_dict(mae_encoder.state_dict())
    print(f'✅ 已同步MAE训练后的编码器到core_model.backbone[0].encoder.encoder.encoder')

def main(args):
    # 1. 初始化分布式训练
    misc.init_distributed_mode(args)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f'脚本路径：{script_dir}')
    print(f'参数配置：\n{args}'.replace(', ', ',\n'))

    # 2. 设备与随机种子
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # 3. 加载RFDETR并提取DinoV2编码器（核心）
    print(f'\n加载RFDETR：{args.rfdetr_pretrain_path}')
    if args.model_size=='medium':
        rfdetr = RFDETRMedium(pretrain_weights=args.rfdetr_pretrain_path)
    elif args.model_size == 'nano':
        rfdetr = RFDETRNano(pretrain_weights=args.rfdetr_pretrain_path)
    core_model = rfdetr.model.model
    core_model = core_model.to(device)
    dino_encoder = core_model.backbone[0].encoder.encoder.encoder
    dino_encoder = dino_encoder.to(device)
    print(f'DinoV2编码器加载完成，设备：{device}')
    config = get_config("small",True)
    config = SimpleNamespace(** config)
    config.num_layers=12
    mae_encoder=WindowedDinov2WithRegistersEncoder(config)
    print("验证MAE编码器参数设备：")
    for name, param in mae_encoder.named_parameters():
        if param.device != device:
            print(f"❌ 编码器参数 {name} 设备错误：{param.device}（应为{device}）")
        else:
            print(f"✅ 编码器参数 {name} 设备正确")
    # 4. 数据预处理与加载
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.6, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 递归加载数据集（修正后无datasets依赖）
    print(f'从 {args.data_path} 加载图像...')
    dataset_train = RecursiveImageDataset(
        root_dir=args.data_path,
        transform=transform_train
    )
    print(f'数据集加载完成：{len(dataset_train)} 张图像')

    # 采样器与数据加载器（直接用torch.utils.data的类，无需datasets）
    if args.distributed:
        sampler_train = DistributedSampler(
            dataset_train, num_replicas=args.world_size, rank=misc.get_rank(), shuffle=True
        )
        print(f'分布式采样器：GPU数量={args.world_size}')
    else:
        sampler_train = RandomSampler(dataset_train)

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    # 5. 创建MAE模型（用DinoV2编码器）
    print(f'\n创建MAE模型（基于DinoV2编码器）')
    mae_model = MaskedAutoencoderViT(
        mae_encoder,
        img_size=args.input_size,
        patch_size=16,
        norm_pix_loss=args.norm_pix_loss
    )
    mae_model = mae_model.to(device)
    def print_memory_usage(device):
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # 已分配显存
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)    # 已预留显存
        print(f"当前GPU显存使用（rank={misc.get_rank()}）：")
        print(f"  - 已分配：{allocated:.2f} GB")
        print(f"  - 已预留：{reserved:.2f} GB")
        print(f"  - 剩余显存：{32 - reserved:.2f} GB")
    print_memory_usage(device)
    model_without_ddp = mae_model
    # 6. 有效批次与学习率计算
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256  # 按MAE论文缩放
    print(f'\n训练配置：')
    print(f'基础LR：{args.lr * 256 / eff_batch_size:.2e} | 实际LR：{args.lr:.2e}')
    print(f'梯度累积：{args.accum_iter} | 有效批次：{eff_batch_size}')

    # 7. 动态计算epoch数（保证总iter一致）
    iter_per_epoch = len(data_loader_train)
    total_needed_epochs = (args.total_iter + iter_per_epoch - 1) // iter_per_epoch
    args.epochs = max(total_needed_epochs, args.warmup_epochs)  # 确保热身完成
    print(f'单epoch iter：{iter_per_epoch} | 目标总iter：{args.total_iter}')
    print(f'自动调整epoch数：{args.epochs}（总iter≈{args.epochs * iter_per_epoch}）')

    # 8. 分布式模型包装
    if args.distributed:
        mae_model = torch.nn.parallel.DistributedDataParallel(
            mae_model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = mae_model.module

    # 9. 优化器（✅ 用直接导入的add_weight_decay，无timm警告）
    param_groups = [
        {"params": [], "weight_decay": args.weight_decay},
        {"params": [], "weight_decay": 0.0}
    ]

    for name, param in model_without_ddp.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name:
            param_groups[1]["params"].append(param)
        else:
            param_groups[0]["params"].append(param)

    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, betas=(0.9, 0.95)
    )
    loss_scaler = NativeScaler()
    print(f'\n优化器：\n{optimizer}')


    # 10. 断点续训
    start_epoch = args.start_epoch
    if args.resume:
        if os.path.isfile(args.resume):
            print(f'\n从续训文件 {args.resume} 加载完整训练状态')
            checkpoint = torch.load(args.resume, map_location=device)
            
            # 恢复核心模型（含预训练后的编码器）
            core_model.load_state_dict(checkpoint['core_model_state_dict'])
            # 恢复MAE模型（含解码器权重，续训必须）
            mae_model.load_state_dict(checkpoint['mae_model_state_dict'])
            # 恢复优化器和损失缩放器
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loss_scaler.load_state_dict(checkpoint['loss_scaler_state_dict'])
            # 恢复起始epoch
            start_epoch = checkpoint['epoch'] + 1
            
            print(f'断点加载完成：')
            print(f'  - 起始epoch：{start_epoch}')
            print(f'  - 恢复内容：MAE模型(解码器+编码器) + core_model + 优化器 + 损失缩放器')
        else:
            raise FileNotFoundError(f'续训文件不存在：{args.resume}')

    # 11. 日志初始化（仅主进程）
    log_writer = None
    exp_dir = None
    if misc.is_main_process():
        dataset_name = os.path.basename(os.path.normpath(args.data_path))
        exp_suffix = f'iter{args.total_iter}_img_size{args.input_size}'
        exp_dir = os.path.join(args.output_dir, f'{dataset_name}_{exp_suffix}_{args.model_size}_{args.batch_size}')
        os.makedirs(exp_dir, exist_ok=True)
        log_dir = os.path.join(exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_dir)
        print(f'\n实验目录：{exp_dir}')
        print(f'日志路径：{log_dir}')
        print(f'core_model保存路径：{exp_dir}')

    # 12. 训练主循环
    print(f'\n{"="*50}')
    print(f'开始训练：起始epoch={start_epoch}，总epoch={args.epochs}')
    print(f'{"="*50}')
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # 单epoch训练（依赖engine_pretrain.py的train_one_epoch）
        train_stats = train_one_epoch(
            model=mae_model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if misc.is_main_process() and exp_dir is not None:
            sync_mae_encoder_to_core_model(
                mae_model=mae_model,
                core_model=core_model,
                is_distributed=args.distributed
            )
            latest_path = os.path.join(exp_dir, 'train_model_latest.pth')
            torch.save({
                'epoch': epoch,
                'mae_model_state_dict': mae_model.state_dict(),
                'core_model_state_dict': core_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_scaler_state_dict': loss_scaler.state_dict()
            }, latest_path)

            model_path = os.path.join(exp_dir, 'latest_model.pth')
            torch.save({
                    'model': core_model.state_dict()
                }, model_path)
            print(f'✅ 已保存core_model：{model_path}')

        # 记录日志
        if misc.is_main_process() and exp_dir is not None:
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr'],
                'core_model_save_path': model_path
            }
            log_file = os.path.join(exp_dir, 'train_log.txt')
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_stats) + '\n')
            if log_writer is not None:
                log_writer.flush()

    # 13. 训练结束
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if misc.is_main_process():
        print(f'\n{"="*50}')
        print(f'训练完成！总耗时：{total_time_str}')
        print(f'最终core_model保存路径：{os.path.join(exp_dir, "core_model_latest.pth")}')
        print(f'训练日志路径：{os.path.join(exp_dir, "train_log.txt")}')
        print(f'{"="*50}')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if misc.is_main_process() and args.output_dir is not None:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)