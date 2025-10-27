#!/bin/bash
set -e  # 命令出错时自动退出
PLUS=""
MODEL_SIZE="medium${PLUS}"
#MODEL_SIZE="medium"
BATCH_SIZE=8
GRAD_ACCUM_STEPS=2
IMG_SIZE=320 #320,large需要336
#ENV_NAME="rfdetr"
ENV_NAME="rf-1"
GPUS="5,6"   # 显卡编号，例如 "6" 或 "0,1,2,3"
#DATASET_DIR="/home/cobot/数据服务器/自主超声数据/标注数据/病灶标注/甲状腺/jzx_coco"
QIGUAN="shen"
DATASET_DIR="/home/cobot/github_code/data/shen_coco"
DECODER_MODE=0
FREEZE_ENCODER=0
LR=0.00045
LR_ENCODER=0.0001
SCALES="P4 P5 P6" # P3-P6
USE_FDAM=0
USE_FEATAUG=2

if [ "${FREEZE_ENCODER}" -eq 1 ]; then
    FREEZE_SUFFIX="_freeze"
else
    FREEZE_SUFFIX=""
fi

DECODER_SA_TYPE="normal" # normal or diff
if [[ "${DECODER_SA_TYPE}" == "diff" ]]; then
    echo "Using Differential Attention, set BATCH_SIZE=16 (from 32)"
    BATCH_SIZE=16
    GRAD_ACCUM_STEPS=4
    DIFF="_diff"
else
    DIFF=""
fi
SELECT_MODE=2
#WEIGHT_PATH="/home/cobot/github_code/RF-DETR-Dinov3/rfdetr/checkpoint/medium-dinov3plus-coco.pth"
WEIGHT_PATH="/home/cobot/github_code/RF-DETR-Dinov3/rfdetr/checkpoint/medium-dinov3${PLUS}-randomdecoder.pth"
LOG_FILE="log/dinov3/${MODEL_SIZE}_bs:${BATCH_SIZE}*${GRAD_ACCUM_STEPS}_${QIGUAN}_LR:${LR}_${LR_ENCODER}${FREEZE_SUFFIX}_randomdecoder_${SCALES}${DIFF}_${SELECT_MODE}_dec_rope_fdam_${USE_FDAM}.log"
OUT_DIR="/home/cobot/github_code/RF-DETR-Dinov3/output/dinov3/${MODEL_SIZE}_bs:${BATCH_SIZE}*${GRAD_ACCUM_STEPS}_${QIGUAN}_LR:${LR}_${LR_ENCODER}${FREEZE_SUFFIX}_randomdecoder_${SCALES}${DIFF}_${SELECT_MODE}_dec_rope_fdam_${USE_FDAM}"
# 激活 Conda 环境
echo "→ 激活 Conda 环境: ${ENV_NAME}"
source activate "${ENV_NAME}" || {
    echo "ERROR: 无法激活 Conda 环境 ${ENV_NAME}" >&2
    exit 1
}

echo "→ 环境检查："
echo "  • Python 路径: $(which python)"
echo "  • PyTorch 版本: $(python -c "import torch; print(torch.__version__)")"

# GPU 可见性检查
CUDA_VISIBLE_DEVICES=${GPUS} python -c "import torch; \
print('  • PyTorch 检测到', torch.cuda.device_count(), '块 GPU'); \
assert torch.cuda.is_available(), 'PyTorch 未检测到 GPU 支持'"


# 启动分布式训练（传递 model_size 和 batch_size 参数给 train.py）
echo "→ 启动分布式训练 (model=${MODEL_SIZE}, batch_size=${BATCH_SIZE}, GPUs=${GPUS})，日志保存至 ${LOG_FILE}"
TORCHELASTIC_ERROR_FILE=error.json CUDA_VISIBLE_DEVICES=${GPUS} torchrun \
    --nproc_per_node=$(echo ${GPUS} | awk -F',' '{print NF}') \
    --standalone \
    train_dinov3.py \
    --model_size "${MODEL_SIZE}" \
    --weight_path "${WEIGHT_PATH}"\
    --out_dir "${OUT_DIR}"\
    --decoder_mode "${DECODER_MODE}"\
    --batch_size "${BATCH_SIZE}" \
    --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
    --img_size "${IMG_SIZE}"\
    --lr "${LR}"\
    --lr_encoder "${LR_ENCODER}"\
    --freeze_encoder "${FREEZE_ENCODER}"\
    --dataset_dir "${DATASET_DIR}" \
    --decoder_sa_type "${DECODER_SA_TYPE}"\
    --projector_scale $SCALES \
    --select_mode $SELECT_MODE\
    --use_fdam $USE_FDAM\
    --use_featAug $USE_FEATAUG\
    > "${LOG_FILE}" 2>&1

# 退出环境
conda deactivate
echo "→ 训练结束，日志文件：${LOG_FILE}"

