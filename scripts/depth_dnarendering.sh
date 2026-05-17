#!/bin/bash
set -euo pipefail

# ================= GPU 设置 =================
GPU_id=2
PYTHON_BIN=${PYTHON_BIN:-/media/image/mxz/.conda/envs/seqavatar/bin/python}
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

# ================= 序列列表 =================
SEQUENCES=(0051_09 0206_04 0813_05)

# ================= WANDB 项目名称 =================
export WANDB_PROJECT="SeqAvatar_DNA_Rendering"

# ================= 数据集路径 =================
data_path=/media/image/mxz/human/SeqAvatar/DNA-Rendering
model_root=/media/image/mxz/human/SeqAvatar/output/DNA-Rendering

# ================= 训练参数 =================
iter=25000
densify_until_iter=1500
seq_len=8
seq_xyz_knn=8
time_step_num=3
max_time_step=3
minimal_time_step=1

# ================= 损失权重 =================
l1_loss_w=1.0
ssim_loss_w=0.01
lpips_loss_w=0.01

# ================= 进入 SeqAvatar 根目录 =================
cd /media/image/mxz/human/SeqAvatar/ || exit

# ================= 循环处理序列 =================
for SEQUENCE in "${SEQUENCES[@]}"; do
    echo "Processing sequence $SEQUENCE"

    dataset_path=$data_path/$SEQUENCE/
    model_path=$model_root/$SEQUENCE/
    exp_name=DNA-Rendering/$SEQUENCE/

    # 创建输出目录
    mkdir -p "$model_path/logs"
    mkdir -p "$dataset_path/depth"

    # WANDB 动态命名
    export WANDB_NAME="train_${SEQUENCE}"

    # ------------------- 训练 -------------------
    echo "Training on GPU $GPU_id for sequence $SEQUENCE"
    CUDA_VISIBLE_DEVICES=$GPU_id "$PYTHON_BIN" train.py -s "$dataset_path" --eval --exp_name "$exp_name" \
        --motion_offset_flag --smpl_type smplx --actor_gender neutral \
        --iterations $iter --densify_until_iter $densify_until_iter \
        --seq_len $seq_len --seq_xyz_knn $seq_xyz_knn \
        --time_step_num $time_step_num --max_time_step $max_time_step --minimal_time_step $minimal_time_step \
        --l1_loss_w $l1_loss_w --ssim_loss_w $ssim_loss_w --lpips_loss_w $lpips_loss_w \
        2>&1 | tee "$model_path/logs/train_${SEQUENCE}.log"

    # ------------------- 渲染 RGB + Depth (novelview only) -------------------
    echo "Rendering RGB + Depth on GPU $GPU_id for sequence $SEQUENCE"
    CUDA_VISIBLE_DEVICES=$GPU_id "$PYTHON_BIN" depth_render.py -s "$dataset_path" -m "$model_path" \
        --motion_offset_flag --smpl_type smplx --actor_gender neutral --iteration $iter \
        --seq_len $seq_len --seq_xyz_knn $seq_xyz_knn \
        --time_step_num $time_step_num --max_time_step $max_time_step --minimal_time_step $minimal_time_step \
        --skip_train \
        2>&1 | tee "$model_path/logs/depth_render_${SEQUENCE}.log"

    echo "Finished sequence $SEQUENCE"
done

echo "All sequences finished."
