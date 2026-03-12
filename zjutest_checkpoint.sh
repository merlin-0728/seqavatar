#!/bin/bash

# 指定你的 4090 显卡
GPU_id=3

# ZJU-MoCap 序列列表
SEQUENCES=(CoreView_377 CoreView_386 CoreView_387 CoreView_392 CoreView_393 CoreView_394)

# 数据集和检查点的绝对路径
data_path=/media/image/mxz/human/SeqAvatar/ZJU-MoCap
checkpoint_base=/media/image/mxz/human/SeqAvatar/checkpoint/ZJU-MoCap

# 核心参数 (来自 ZJU-MoCap 专属配置)
iter=3000
seq_len=3
seq_xyz_knn=6
time_step_num=2
max_time_step=6
minimal_time_step=3

for SEQUENCE in "${SEQUENCES[@]}"; do
    dataset_path=${data_path}/${SEQUENCE}
    model_path=${checkpoint_base}/${SEQUENCE}
    
    echo "========================================================="
    echo "Evaluating on GPU $GPU_id for ZJU-MoCap sequence: $SEQUENCE"
    echo "========================================================="
    
    CUDA_VISIBLE_DEVICES=$GPU_id python render.py \
        -s $dataset_path \
        -m $model_path \
        --motion_offset_flag \
        --smpl_type smpl \
        --actor_gender neutral \
        --iteration $iter \
        --skip_train \
        --seq_len $seq_len \
        --seq_xyz_knn $seq_xyz_knn \
        --time_step_num $time_step_num \
        --max_time_step $max_time_step \
        --minimal_time_step $minimal_time_step
done