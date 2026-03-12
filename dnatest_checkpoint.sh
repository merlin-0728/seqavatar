#!/bin/bash

# 指定显卡
GPU_id=3

# 序列列表
SEQUENCES=(0007_04 0019_10 0044_11 0051_09 0206_04 0813_05)

# 数据集和检查点的绝对路径
data_path=/media/image/mxz/human/SeqAvatar/DNA-Rendering
checkpoint_base=/media/image/mxz/human/SeqAvatar/checkpoint/DNA-Rendering

# 核心参数 (来自原训练脚本)
iter=25000
seq_len=8
seq_xyz_knn=8
time_step_num=3
max_time_step=3
minimal_time_step=1

for SEQUENCE in "${SEQUENCES[@]}"; do
    dataset_path=$data_path/$SEQUENCE/
    model_path=$checkpoint_base/$SEQUENCE/
    
    echo "========================================================="
    echo "Evaluating on GPU $GPU_id for sequence: $SEQUENCE"
    echo "========================================================="
    
    CUDA_VISIBLE_DEVICES=$GPU_id python render.py \
        -s $dataset_path \
        -m $model_path \
        --motion_offset_flag \
        --smpl_type smplx \
        --actor_gender neutral \
        --iteration $iter \
        --skip_train \
        --seq_len $seq_len \
        --seq_xyz_knn $seq_xyz_knn \
        --time_step_num $time_step_num \
        --max_time_step $max_time_step \
        --minimal_time_step $minimal_time_step
done