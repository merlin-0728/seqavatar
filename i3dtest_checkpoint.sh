#!/bin/bash

# 指定你的 4090 显卡
GPU_id=3

# I3D-Human 序列列表
SEQUENCES=(ID1_1 ID1_2 ID2_1 ID3_1)

# 数据集和检查点的绝对路径
data_path=/media/image/mxz/human/SeqAvatar/I3D-Human
checkpoint_base=/media/image/mxz/human/SeqAvatar/checkpoint/I3D-Human

# 核心参数 (来自 I3D-Human 专属配置)
iter=15000
seq_len=8
seq_xyz_knn=8
time_step_num=3
max_time_step=42
minimal_time_step=24

for SEQUENCE in "${SEQUENCES[@]}"; do
    # 重点注意：I3D-Human 的数据集路径名拼接了 -train 
    dataset_path=${data_path}/${SEQUENCE}-train
    model_path=${checkpoint_base}/${SEQUENCE}
    
    echo "========================================================="
    echo "Evaluating on GPU $GPU_id for I3D-Human sequence: $SEQUENCE"
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