#!/bin/bash
set -euo pipefail

# ================= 总日志设置 =================
RUN_TIME=$(date +%Y%m%d_%H%M%S)
GLOBAL_LOG_DIR="logs"
mkdir -p "$GLOBAL_LOG_DIR"
GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/run_${RUN_TIME}.log"

# 让整个脚本的所有输出同时显示在终端并保存到总日志
exec > >(tee -a "$GLOBAL_LOG_FILE") 2>&1

echo "================================================="
echo "[INFO] Global log file: $GLOBAL_LOG_FILE"
echo "[INFO] Start time: $(date)"
echo "================================================="

# ================= 基础设置 =================
GPU_id=${GPU_id:-3}
PYTHON_BIN=${PYTHON_BIN:-python}
if [ -n "${SEQUENCES_OVERRIDE:-}" ]; then
    read -r -a SEQUENCES <<< "$SEQUENCES_OVERRIDE"
else
    SEQUENCES=("0051_09" "0813_05")
fi
SKIP_COMPLETED=${SKIP_COMPLETED:-0}

# 改成你的 DNA-Rendering 路径；也可以运行时用 DATA_PATH=... 覆盖
data_path=${DATA_PATH:-/media/image/mxz/human/SeqAvatar/DNA-Rendering}

iter=25000
densify_until_iter=1500

seq_len=8
seq_xyz_knn=8
time_step_num=3
max_time_step=3
minimal_time_step=1

l1_loss_w=1.0
ssim_loss_w=0.01
lpips_loss_w=0.01

# ---------------- 深度相关参数 ----------------
lambda_depth=0.01
depth_loss_start=15000
depth_loss_end=20000
vggt_conf_threshold=0.5
depth_split_start=${DEPTH_SPLIT_START:-11000}
depth_split_end=${DEPTH_SPLIT_END:-12000}

depth_split_threshold=0.05
depth_split_interval=25
depth_split_count=4
depth_split_scale=0.5
max_depth_split_points_per_frame=8
max_depth_split_points=65000
max_depth_split_new_points=960
max_split_extra_new_points=960

# ---------------- 消融实验设置 ----------------
# 用法：
# bash scripts/exps_dnarendering.sh depth_loss_and_split
# bash scripts/exps_dnarendering.sh depth_loss_only
# bash scripts/exps_dnarendering.sh depth_loss_high_conf_only
# bash scripts/exps_dnarendering.sh split_only
# bash scripts/exps_dnarendering.sh split_extra
# bash scripts/exps_dnarendering.sh no_depth_no_split

ablation_name=${1:-depth_loss_and_split}

echo "[INFO] Ablation setting: $ablation_name"
echo "[INFO] GPU_id: $GPU_id"
echo "[INFO] PYTHON_BIN: $PYTHON_BIN"
echo "[INFO] DATA_PATH: $data_path"
echo "[INFO] Sequences: ${SEQUENCES[*]}"
echo "[INFO] SKIP_COMPLETED: $SKIP_COMPLETED"

# 默认各类开关都关闭，需要哪个消融就在下面分支里打开
# depth_loss_high_conf_flag 对应 train.py 里的 --depth_loss_high_conf_only
# split_extra_flag 对应 train.py 里的 --enable_split_extra
if [ "$ablation_name" = "depth_loss_and_split" ]; then
    depth_loss_flag="--enable_depth_loss"
    gaussian_split_flag="--enable_gaussian_split"
    depth_loss_high_conf_flag=""
    split_extra_flag=""
elif [ "$ablation_name" = "depth_loss_only" ]; then
    depth_loss_flag="--enable_depth_loss"
    gaussian_split_flag=""
    depth_loss_high_conf_flag=""
    split_extra_flag=""
elif [ "$ablation_name" = "depth_loss_high_conf_only" ]; then
    depth_loss_flag="--enable_depth_loss"
    gaussian_split_flag=""
    depth_loss_high_conf_flag="--depth_loss_high_conf_only"
    split_extra_flag=""
elif [ "$ablation_name" = "split_only" ]; then
    depth_loss_flag=""
    gaussian_split_flag="--enable_gaussian_split"
    depth_loss_high_conf_flag=""
    split_extra_flag=""
elif [ "$ablation_name" = "split_extra" ]; then
    depth_loss_flag=""
    gaussian_split_flag=""
    depth_loss_high_conf_flag=""
    split_extra_flag="--enable_split_extra"
elif [ "$ablation_name" = "no_depth_no_split" ]; then
    depth_loss_flag=""
    gaussian_split_flag=""
    depth_loss_high_conf_flag=""
    split_extra_flag=""
else
    echo "[ERROR] Unknown ablation_name: $ablation_name"
    echo "Available options:"
    echo "  depth_loss_and_split"
    echo "  depth_loss_only"
    echo "  depth_loss_high_conf_only"
    echo "  split_only"
    echo "  split_extra"
    echo "  no_depth_no_split"
    exit 1
fi

for SEQUENCE in "${SEQUENCES[@]}"; do
    exp_name=DNA-Rendering/${SEQUENCE}/${ablation_name}/${RUN_TIME}/
    dataset_path=${data_path}/${SEQUENCE}/
    model_path=output/${exp_name}/

    if [ "$SKIP_COMPLETED" = "1" ]; then
        completed_dir=$(find "output/DNA-Rendering/${SEQUENCE}/${ablation_name}" -mindepth 1 -maxdepth 1 -type d \
            -path "*/${RUN_TIME}" -prune -o \
            -exec test -f "{}/point_cloud/iteration_${iter}/point_cloud.ply" \; \
            -exec test -f "{}/metrics/results_novelview_${iter}.json" \; \
            -print 2>/dev/null | sort | tail -1 || true)
        if [ -n "$completed_dir" ]; then
            echo "[INFO] Skip completed sequence: $SEQUENCE"
            echo "[INFO] Completed output: $completed_dir"
            continue
        fi
    fi

    mkdir -p "$model_path/logs"

    smc_file="${dataset_path}/${SEQUENCE}.smc"
    if [ ! -f "$smc_file" ]; then
        echo "[ERROR] Missing SMC file: $smc_file"
        exit 1
    fi

    echo "================================================="
    echo "[INFO] Sequence: $SEQUENCE"
    echo "[INFO] Ablation: $ablation_name"
    echo "[INFO] Dataset path: $dataset_path"
    echo "[INFO] Model path: $model_path"
    echo "================================================="

    # Train
    echo "Training on GPU $GPU_id for sequence $SEQUENCE"
    CUDA_VISIBLE_DEVICES=$GPU_id "$PYTHON_BIN" train.py -s "$dataset_path" --eval --exp_name "$exp_name" \
        --motion_offset_flag --smpl_type smplx --actor_gender neutral \
        --iterations $iter --densify_until_iter $densify_until_iter \
        --seq_len $seq_len --seq_xyz_knn $seq_xyz_knn \
        --time_step_num $time_step_num --max_time_step $max_time_step --minimal_time_step $minimal_time_step \
        --l1_loss_w $l1_loss_w --ssim_loss_w $ssim_loss_w --lpips_loss_w $lpips_loss_w \
        $depth_loss_flag $gaussian_split_flag $depth_loss_high_conf_flag $split_extra_flag \
        --lambda_depth $lambda_depth \
        --depth_loss_start $depth_loss_start \
        --depth_loss_end $depth_loss_end \
        --vggt_conf_threshold $vggt_conf_threshold \
        --depth_split_start $depth_split_start \
        --depth_split_end $depth_split_end \
        --depth_split_threshold $depth_split_threshold \
        --depth_split_interval $depth_split_interval \
        --depth_split_count $depth_split_count \
        --depth_split_scale $depth_split_scale \
        --max_depth_split_points_per_frame $max_depth_split_points_per_frame \
        --max_depth_split_points $max_depth_split_points \
        --max_depth_split_new_points $max_depth_split_new_points \
        --max_split_extra_new_points $max_split_extra_new_points \
        --split_extra_start $depth_split_start \
        --split_extra_end $depth_split_end \
        --split_extra_interval $depth_split_interval \
        --split_extra_max_points $max_depth_split_points \
        2>&1 | tee "$model_path/logs/train_${SEQUENCE}_${ablation_name}.log"

    # Evaluation
    echo "Evaluating on GPU $GPU_id for sequence $SEQUENCE"
    CUDA_VISIBLE_DEVICES=$GPU_id "$PYTHON_BIN" render.py -s "$dataset_path" -m "$model_path" \
        --motion_offset_flag --smpl_type smplx --actor_gender neutral --iteration $iter --skip_train \
        --seq_len $seq_len --seq_xyz_knn $seq_xyz_knn \
        --time_step_num $time_step_num --max_time_step $max_time_step --minimal_time_step $minimal_time_step \
        2>&1 | tee "$model_path/logs/render_${SEQUENCE}_${ablation_name}.log"

    echo "[INFO] Finished sequence: $SEQUENCE"
done

echo "================================================="
echo "[INFO] End time: $(date)"
echo "[INFO] All sequences finished."
echo "[INFO] Global log saved to: $GLOBAL_LOG_FILE"
echo "================================================="
