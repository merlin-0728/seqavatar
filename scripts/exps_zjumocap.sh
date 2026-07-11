#!/bin/bash
set -euo pipefail

# Usage:
#   bash scripts/exps_zjumocap.sh
#   bash scripts/exps_zjumocap.sh orginal
#   bash scripts/exps_zjumocap.sh use_part_moe
#   bash scripts/exps_zjumocap.sh part_moe_leg
#   bash scripts/exps_zjumocap.sh part_moe_foot
#   bash scripts/exps_zjumocap.sh part_moe_arm
#
# 常用覆盖方式：
#   GPU_id=3 bash scripts/exps_zjumocap.sh use_part_moe
#   SEQUENCES_OVERRIDE="CoreView_377" GPU_id=3 bash scripts/exps_zjumocap.sh use_part_moe

# ================= 消融模式 =================
MODE=${1:-orginal}
part_label_schema=anatomy5
num_parts=5
case "$MODE" in
    orginal|original)
        experiment_name=orginal
        part_moe_enabled=0
        ;;
    use_part_moe|part_moe)
        experiment_name=part_moe
        part_moe_enabled=1
        ;;
    use_part_moe_leg|part_moe_leg)
        experiment_name=part_moe_leg
        part_moe_enabled=1
        part_label_schema=part_moe_leg
        num_parts=7
        ;;
    use_part_moe_foot|part_moe_foot)
        experiment_name=part_moe_foot
        part_moe_enabled=1
        part_label_schema=part_moe_foot
        num_parts=7
        ;;
    use_part_moe_arm|part_moe_arm)
        experiment_name=part_moe_arm
        part_moe_enabled=1
        part_label_schema=part_moe_arm
        num_parts=7
        ;;
    *)
        echo "[ERROR] Unknown mode: $MODE"
        echo "        Supported modes: orginal, use_part_moe, part_moe_leg, part_moe_foot, part_moe_arm"
        exit 1
        ;;
esac

# ================= 路径和基础设置 =================
REPO_ROOT=${REPO_ROOT:-/media/image/mxz/human/SeqAvatar}
RUN_TIME=${RUN_TIME:-$(date +%Y%m%d_%H%M%S)}
GPU_id=${GPU_id:-3}
PYTHON_BIN=${PYTHON_BIN:-/media/image/mxz/.conda/envs/seqavatar/bin/python}
DATA_PATH=${DATA_PATH:-/media/image/mxz/human/SeqAvatar/ZJU-MoCap}
PART_LOG_DIR=${PART_LOG_DIR:-/media/image/mxz/human/SeqAvatar/logs/part}
SMPL_VERTEX_SEG_PATH=${SMPL_VERTEX_SEG_PATH:-/media/image/mxz/human/SeqAvatar/smpl_model/smpl_vert_segmentation.json}

cd "$REPO_ROOT"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export WANDB_PROJECT=${WANDB_PROJECT:-SeqAvatar_ZJU_MoCap}

if [ -n "${SEQUENCES_OVERRIDE:-}" ]; then
    read -r -a SEQUENCES <<< "$SEQUENCES_OVERRIDE"
else
    SEQUENCES=("CoreView_377" "CoreView_386" "CoreView_387" "CoreView_392" "CoreView_393" "CoreView_394")
fi

SKIP_COMPLETED=${SKIP_COMPLETED:-0}

# ================= 训练参数 =================
iter=${ITERATIONS:-3000}
base_densify_until_iter=${DENSIFY_UNTIL_ITER:-1200}

seq_len=${SEQ_LEN:-3}
seq_xyz_knn=${SEQ_XYZ_KNN:-6}
time_step_num=${TIME_STEP_NUM:-2}
max_time_step=${MAX_TIME_STEP:-6}
minimal_time_step=${MINIMAL_TIME_STEP:-3}
non_rigid_mlp_depth=${NON_RIGID_MLP_DEPTH:-3}
non_rigid_mlp_width=${NON_RIGID_MLP_WIDTH:-512}

l1_loss_w=${L1_LOSS_W:-1.0}
ssim_loss_w=${SSIM_LOSS_W:-0.1}
lpips_loss_w=${LPIPS_LOSS_W:-0.1}

# ZJU 总训练 3000 步；Part-MoE 在 1000 步分层，并要求分层后不再增密。
part_moe_start_iter=${PART_MOE_START_ITER:-1000}
part_moe_warmup=${PART_MOE_WARMUP:-500}
part_moe_global_keep=${PART_MOE_GLOBAL_KEEP:-0.1}

if [ "$part_moe_enabled" = "1" ]; then
    densify_until_iter=${PART_MOE_DENSIFY_UNTIL_ITER:-$part_moe_start_iter}
else
    densify_until_iter=$base_densify_until_iter
fi

if [ "$part_moe_enabled" = "1" ] && [ "$densify_until_iter" -gt "$part_moe_start_iter" ]; then
    echo "[ERROR] Part-MoE requires densify_until_iter <= part_moe_start_iter."
    echo "        densify_until_iter=$densify_until_iter part_moe_start_iter=$part_moe_start_iter"
    exit 1
fi

# ================= 总日志设置 =================
if [ "$part_moe_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$PART_LOG_DIR"
else
    GLOBAL_LOG_DIR="$REPO_ROOT/logs"
fi
GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_ZJU-MoCap_${experiment_name}.log"
mkdir -p "$GLOBAL_LOG_DIR"

# train.py/render.py 在 --use_part_moe 时会自动建立自己的 part 日志。
# 这里把那些拆分日志放到临时目录，最终保留上面的训练+测试总日志。
AUTO_PART_LOG_DIR="$PART_LOG_DIR/.auto_${RUN_TIME}_${experiment_name}"
cleanup_auto_part_logs() {
    if [ "${KEEP_SPLIT_PART_LOGS:-0}" != "1" ]; then
        rm -rf "$AUTO_PART_LOG_DIR"
    fi
}
trap cleanup_auto_part_logs EXIT

exec > >(tee -a "$GLOBAL_LOG_FILE") 2>&1

echo "================================================="
echo "[INFO] Dataset: ZJU-MoCap"
echo "[INFO] Mode: $MODE"
echo "[INFO] Experiment: $experiment_name"
echo "[INFO] Run time: $RUN_TIME"
echo "[INFO] GPU_id: $GPU_id"
echo "[INFO] PYTHON_BIN: $PYTHON_BIN"
echo "[INFO] DATA_PATH: $DATA_PATH"
echo "[INFO] SMPL_VERTEX_SEG_PATH: $SMPL_VERTEX_SEG_PATH"
echo "[INFO] Iterations: $iter"
echo "[INFO] Densify until iter: $densify_until_iter"
echo "[INFO] PART_MOE_START_ITER: $part_moe_start_iter"
echo "[INFO] PART_MOE_WARMUP: $part_moe_warmup"
echo "[INFO] PART_MOE_GLOBAL_KEEP: $part_moe_global_keep"
echo "[INFO] PART_LABEL_SCHEMA: $part_label_schema"
echo "[INFO] NUM_PARTS: $num_parts"
echo "[INFO] NON_RIGID_MLP_DEPTH: $non_rigid_mlp_depth"
echo "[INFO] NON_RIGID_MLP_WIDTH: $non_rigid_mlp_width"
echo "[INFO] Sequences: ${SEQUENCES[*]}"
echo "[INFO] SKIP_COMPLETED: $SKIP_COMPLETED"
echo "[INFO] Global log file: $GLOBAL_LOG_FILE"
echo "[INFO] Start time: $(date)"
echo "================================================="

COMMON_TRAIN_ARGS=(
    --motion_offset_flag
    --smpl_type smpl
    --actor_gender neutral
    --iterations "$iter"
    --densify_until_iter "$densify_until_iter"
    --seq_len "$seq_len"
    --seq_xyz_knn "$seq_xyz_knn"
    --time_step_num "$time_step_num"
    --max_time_step "$max_time_step"
    --minimal_time_step "$minimal_time_step"
    --non_rigid_mlp_depth "$non_rigid_mlp_depth"
    --non_rigid_mlp_width "$non_rigid_mlp_width"
    --l1_loss_w "$l1_loss_w"
    --ssim_loss_w "$ssim_loss_w"
    --lpips_loss_w "$lpips_loss_w"
    --test_iterations "$part_moe_start_iter" "$iter"
    --save_iterations "$part_moe_start_iter" "$iter"
)

COMMON_RENDER_ARGS=(
    --motion_offset_flag
    --smpl_type smpl
    --actor_gender neutral
    --iteration "$iter"
    --skip_train
    --seq_len "$seq_len"
    --seq_xyz_knn "$seq_xyz_knn"
    --time_step_num "$time_step_num"
    --max_time_step "$max_time_step"
    --minimal_time_step "$minimal_time_step"
    --non_rigid_mlp_depth "$non_rigid_mlp_depth"
    --non_rigid_mlp_width "$non_rigid_mlp_width"
)

PART_MOE_ARGS=()
if [ "$part_moe_enabled" = "1" ]; then
    if [ ! -f "$SMPL_VERTEX_SEG_PATH" ]; then
        echo "[ERROR] Missing SMPL vertex segmentation file: $SMPL_VERTEX_SEG_PATH"
        exit 1
    fi
    PART_MOE_ARGS=(
        --use_part_moe
        --part_moe_start_iter "$part_moe_start_iter"
        --part_moe_warmup "$part_moe_warmup"
        --part_moe_global_keep "$part_moe_global_keep"
        --part_grouping_mode smpl_vertex_seg
        --smpl_vertex_seg_path "$SMPL_VERTEX_SEG_PATH"
        --num_parts "$num_parts"
        --part_label_schema "$part_label_schema"
        --part_log_dir "$AUTO_PART_LOG_DIR"
    )
fi

for SEQUENCE in "${SEQUENCES[@]}"; do
    exp_name=ZJU-MoCap/${SEQUENCE}/${experiment_name}/${RUN_TIME}/
    dataset_path=${DATA_PATH}/${SEQUENCE}/
    model_path=output/${exp_name}

    if [ "$SKIP_COMPLETED" = "1" ]; then
        completed_dir=$(find "output/ZJU-MoCap/${SEQUENCE}/${experiment_name}" -mindepth 1 -maxdepth 1 -type d \
            -path "*/${RUN_TIME}" -prune -o \
            -exec test -f "{}/point_cloud/iteration_${iter}/point_cloud.ply" \; \
            -exec test -f "{}/metrics/results_test_${iter}.json" \; \
            -print 2>/dev/null | sort | tail -1 || true)
        if [ -n "$completed_dir" ]; then
            echo "[INFO] Skip completed sequence: $SEQUENCE"
            echo "[INFO] Completed output: $completed_dir"
            continue
        fi
    fi

    if [ ! -d "$dataset_path" ]; then
        echo "[ERROR] Missing dataset directory: $dataset_path"
        exit 1
    fi

    export WANDB_NAME="train_${SEQUENCE}_${experiment_name}_${RUN_TIME}"

    echo "================================================="
    echo "[INFO] Sequence: $SEQUENCE"
    echo "[INFO] Experiment: $experiment_name"
    echo "[INFO] Dataset path: $dataset_path"
    echo "[INFO] Model path: $model_path"
    echo "================================================="

    echo "[INFO] Training on GPU $GPU_id for sequence $SEQUENCE"
    CUDA_VISIBLE_DEVICES=$GPU_id "$PYTHON_BIN" train.py \
        -s "$dataset_path" --eval --exp_name "$exp_name" \
        "${COMMON_TRAIN_ARGS[@]}" \
        "${PART_MOE_ARGS[@]}"

    echo "[INFO] Evaluating on GPU $GPU_id for sequence $SEQUENCE"
    CUDA_VISIBLE_DEVICES=$GPU_id "$PYTHON_BIN" render.py \
        -s "$dataset_path" -m "$model_path" \
        "${COMMON_RENDER_ARGS[@]}" \
        "${PART_MOE_ARGS[@]}"

    echo "[INFO] Finished sequence: $SEQUENCE"
done

echo "================================================="
echo "[INFO] End time: $(date)"
echo "[INFO] All sequences finished."
echo "[INFO] Global log saved to: $GLOBAL_LOG_FILE"
echo "================================================="
