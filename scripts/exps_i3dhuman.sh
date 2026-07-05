#!/bin/bash
set -euo pipefail

# Usage:
#   bash scripts/exps_i3dhuman.sh
#   bash scripts/exps_i3dhuman.sh orginal
#   bash scripts/exps_i3dhuman.sh use_part_moe
#   bash scripts/exps_i3dhuman.sh part_moe_leg
#   bash scripts/exps_i3dhuman.sh part_moe_foot
#   bash scripts/exps_i3dhuman.sh part_moe_arm
#   bash scripts/exps_i3dhuman.sh tdp_semantic
#
# 常用覆盖方式：
#   GPU_id=3 bash scripts/exps_i3dhuman.sh use_part_moe
#   SEQUENCES_OVERRIDE="ID1_1 ID1_2" GPU_id=3 bash scripts/exps_i3dhuman.sh use_part_moe
#   PART_MAX_SMPL_DIST=0.05 GPU_id=3 bash scripts/exps_i3dhuman.sh part_moe_arm

# ================= 消融模式 =================
MODE=${1:-orginal}
part_label_schema=anatomy5
num_parts=5
use_tdp=0
tdp_mode=${TDP_MODE:-keep_base}
use_tdp_semantic_encoder=0
tdp_semantic_mode=${TDP_SEMANTIC_MODE:-gated_residual}
tdp_gate_init_bias=${TDP_GATE_INIT_BIAS:--4.0}
tdp_debug_stats=${TDP_DEBUG_STATS:-0}
tdp_debug_interval=${TDP_DEBUG_INTERVAL:-1000}
final_eval_only=0
case "$MODE" in
    orginal|original)
        experiment_name=orginal
        part_moe_enabled=0
        ;;
    tdp_semantic|tdp_branch_gate|tdp_semantic_gate)
        experiment_name=tdp_semantic
        part_moe_enabled=0
        use_tdp=1
        tdp_mode=keep_base
        use_tdp_semantic_encoder=1
        final_eval_only=1
        ;;
    tdp_adapter|tdp_residual|tdp_baseline_residual)
        experiment_name=tdp_adapter
        part_moe_enabled=0
        use_tdp=1
        tdp_mode=keep_base
        use_tdp_semantic_encoder=1
        tdp_semantic_mode=baseline_residual
        tdp_debug_stats=${TDP_DEBUG_STATS:-1}
        final_eval_only=1
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
        echo "        Supported modes: orginal, tdp_semantic, tdp_adapter, use_part_moe, part_moe_leg, part_moe_foot, part_moe_arm"
        exit 1
        ;;
esac

# ================= 路径和基础设置 =================
REPO_ROOT=${REPO_ROOT:-/media/image/mxz/human/SeqAvatar}
RUN_TIME=${RUN_TIME:-$(date +%Y%m%d_%H%M%S)}
GPU_id=${GPU_id:-3}
PYTHON_BIN=${PYTHON_BIN:-/media/image/mxz/.conda/envs/seqavatar/bin/python}
DATA_PATH=${DATA_PATH:-/media/image/mxz/human/SeqAvatar/I3D-Human}
PART_LOG_DIR=${PART_LOG_DIR:-/media/image/mxz/human/SeqAvatar/logs/part}
TDP_LOG_DIR=${TDP_LOG_DIR:-/media/image/mxz/human/SeqAvatar/logs/TDP}
SMPL_VERTEX_SEG_PATH=${SMPL_VERTEX_SEG_PATH:-/media/image/mxz/human/SeqAvatar/smpl_model/smpl_vert_segmentation.json}
GLOBAL_LOG_SUFFIX=${GLOBAL_LOG_SUFFIX:-${LOG_SUFFIX:-}}
if [ -n "$GLOBAL_LOG_SUFFIX" ] && [[ "$GLOBAL_LOG_SUFFIX" != _* ]]; then
    GLOBAL_LOG_SUFFIX="_${GLOBAL_LOG_SUFFIX}"
fi

cd "$REPO_ROOT"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export WANDB_PROJECT=${WANDB_PROJECT:-SeqAvatar_I3D_Human}

if [ -n "${SEQUENCES_OVERRIDE:-}" ]; then
    read -r -a SEQUENCES <<< "$SEQUENCES_OVERRIDE"
else
    SEQUENCES=("ID1_1" "ID1_2" "ID2_1" "ID3_1")
fi

SKIP_COMPLETED=${SKIP_COMPLETED:-0}
skip_load_test_cameras=${SKIP_LOAD_TEST_CAMERAS:-0}
image_data_device=${IMAGE_DATA_DEVICE:-cuda}

# ================= 训练参数 =================
# Original I3D baseline setting is 15000. Keep TDP-semantic aligned by default.
iter=${ITER:-15000}
densify_until_iter=1800

seq_len=8
seq_xyz_knn=8
time_step_num=3
max_time_step=42
minimal_time_step=24
if [ "$use_tdp" = "1" ]; then
    if [ "$tdp_mode" = "local" ]; then
        motion_cond_time_step_num=$((time_step_num * 2))
    elif [ "$tdp_mode" = "keep_base" ]; then
        motion_cond_time_step_num=$((time_step_num + 2 * time_step_num - 1))
    else
        echo "[ERROR] Unsupported TDP mode: $tdp_mode"
        exit 1
    fi
else
    motion_cond_time_step_num=$time_step_num
fi
non_rigid_mlp_depth=${NON_RIGID_MLP_DEPTH:-3}
non_rigid_mlp_width=${NON_RIGID_MLP_WIDTH:-512}

l1_loss_w=1.0
ssim_loss_w=0.1
lpips_loss_w=0.1

# I3D 官方训练终点是 15000；Part-MoE 需要在中途分层后继续训练。
# 默认 5000，可通过环境变量 PART_MOE_START_ITER 覆盖做调参。
part_moe_start_iter=${PART_MOE_START_ITER:-4000}
part_moe_warmup=${PART_MOE_WARMUP:-1000}
part_moe_global_keep=${PART_MOE_GLOBAL_KEEP:-0.1}
part_max_smpl_dist=${PART_MAX_SMPL_DIST:-0.08}

# ================= 总日志设置 =================
if [ "$part_moe_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$PART_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_I3D-Human_${experiment_name}${GLOBAL_LOG_SUFFIX}.log"
elif [ "$use_tdp" = "1" ]; then
    GLOBAL_LOG_DIR="$TDP_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_I3D-Human_${experiment_name}${GLOBAL_LOG_SUFFIX}.log"
else
    GLOBAL_LOG_DIR="$REPO_ROOT/logs"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_I3D-Human_${experiment_name}${GLOBAL_LOG_SUFFIX}.log"
fi
mkdir -p "$GLOBAL_LOG_DIR"

# train.py/render.py 在 --use_part_moe 时会自动建立自己的 part 日志。
# 这里把那些拆分日志放到临时目录，最终只保留上面的训练+测试总日志。
AUTO_PART_LOG_DIR="$PART_LOG_DIR/.auto_${RUN_TIME}_${experiment_name}${GLOBAL_LOG_SUFFIX}"
cleanup_auto_part_logs() {
    if [ "${KEEP_SPLIT_PART_LOGS:-0}" != "1" ]; then
        rm -rf "$AUTO_PART_LOG_DIR"
    fi
}
trap cleanup_auto_part_logs EXIT

exec > >(tee -a "$GLOBAL_LOG_FILE") 2>&1

echo "================================================="
echo "[INFO] Dataset: I3D-Human"
echo "[INFO] Mode: $MODE"
echo "[INFO] Experiment: $experiment_name"
echo "[INFO] Run time: $RUN_TIME"
echo "[INFO] Global log suffix: ${GLOBAL_LOG_SUFFIX:-<none>}"
echo "[INFO] GPU_id: $GPU_id"
echo "[INFO] PYTHON_BIN: $PYTHON_BIN"
echo "[INFO] DATA_PATH: $DATA_PATH"
echo "[INFO] SMPL_VERTEX_SEG_PATH: $SMPL_VERTEX_SEG_PATH"
echo "[INFO] PART_MOE_START_ITER: $part_moe_start_iter"
echo "[INFO] PART_MOE_WARMUP: $part_moe_warmup"
echo "[INFO] PART_MOE_GLOBAL_KEEP: $part_moe_global_keep"
echo "[INFO] PART_MAX_SMPL_DIST: $part_max_smpl_dist"
echo "[INFO] PART_LABEL_SCHEMA: $part_label_schema"
echo "[INFO] NUM_PARTS: $num_parts"
echo "[INFO] NON_RIGID_MLP_DEPTH: $non_rigid_mlp_depth"
echo "[INFO] NON_RIGID_MLP_WIDTH: $non_rigid_mlp_width"
echo "[INFO] USE_TDP: $use_tdp"
echo "[INFO] TDP_MODE: $tdp_mode"
echo "[INFO] USE_TDP_SEMANTIC_ENCODER: $use_tdp_semantic_encoder"
echo "[INFO] TDP_SEMANTIC_MODE: $tdp_semantic_mode"
echo "[INFO] TDP_GATE_INIT_BIAS: $tdp_gate_init_bias"
echo "[INFO] TDP_DEBUG_STATS: $tdp_debug_stats"
echo "[INFO] TDP_DEBUG_INTERVAL: $tdp_debug_interval"
echo "[INFO] ITER: $iter"
echo "[INFO] TIME_STEP_NUM(base): $time_step_num"
echo "[INFO] MOTION_COND_TIME_STEP_NUM: $motion_cond_time_step_num"
echo "[INFO] DENSIFY_UNTIL_ITER: $densify_until_iter"
echo "[INFO] FINAL_EVAL_ONLY: $final_eval_only"
echo "[INFO] SKIP_LOAD_TEST_CAMERAS: $skip_load_test_cameras"
echo "[INFO] IMAGE_DATA_DEVICE: $image_data_device"
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
    --motion_cond_time_step_num "$motion_cond_time_step_num"
    --max_time_step "$max_time_step"
    --minimal_time_step "$minimal_time_step"
    --non_rigid_mlp_depth "$non_rigid_mlp_depth"
    --non_rigid_mlp_width "$non_rigid_mlp_width"
    --l1_loss_w "$l1_loss_w"
    --ssim_loss_w "$ssim_loss_w"
    --lpips_loss_w "$lpips_loss_w"
)
if [ "$final_eval_only" = "1" ]; then
    COMMON_TRAIN_ARGS+=(--test_iterations "$iter" --save_iterations "$iter")
else
    COMMON_TRAIN_ARGS+=(--test_iterations 3000 "$part_moe_start_iter" "$iter")
    COMMON_TRAIN_ARGS+=(--save_iterations 3000 "$part_moe_start_iter" "$iter")
fi

COMMON_RENDER_ARGS=(
    --motion_offset_flag
    --smpl_type smpl
    --actor_gender neutral
    --iteration "$iter"
    --skip_train
    --seq_len "$seq_len"
    --seq_xyz_knn "$seq_xyz_knn"
    --time_step_num "$time_step_num"
    --motion_cond_time_step_num "$motion_cond_time_step_num"
    --max_time_step "$max_time_step"
    --minimal_time_step "$minimal_time_step"
    --non_rigid_mlp_depth "$non_rigid_mlp_depth"
    --non_rigid_mlp_width "$non_rigid_mlp_width"
)

TDP_ARGS=()
if [ "$use_tdp" = "1" ]; then
    TDP_ARGS=(
        --use_tdp
        --tdp_mode "$tdp_mode"
    )
    if [ "$use_tdp_semantic_encoder" = "1" ]; then
        TDP_ARGS+=(
            --use_tdp_semantic_encoder
            --tdp_semantic_mode "$tdp_semantic_mode"
            --tdp_gate_init_bias "$tdp_gate_init_bias"
            --tdp_debug_interval "$tdp_debug_interval"
        )
        if [ "$tdp_debug_stats" = "1" ]; then
            TDP_ARGS+=(--tdp_debug_stats)
        fi
    fi
fi

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
        --part_max_smpl_dist "$part_max_smpl_dist"
        --part_grouping_mode smpl_vertex_seg
        --smpl_vertex_seg_path "$SMPL_VERTEX_SEG_PATH"
        --num_parts "$num_parts"
        --part_label_schema "$part_label_schema"
        --part_log_dir "$AUTO_PART_LOG_DIR"
    )
fi

for SEQUENCE in "${SEQUENCES[@]}"; do
    exp_name=I3D-Human/${SEQUENCE}/${experiment_name}/${RUN_TIME}/
    dataset_path=${DATA_PATH}/${SEQUENCE}-train/
    model_path=output/${exp_name}

    if [ "$SKIP_COMPLETED" = "1" ]; then
        completed_dir=$(find "output/I3D-Human/${SEQUENCE}/${experiment_name}" -mindepth 1 -maxdepth 1 -type d \
            -path "*/${RUN_TIME}" -prune -o \
            -exec test -f "{}/point_cloud/iteration_${iter}/point_cloud.ply" \; \
            -exec test -f "{}/metrics/results_novelview_${iter}.json" \; \
            -exec test -f "{}/metrics/results_novelpose_${iter}.json" \; \
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

    TRAIN_ENV=(CUDA_VISIBLE_DEVICES=$GPU_id)
    RENDER_ENV=(CUDA_VISIBLE_DEVICES=$GPU_id)
    if [ "$skip_load_test_cameras" = "1" ]; then
        TRAIN_ENV+=(SEQAVATAR_SKIP_LOAD_TEST_CAMERAS=1)
    fi
    if [ -n "$image_data_device" ]; then
        TRAIN_ENV+=(SEQAVATAR_IMAGE_DATA_DEVICE="$image_data_device")
        RENDER_ENV+=(SEQAVATAR_IMAGE_DATA_DEVICE="$image_data_device")
    fi

    echo "[INFO] Training on GPU $GPU_id for sequence $SEQUENCE"
    env "${TRAIN_ENV[@]}" "$PYTHON_BIN" train.py \
        -s "$dataset_path" --eval --exp_name "$exp_name" \
        "${COMMON_TRAIN_ARGS[@]}" \
        "${TDP_ARGS[@]}" \
        "${PART_MOE_ARGS[@]}"

    echo "[INFO] Evaluating on GPU $GPU_id for sequence $SEQUENCE"
    env "${RENDER_ENV[@]}" "$PYTHON_BIN" render.py \
        -s "$dataset_path" -m "$model_path" \
        "${COMMON_RENDER_ARGS[@]}" \
        "${TDP_ARGS[@]}" \
        "${PART_MOE_ARGS[@]}"

    echo "[INFO] Finished sequence: $SEQUENCE"
done

echo "================================================="
echo "[INFO] End time: $(date)"
echo "[INFO] All sequences finished."
echo "[INFO] Global log saved to: $GLOBAL_LOG_FILE"
echo "================================================="
