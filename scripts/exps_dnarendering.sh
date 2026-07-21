#!/bin/bash
set -euo pipefail

# Usage:
#   bash scripts/exps_dnarendering.sh
#   bash scripts/exps_dnarendering.sh orginal
#   bash scripts/exps_dnarendering.sh use_part_moe
#   bash scripts/exps_dnarendering.sh part_moe_leg
#   bash scripts/exps_dnarendering.sh part_moe_foot
#   bash scripts/exps_dnarendering.sh part_moe_arm
#   bash scripts/exps_dnarendering.sh part_pamo
#   bash scripts/exps_dnarendering.sh part_pamo_gate_floor
#   bash scripts/exps_dnarendering.sh part_pamo_step1_only
#   bash scripts/exps_dnarendering.sh part_pamo_motion_film
#   bash scripts/exps_dnarendering.sh part_pamo_motion_film_rich
#   bash scripts/exps_dnarendering.sh part_pamo_r_fixed_0.2
#
# 常用覆盖方式：
#   GPU_id=3 bash scripts/exps_dnarendering.sh use_part_moe
#   SEQUENCES_OVERRIDE="0007_04 0019_10" GPU_id=3 bash scripts/exps_dnarendering.sh use_part_moe

# ================= 消融模式 =================
MODE=${1:-orginal}
part_label_schema=anatomy5
num_parts=5
final_eval_only=0
part_pamo_enabled=0
part_pamo_rigidity_min=0.0
part_pamo_step1_only=0
part_pamo_fixed_rigidity=-1.0
part_pamo_motion_film=0
part_pamo_motion_feat_mode=mean
part_pamo_motion_lr_mult=1.0
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
        final_eval_only=1
        ;;
    use_part_pamo|part_pamo)
        experiment_name=part_pamo
        part_moe_enabled=1
        part_pamo_enabled=1
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    use_part_pamo_gate_floor|part_pamo_gate_floor)
        experiment_name=part_pamo_gate_floor
        part_moe_enabled=1
        part_pamo_enabled=1
        part_pamo_rigidity_min=${PART_PAMO_RIGIDITY_MIN:-0.05}
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    part_pamo_gate_floor_0.2)
        experiment_name=part_pamo_gate_floor_0.2
        part_moe_enabled=1
        part_pamo_enabled=1
        part_pamo_rigidity_min=${PART_PAMO_RIGIDITY_MIN:-0.2}
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    part_pamo_step1_only)
        experiment_name=part_pamo_step1_only
        part_moe_enabled=1
        part_pamo_enabled=1
        part_pamo_step1_only=1
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    part_pamo_motion_film)
        experiment_name=part_pamo_motion_film
        part_moe_enabled=1
        part_pamo_enabled=1
        part_pamo_step1_only=1
        part_pamo_motion_film=1
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    part_pamo_motion_film_rich)
        experiment_name=part_pamo_motion_film_rich
        part_moe_enabled=1
        part_pamo_enabled=1
        part_pamo_step1_only=1
        part_pamo_motion_film=1
        part_pamo_motion_feat_mode=rich
        part_pamo_motion_lr_mult=${PART_PAMO_MOTION_LR_MULT:-2.0}
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    part_pamo_r_fixed_0.2)
        experiment_name=part_pamo_r_fixed_0.2
        part_moe_enabled=1
        part_pamo_enabled=1
        part_pamo_fixed_rigidity=0.2
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    part_pamo_r_fixed_0.5)
        experiment_name=part_pamo_r_fixed_0.5
        part_moe_enabled=1
        part_pamo_enabled=1
        part_pamo_fixed_rigidity=0.5
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    part_pamo_r_fixed_1.0)
        experiment_name=part_pamo_r_fixed_1.0
        part_moe_enabled=1
        part_pamo_enabled=1
        part_pamo_fixed_rigidity=1.0
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    use_part_moe_foot|part_moe_foot)
        experiment_name=part_moe_foot
        part_moe_enabled=1
        part_label_schema=part_moe_foot
        num_parts=7
        final_eval_only=1
        ;;
    use_part_moe_arm|part_moe_arm)
        experiment_name=part_moe_arm
        part_moe_enabled=1
        part_label_schema=part_moe_arm
        num_parts=7
        final_eval_only=1
        ;;
    *)
        echo "[ERROR] Unknown mode: $MODE"
        echo "        Supported modes: orginal, use_part_moe, part_moe_leg, part_moe_foot, part_moe_arm, part_pamo, part_pamo_gate_floor, part_pamo_gate_floor_0.2, part_pamo_step1_only, part_pamo_motion_film, part_pamo_motion_film_rich, part_pamo_r_fixed_0.2, part_pamo_r_fixed_0.5, part_pamo_r_fixed_1.0"
        exit 1
        ;;
esac

# ================= 路径和基础设置 =================
REPO_ROOT=${REPO_ROOT:-/media/image/mxz/human/SeqAvatar}
RUN_TIME=${RUN_TIME:-$(date +%Y%m%d_%H%M%S)}
GPU_id=${GPU_id:-3}
PYTHON_BIN=${PYTHON_BIN:-/media/image/mxz/.conda/envs/seqavatar/bin/python}
DATA_PATH=${DATA_PATH:-/media/image/mxz/human/SeqAvatar/DNA-Rendering}
PART_LOG_DIR=${PART_LOG_DIR:-/media/image/mxz/human/SeqAvatar/logs/part}
PAMO_LOG_DIR=${PAMO_LOG_DIR:-/media/image/mxz/human/SeqAvatar/logs/pamo}

cd "$REPO_ROOT"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export WANDB_PROJECT=${WANDB_PROJECT:-SeqAvatar_DNA_Rendering}

if [ -n "${SEQUENCES_OVERRIDE:-}" ]; then
    read -r -a SEQUENCES <<< "$SEQUENCES_OVERRIDE"
else
    SEQUENCES=("0044_11" "0051_09" "0206_04" "0813_05" "0007_04" "0019_10")
fi

SKIP_COMPLETED=${SKIP_COMPLETED:-0}
skip_load_test_cameras=${SKIP_LOAD_TEST_CAMERAS:-0}
image_data_device=${IMAGE_DATA_DEVICE:-cuda}

# ================= 训练参数 =================
iter=25000
densify_until_iter=1500

seq_len=8
seq_xyz_knn=8
time_step_num=3
max_time_step=3
minimal_time_step=1
non_rigid_mlp_depth=${NON_RIGID_MLP_DEPTH:-3}
non_rigid_mlp_width=${NON_RIGID_MLP_WIDTH:-512}

l1_loss_w=1.0
ssim_loss_w=0.01
lpips_loss_w=0.01

# ================= Part-MoE 参数 =================
part_moe_start_iter=10000
part_moe_warmup=1000
part_moe_global_keep=0.1
part_pamo_dim=${PART_PAMO_DIM:-32}
part_pamo_log_interval=${PART_PAMO_LOG_INTERVAL:-1000}

test_iterations=(3000 "$part_moe_start_iter" "$iter")
save_iterations=(3000 "$part_moe_start_iter" "$iter")
if [ "$final_eval_only" = "1" ]; then
    # part_moe_leg on DNA is memory tight during intermediate full-set eval.
    # Keep label activation at part_moe_start_iter, but only evaluate/save final outputs.
    test_iterations=("$iter")
    save_iterations=("$iter")
fi

# ================= 总日志设置 =================
if [ "$part_pamo_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$PAMO_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_DNA-Rendering_${experiment_name}.log"
elif [ "$part_moe_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$PART_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_DNA-Rendering_${experiment_name}.log"
else
    GLOBAL_LOG_DIR="$REPO_ROOT/logs"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_DNA-Rendering_${experiment_name}.log"
fi
mkdir -p "$GLOBAL_LOG_DIR"

# train.py/render.py 在 --use_part_moe 时会自动建立自己的 part 日志。
# 这里把那些拆分日志放到临时目录，最终只保留上面的训练+测试总日志。
if [ "$part_pamo_enabled" = "1" ]; then
    AUTO_PART_LOG_DIR="$PAMO_LOG_DIR/.auto_${RUN_TIME}_${experiment_name}"
else
    AUTO_PART_LOG_DIR="$PART_LOG_DIR/.auto_${RUN_TIME}_${experiment_name}"
fi
cleanup_auto_part_logs() {
    if [ "${KEEP_SPLIT_PART_LOGS:-0}" != "1" ]; then
        rm -rf "$AUTO_PART_LOG_DIR"
    fi
}
trap cleanup_auto_part_logs EXIT

exec > >(tee -a "$GLOBAL_LOG_FILE") 2>&1

echo "================================================="
echo "[INFO] Dataset: DNA-Rendering"
echo "[INFO] Mode: $MODE"
echo "[INFO] Experiment: $experiment_name"
echo "[INFO] Run time: $RUN_TIME"
echo "[INFO] GPU_id: $GPU_id"
echo "[INFO] PYTHON_BIN: $PYTHON_BIN"
echo "[INFO] DATA_PATH: $DATA_PATH"
echo "[INFO] PART_LABEL_SCHEMA: $part_label_schema"
echo "[INFO] PART_PAMO_ENABLED: $part_pamo_enabled"
echo "[INFO] PART_PAMO_DIM: $part_pamo_dim"
echo "[INFO] PART_PAMO_LOG_INTERVAL: $part_pamo_log_interval"
echo "[INFO] PART_PAMO_RIGIDITY_MIN: $part_pamo_rigidity_min"
echo "[INFO] PART_PAMO_STEP1_ONLY: $part_pamo_step1_only"
echo "[INFO] PART_PAMO_FIXED_RIGIDITY: $part_pamo_fixed_rigidity"
echo "[INFO] PART_PAMO_MOTION_FILM: $part_pamo_motion_film"
echo "[INFO] PART_PAMO_MOTION_FEAT_MODE: $part_pamo_motion_feat_mode"
echo "[INFO] PART_PAMO_MOTION_LR_MULT: $part_pamo_motion_lr_mult"
echo "[INFO] NUM_PARTS: $num_parts"
echo "[INFO] NON_RIGID_MLP_DEPTH: $non_rigid_mlp_depth"
echo "[INFO] NON_RIGID_MLP_WIDTH: $non_rigid_mlp_width"
echo "[INFO] FINAL_EVAL_ONLY: $final_eval_only"
echo "[INFO] SKIP_LOAD_TEST_CAMERAS: $skip_load_test_cameras"
echo "[INFO] IMAGE_DATA_DEVICE: $image_data_device"
echo "[INFO] TEST_ITERATIONS: ${test_iterations[*]}"
echo "[INFO] SAVE_ITERATIONS: ${save_iterations[*]}"
echo "[INFO] Sequences: ${SEQUENCES[*]}"
echo "[INFO] SKIP_COMPLETED: $SKIP_COMPLETED"
echo "[INFO] Global log file: $GLOBAL_LOG_FILE"
echo "[INFO] Start time: $(date)"
echo "================================================="

COMMON_TRAIN_ARGS=(
    --motion_offset_flag
    --smpl_type smplx
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
    --test_iterations "${test_iterations[@]}"
    --save_iterations "${save_iterations[@]}"
)

COMMON_RENDER_ARGS=(
    --motion_offset_flag
    --smpl_type smplx
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
    PART_MOE_ARGS=(
        --use_part_moe
        --part_moe_start_iter "$part_moe_start_iter"
        --part_moe_warmup "$part_moe_warmup"
        --part_moe_global_keep "$part_moe_global_keep"
        --num_parts "$num_parts"
        --part_label_schema "$part_label_schema"
        --part_log_dir "$AUTO_PART_LOG_DIR"
    )
    if [ "$part_pamo_enabled" = "1" ]; then
        PART_MOE_ARGS+=(
            --use_part_pamo
            --part_pamo_dim "$part_pamo_dim"
            --part_pamo_log_interval "$part_pamo_log_interval"
            --part_pamo_rigidity_min "$part_pamo_rigidity_min"
            --part_pamo_fixed_rigidity "$part_pamo_fixed_rigidity"
            --part_pamo_motion_feat_mode "$part_pamo_motion_feat_mode"
            --part_pamo_motion_lr_mult "$part_pamo_motion_lr_mult"
        )
        if [ "$part_pamo_motion_film" = "1" ]; then
            PART_MOE_ARGS+=(
                --part_pamo_motion_film
            )
        fi
        if [ "$part_pamo_step1_only" = "1" ]; then
            PART_MOE_ARGS+=(
                --part_pamo_step1_only
            )
        fi
    fi
fi

for SEQUENCE in "${SEQUENCES[@]}"; do
    exp_name=DNA-Rendering/${SEQUENCE}/${experiment_name}/${RUN_TIME}/
    dataset_path=${DATA_PATH}/${SEQUENCE}/
    model_path=output/${exp_name}

    if [ "$SKIP_COMPLETED" = "1" ]; then
        completed_dir=$(find "output/DNA-Rendering/${SEQUENCE}/${experiment_name}" -mindepth 1 -maxdepth 1 -type d \
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

    export WANDB_NAME="train_${SEQUENCE}_${experiment_name}_${RUN_TIME}"
    TRAIN_ENV=(CUDA_VISIBLE_DEVICES="$GPU_id")
    if [ "$skip_load_test_cameras" = "1" ]; then
        TRAIN_ENV+=(SEQAVATAR_SKIP_LOAD_TEST_CAMERAS=1)
    fi
    if [ "$image_data_device" != "cuda" ]; then
        TRAIN_ENV+=(SEQAVATAR_IMAGE_DATA_DEVICE="$image_data_device")
    fi

    echo "================================================="
    echo "[INFO] Sequence: $SEQUENCE"
    echo "[INFO] Experiment: $experiment_name"
    echo "[INFO] Dataset path: $dataset_path"
    echo "[INFO] Model path: $model_path"
    echo "================================================="

    echo "[INFO] Training on GPU $GPU_id for sequence $SEQUENCE"
    env "${TRAIN_ENV[@]}" "$PYTHON_BIN" train.py \
        -s "$dataset_path" --eval --exp_name "$exp_name" \
        "${COMMON_TRAIN_ARGS[@]}" \
        "${PART_MOE_ARGS[@]}" \
        2>&1 | tee "$model_path/logs/train_${SEQUENCE}_${experiment_name}.log"

    echo "[INFO] Evaluating on GPU $GPU_id for sequence $SEQUENCE"
    RENDER_ENV=(CUDA_VISIBLE_DEVICES="$GPU_id")
    if [ "$image_data_device" != "cuda" ]; then
        RENDER_ENV+=(SEQAVATAR_IMAGE_DATA_DEVICE="$image_data_device")
    fi
    env "${RENDER_ENV[@]}" "$PYTHON_BIN" render.py \
        -s "$dataset_path" -m "$model_path" \
        "${COMMON_RENDER_ARGS[@]}" \
        "${PART_MOE_ARGS[@]}" \
        2>&1 | tee "$model_path/logs/render_${SEQUENCE}_${experiment_name}.log"

    echo "[INFO] Finished sequence: $SEQUENCE"
done

echo "================================================="
echo "[INFO] End time: $(date)"
echo "[INFO] All sequences finished."
echo "[INFO] Global log saved to: $GLOBAL_LOG_FILE"
echo "================================================="
