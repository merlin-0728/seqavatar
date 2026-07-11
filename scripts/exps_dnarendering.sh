#!/bin/bash
set -euo pipefail

# Usage:
#   bash scripts/exps_dnarendering.sh
#   bash scripts/exps_dnarendering.sh orginal
#   bash scripts/exps_dnarendering.sh use_part_moe
#   bash scripts/exps_dnarendering.sh part_moe_leg
#   bash scripts/exps_dnarendering.sh part_moe_foot
#   bash scripts/exps_dnarendering.sh part_moe_arm
#   bash scripts/exps_dnarendering.sh state
#   bash scripts/exps_dnarendering.sh state_warm
#   bash scripts/exps_dnarendering.sh state_warm_late
#   bash scripts/exps_dnarendering.sh state_warm_half
#   bash scripts/exps_dnarendering.sh state_warm_a03
#   bash scripts/exps_dnarendering.sh state_warm_a02
#   bash scripts/exps_dnarendering.sh state_warm_a04
#   bash scripts/exps_dnarendering.sh state_gate
#
# 常用覆盖方式：
#   GPU_id=3 bash scripts/exps_dnarendering.sh use_part_moe
#   SEQUENCES_OVERRIDE="0007_04 0019_10" GPU_id=3 bash scripts/exps_dnarendering.sh use_part_moe

# ================= 消融模式 =================
MODE=${1:-orginal}
part_label_schema=anatomy5
num_parts=5
final_eval_only=0
state_enabled=0
state_warm_enabled=0
state_gate_enabled=0
state_start_iter_default=1500
state_ramp_iter_default=3000
state_max_alpha_default=1.0
state_gate_hidden_dim_default=128
state_gate_bias_default=-1.0
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
    state)
        experiment_name=state
        part_moe_enabled=0
        state_enabled=1
        final_eval_only=1
        ;;
    state_warm)
        experiment_name=state_warm
        part_moe_enabled=0
        state_warm_enabled=1
        final_eval_only=1
        ;;
    state_warm_late)
        experiment_name=state_warm_late
        part_moe_enabled=0
        state_warm_enabled=1
        final_eval_only=1
        state_start_iter_default=3000
        state_ramp_iter_default=5000
        ;;
    state_warm_half)
        experiment_name=state_warm_half
        part_moe_enabled=0
        state_warm_enabled=1
        final_eval_only=1
        state_max_alpha_default=0.5
        ;;
    state_warm_a02)
        experiment_name=state_warm_a02
        part_moe_enabled=0
        state_warm_enabled=1
        final_eval_only=1
        state_max_alpha_default=0.2
        ;;
    state_warm_a03)
        experiment_name=state_warm_a03
        part_moe_enabled=0
        state_warm_enabled=1
        final_eval_only=1
        state_max_alpha_default=0.3
        ;;
    state_warm_a04)
        experiment_name=state_warm_a04
        part_moe_enabled=0
        state_warm_enabled=1
        final_eval_only=1
        state_max_alpha_default=0.4
        ;;
    state_gate)
        experiment_name=state_gate
        part_moe_enabled=0
        state_warm_enabled=1
        state_gate_enabled=1
        final_eval_only=1
        state_max_alpha_default=0.6
        ;;
    *)
        echo "[ERROR] Unknown mode: $MODE"
        echo "        Supported modes: orginal, use_part_moe, part_moe_leg, part_moe_foot, part_moe_arm, state, state_warm, state_warm_late, state_warm_half, state_warm_a02, state_warm_a03, state_warm_a04, state_gate"
        exit 1
        ;;
esac

# ================= 路径和基础设置 =================
REPO_ROOT=${REPO_ROOT:-/media/image/mxz/human/SeqAvatar}
RUN_TIME=${RUN_TIME:-$(date +%Y%m%d_%H%M%S)}
GPU_id=${GPU_id:-2}
PYTHON_BIN=${PYTHON_BIN:-/media/image/mxz/.conda/envs/seqavatar/bin/python}
DATA_PATH=${DATA_PATH:-/media/image/mxz/human/SeqAvatar/DNA-Rendering}
PART_LOG_DIR=${PART_LOG_DIR:-/media/image/mxz/human/SeqAvatar/logs/part}

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

# ================= State warm 参数 =================
state_start_iter=${STATE_START_ITER:-$state_start_iter_default}
state_ramp_iter=${STATE_RAMP_ITER:-$state_ramp_iter_default}
state_max_alpha=${STATE_MAX_ALPHA:-$state_max_alpha_default}
state_gate_hidden_dim=${STATE_GATE_HIDDEN_DIM:-$state_gate_hidden_dim_default}
state_gate_bias=${STATE_GATE_BIAS:-$state_gate_bias_default}

test_iterations=(3000 "$part_moe_start_iter" "$iter")
save_iterations=(3000 "$part_moe_start_iter" "$iter")
if [ "$final_eval_only" = "1" ]; then
    # part_moe_leg on DNA is memory tight during intermediate full-set eval.
    # Keep label activation at part_moe_start_iter, but only evaluate/save final outputs.
    test_iterations=("$iter")
    save_iterations=("$iter")
fi

# ================= 总日志设置 =================
if [ "$state_enabled" = "1" ] || [ "$state_warm_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$REPO_ROOT/logs/state"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_DNA-Rendering_${experiment_name}_gpu${GPU_id}.log"
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
AUTO_PART_LOG_DIR="$PART_LOG_DIR/.auto_${RUN_TIME}_${experiment_name}"
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
echo "[INFO] NUM_PARTS: $num_parts"
echo "[INFO] NON_RIGID_MLP_DEPTH: $non_rigid_mlp_depth"
echo "[INFO] NON_RIGID_MLP_WIDTH: $non_rigid_mlp_width"
echo "[INFO] FINAL_EVAL_ONLY: $final_eval_only"
echo "[INFO] STATE_ENABLED: $state_enabled"
echo "[INFO] STATE_WARM_ENABLED: $state_warm_enabled"
echo "[INFO] STATE_GATE_ENABLED: $state_gate_enabled"
echo "[INFO] STATE_START_ITER: $state_start_iter"
echo "[INFO] STATE_RAMP_ITER: $state_ramp_iter"
echo "[INFO] STATE_MAX_ALPHA: $state_max_alpha"
echo "[INFO] STATE_GATE_HIDDEN_DIM: $state_gate_hidden_dim"
echo "[INFO] STATE_GATE_BIAS: $state_gate_bias"
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
fi

STATE_ARGS=()
if [ "$state_enabled" = "1" ]; then
    STATE_ARGS=(
        --use_state
    )
fi
if [ "$state_warm_enabled" = "1" ]; then
    STATE_ARGS=(
        --use_state_warm
        --state_start_iter "$state_start_iter"
        --state_ramp_iter "$state_ramp_iter"
        --state_max_alpha "$state_max_alpha"
    )
    if [ "$state_gate_enabled" = "1" ]; then
        STATE_ARGS+=(
            --use_state_gate
            --state_gate_hidden_dim "$state_gate_hidden_dim"
            --state_gate_bias "$state_gate_bias"
        )
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
        "${STATE_ARGS[@]}" \
        2>&1 | tee "$model_path/logs/train_${SEQUENCE}_${experiment_name}.log"

    echo "[INFO] Evaluating on GPU $GPU_id for sequence $SEQUENCE"
    CUDA_VISIBLE_DEVICES=$GPU_id "$PYTHON_BIN" render.py \
        -s "$dataset_path" -m "$model_path" \
        "${COMMON_RENDER_ARGS[@]}" \
        "${PART_MOE_ARGS[@]}" \
        "${STATE_ARGS[@]}" \
        2>&1 | tee "$model_path/logs/render_${SEQUENCE}_${experiment_name}.log"

    echo "[INFO] Finished sequence: $SEQUENCE"
done

echo "================================================="
echo "[INFO] End time: $(date)"
echo "[INFO] All sequences finished."
echo "[INFO] Global log saved to: $GLOBAL_LOG_FILE"
echo "================================================="
