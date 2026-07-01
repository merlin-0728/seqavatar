#!/bin/bash
set -euo pipefail

# Usage:
#   bash scripts/exps_dnarendering.sh
#   bash scripts/exps_dnarendering.sh orginal
#   bash scripts/exps_dnarendering.sh use_part_moe
#   bash scripts/exps_dnarendering.sh part_moe_leg
#   bash scripts/exps_dnarendering.sh part_moe_foot
#   bash scripts/exps_dnarendering.sh part_moe_arm
#   bash scripts/exps_dnarendering.sh msti
#   bash scripts/exps_dnarendering.sh part_moe_leg_msti
#   bash scripts/exps_dnarendering.sh amc_pair
#   bash scripts/exps_dnarendering.sh amc_causal
#
# 常用覆盖方式：
#   GPU_id=3 bash scripts/exps_dnarendering.sh use_part_moe
#   SEQUENCES_OVERRIDE="0007_04 0019_10" GPU_id=3 bash scripts/exps_dnarendering.sh use_part_moe
#   SEQUENCES_OVERRIDE="0044_11 0051_09 0206_04" GPU_id=2 bash scripts/exps_dnarendering.sh msti
#   SEQUENCES_OVERRIDE="0051_09 0206_04 0813_05 0007_04 0019_10" GPU_id=2 bash scripts/exps_dnarendering.sh part_moe_leg_msti
#   SEQUENCES_OVERRIDE="0044_11 0051_09 0206_04" GPU_id=2 bash scripts/exps_dnarendering.sh amc_pair
#   SEQUENCES_OVERRIDE="0044_11 0051_09 0206_04" GPU_id=2 bash scripts/exps_dnarendering.sh amc_causal

# ================= 消融模式 =================
MODE=${1:-orginal}
part_label_schema=anatomy5
num_parts=5
final_eval_only=0
use_msti=0
msti_mode=none
msti_mid_type=real
use_amc_pair=0
amc_pair_mode=baseline_full
use_amc_causal=0
amc_causal_mode=gated_residual
case "$MODE" in
    orginal|original)
        experiment_name=orginal
        part_moe_enabled=0
        ;;
    msti|msti_lite|real_mid_msti_lite)
        experiment_name=msti_lite
        part_moe_enabled=0
        use_msti=1
        msti_mode=lite
        msti_mid_type=real
        final_eval_only=1
        ;;
    amc_pair)
        experiment_name=amc_pair
        part_moe_enabled=0
        use_amc_pair=1
        amc_pair_mode=baseline_full
        final_eval_only=1
        ;;
    amc|amc_causal|causal_amc)
        experiment_name=amc_causal
        part_moe_enabled=0
        use_amc_causal=1
        amc_causal_mode=gated_residual
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
        final_eval_only=1
        ;;
    use_part_moe_leg_msti|part_moe_leg_msti|msti_part_moe_leg)
        experiment_name=part_moe_leg_msti
        part_moe_enabled=1
        part_label_schema=part_moe_leg
        num_parts=7
        use_msti=1
        msti_mode=lite
        msti_mid_type=real
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
        echo "        Supported modes: orginal, msti, amc_pair, amc_causal, use_part_moe, part_moe_leg, part_moe_leg_msti, part_moe_foot, part_moe_arm"
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
MSTI_LOG_DIR=${MSTI_LOG_DIR:-/media/image/mxz/human/SeqAvatar/logs/msti}
AMC_LOG_DIR=${AMC_LOG_DIR:-/media/image/mxz/human/SeqAvatar/logs/AMC}

cd "$REPO_ROOT"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export WANDB_PROJECT=${WANDB_PROJECT:-SeqAvatar_DNA_Rendering}

if [ -n "${SEQUENCES_OVERRIDE:-}" ]; then
    read -r -a SEQUENCES <<< "$SEQUENCES_OVERRIDE"
else
    SEQUENCES=("0044_11")
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
if [ $((use_msti + use_amc_pair + use_amc_causal)) -gt 1 ]; then
    echo "[ERROR] use_msti, use_amc_pair, and use_amc_causal are mutually exclusive."
    exit 1
elif [ "$use_msti" = "1" ]; then
    if [ "$msti_mode" = "lite" ]; then
        motion_cond_time_step_num=$((time_step_num + 2))
    elif [ "$msti_mode" = "full" ]; then
        motion_cond_time_step_num=$((time_step_num * 3))
    else
        echo "[ERROR] Unsupported MSTI mode: $msti_mode"
        exit 1
    fi
elif [ "$use_amc_pair" = "1" ]; then
    motion_cond_time_step_num=$((time_step_num + time_step_num * (time_step_num - 1) / 2))
elif [ "$use_amc_causal" = "1" ]; then
    motion_cond_time_step_num=$time_step_num
else
    motion_cond_time_step_num=$time_step_num
fi
amc_causal_window=${AMC_CAUSAL_WINDOW:-3}
amc_motion_gate_alpha=${AMC_MOTION_GATE_ALPHA:-1.0}
amc_motion_gate_temp=${AMC_MOTION_GATE_TEMP:-0.5}
non_rigid_mlp_depth=${NON_RIGID_MLP_DEPTH:-3}
non_rigid_mlp_width=${NON_RIGID_MLP_WIDTH:-512}

l1_loss_w=1.0
ssim_loss_w=0.01
lpips_loss_w=0.01

# ================= Part-MoE 参数 =================
part_moe_start_iter=10000
part_moe_warmup=1000
part_moe_global_keep=0.1

test_iterations=(3000 "$part_moe_start_iter" "$iter")
save_iterations=(3000 "$part_moe_start_iter" "$iter")
if [ "$final_eval_only" = "1" ]; then
    # part_moe_leg on DNA is memory tight during intermediate full-set eval.
    # Keep label activation at part_moe_start_iter, but only evaluate/save final outputs.
    test_iterations=("$iter")
    save_iterations=("$iter")
fi

# ================= 总日志设置 =================
if [ "$part_moe_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$PART_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_DNA-Rendering_${experiment_name}.log"
elif [ "$use_msti" = "1" ]; then
    GLOBAL_LOG_DIR="$MSTI_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_DNA-Rendering_msti.log"
elif [ "$use_amc_pair" = "1" ] || [ "$use_amc_causal" = "1" ]; then
    GLOBAL_LOG_DIR="$AMC_LOG_DIR"
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
echo "[INFO] USE_MSTI: $use_msti"
echo "[INFO] MSTI_MODE: $msti_mode"
echo "[INFO] MSTI_MID_TYPE: $msti_mid_type"
echo "[INFO] USE_AMC_PAIR: $use_amc_pair"
echo "[INFO] AMC_PAIR_MODE: $amc_pair_mode"
echo "[INFO] USE_AMC_CAUSAL: $use_amc_causal"
echo "[INFO] AMC_CAUSAL_MODE: $amc_causal_mode"
echo "[INFO] AMC_CAUSAL_WINDOW: $amc_causal_window"
echo "[INFO] AMC_MOTION_GATE_ALPHA: $amc_motion_gate_alpha"
echo "[INFO] AMC_MOTION_GATE_TEMP: $amc_motion_gate_temp"
echo "[INFO] TIME_STEP_NUM(base): $time_step_num"
echo "[INFO] MOTION_COND_TIME_STEP_NUM: $motion_cond_time_step_num"
echo "[INFO] DENSIFY_UNTIL_ITER: $densify_until_iter"
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
    --motion_cond_time_step_num "$motion_cond_time_step_num"
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
    --motion_cond_time_step_num "$motion_cond_time_step_num"
    --max_time_step "$max_time_step"
    --minimal_time_step "$minimal_time_step"
    --non_rigid_mlp_depth "$non_rigid_mlp_depth"
    --non_rigid_mlp_width "$non_rigid_mlp_width"
)

MSTI_ARGS=()
if [ "$use_msti" = "1" ]; then
    MSTI_ARGS=(
        --use_msti
        --msti_mode "$msti_mode"
        --msti_mid_type "$msti_mid_type"
    )
fi

AMC_ARGS=()
if [ "$use_amc_pair" = "1" ]; then
    AMC_ARGS=(
        --use_amc_pair
        --amc_pair_mode "$amc_pair_mode"
    )
elif [ "$use_amc_causal" = "1" ]; then
    AMC_ARGS=(
        --use_amc_causal
        --amc_causal_mode "$amc_causal_mode"
        --amc_causal_window "$amc_causal_window"
        --amc_motion_gate_alpha "$amc_motion_gate_alpha"
        --amc_motion_gate_temp "$amc_motion_gate_temp"
    )
fi

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

    if [ "$use_msti" != "1" ] && [ "$use_amc_pair" != "1" ]; then
        mkdir -p "$model_path/logs"
    fi

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
    if [ "$use_msti" = "1" ] || [ "$use_amc_pair" = "1" ] || [ "$use_amc_causal" = "1" ]; then
        if env "${TRAIN_ENV[@]}" "$PYTHON_BIN" train.py \
            -s "$dataset_path" --eval --exp_name "$exp_name" \
            "${COMMON_TRAIN_ARGS[@]}" \
            "${MSTI_ARGS[@]}" \
            "${AMC_ARGS[@]}" \
            "${PART_MOE_ARGS[@]}"
        then
            :
        else
            train_status=$?
            echo "[WARN] Training failed for sequence $SEQUENCE with status $train_status."
            if [ "$use_amc_pair" = "1" ] || [ "$use_amc_causal" = "1" ]; then
                echo "[WARN] AMC mode: skip failed sequence and continue."
                continue
            fi
            exit "$train_status"
        fi
    else
        env "${TRAIN_ENV[@]}" "$PYTHON_BIN" train.py \
            -s "$dataset_path" --eval --exp_name "$exp_name" \
            "${COMMON_TRAIN_ARGS[@]}" \
            "${MSTI_ARGS[@]}" \
            "${AMC_ARGS[@]}" \
            "${PART_MOE_ARGS[@]}" \
            2>&1 | tee "$model_path/logs/train_${SEQUENCE}_${experiment_name}.log"
    fi

    echo "[INFO] Evaluating on GPU $GPU_id for sequence $SEQUENCE"
    if [ "$use_msti" = "1" ] || [ "$use_amc_pair" = "1" ] || [ "$use_amc_causal" = "1" ]; then
        if CUDA_VISIBLE_DEVICES=$GPU_id "$PYTHON_BIN" render.py \
            -s "$dataset_path" -m "$model_path" \
            "${COMMON_RENDER_ARGS[@]}" \
            "${MSTI_ARGS[@]}" \
            "${AMC_ARGS[@]}" \
            "${PART_MOE_ARGS[@]}"
        then
            :
        else
            render_status=$?
            echo "[WARN] Render failed for sequence $SEQUENCE with status $render_status."
            if [ "$use_amc_pair" = "1" ] || [ "$use_amc_causal" = "1" ]; then
                echo "[WARN] AMC mode: skip failed render and continue."
                continue
            fi
            exit "$render_status"
        fi
    else
        CUDA_VISIBLE_DEVICES=$GPU_id "$PYTHON_BIN" render.py \
            -s "$dataset_path" -m "$model_path" \
            "${COMMON_RENDER_ARGS[@]}" \
            "${MSTI_ARGS[@]}" \
            "${AMC_ARGS[@]}" \
            "${PART_MOE_ARGS[@]}" \
            2>&1 | tee "$model_path/logs/render_${SEQUENCE}_${experiment_name}.log"
    fi

    echo "[INFO] Finished sequence: $SEQUENCE"
done

echo "================================================="
echo "[INFO] End time: $(date)"
echo "[INFO] All sequences finished."
echo "[INFO] Global log saved to: $GLOBAL_LOG_FILE"
echo "================================================="
