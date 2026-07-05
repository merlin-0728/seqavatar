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
#   bash scripts/exps_dnarendering.sh tdp
#   bash scripts/exps_dnarendering.sh tdp_semantic
#   bash scripts/exps_dnarendering.sh dif
#   bash scripts/exps_dnarendering.sh dif_sigma_rectifier
#   bash scripts/exps_dnarendering.sh dif_sigma_rectifier_v2
#   bash scripts/exps_dnarendering.sh dif_uncert_loss
#   bash scripts/exps_dnarendering.sh fix_stms
#   bash scripts/exps_dnarendering.sh acc
#   bash scripts/exps_dnarendering.sh token_fix_stms
#   bash scripts/exps_dnarendering.sh token_acc
#   bash scripts/exps_dnarendering.sh token_part
#   bash scripts/exps_dnarendering.sh token_codebook
#   bash scripts/exps_dnarendering.sh token_full
#
# 常用覆盖方式：
#   GPU_id=3 bash scripts/exps_dnarendering.sh use_part_moe
#   SEQUENCES_OVERRIDE="0007_04 0019_10" GPU_id=3 bash scripts/exps_dnarendering.sh use_part_moe
#   SEQUENCES_OVERRIDE="0044_11 0051_09 0206_04" GPU_id=2 bash scripts/exps_dnarendering.sh msti
#   SEQUENCES_OVERRIDE="0051_09 0206_04 0813_05 0007_04 0019_10" GPU_id=2 bash scripts/exps_dnarendering.sh part_moe_leg_msti
#   SEQUENCES_OVERRIDE="0044_11 0051_09 0206_04" GPU_id=2 bash scripts/exps_dnarendering.sh amc_pair
#   SEQUENCES_OVERRIDE="0044_11 0051_09 0206_04" GPU_id=2 bash scripts/exps_dnarendering.sh amc_causal
#   SEQUENCES_OVERRIDE="0044_11 0051_09 0206_04" GPU_id=2 bash scripts/exps_dnarendering.sh tdp
#   SEQUENCES_OVERRIDE="0044_11 0051_09 0206_04" GPU_id=2 bash scripts/exps_dnarendering.sh tdp_semantic

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
use_tdp=0
tdp_mode=keep_base
use_tdp_semantic_encoder=0
tdp_semantic_mode=${TDP_SEMANTIC_MODE:-gated_residual}
tdp_gate_init_bias=${TDP_GATE_INIT_BIAS:--4.0}
tdp_debug_stats=${TDP_DEBUG_STATS:-0}
tdp_debug_interval=${TDP_DEBUG_INTERVAL:-1000}
use_dif=0
dif_mode=${DIF_MODE:-peak}
dif_sigma_init=${DIF_SIGMA_INIT:--7.0}
dif_eps=${DIF_EPS:-1e-6}
dif_sigma_min=${DIF_SIGMA_MIN:-1e-4}
dif_sigma_max=${DIF_SIGMA_MAX:-0.05}
dif_residual_beta=${DIF_RESIDUAL_BETA:-1.0}
dif_residual_warmup=${DIF_RESIDUAL_WARMUP:-3000}
dif_sigma_prior=${DIF_SIGMA_PRIOR:-0.02}
dif_sigma_prior_w=${DIF_SIGMA_PRIOR_W:-0.0}
dif_uncert_loss_w=${DIF_UNCERT_LOSS_W:-0.0}
dif_uncert_s_min=${DIF_UNCERT_S_MIN:--6.0}
dif_uncert_s_max=${DIF_UNCERT_S_MAX:-3.0}
dif_debug_interval=${DIF_DEBUG_INTERVAL:-1000}
fix_stms=0
use_acc_cond=0
seq_acc_cond_dim=${SEQ_ACC_COND_DIM:-64}
use_motion_token=0
motion_token_mode=none
motion_token_num=${MOTION_TOKEN_NUM:-32}
motion_token_dim=${MOTION_TOKEN_DIM:-64}
motion_token_part_dim=${MOTION_TOKEN_PART_DIM:-16}
motion_token_acc_dim=${MOTION_TOKEN_ACC_DIM:-64}
motion_token_debug_stats=${MOTION_TOKEN_DEBUG_STATS:-0}
motion_token_debug_interval=${MOTION_TOKEN_DEBUG_INTERVAL:-1000}
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
    tdp|tdp_keep_base|tdp_keepbase)
        experiment_name=tdp
        part_moe_enabled=0
        use_tdp=1
        tdp_mode=keep_base
        final_eval_only=1
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
    dif)
        experiment_name=dif
        part_moe_enabled=0
        use_dif=1
        final_eval_only=1
        ;;
    dif_sigma_rectifier|dif_rectifier)
        experiment_name=dif_sigma_rectifier
        part_moe_enabled=0
        use_dif=1
        dif_mode=sigma_rectifier
        dif_sigma_init=${DIF_SIGMA_INIT:--4.0}
        final_eval_only=1
        ;;
    dif_sigma_rectifier_v2|dif_rectifier_v2|dif_sigma_v2)
        experiment_name=dif_sigma_rectifier_v2
        part_moe_enabled=0
        use_dif=1
        dif_mode=sigma_rectifier_v2
        dif_sigma_init=${DIF_SIGMA_INIT:--3.5}
        final_eval_only=1
        ;;
    dif_uncert_loss|dif_image_uncert|dif_uncertainty)
        experiment_name=dif_uncert_loss
        part_moe_enabled=0
        use_dif=1
        dif_mode=uncert_loss
        dif_sigma_init=${DIF_SIGMA_INIT:--3.0}
        dif_uncert_loss_w=${DIF_UNCERT_LOSS_W:-0.01}
        final_eval_only=1
        ;;
    fix_stms)
        experiment_name=fix_stms
        part_moe_enabled=0
        fix_stms=1
        final_eval_only=1
        ;;
    acc)
        experiment_name=acc
        part_moe_enabled=0
        fix_stms=1
        use_acc_cond=1
        final_eval_only=1
        ;;
    token_fix_stms)
        experiment_name=token_fix_stms
        part_moe_enabled=0
        use_motion_token=1
        motion_token_mode=fix_stms
        final_eval_only=1
        ;;
    token_acc|motion_token_acc)
        experiment_name=token_acc
        part_moe_enabled=0
        use_motion_token=1
        motion_token_mode=acc
        final_eval_only=1
        ;;
    token_part|motion_token_part)
        experiment_name=token_part
        part_moe_enabled=0
        use_motion_token=1
        motion_token_mode=part
        final_eval_only=1
        ;;
    token_codebook|motion_token_codebook|token)
        experiment_name=token_codebook
        part_moe_enabled=0
        use_motion_token=1
        motion_token_mode=codebook
        final_eval_only=1
        ;;
    token_full|token_part_acc|motion_token_full)
        experiment_name=token_full
        part_moe_enabled=0
        use_motion_token=1
        motion_token_mode=full
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
        echo "        Supported modes: orginal, msti, amc_pair, amc_causal, tdp, tdp_semantic, tdp_adapter, dif, dif_sigma_rectifier, dif_sigma_rectifier_v2, dif_uncert_loss, fix_stms, acc, token_fix_stms, token_acc, token_part, token_codebook, token_full, use_part_moe, part_moe_leg, part_moe_leg_msti, part_moe_foot, part_moe_arm"
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
TDP_LOG_DIR=${TDP_LOG_DIR:-/media/image/mxz/human/SeqAvatar/logs/TDP}
DIF_LOG_DIR=${DIF_LOG_DIR:-/media/image/mxz/human/SeqAvatar/logs/dif}
ACC_LOG_DIR=${ACC_LOG_DIR:-/media/image/mxz/human/SeqAvatar/logs/acc}
TOKEN_LOG_DIR=${TOKEN_LOG_DIR:-/media/image/mxz/human/SeqAvatar/logs/token}
GLOBAL_LOG_SUFFIX=${GLOBAL_LOG_SUFFIX:-${LOG_SUFFIX:-}}
if [ -n "$GLOBAL_LOG_SUFFIX" ] && [[ "$GLOBAL_LOG_SUFFIX" != _* ]]; then
    GLOBAL_LOG_SUFFIX="_${GLOBAL_LOG_SUFFIX}"
fi

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
if [ $((use_msti + use_amc_pair + use_amc_causal + use_tdp + use_motion_token + use_acc_cond)) -gt 1 ]; then
    echo "[ERROR] use_msti, use_amc_pair, use_amc_causal, use_tdp, use_motion_token, and use_acc_cond are mutually exclusive."
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
elif [ "$use_tdp" = "1" ]; then
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
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_DNA-Rendering_${experiment_name}${GLOBAL_LOG_SUFFIX}.log"
elif [ "$use_msti" = "1" ]; then
    GLOBAL_LOG_DIR="$MSTI_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_DNA-Rendering_msti${GLOBAL_LOG_SUFFIX}.log"
elif [ "$use_amc_pair" = "1" ] || [ "$use_amc_causal" = "1" ]; then
    GLOBAL_LOG_DIR="$AMC_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_DNA-Rendering_${experiment_name}${GLOBAL_LOG_SUFFIX}.log"
elif [ "$use_tdp" = "1" ]; then
    GLOBAL_LOG_DIR="$TDP_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_DNA-Rendering_${experiment_name}${GLOBAL_LOG_SUFFIX}.log"
elif [ "$use_dif" = "1" ]; then
    GLOBAL_LOG_DIR="$DIF_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_DNA-Rendering_${experiment_name}${GLOBAL_LOG_SUFFIX}.log"
elif [ "$fix_stms" = "1" ] || [ "$use_acc_cond" = "1" ]; then
    GLOBAL_LOG_DIR="$ACC_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_DNA-Rendering_${experiment_name}${GLOBAL_LOG_SUFFIX}.log"
elif [ "$use_motion_token" = "1" ]; then
    GLOBAL_LOG_DIR="$TOKEN_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_DNA-Rendering_${experiment_name}${GLOBAL_LOG_SUFFIX}.log"
else
    GLOBAL_LOG_DIR="$REPO_ROOT/logs"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_DNA-Rendering_${experiment_name}${GLOBAL_LOG_SUFFIX}.log"
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
echo "[INFO] Dataset: DNA-Rendering"
echo "[INFO] Mode: $MODE"
echo "[INFO] Experiment: $experiment_name"
echo "[INFO] Run time: $RUN_TIME"
echo "[INFO] Global log suffix: ${GLOBAL_LOG_SUFFIX:-<none>}"
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
echo "[INFO] USE_TDP: $use_tdp"
echo "[INFO] TDP_MODE: $tdp_mode"
echo "[INFO] USE_TDP_SEMANTIC_ENCODER: $use_tdp_semantic_encoder"
echo "[INFO] TDP_SEMANTIC_MODE: $tdp_semantic_mode"
echo "[INFO] TDP_GATE_INIT_BIAS: $tdp_gate_init_bias"
echo "[INFO] TDP_DEBUG_STATS: $tdp_debug_stats"
echo "[INFO] TDP_DEBUG_INTERVAL: $tdp_debug_interval"
echo "[INFO] USE_DIF: $use_dif"
echo "[INFO] DIF_MODE: $dif_mode"
echo "[INFO] DIF_SIGMA_INIT: $dif_sigma_init"
echo "[INFO] DIF_EPS: $dif_eps"
echo "[INFO] DIF_SIGMA_MIN: $dif_sigma_min"
echo "[INFO] DIF_SIGMA_MAX: $dif_sigma_max"
echo "[INFO] DIF_RESIDUAL_BETA: $dif_residual_beta"
echo "[INFO] DIF_RESIDUAL_WARMUP: $dif_residual_warmup"
echo "[INFO] DIF_SIGMA_PRIOR: $dif_sigma_prior"
echo "[INFO] DIF_SIGMA_PRIOR_W: $dif_sigma_prior_w"
echo "[INFO] DIF_UNCERT_LOSS_W: $dif_uncert_loss_w"
echo "[INFO] DIF_UNCERT_S_MIN: $dif_uncert_s_min"
echo "[INFO] DIF_UNCERT_S_MAX: $dif_uncert_s_max"
echo "[INFO] DIF_DEBUG_INTERVAL: $dif_debug_interval"
echo "[INFO] FIX_STMS: $fix_stms"
echo "[INFO] USE_ACC_COND: $use_acc_cond"
echo "[INFO] SEQ_ACC_COND_DIM: $seq_acc_cond_dim"
echo "[INFO] USE_MOTION_TOKEN: $use_motion_token"
echo "[INFO] MOTION_TOKEN_MODE: $motion_token_mode"
echo "[INFO] MOTION_TOKEN_NUM: $motion_token_num"
echo "[INFO] MOTION_TOKEN_DIM: $motion_token_dim"
echo "[INFO] MOTION_TOKEN_PART_DIM: $motion_token_part_dim"
echo "[INFO] MOTION_TOKEN_ACC_DIM: $motion_token_acc_dim"
echo "[INFO] MOTION_TOKEN_DEBUG_STATS: $motion_token_debug_stats"
echo "[INFO] MOTION_TOKEN_DEBUG_INTERVAL: $motion_token_debug_interval"
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

DIF_ARGS=()
if [ "$use_dif" = "1" ]; then
    DIF_ARGS=(
        --use_dif
        --dif_mode "$dif_mode"
        --dif_sigma_init "$dif_sigma_init"
        --dif_eps "$dif_eps"
        --dif_sigma_min "$dif_sigma_min"
        --dif_sigma_max "$dif_sigma_max"
        --dif_residual_beta "$dif_residual_beta"
        --dif_residual_warmup "$dif_residual_warmup"
        --dif_sigma_prior "$dif_sigma_prior"
        --dif_sigma_prior_w "$dif_sigma_prior_w"
        --dif_uncert_loss_w "$dif_uncert_loss_w"
        --dif_uncert_s_min "$dif_uncert_s_min"
        --dif_uncert_s_max "$dif_uncert_s_max"
        --dif_debug_interval "$dif_debug_interval"
    )
fi

ACC_ARGS=()
if [ "$fix_stms" = "1" ]; then
    ACC_ARGS+=(--fix_stms)
fi
if [ "$use_acc_cond" = "1" ]; then
    ACC_ARGS+=(
        --use_acc_cond
        --seq_acc_cond_dim "$seq_acc_cond_dim"
    )
fi

TOKEN_ARGS=()
if [ "$use_motion_token" = "1" ]; then
    TOKEN_ARGS=(
        --use_motion_token
        --motion_token_mode "$motion_token_mode"
        --motion_token_num "$motion_token_num"
        --motion_token_dim "$motion_token_dim"
        --motion_token_part_dim "$motion_token_part_dim"
        --motion_token_acc_dim "$motion_token_acc_dim"
        --motion_token_debug_interval "$motion_token_debug_interval"
    )
    if [ "$motion_token_debug_stats" = "1" ]; then
        TOKEN_ARGS+=(--motion_token_debug_stats)
    fi
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

    if [ "$use_msti" != "1" ] && [ "$use_amc_pair" != "1" ] && [ "$use_amc_causal" != "1" ] && [ "$use_tdp" != "1" ] && [ "$use_dif" != "1" ] && [ "$fix_stms" != "1" ] && [ "$use_acc_cond" != "1" ] && [ "$use_motion_token" != "1" ]; then
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
    if [ "$use_msti" = "1" ] || [ "$use_amc_pair" = "1" ] || [ "$use_amc_causal" = "1" ] || [ "$use_tdp" = "1" ] || [ "$use_dif" = "1" ] || [ "$fix_stms" = "1" ] || [ "$use_acc_cond" = "1" ] || [ "$use_motion_token" = "1" ]; then
        if env "${TRAIN_ENV[@]}" "$PYTHON_BIN" train.py \
            -s "$dataset_path" --eval --exp_name "$exp_name" \
            "${COMMON_TRAIN_ARGS[@]}" \
            "${MSTI_ARGS[@]}" \
            "${AMC_ARGS[@]}" \
            "${TDP_ARGS[@]}" \
            "${DIF_ARGS[@]}" \
            "${ACC_ARGS[@]}" \
            "${TOKEN_ARGS[@]}" \
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
            "${TDP_ARGS[@]}" \
            "${DIF_ARGS[@]}" \
            "${ACC_ARGS[@]}" \
            "${TOKEN_ARGS[@]}" \
            "${PART_MOE_ARGS[@]}" \
            2>&1 | tee "$model_path/logs/train_${SEQUENCE}_${experiment_name}.log"
    fi

    echo "[INFO] Evaluating on GPU $GPU_id for sequence $SEQUENCE"
    if [ "$use_msti" = "1" ] || [ "$use_amc_pair" = "1" ] || [ "$use_amc_causal" = "1" ] || [ "$use_tdp" = "1" ] || [ "$use_dif" = "1" ] || [ "$fix_stms" = "1" ] || [ "$use_acc_cond" = "1" ] || [ "$use_motion_token" = "1" ]; then
        if CUDA_VISIBLE_DEVICES=$GPU_id "$PYTHON_BIN" render.py \
            -s "$dataset_path" -m "$model_path" \
            "${COMMON_RENDER_ARGS[@]}" \
            "${MSTI_ARGS[@]}" \
            "${AMC_ARGS[@]}" \
            "${TDP_ARGS[@]}" \
            "${DIF_ARGS[@]}" \
            "${ACC_ARGS[@]}" \
            "${TOKEN_ARGS[@]}" \
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
            "${TDP_ARGS[@]}" \
            "${DIF_ARGS[@]}" \
            "${ACC_ARGS[@]}" \
            "${TOKEN_ARGS[@]}" \
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
