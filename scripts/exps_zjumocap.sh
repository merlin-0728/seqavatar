#!/bin/bash
set -euo pipefail

# Usage:
#   bash scripts/exps_zjumocap.sh
#   bash scripts/exps_zjumocap.sh original
#   bash scripts/exps_zjumocap.sh part_moe_leg
#   bash scripts/exps_zjumocap.sh part_budget
#   bash scripts/exps_zjumocap.sh tri
#   bash scripts/exps_zjumocap.sh tri_part
#   bash scripts/exps_zjumocap.sh tri_gate
#
# 常用覆盖方式：
#   GPU_id=3 bash scripts/exps_zjumocap.sh part_moe_leg
#   GPU_id=3 bash scripts/exps_zjumocap.sh part_budget
#   GPU_id=3 bash scripts/exps_zjumocap.sh tri
#   GPU_id=3 bash scripts/exps_zjumocap.sh tri_part
#   GPU_id=3 bash scripts/exps_zjumocap.sh tri_gate
#   SEQUENCES_OVERRIDE="CoreView_377" GPU_id=3 bash scripts/exps_zjumocap.sh part_moe_leg
#   TRI_PLANE_DIM=32 TRI_PLANE_RES=64 TRI_PLANE_EXTENT=1.0 GPU_id=3 bash scripts/exps_zjumocap.sh tri
#
# tri 调参只改 TRI_PLANE_* 等命令行环境变量；实验名固定为 tri。
# 不再新建 triA/triB/triC 这类消融名称，具体参数会写入日志。

# ================= 消融模式 =================
MODE=${1:-original}
part_label_schema=anatomy5
num_parts=5
final_eval_only=0
tri_enabled=0
tri_part_enabled=0
tri_gate_enabled=0
part_budget_enabled=0
case "$MODE" in
    original)
        experiment_name=original
        part_moe_enabled=0
        ;;
    part_moe_leg)
        experiment_name=part_moe_leg
        part_moe_enabled=1
        part_label_schema=part_moe_leg
        num_parts=7
        ;;
    part_budget)
        experiment_name=part_budget
        part_moe_enabled=1
        part_budget_enabled=1
        part_label_schema=part_moe_leg
        num_parts=7
        ;;
    tri)
        experiment_name=tri
        part_moe_enabled=1
        tri_enabled=1
        part_label_schema=part_moe_leg
        num_parts=7
        ;;
    tri_part)
        experiment_name=tri_part
        part_moe_enabled=1
        tri_enabled=1
        tri_part_enabled=1
        part_label_schema=part_moe_leg
        num_parts=7
        ;;
    tri_gate)
        experiment_name=tri_gate
        part_moe_enabled=1
        tri_enabled=1
        tri_gate_enabled=1
        part_label_schema=part_moe_leg
        num_parts=7
        ;;
    *)
        echo "[ERROR] Unknown mode: $MODE"
        echo "        Supported modes: original, part_moe_leg, part_budget, tri, tri_part, tri_gate"
        exit 1
        ;;
esac

# ================= 路径和基础设置 =================
REPO_ROOT=${REPO_ROOT:-/media/image/mxz/human/SeqAvatar}
RUN_TIME=${RUN_TIME:-$(date +%Y%m%d_%H%M%S)}
GPU_id=${GPU_id:-2}
PYTHON_BIN=${PYTHON_BIN:-/media/image/mxz/.conda/envs/seqavatar/bin/python}
DATA_PATH=${DATA_PATH:-/media/image/mxz/human/SeqAvatar/ZJU-MoCap}
PART_LOG_DIR=${PART_LOG_DIR:-/media/image/mxz/human/SeqAvatar/logs/part}
PART_BUDGET_LOG_DIR=${PART_BUDGET_LOG_DIR:-/media/image/mxz/human/SeqAvatar/logs/budget}
TRI_LOG_DIR=${TRI_LOG_DIR:-/media/image/mxz/human/SeqAvatar/logs/tri}
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
tri_plane_dim=${TRI_PLANE_DIM:-32}
tri_plane_res=${TRI_PLANE_RES:-64}
tri_plane_extent=${TRI_PLANE_EXTENT:-1.0}
tri_gate_alpha=${TRI_GATE_ALPHA:-0.2}
tri_gate_init=${TRI_GATE_INIT:-0.5}
tri_gate_hidden_dim=${TRI_GATE_HIDDEN_DIM:-128}
tri_part_alpha=${TRI_PART_ALPHA:-1.0}
tri_part_motion_gain=${TRI_PART_MOTION_GAIN:-0.5}
tri_part_boundary_gain=${TRI_PART_BOUNDARY_GAIN:-0.5}
tri_part_hidden_dim=${TRI_PART_HIDDEN_DIM:-64}
tri_part_reg_w=${TRI_PART_REG_W:-0.0001}
part_budget_alpha=${PART_BUDGET_ALPHA:-1.0}
part_budget_start_iter=${PART_BUDGET_START_ITER:-$((part_moe_start_iter + 1000))}
part_budget_warmup=${PART_BUDGET_WARMUP:-1000}
part_budget_hidden_dim=${PART_BUDGET_HIDDEN_DIM:-128}
part_budget_token_dim=${PART_BUDGET_TOKEN_DIM:-32}

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

test_iterations=("$part_moe_start_iter" "$iter")
save_iterations=("$part_moe_start_iter" "$iter")
if [ "$final_eval_only" = "1" ]; then
    test_iterations=("$iter")
    save_iterations=("$iter")
fi

# ================= 总日志设置 =================
if [ "$tri_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$TRI_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_ZJU-MoCap_${experiment_name}.log"
elif [ "$part_budget_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$PART_BUDGET_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_ZJU-MoCap_${experiment_name}.log"
elif [ "$part_moe_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$PART_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_ZJU-MoCap_${experiment_name}.log"
else
    GLOBAL_LOG_DIR="$REPO_ROOT/logs"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${RUN_TIME}_ZJU-MoCap_${experiment_name}.log"
fi
mkdir -p "$GLOBAL_LOG_DIR"

# train.py/render.py 在 --use_part_moe 时会自动建立自己的 part 日志。
# 这里把那些拆分日志放到临时目录，最终保留上面的训练+测试总日志。
if [ "$tri_enabled" = "1" ]; then
    AUTO_PART_LOG_DIR="$TRI_LOG_DIR/.auto_${RUN_TIME}_${experiment_name}"
elif [ "$part_budget_enabled" = "1" ]; then
    AUTO_PART_LOG_DIR="$PART_BUDGET_LOG_DIR/.auto_${RUN_TIME}_${experiment_name}"
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
echo "[INFO] Dataset: ZJU-MoCap"
echo "[INFO] Mode: $MODE"
echo "[INFO] Experiment: $experiment_name"
echo "[INFO] Run time: $RUN_TIME"
echo "[INFO] GPU_id: $GPU_id"
echo "[INFO] PYTHON_BIN: $PYTHON_BIN"
echo "[INFO] DATA_PATH: $DATA_PATH"
echo "[INFO] SMPL_VERTEX_SEG_PATH: $SMPL_VERTEX_SEG_PATH"
echo "[INFO] TRI_LOG_DIR: $TRI_LOG_DIR"
echo "[INFO] PART_BUDGET_LOG_DIR: $PART_BUDGET_LOG_DIR"
echo "[INFO] Iterations: $iter"
echo "[INFO] Densify until iter: $densify_until_iter"
echo "[INFO] PART_MOE_START_ITER: $part_moe_start_iter"
echo "[INFO] PART_MOE_WARMUP: $part_moe_warmup"
echo "[INFO] PART_MOE_GLOBAL_KEEP: $part_moe_global_keep"
echo "[INFO] TRI_ENABLED: $tri_enabled"
echo "[INFO] TRI_PART_ENABLED: $tri_part_enabled"
echo "[INFO] TRI_GATE_ENABLED: $tri_gate_enabled"
echo "[INFO] PART_BUDGET_ENABLED: $part_budget_enabled"
echo "[INFO] PART_BUDGET_ALPHA: $part_budget_alpha"
echo "[INFO] PART_BUDGET_START_ITER: $part_budget_start_iter"
echo "[INFO] PART_BUDGET_WARMUP: $part_budget_warmup"
echo "[INFO] PART_BUDGET_HIDDEN_DIM: $part_budget_hidden_dim"
echo "[INFO] PART_BUDGET_TOKEN_DIM: $part_budget_token_dim"
echo "[INFO] TRI_PLANE_DIM: $tri_plane_dim"
echo "[INFO] TRI_PLANE_RES: $tri_plane_res"
echo "[INFO] TRI_PLANE_EXTENT: $tri_plane_extent"
echo "[INFO] TRI_GATE_ALPHA: $tri_gate_alpha"
echo "[INFO] TRI_GATE_INIT: $tri_gate_init"
echo "[INFO] TRI_GATE_HIDDEN_DIM: $tri_gate_hidden_dim"
echo "[INFO] TRI_PART_ALPHA: $tri_part_alpha"
echo "[INFO] TRI_PART_MOTION_GAIN: $tri_part_motion_gain"
echo "[INFO] TRI_PART_BOUNDARY_GAIN: $tri_part_boundary_gain"
echo "[INFO] TRI_PART_HIDDEN_DIM: $tri_part_hidden_dim"
echo "[INFO] TRI_PART_REG_W: $tri_part_reg_w"
echo "[INFO] PART_LABEL_SCHEMA: $part_label_schema"
echo "[INFO] NUM_PARTS: $num_parts"
echo "[INFO] NON_RIGID_MLP_DEPTH: $non_rigid_mlp_depth"
echo "[INFO] NON_RIGID_MLP_WIDTH: $non_rigid_mlp_width"
echo "[INFO] FINAL_EVAL_ONLY: $final_eval_only"
echo "[INFO] TEST_ITERATIONS: ${test_iterations[*]}"
echo "[INFO] SAVE_ITERATIONS: ${save_iterations[*]}"
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
    --test_iterations "${test_iterations[@]}"
    --save_iterations "${save_iterations[@]}"
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
    if [ "$tri_enabled" = "1" ]; then
        PART_MOE_ARGS+=(
            --use_tri
            --tri_plane_dim "$tri_plane_dim"
            --tri_plane_res "$tri_plane_res"
            --tri_plane_extent "$tri_plane_extent"
        )
        if [ "$tri_part_enabled" = "1" ]; then
            PART_MOE_ARGS+=(
                --use_tri_part
                --tri_part_alpha "$tri_part_alpha"
                --tri_part_motion_gain "$tri_part_motion_gain"
                --tri_part_boundary_gain "$tri_part_boundary_gain"
                --tri_part_hidden_dim "$tri_part_hidden_dim"
                --tri_part_reg_w "$tri_part_reg_w"
            )
        fi
        if [ "$tri_gate_enabled" = "1" ]; then
            PART_MOE_ARGS+=(
                --use_tri_gate
                --tri_gate_alpha "$tri_gate_alpha"
                --tri_gate_init "$tri_gate_init"
                --tri_gate_hidden_dim "$tri_gate_hidden_dim"
            )
        fi
    fi
    if [ "$part_budget_enabled" = "1" ]; then
        PART_MOE_ARGS+=(
            --use_part_budget
            --part_budget_alpha "$part_budget_alpha"
            --part_budget_start_iter "$part_budget_start_iter"
            --part_budget_warmup "$part_budget_warmup"
            --part_budget_hidden_dim "$part_budget_hidden_dim"
            --part_budget_token_dim "$part_budget_token_dim"
        )
    fi
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
