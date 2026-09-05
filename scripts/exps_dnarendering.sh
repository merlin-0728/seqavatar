#!/bin/bash
set -euo pipefail

# Usage:
#   bash scripts/exps_dnarendering.sh
#   bash scripts/exps_dnarendering.sh original
#   bash scripts/exps_dnarendering.sh part_moe_leg
#   bash scripts/exps_dnarendering.sh part_moe_leg_robust
#   bash scripts/exps_dnarendering.sh part_budget
#   bash scripts/exps_dnarendering.sh part_budget_v2
#   bash scripts/exps_dnarendering.sh part_budget_sup
#   bash scripts/exps_dnarendering.sh part_budget_route
#   bash scripts/exps_dnarendering.sh part_budget_full
#   bash scripts/exps_dnarendering.sh tri
#   bash scripts/exps_dnarendering.sh tri_part
#   bash scripts/exps_dnarendering.sh tri_gate
#   bash scripts/exps_dnarendering.sh tri_token
#   bash scripts/exps_dnarendering.sh tri_token_residual
#   bash scripts/exps_dnarendering.sh tri_token_route
#   bash scripts/exps_dnarendering.sh tri_token_part_fusion
#   bash scripts/exps_dnarendering.sh tri_token_route_nopart
#   bash scripts/exps_dnarendering.sh tri_token_route_hard
#   bash scripts/exps_dnarendering.sh tri_token_route_output
#   bash scripts/exps_dnarendering.sh tri_token_route_boundary
#   bash scripts/exps_dnarendering.sh time
#   bash scripts/exps_dnarendering.sh point
#   bash scripts/exps_dnarendering.sh point_anchor
#   bash scripts/exps_dnarendering.sh point_anchor_tb
#   bash scripts/exps_dnarendering.sh point_tem
#   bash scripts/exps_dnarendering.sh dynomo_c
#   bash scripts/exps_dnarendering.sh point_depth
#   bash scripts/exps_dnarendering.sh point_cloth_boundary
#   bash scripts/exps_dnarendering.sh point_cloth_boundary_hf
#   bash scripts/exps_dnarendering.sh point_cloth_budget
#   bash scripts/exps_dnarendering.sh mapo_all_dynamic_l1
#   bash scripts/exps_dnarendering.sh mapo_all_dynamic_l2
#   bash scripts/exps_dnarendering.sh mapo_all_dynamic_l2_soft
#   bash scripts/exps_dnarendering.sh mapo_all_dynamic_l2_wide
#   bash scripts/exps_dnarendering.sh mapo_all_dynamic_l2_residual
#   bash scripts/exps_dnarendering.sh mapo_all_dynamic_l2_partial
#   bash scripts/exps_dnarendering.sh mapo_all_dynamic_l2_residual_dynamic
#   bash scripts/exps_dnarendering.sh part_time
#   bash scripts/exps_dnarendering.sh part_time_moe
#   bash scripts/exps_dnarendering.sh mapo_single_mlp_partial_match
#   bash scripts/exps_dnarendering.sh motion_temporal_temperature
#   bash scripts/exps_dnarendering.sh motion_temporal_temperature_velocity
#   bash scripts/exps_dnarendering.sh motion_temporal_temperature_acceleration
#
# 常用覆盖方式：
#   GPU_id=3 bash scripts/exps_dnarendering.sh part_moe_leg
#   GPU_id=3 bash scripts/exps_dnarendering.sh part_budget
#   GPU_id=3 bash scripts/exps_dnarendering.sh part_budget_v2
#   GPU_id=3 bash scripts/exps_dnarendering.sh part_budget_sup
#   GPU_id=3 bash scripts/exps_dnarendering.sh part_budget_route
#   GPU_id=3 bash scripts/exps_dnarendering.sh part_budget_full
#   GPU_id=3 bash scripts/exps_dnarendering.sh tri
#   GPU_id=3 bash scripts/exps_dnarendering.sh tri_part
#   GPU_id=3 bash scripts/exps_dnarendering.sh tri_gate
#   GPU_id=3 bash scripts/exps_dnarendering.sh tri_token
#   GPU_id=3 bash scripts/exps_dnarendering.sh tri_token_residual
#   GPU_id=3 bash scripts/exps_dnarendering.sh tri_token_route
#   GPU_id=3 bash scripts/exps_dnarendering.sh tri_token_part_fusion
#   GPU_id=3 bash scripts/exps_dnarendering.sh tri_token_route_nopart
#   GPU_id=3 bash scripts/exps_dnarendering.sh tri_token_route_hard
#   GPU_id=3 bash scripts/exps_dnarendering.sh tri_token_route_output
#   GPU_id=3 bash scripts/exps_dnarendering.sh tri_token_route_boundary
#   GPU_id=3 bash scripts/exps_dnarendering.sh time
#   GPU_id=3 bash scripts/exps_dnarendering.sh point
#   GPU_id=3 bash scripts/exps_dnarendering.sh point_anchor
#   GPU_id=3 bash scripts/exps_dnarendering.sh point_anchor_tb
#   GPU_id=3 bash scripts/exps_dnarendering.sh point_depth
#   GPU_id=3 bash scripts/exps_dnarendering.sh vggt_garment
#   SEQUENCES_OVERRIDE="0007_04 0019_10" GPU_id=3 bash scripts/exps_dnarendering.sh part_moe_leg
#   TRI_PLANE_DIM=32 TRI_PLANE_RES=64 TRI_PLANE_EXTENT=1.0 GPU_id=3 bash scripts/exps_dnarendering.sh tri
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
tri_token_enabled=0
time_enabled=0
point_enabled=0
point_anchor_enabled=0
point_anchor_tb_enabled=0
part_point_enabled=0
point_update_enabled=0
part_score_route_enabled=0
point_cloth_budget_enabled=0
point_depth_enabled=0
point_cloth_boundary_beta_default=1.0
point_cloth_hf_beta_default=0.0
point_cloth_nonrigid_beta_default=0.0
token_tri_fusion_mode=concat
token_tri_fusion_hidden_dim=${TOKEN_TRI_FUSION_HIDDEN_DIM:-128}
part_budget_enabled=0
part_budget_mode=base
dynomo_c_enabled=0
vggt_garment_enabled=0
vggt_strict_budget_enabled=0
vggt_high_error_enabled=0
mapo_all_dynamic_enabled=0
mapo_l2_wide_enabled=0
single_mlp_partial_match_enabled=0
mapo_partial_sharing_enabled=0
temporal_conditioned_part_enabled=0
temporal_conditioned_part_fusion_mode=replace
temporal_conditioned_part_conf_threshold=${TEMPORAL_CONDITIONED_PART_CONF_THRESHOLD:-0.5}
temporal_conditioned_part_max_mix=${TEMPORAL_CONDITIONED_PART_MAX_MIX:-0.75}
part_label_robust_enabled=0
part_confidence_route_enabled=0
mapo_max_partition_level=2
motion_temperature_enabled=0
motion_temperature_variant=none
case "$MODE" in
    original)
        experiment_name=original
        part_moe_enabled=0
        ;;
    time)
        experiment_name=time
        part_moe_enabled=0
        time_enabled=1
        final_eval_only=1
        ;;
    point)
        experiment_name=point
        part_moe_enabled=0
        point_enabled=1
        final_eval_only=0
        ;;
    point_anchor)
        experiment_name=point_anchor
        part_moe_enabled=0
        point_anchor_enabled=1
        final_eval_only=0
        ;;
    point_anchor_tb)
        experiment_name=point_anchor_tb
        part_moe_enabled=0
        point_anchor_tb_enabled=1
        final_eval_only=0
        ;;
    point_tem)
        experiment_name=point_tem
        part_moe_enabled=0
        point_anchor_tb_enabled=1
        final_eval_only=0
        ;;
    point_tem_mul)
        experiment_name=point_tem_mul
        part_moe_enabled=0
        point_anchor_tb_enabled=1
        final_eval_only=0
        ;;
    dynomo_c)
        experiment_name=dynomo_c
        part_moe_enabled=0
        dynomo_c_enabled=1
        final_eval_only=0
        ;;
    vggt_garment)
        experiment_name=vggt_garment
        part_moe_enabled=0
        vggt_garment_enabled=1
        final_eval_only=0
        ;;
    vggt_garment_strict)
        experiment_name=vggt_garment_strict
        part_moe_enabled=0
        vggt_garment_enabled=1
        vggt_strict_budget_enabled=1
        final_eval_only=1
        ;;
    vggt_garment_strict_high_error)
        experiment_name=vggt_garment_strict_high_error
        part_moe_enabled=0
        vggt_garment_enabled=1
        vggt_strict_budget_enabled=1
        vggt_high_error_enabled=1
        final_eval_only=1
        ;;
    mapo_all_dynamic_l1)
        experiment_name=mapo_all_dynamic_l1
        part_moe_enabled=0
        mapo_all_dynamic_enabled=1
        mapo_max_partition_level=1
        final_eval_only=1
        ;;
    mapo_all_dynamic_l2)
        experiment_name=mapo_all_dynamic_l2
        part_moe_enabled=0
        mapo_all_dynamic_enabled=1
        mapo_max_partition_level=2
        final_eval_only=1
        ;;
    mapo_all_dynamic_l2_soft)
        experiment_name=mapo_all_dynamic_l2_soft
        part_moe_enabled=0
        mapo_all_dynamic_enabled=1
        mapo_max_partition_level=2
        mapo_soft_routing_enabled=1
        final_eval_only=1
        ;;
    mapo_all_dynamic_l2_partial)
        experiment_name=mapo_all_dynamic_l2_partial
        part_moe_enabled=0
        mapo_all_dynamic_enabled=1
        mapo_max_partition_level=2
        mapo_soft_routing_enabled=1
        mapo_partial_sharing_enabled=1
        final_eval_only=1
        ;;
    mapo_temporal_conditioned_part)
        experiment_name=mapo_temporal_conditioned_part
        part_moe_enabled=1
        part_label_schema=part_moe_leg
        num_parts=7
        mapo_all_dynamic_enabled=1
        mapo_max_partition_level=2
        mapo_soft_routing_enabled=1
        mapo_partial_sharing_enabled=1
        temporal_conditioned_part_enabled=1
        final_eval_only=1
        ;;
    mapo_temporal_conditioned_part_confidence)
        experiment_name=mapo_temporal_conditioned_part_confidence
        part_moe_enabled=1
        part_label_schema=part_moe_leg
        num_parts=7
        mapo_all_dynamic_enabled=1
        mapo_max_partition_level=2
        mapo_soft_routing_enabled=1
        mapo_partial_sharing_enabled=1
        temporal_conditioned_part_enabled=1
        temporal_conditioned_part_fusion_mode=confidence
        final_eval_only=1
        ;;
    mapo_temporal_conditioned_part_full_confidence)
        experiment_name=mapo_temporal_conditioned_part_full_confidence
        part_moe_enabled=1
        part_label_schema=part_moe_leg
        num_parts=7
        mapo_all_dynamic_enabled=1
        mapo_max_partition_level=2
        mapo_soft_routing_enabled=1
        mapo_partial_sharing_enabled=0
        temporal_conditioned_part_enabled=1
        temporal_conditioned_part_fusion_mode=confidence
        part_label_robust_enabled=1
        part_confidence_route_enabled=1
        final_eval_only=1
        ;;
    part_time)
        experiment_name=part_time
        part_moe_enabled=1
        part_label_schema=part_moe_leg
        num_parts=7
        mapo_all_dynamic_enabled=1
        mapo_max_partition_level=2
        mapo_soft_routing_enabled=1
        mapo_partial_sharing_enabled=0
        temporal_conditioned_part_enabled=1
        temporal_conditioned_part_fusion_mode=confidence
        part_label_robust_enabled=1
        part_confidence_route_enabled=1
        final_eval_only=1
        ;;
    part_time_moe)
        experiment_name=part_time_moe
        part_moe_enabled=1
        part_label_schema=part_moe_leg
        num_parts=7
        mapo_all_dynamic_enabled=1
        mapo_max_partition_level=2
        mapo_soft_routing_enabled=1
        mapo_partial_sharing_enabled=0
        temporal_conditioned_part_enabled=1
        temporal_conditioned_part_fusion_mode=replace
        part_label_robust_enabled=0
        part_confidence_route_enabled=0
        final_eval_only=1
        ;;
    motion_temporal_temperature)
        experiment_name=motion_temporal_temperature
        part_moe_enabled=0
        mapo_all_dynamic_enabled=1
        mapo_max_partition_level=2
        mapo_soft_routing_enabled=1
        motion_temperature_enabled=1
        motion_temperature_variant=both
        final_eval_only=1
        ;;
    motion_temporal_temperature_velocity)
        experiment_name=motion_temporal_temperature_velocity
        part_moe_enabled=0
        mapo_all_dynamic_enabled=1
        mapo_max_partition_level=2
        mapo_soft_routing_enabled=1
        motion_temperature_enabled=1
        motion_temperature_variant=velocity
        final_eval_only=1
        ;;
    motion_temporal_temperature_acceleration)
        experiment_name=motion_temporal_temperature_acceleration
        part_moe_enabled=0
        mapo_all_dynamic_enabled=1
        mapo_max_partition_level=2
        mapo_soft_routing_enabled=1
        motion_temperature_enabled=1
        motion_temperature_variant=acceleration
        final_eval_only=1
        ;;
    motion_temporal_temperature_both)
        experiment_name=motion_temporal_temperature_both
        part_moe_enabled=0
        mapo_all_dynamic_enabled=1
        mapo_max_partition_level=2
        mapo_soft_routing_enabled=1
        motion_temperature_enabled=1
        motion_temperature_variant=both
        final_eval_only=1
        ;;
    mapo_all_dynamic_l2_wide)
        experiment_name=mapo_all_dynamic_l2_wide
        part_moe_enabled=0
        mapo_l2_wide_enabled=1
        final_eval_only=1
        ;;
    mapo_single_mlp_partial_match)
        experiment_name=mapo_single_mlp_partial_match
        part_moe_enabled=0
        single_mlp_partial_match_enabled=1
        final_eval_only=1
        ;;
    mapo_all_dynamic_l2_residual)
        experiment_name=mapo_all_dynamic_l2_residual
        part_moe_enabled=0
        mapo_all_dynamic_enabled=1
        mapo_max_partition_level=2
        mapo_shared_trunk_enabled=1
        mapo_soft_routing_enabled=1
        final_eval_only=1
        ;;
    mapo_all_dynamic_l2_residual_dynamic)
        experiment_name=mapo_all_dynamic_l2_residual_dynamic
        part_moe_enabled=0
        mapo_all_dynamic_enabled=1
        mapo_max_partition_level=2
        mapo_shared_trunk_enabled=1
        mapo_soft_routing_enabled=1
        mapo_dynamic_score_enabled=1
        final_eval_only=1
        ;;
    part_point)
        experiment_name=part_point
        part_moe_enabled=1
        part_score_route_enabled=1
        PART_SCORE_ROUTE_MODE=${PART_SCORE_ROUTE_MODE:-delta}
        PART_SCORE_ROUTE_ALPHA=${PART_SCORE_ROUTE_ALPHA:-1.5}
        PART_SCORE_ROUTE_GATE_BIAS=${PART_SCORE_ROUTE_GATE_BIAS:-0.0}
        part_point_enabled=1
        point_anchor_tb_enabled=1
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    point_update)
        experiment_name=point_update
        part_moe_enabled=0
        point_anchor_tb_enabled=1
        point_update_enabled=1
        final_eval_only=0
        ;;
    point_update_soft)
        experiment_name=point_update_soft
        part_moe_enabled=0
        point_anchor_tb_enabled=1
        point_update_enabled=1
        final_eval_only=0
        ;;
    point_update_perf)
        experiment_name=point_update_perf
        part_moe_enabled=0
        point_anchor_tb_enabled=1
        point_update_enabled=1
        final_eval_only=0
        ;;
    point_update_edge)
        experiment_name=point_update_edge
        part_moe_enabled=0
        point_anchor_tb_enabled=1
        point_update_enabled=1
        POINT_UPDATE_EDGE_BETA=0.5
        POINT_UPDATE_EDGE_KERNEL=3
        POINT_UPDATE_NONRIGID_BETA=0.0
        final_eval_only=0
        ;;
    point_update_nonrigid)
        experiment_name=point_update_nonrigid
        part_moe_enabled=0
        point_anchor_tb_enabled=1
        point_update_enabled=1
        POINT_UPDATE_EDGE_BETA=0.0
        POINT_UPDATE_NONRIGID_BETA=0.5
        final_eval_only=0
        ;;
    point_cloth_boundary)
        experiment_name=point_cloth_boundary
        part_moe_enabled=0
        point_cloth_budget_enabled=1
        point_cloth_boundary_beta_default=1.0
        point_cloth_hf_beta_default=0.0
        point_cloth_nonrigid_beta_default=0.0
        final_eval_only=0
        ;;
    point_cloth_boundary_hf)
        experiment_name=point_cloth_boundary_hf
        part_moe_enabled=0
        point_cloth_budget_enabled=1
        point_cloth_boundary_beta_default=1.0
        point_cloth_hf_beta_default=0.5
        point_cloth_nonrigid_beta_default=0.0
        final_eval_only=0
        ;;
    point_cloth_budget)
        experiment_name=point_cloth_budget
        part_moe_enabled=0
        point_cloth_budget_enabled=1
        point_cloth_boundary_beta_default=1.0
        point_cloth_hf_beta_default=0.5
        point_cloth_nonrigid_beta_default=0.75
        final_eval_only=0
        ;;
    point_depth)
        experiment_name=point_depth
        part_moe_enabled=0
        point_depth_enabled=1
        final_eval_only=0
        ;;
    part_moe_leg)
        experiment_name=part_moe_leg
        part_moe_enabled=1
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    part_moe_leg_robust)
        experiment_name=part_moe_leg_robust
        part_moe_enabled=1
        part_label_schema=part_moe_leg
        num_parts=7
        part_label_robust_enabled=1
        part_confidence_route_enabled=1
        final_eval_only=1
        ;;
    part_moe_leg_unknown_route)
        experiment_name=part_moe_leg_unknown_route
        part_moe_enabled=1
        part_score_route_enabled=1
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    part_moe_leg_unknown_route_strong)
        experiment_name=part_moe_leg_unknown_route_strong
        part_moe_enabled=1
        part_score_route_enabled=1
        PART_SCORE_ROUTE_MODE=${PART_SCORE_ROUTE_MODE:-delta}
        PART_SCORE_ROUTE_ALPHA=${PART_SCORE_ROUTE_ALPHA:-1.5}
        PART_SCORE_ROUTE_GATE_BIAS=${PART_SCORE_ROUTE_GATE_BIAS:-0.0}
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    part_budget)
        experiment_name=part_budget
        part_moe_enabled=1
        part_budget_enabled=1
        part_budget_mode=base
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    part_budget_v2)
        experiment_name=part_budget_v2
        part_moe_enabled=1
        part_budget_enabled=1
        part_budget_mode=v2_sup
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    part_budget_sup)
        experiment_name=part_budget_sup
        part_moe_enabled=1
        part_budget_enabled=1
        part_budget_mode=sup
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    part_budget_route)
        experiment_name=part_budget_route
        part_moe_enabled=1
        part_budget_enabled=1
        part_budget_mode=route
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    part_budget_full)
        experiment_name=part_budget_full
        part_moe_enabled=1
        part_budget_enabled=1
        part_budget_mode=full
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    tri)
        experiment_name=tri
        part_moe_enabled=1
        tri_enabled=1
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    tri_part)
        experiment_name=tri_part
        part_moe_enabled=1
        tri_enabled=1
        tri_part_enabled=1
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    tri_gate)
        experiment_name=tri_gate
        part_moe_enabled=1
        tri_enabled=1
        tri_gate_enabled=1
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    tri_token)
        experiment_name=tri_token
        part_moe_enabled=1
        tri_token_enabled=1
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    tri_token_residual)
        experiment_name=tri_token_residual
        part_moe_enabled=1
        tri_token_enabled=1
        token_tri_fusion_mode=residual
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    tri_token_route)
        experiment_name=tri_token_route
        part_moe_enabled=1
        tri_token_enabled=1
        token_tri_fusion_mode=route
        token_tri_start_iter=${TOKEN_TRI_START_ITER:-7000}
        token_tri_warmup=${TOKEN_TRI_WARMUP:-2000}
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    tri_token_part_fusion)
        experiment_name=tri_token_part_fusion
        part_moe_enabled=1
        tri_token_enabled=1
        token_tri_fusion_mode=part_fusion
        token_tri_start_iter=${TOKEN_TRI_START_ITER:-7000}
        token_tri_warmup=${TOKEN_TRI_WARMUP:-2000}
        token_tri_part_fusion_delta_scale=${TOKEN_TRI_PART_FUSION_DELTA_SCALE:-0.35}
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    tri_token_part_fusion_spatial)
        experiment_name=tri_token_part_fusion_spatial
        part_moe_enabled=1
        tri_token_enabled=1
        token_tri_fusion_mode=part_fusion_spatial
        token_tri_start_iter=${TOKEN_TRI_START_ITER:-7000}
        token_tri_warmup=${TOKEN_TRI_WARMUP:-2000}
        token_tri_part_fusion_spatial_delta_scale=${TOKEN_TRI_PART_FUSION_SPATIAL_DELTA_SCALE:-0.45}
        token_tri_part_fusion_spatial_w=${TOKEN_TRI_PART_FUSION_SPATIAL_W:-0.01}
        token_tri_part_fusion_spatial_std_floor=${TOKEN_TRI_PART_FUSION_SPATIAL_STD_FLOOR:-0.08}
        token_tri_part_fusion_spatial_motion_mix=${TOKEN_TRI_PART_FUSION_SPATIAL_MOTION_MIX:-0.5}
        token_tri_part_fusion_spatial_boundary_mix=${TOKEN_TRI_PART_FUSION_SPATIAL_BOUNDARY_MIX:-0.5}
        token_tri_part_fusion_spatial_part_mix=${TOKEN_TRI_PART_FUSION_SPATIAL_PART_MIX:-0.5}
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    tri_token_route_nopart)
        experiment_name=tri_token_route_nopart
        part_moe_enabled=0
        tri_token_enabled=1
        token_tri_fusion_mode=route
        token_tri_start_iter=${TOKEN_TRI_START_ITER:-7000}
        token_tri_warmup=${TOKEN_TRI_WARMUP:-2000}
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    tri_token_route_hard)
        experiment_name=tri_token_route_hard
        part_moe_enabled=1
        tri_token_enabled=1
        token_tri_fusion_mode=route_hard
        token_tri_start_iter=${TOKEN_TRI_START_ITER:-2500}
        token_tri_warmup=${TOKEN_TRI_WARMUP:-3500}
        token_tri_route_hard_w=${TOKEN_TRI_ROUTE_HARD_W:-0.01}
        token_tri_route_hard_boundary_mix=${TOKEN_TRI_ROUTE_HARD_BOUNDARY_MIX:-0.65}
        token_tri_route_hard_motion_mix=${TOKEN_TRI_ROUTE_HARD_MOTION_MIX:-0.35}
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    tri_token_route_output)
        experiment_name=tri_token_route_output
        part_moe_enabled=1
        tri_token_enabled=1
        token_tri_fusion_mode=route_output
        token_tri_start_iter=${TOKEN_TRI_START_ITER:-3000}
        token_tri_warmup=${TOKEN_TRI_WARMUP:-3000}
        token_tri_route_output_alpha=${TOKEN_TRI_ROUTE_OUTPUT_ALPHA:-0.2}
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    tri_token_route_boundary)
        experiment_name=tri_token_route_boundary
        part_moe_enabled=1
        tri_token_enabled=1
        token_tri_fusion_mode=route
        token_tri_start_iter=${TOKEN_TRI_START_ITER:-5000}
        token_tri_warmup=${TOKEN_TRI_WARMUP:-2500}
        token_tri_route_boundary_w=${TOKEN_TRI_ROUTE_BOUNDARY_W:-0.01}
        token_tri_route_boundary_floor=${TOKEN_TRI_ROUTE_BOUNDARY_FLOOR:-0.15}
        part_label_schema=part_moe_leg
        num_parts=7
        final_eval_only=1
        ;;
    *)
        echo "[ERROR] Unknown mode: $MODE"
        echo "        Supported modes include original, part_moe_leg, mapo_all_dynamic_l2_soft, mapo_temporal_conditioned_part, mapo_temporal_conditioned_part_confidence, mapo_temporal_conditioned_part_full_confidence, and existing ablations"
        exit 1
        ;;
esac

# ================= 路径和基础设置 =================
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_TIME_RAW=${RUN_TIME:-$(date +%Y%m%d_%H%M%S)}
RUN_TIME_BASE=${RUN_TIME_RAW%%_gpu*}
RUN_TIME=${RUN_TIME_BASE}
GPU_id=${GPU_id:-3}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
DATA_PATH=${DATA_PATH:-$REPO_ROOT/DNA-Rendering}
PART_LOG_DIR=${PART_LOG_DIR:-$REPO_ROOT/logs/part}
PART_SCORE_ROUTE_LOG_DIR=${PART_SCORE_ROUTE_LOG_DIR:-$REPO_ROOT/logs/part_score_route}
PART_BUDGET_LOG_DIR=${PART_BUDGET_LOG_DIR:-$REPO_ROOT/logs/budget}
TRI_LOG_DIR=${TRI_LOG_DIR:-$REPO_ROOT/logs/tri}
TIME_LOG_DIR=${TIME_LOG_DIR:-$REPO_ROOT/logs/time}
POINT_LOG_DIR=${POINT_LOG_DIR:-$REPO_ROOT/logs/point}
POINT_ANCHOR_LOG_DIR=${POINT_ANCHOR_LOG_DIR:-$REPO_ROOT/logs/point_anchor}
PART_POINT_LOG_DIR=${PART_POINT_LOG_DIR:-$REPO_ROOT/logs/part_point}
if [ "$MODE" = "point_tem" ]; then
    POINT_ANCHOR_TB_LOG_DIR=${POINT_ANCHOR_TB_LOG_DIR:-$REPO_ROOT/logs/point_tem}
elif [ "$MODE" = "point_tem_mul" ]; then
    POINT_ANCHOR_TB_LOG_DIR=${POINT_ANCHOR_TB_LOG_DIR:-$REPO_ROOT/logs/point_tem_mul}
elif [ "$MODE" = "point_update" ]; then
    POINT_ANCHOR_TB_LOG_DIR=${POINT_ANCHOR_TB_LOG_DIR:-$REPO_ROOT/logs/point_update}
elif [ "$MODE" = "point_update_soft" ]; then
    POINT_ANCHOR_TB_LOG_DIR=${POINT_ANCHOR_TB_LOG_DIR:-$REPO_ROOT/logs/point_update_soft}
elif [ "$MODE" = "point_update_perf" ]; then
    POINT_ANCHOR_TB_LOG_DIR=${POINT_ANCHOR_TB_LOG_DIR:-$REPO_ROOT/logs/point_update_perf}
elif [ "$MODE" = "point_update_edge" ]; then
    POINT_ANCHOR_TB_LOG_DIR=${POINT_ANCHOR_TB_LOG_DIR:-$REPO_ROOT/logs/point_update_edge}
elif [ "$MODE" = "point_update_nonrigid" ]; then
    POINT_ANCHOR_TB_LOG_DIR=${POINT_ANCHOR_TB_LOG_DIR:-$REPO_ROOT/logs/point_update_nonrigid}
else
    POINT_ANCHOR_TB_LOG_DIR=${POINT_ANCHOR_TB_LOG_DIR:-$REPO_ROOT/logs/point_anchor_tb}
fi
POINT_CLOTH_LOG_DIR=${POINT_CLOTH_LOG_DIR:-$REPO_ROOT/logs/point_cloth}
POINT_DEPTH_LOG_DIR=${POINT_DEPTH_LOG_DIR:-$REPO_ROOT/logs/point_depth}

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
iter=${ITERATIONS:-25000}
densify_until_iter=${DENSIFY_UNTIL_ITER:-1800}
seed=${SEED:-0}

seq_len=8
seq_xyz_knn=8
time_step_num=3
max_time_step=3
minimal_time_step=1
non_rigid_mlp_depth=${NON_RIGID_MLP_DEPTH:-3}
non_rigid_mlp_width=${NON_RIGID_MLP_WIDTH:-512}
if [ "$mapo_l2_wide_enabled" = "1" ]; then
    if [ -n "${NON_RIGID_MLP_WIDTH:-}" ] && [ "$NON_RIGID_MLP_WIDTH" != "1078" ]; then
        echo "[ERROR] mapo_all_dynamic_l2_wide requires NON_RIGID_MLP_WIDTH=1078 for parameter matching."
        exit 1
    fi
    # Matches the full l2 four-branch deformation predictor within 0.075%.
    non_rigid_mlp_width=1078
fi
if [ "$single_mlp_partial_match_enabled" = "1" ]; then
    if [ -n "${NON_RIGID_MLP_WIDTH:-}" ] && [ "$NON_RIGID_MLP_WIDTH" != "794" ]; then
        echo "[ERROR] mapo_single_mlp_partial_match requires NON_RIGID_MLP_WIDTH=794 for parameter matching."
        exit 1
    fi
    # Matches partial-sharing within 222 NonrigidDeformer parameters.
    non_rigid_mlp_width=794
fi

l1_loss_w=1.0
ssim_loss_w=0.01
lpips_loss_w=0.01

# ================= Part-MoE 参数 =================
part_moe_start_iter=10000
part_moe_warmup=1000
part_moe_global_keep=0.1
part_label_knn=${PART_LABEL_KNN:-8}
part_label_vote_temperature=${PART_LABEL_VOTE_TEMPERATURE:-0.01}
part_label_refresh_interval=${PART_LABEL_REFRESH_INTERVAL:-5000}
part_moe_conf_threshold=${PART_MOE_CONF_THRESHOLD:-0.5}
tri_plane_dim=${TRI_PLANE_DIM:-32}
tri_plane_res=${TRI_PLANE_RES:-64}
tri_plane_extent=${TRI_PLANE_EXTENT:-1.0}
token_tri_dim=${TOKEN_TRI_DIM:-32}
token_tri_res=${TOKEN_TRI_RES:-16}
token_tri_extent=${TOKEN_TRI_EXTENT:-1.0}
token_tri_heads=${TOKEN_TRI_HEADS:-4}
token_tri_layers=${TOKEN_TRI_LAYERS:-2}
token_tri_hidden_dim=${TOKEN_TRI_HIDDEN_DIM:-128}
token_tri_alpha=${TOKEN_TRI_ALPHA:-1.0}
token_tri_start_iter=${TOKEN_TRI_START_ITER:-$part_moe_start_iter}
token_tri_warmup=${TOKEN_TRI_WARMUP:-1000}
token_tri_route_boundary_w=${TOKEN_TRI_ROUTE_BOUNDARY_W:-0.0}
token_tri_route_boundary_floor=${TOKEN_TRI_ROUTE_BOUNDARY_FLOOR:-0.15}
token_tri_route_output_alpha=${TOKEN_TRI_ROUTE_OUTPUT_ALPHA:-0.2}
token_tri_route_hard_w=${TOKEN_TRI_ROUTE_HARD_W:-0.0}
token_tri_route_hard_boundary_mix=${TOKEN_TRI_ROUTE_HARD_BOUNDARY_MIX:-0.65}
token_tri_route_hard_motion_mix=${TOKEN_TRI_ROUTE_HARD_MOTION_MIX:-0.35}
token_tri_part_fusion_delta_scale=${TOKEN_TRI_PART_FUSION_DELTA_SCALE:-0.35}
token_tri_part_fusion_spatial_delta_scale=${TOKEN_TRI_PART_FUSION_SPATIAL_DELTA_SCALE:-0.45}
token_tri_part_fusion_spatial_w=${TOKEN_TRI_PART_FUSION_SPATIAL_W:-0.01}
token_tri_part_fusion_spatial_std_floor=${TOKEN_TRI_PART_FUSION_SPATIAL_STD_FLOOR:-0.08}
token_tri_part_fusion_spatial_motion_mix=${TOKEN_TRI_PART_FUSION_SPATIAL_MOTION_MIX:-0.5}
token_tri_part_fusion_spatial_boundary_mix=${TOKEN_TRI_PART_FUSION_SPATIAL_BOUNDARY_MIX:-0.5}
token_tri_part_fusion_spatial_part_mix=${TOKEN_TRI_PART_FUSION_SPATIAL_PART_MIX:-0.5}
time_scale_emb_dim=${TIME_SCALE_EMB_DIM:-16}
time_scale_temperature=${TIME_SCALE_TEMPERATURE:-1.5}
tri_gate_alpha=${TRI_GATE_ALPHA:-0.2}
tri_gate_init=${TRI_GATE_INIT:-0.5}
tri_gate_hidden_dim=${TRI_GATE_HIDDEN_DIM:-128}
tri_gate_mode=${TRI_GATE_MODE:-additive}
tri_gate_start_iter=${TRI_GATE_START_ITER:-3000}
tri_gate_warmup=${TRI_GATE_WARMUP:-3000}
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
part_budget_sup_w=${PART_BUDGET_SUP_W:-0.02}
part_budget_balance_w=${PART_BUDGET_BALANCE_W:-0.005}
part_budget_target_mix=${PART_BUDGET_TARGET_MIX:-0.6}
part_budget_target_sharpness=${PART_BUDGET_TARGET_SHARPNESS:-2.0}
part_budget_router_sharpness=${PART_BUDGET_ROUTER_SHARPNESS:-1.0}
point_patch_size=${POINT_PATCH_SIZE:-32}
point_start_iter=${POINT_START_ITER:-800}
point_interval=${POINT_INTERVAL:-100}
point_topk_patches=${POINT_TOPK_PATCHES:-16}
point_anchor_radius=${POINT_ANCHOR_RADIUS:-20.0}
point_max_anchors=${POINT_MAX_ANCHORS:-512}
point_grad_boost=${POINT_GRAD_BOOST:-2.0}
point_min_patch_coverage=${POINT_MIN_PATCH_COVERAGE:-0.20}
point_anchor_start_iter=${POINT_ANCHOR_START_ITER:-800}
point_anchor_end_iter=${POINT_ANCHOR_END_ITER:-1800}
point_anchor_interval=${POINT_ANCHOR_INTERVAL:-100}
point_anchor_patch_size=${POINT_ANCHOR_PATCH_SIZE:-32}
point_anchor_topk=${POINT_ANCHOR_TOPK:-16}
point_anchor_min_coverage=${POINT_ANCHOR_MIN_COVERAGE:-0.20}
point_anchor_spawn_radius=${POINT_ANCHOR_SPAWN_RADIUS:-20.0}
point_anchor_spawn_max_anchors=${POINT_ANCHOR_SPAWN_MAX_ANCHORS:-512}
point_anchor_children_per_anchor=${POINT_ANCHOR_CHILDREN_PER_ANCHOR:-1}
point_anchor_offset_scale=${POINT_ANCHOR_OFFSET_SCALE:-0.35}
point_anchor_scale_ratio=${POINT_ANCHOR_SCALE_RATIO:-0.7}
point_anchor_opacity_ratio=${POINT_ANCHOR_OPACITY_RATIO:-0.8}
point_anchor_opacity_min=${POINT_ANCHOR_OPACITY_MIN:-0.01}
point_anchor_max_points=${POINT_ANCHOR_MAX_POINTS:-120000}
point_tb_start_iter=${POINT_TB_START_ITER:-800}
point_tb_end_iter=${POINT_TB_END_ITER:-1800}
point_tb_interval=${POINT_TB_INTERVAL:-100}
point_tb_patch_size=${POINT_TB_PATCH_SIZE:-32}
point_tb_topk=${POINT_TB_TOPK:-16}
point_tb_min_coverage=${POINT_TB_MIN_COVERAGE:-0.20}
point_tb_anchor_radius=${POINT_TB_ANCHOR_RADIUS:-20.0}
point_tb_max_anchors=${POINT_TB_MAX_ANCHORS:-512}
point_tb_children_per_anchor=${POINT_TB_CHILDREN_PER_ANCHOR:-1}
point_tb_offset_scale=${POINT_TB_OFFSET_SCALE:-0.35}
point_tb_scale_ratio=${POINT_TB_SCALE_RATIO:-0.7}
point_tb_opacity_ratio=${POINT_TB_OPACITY_RATIO:-0.8}
point_tb_opacity_min=${POINT_TB_OPACITY_MIN:-0.01}
point_tb_max_points=${POINT_TB_MAX_POINTS:-120000}
point_tb_temporal_alpha=${POINT_TB_TEMPORAL_ALPHA:-0.5}
point_tb_history_momentum=${POINT_TB_HISTORY_MOMENTUM:-0.8}
point_tb_min_persistence=${POINT_TB_MIN_PERSISTENCE:-0.0}
point_tb_boundary_beta=${POINT_TB_BOUNDARY_BETA:-0.5}
point_tb_boundary_kernel=${POINT_TB_BOUNDARY_KERNEL:-5}
point_tb_target_points=${POINT_TB_TARGET_POINTS:-0}
point_tb_replace_after_iter=${POINT_TB_REPLACE_AFTER_ITER:-1500}
if [ "$point_update_enabled" = "1" ] && [ -z "${POINT_TB_REPLACE_AFTER_ITER:-}" ]; then
    point_tb_replace_after_iter=800
fi
point_tb_replacement_ratio=${POINT_TB_REPLACEMENT_RATIO:-1.0}
point_tb_replace_opacity_w=${POINT_TB_REPLACE_OPACITY_W:-1.0}
point_tb_replace_gradient_w=${POINT_TB_REPLACE_GRADIENT_W:-0.25}
point_tb_replace_visibility_w=${POINT_TB_REPLACE_VISIBILITY_W:-0.10}
point_update_ema_momentum=${POINT_UPDATE_EMA_MOMENTUM:-0.95}
point_update_protect_anchors=${POINT_UPDATE_PROTECT_ANCHORS:-1}
point_update_min_keep=${POINT_UPDATE_MIN_KEEP:-1024}
point_update_final_clamp_iter=${POINT_UPDATE_FINAL_CLAMP_ITER:-0}
point_update_clamp_interval=${POINT_UPDATE_CLAMP_INTERVAL:-100}
point_update_clamp_ratio=${POINT_UPDATE_CLAMP_RATIO:-1.0}
point_update_edge_beta=${POINT_UPDATE_EDGE_BETA:-0.0}
point_update_edge_kernel=${POINT_UPDATE_EDGE_KERNEL:-3}
point_update_nonrigid_beta=${POINT_UPDATE_NONRIGID_BETA:-0.0}
part_score_route_hidden_dim=${PART_SCORE_ROUTE_HIDDEN_DIM:-128}
part_score_route_alpha=${PART_SCORE_ROUTE_ALPHA:-1.0}
part_score_route_gate_bias=${PART_SCORE_ROUTE_GATE_BIAS:--2.0}
part_score_route_mode=${PART_SCORE_ROUTE_MODE:-boost}
point_cloth_start_iter=${POINT_CLOTH_START_ITER:-800}
point_cloth_end_iter=${POINT_CLOTH_END_ITER:-1800}
point_cloth_interval=${POINT_CLOTH_INTERVAL:-100}
point_cloth_patch_size=${POINT_CLOTH_PATCH_SIZE:-32}
point_cloth_topk=${POINT_CLOTH_TOPK:-16}
point_cloth_min_coverage=${POINT_CLOTH_MIN_COVERAGE:-0.20}
point_cloth_anchor_radius=${POINT_CLOTH_ANCHOR_RADIUS:-20.0}
point_cloth_max_anchors=${POINT_CLOTH_MAX_ANCHORS:-512}
point_cloth_children_per_anchor=${POINT_CLOTH_CHILDREN_PER_ANCHOR:-1}
point_cloth_offset_scale=${POINT_CLOTH_OFFSET_SCALE:-0.35}
point_cloth_scale_ratio=${POINT_CLOTH_SCALE_RATIO:-0.7}
point_cloth_opacity_ratio=${POINT_CLOTH_OPACITY_RATIO:-0.8}
point_cloth_opacity_min=${POINT_CLOTH_OPACITY_MIN:-0.01}
point_cloth_max_points=${POINT_CLOTH_MAX_POINTS:-120000}
point_cloth_temporal_alpha=${POINT_CLOTH_TEMPORAL_ALPHA:-0.5}
point_cloth_history_momentum=${POINT_CLOTH_HISTORY_MOMENTUM:-0.8}
point_cloth_min_persistence=${POINT_CLOTH_MIN_PERSISTENCE:-0.0}
point_cloth_boundary_beta=${POINT_CLOTH_BOUNDARY_BETA:-$point_cloth_boundary_beta_default}
point_cloth_hf_beta=${POINT_CLOTH_HF_BETA:-$point_cloth_hf_beta_default}
point_cloth_nonrigid_beta=${POINT_CLOTH_NONRIGID_BETA:-$point_cloth_nonrigid_beta_default}
point_cloth_boundary_kernel=${POINT_CLOTH_BOUNDARY_KERNEL:-5}
point_cloth_protect_thresh=${POINT_CLOTH_PROTECT_THRESH:-0.60}
point_cloth_replace_after_iter=${POINT_CLOTH_REPLACE_AFTER_ITER:-800}
point_cloth_replacement_ratio=${POINT_CLOTH_REPLACEMENT_RATIO:-1.0}
point_depth_start_iter=${POINT_DEPTH_START_ITER:-800}
point_depth_end_iter=${POINT_DEPTH_END_ITER:-1800}
point_depth_interval=${POINT_DEPTH_INTERVAL:-100}
point_depth_patch_size=${POINT_DEPTH_PATCH_SIZE:-32}
point_depth_topk=${POINT_DEPTH_TOPK:-16}
point_depth_min_coverage=${POINT_DEPTH_MIN_COVERAGE:-0.20}
point_depth_anchor_radius=${POINT_DEPTH_ANCHOR_RADIUS:-20.0}
point_depth_max_anchors=${POINT_DEPTH_MAX_ANCHORS:-512}
point_depth_surface_threshold=${POINT_DEPTH_SURFACE_THRESHOLD:-0.12}
point_depth_depth_threshold=${POINT_DEPTH_DEPTH_THRESHOLD:-0.08}
point_depth_depth_relative=${POINT_DEPTH_DEPTH_RELATIVE:-0.04}
point_depth_children_per_anchor=${POINT_DEPTH_CHILDREN_PER_ANCHOR:-1}
point_depth_offset_scale=${POINT_DEPTH_OFFSET_SCALE:-0.25}
point_depth_scale_ratio=${POINT_DEPTH_SCALE_RATIO:-0.7}
point_depth_opacity_ratio=${POINT_DEPTH_OPACITY_RATIO:-0.8}
point_depth_opacity_min=${POINT_DEPTH_OPACITY_MIN:-0.01}
point_depth_max_points=${POINT_DEPTH_MAX_POINTS:-120000}
point_depth_fixed_budget=${POINT_DEPTH_FIXED_BUDGET:-1}
point_depth_replace_opacity_w=${POINT_DEPTH_REPLACE_OPACITY_W:-1.0}
point_depth_replace_gradient_w=${POINT_DEPTH_REPLACE_GRADIENT_W:-0.25}
point_depth_replace_visibility_w=${POINT_DEPTH_REPLACE_VISIBILITY_W:-0.10}
dynomo_c_label_iter=${DYNOMO_C_LABEL_ITER:-0}
dynomo_c_affinity_dim=${DYNOMO_C_AFFINITY_DIM:-32}
dynomo_c_knn=${DYNOMO_C_KNN:-5}
dynomo_c_part_same_w=${DYNOMO_C_PART_SAME_W:-1.0}
dynomo_c_part_adj_w=${DYNOMO_C_PART_ADJ_W:-0.1}
dynomo_c_part_other_w=${DYNOMO_C_PART_OTHER_W:-0.0}
dynomo_c_motion_w=${DYNOMO_C_MOTION_W:-0.01}
dynomo_c_rotation_w=${DYNOMO_C_ROTATION_W:-0.01}
vggt_candidate_path=${VGGT_CANDIDATE_PATH:-}
vggt_target_points_file=${VGGT_TARGET_POINTS_FILE:-}
vggt_garment_start_iter=${VGGT_GARMENT_START_ITER:-800}
vggt_garment_end_iter=${VGGT_GARMENT_END_ITER:-1500}
vggt_garment_interval=${VGGT_GARMENT_INTERVAL:-100}
vggt_garment_max_spawn=${VGGT_GARMENT_MAX_SPAWN:-512}
vggt_candidate_opacity=${VGGT_CANDIDATE_OPACITY:-0.04}
vggt_candidate_scale_ratio=${VGGT_CANDIDATE_SCALE_RATIO:-0.65}
vggt_candidate_min_distance=${VGGT_CANDIDATE_MIN_DISTANCE:-0.004}
vggt_max_points=${VGGT_MAX_POINTS:-120000}
vggt_replace_opacity_w=${VGGT_REPLACE_OPACITY_W:-1.0}
vggt_replace_gradient_w=${VGGT_REPLACE_GRADIENT_W:-0.25}
vggt_replace_visibility_w=${VGGT_REPLACE_VISIBILITY_W:-0.10}
vggt_min_keep=${VGGT_MIN_KEEP:-1024}
vggt_high_error_quantile=${VGGT_HIGH_ERROR_QUANTILE:-0.75}
vggt_high_error_candidate_w=${VGGT_HIGH_ERROR_CANDIDATE_W:-1.0}
mapo_partition_level1_iter=${MAPO_PARTITION_LEVEL1_ITER:-5000}
mapo_partition_level2_iter=${MAPO_PARTITION_LEVEL2_ITER:-10000}
mapo_partition_level3_iter=${MAPO_PARTITION_LEVEL3_ITER:-15000}
mapo_num_frames=${MAPO_NUM_FRAMES:-100}
mapo_soft_blend_width=${MAPO_SOFT_BLEND_WIDTH:-4.0}
mapo_soft_routing_enabled=${mapo_soft_routing_enabled:-0}
mapo_shared_trunk_enabled=${mapo_shared_trunk_enabled:-0}
mapo_partial_sharing_enabled=${mapo_partial_sharing_enabled:-0}
mapo_residual_alpha=${MAPO_RESIDUAL_ALPHA:-1.0}
mapo_dynamic_score_enabled=${mapo_dynamic_score_enabled:-0}
mapo_dynamic_score_momentum=${MAPO_DYNAMIC_SCORE_MOMENTUM:-0.95}
mapo_dynamic_score_alpha=${MAPO_DYNAMIC_SCORE_ALPHA:-0.5}
motion_temperature_min=${MOTION_TEMPERATURE_MIN:-0.50}
motion_temperature_max=${MOTION_TEMPERATURE_MAX:-1.50}
motion_velocity_weight=${MOTION_VELOCITY_WEIGHT:-0.50}
motion_acceleration_weight=${MOTION_ACCELERATION_WEIGHT:-0.50}
if [ "$motion_temperature_variant" = "velocity" ]; then
    motion_velocity_weight=1.0
    motion_acceleration_weight=0.0
elif [ "$motion_temperature_variant" = "acceleration" ]; then
    motion_velocity_weight=0.0
    motion_acceleration_weight=1.0
elif [ "$motion_temperature_variant" = "both" ]; then
    motion_velocity_weight=0.5
    motion_acceleration_weight=0.5
fi

if [ "$part_budget_enabled" = "1" ]; then
    if [ -z "${SKIP_LOAD_TEST_CAMERAS:-}" ]; then
        skip_load_test_cameras=1
    fi
    if [ -z "${IMAGE_DATA_DEVICE:-}" ]; then
        image_data_device=cpu
    fi
fi

if [ "$tri_token_enabled" = "1" ]; then
    if [ -z "${SKIP_LOAD_TEST_CAMERAS:-}" ]; then
        skip_load_test_cameras=1
    fi
    if [ -z "${IMAGE_DATA_DEVICE:-}" ]; then
        image_data_device=cpu
    fi
fi

if [ "$MODE" = "part_budget_v2" ] && [ -z "${PART_BUDGET_ROUTER_SHARPNESS:-}" ]; then
    part_budget_router_sharpness=2.0
fi

if [ "$MODE" = "tri_token_route" ] || [ "$MODE" = "tri_token_route_nopart" ]; then
    token_tri_start_iter=${TOKEN_TRI_START_ITER:-7000}
    token_tri_warmup=${TOKEN_TRI_WARMUP:-2000}
fi

if [ "$MODE" = "tri_token_part_fusion" ]; then
    token_tri_start_iter=${TOKEN_TRI_START_ITER:-7000}
    token_tri_warmup=${TOKEN_TRI_WARMUP:-2000}
    token_tri_part_fusion_delta_scale=${TOKEN_TRI_PART_FUSION_DELTA_SCALE:-0.35}
fi

if [ "$MODE" = "tri_token_part_fusion_spatial" ]; then
    token_tri_start_iter=${TOKEN_TRI_START_ITER:-7000}
    token_tri_warmup=${TOKEN_TRI_WARMUP:-2000}
    token_tri_part_fusion_spatial_delta_scale=${TOKEN_TRI_PART_FUSION_SPATIAL_DELTA_SCALE:-0.45}
    token_tri_part_fusion_spatial_w=${TOKEN_TRI_PART_FUSION_SPATIAL_W:-0.01}
    token_tri_part_fusion_spatial_std_floor=${TOKEN_TRI_PART_FUSION_SPATIAL_STD_FLOOR:-0.08}
    token_tri_part_fusion_spatial_motion_mix=${TOKEN_TRI_PART_FUSION_SPATIAL_MOTION_MIX:-0.5}
    token_tri_part_fusion_spatial_boundary_mix=${TOKEN_TRI_PART_FUSION_SPATIAL_BOUNDARY_MIX:-0.5}
    token_tri_part_fusion_spatial_part_mix=${TOKEN_TRI_PART_FUSION_SPATIAL_PART_MIX:-0.5}
fi

if [ "$MODE" = "tri_token_route_hard" ]; then
    token_tri_start_iter=${TOKEN_TRI_START_ITER:-2500}
    token_tri_warmup=${TOKEN_TRI_WARMUP:-3500}
    token_tri_route_hard_w=${TOKEN_TRI_ROUTE_HARD_W:-0.01}
    token_tri_route_hard_boundary_mix=${TOKEN_TRI_ROUTE_HARD_BOUNDARY_MIX:-0.65}
    token_tri_route_hard_motion_mix=${TOKEN_TRI_ROUTE_HARD_MOTION_MIX:-0.35}
fi

if [ "$MODE" = "tri_token_route_boundary" ]; then
    token_tri_start_iter=${TOKEN_TRI_START_ITER:-5000}
    token_tri_warmup=${TOKEN_TRI_WARMUP:-2500}
    token_tri_route_boundary_w=${TOKEN_TRI_ROUTE_BOUNDARY_W:-0.01}
    token_tri_route_boundary_floor=${TOKEN_TRI_ROUTE_BOUNDARY_FLOOR:-0.15}
fi

if [ "$MODE" = "tri_token_route_output" ]; then
    token_tri_start_iter=${TOKEN_TRI_START_ITER:-3000}
    token_tri_warmup=${TOKEN_TRI_WARMUP:-3000}
    token_tri_route_output_alpha=${TOKEN_TRI_ROUTE_OUTPUT_ALPHA:-0.2}
fi

test_iterations=(3000 "$part_moe_start_iter" "$iter")
save_iterations=(3000 "$part_moe_start_iter" "$iter")
if [ "$MODE" = "original" ]; then
    # Match the reference DNA script's train.py default evaluation schedule.
    test_iterations=(3000 15000 "$iter")
    save_iterations=(3000 15000 "$iter")
fi
if [ "$final_eval_only" = "1" ]; then
    # part_moe_leg on DNA is memory tight during intermediate full-set eval.
    # Keep label activation at part_moe_start_iter, but only evaluate/save final outputs.
    test_iterations=("$iter")
    save_iterations=("$iter")
fi

# ================= 总日志设置 =================
log_tag="${RUN_TIME}_gpu${GPU_id}_DNA-Rendering_${experiment_name}"
if [ "$tri_enabled" = "1" ] || [ "$tri_token_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$TRI_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${log_tag}.log"
elif [ "$time_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$TIME_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${log_tag}.log"
elif [ "$point_anchor_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$POINT_ANCHOR_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${log_tag}.log"
elif [ "$point_anchor_tb_enabled" = "1" ]; then
    if [ "$part_point_enabled" = "1" ]; then
        GLOBAL_LOG_DIR="$PART_POINT_LOG_DIR"
    else
        GLOBAL_LOG_DIR="$POINT_ANCHOR_TB_LOG_DIR"
    fi
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${log_tag}.log"
elif [ "$part_point_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$PART_POINT_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${log_tag}.log"
elif [ "$point_cloth_budget_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$POINT_CLOTH_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${log_tag}.log"
elif [ "$point_depth_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$POINT_DEPTH_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${log_tag}.log"
elif [ "$dynomo_c_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$REPO_ROOT/logs/dynomo_c"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${log_tag}.log"
elif [ "$mapo_l2_wide_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$REPO_ROOT/logs/mapo_all_dynamic_l2_wide"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${log_tag}.log"
elif [ "$single_mlp_partial_match_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$REPO_ROOT/logs/mapo_single_mlp_partial_match"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${log_tag}.log"
elif [ "$mapo_all_dynamic_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$REPO_ROOT/logs/mapo_all_dynamic"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${log_tag}.log"
elif [ "$part_score_route_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$PART_SCORE_ROUTE_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${log_tag}.log"
elif [ "$point_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$POINT_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${log_tag}.log"
elif [ "$part_budget_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$PART_BUDGET_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${log_tag}.log"
elif [ "$part_moe_enabled" = "1" ]; then
    GLOBAL_LOG_DIR="$PART_LOG_DIR"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${log_tag}.log"
else
    GLOBAL_LOG_DIR="$REPO_ROOT/logs"
    GLOBAL_LOG_FILE="$GLOBAL_LOG_DIR/${log_tag}.log"
fi
mkdir -p "$GLOBAL_LOG_DIR"

# train.py/render.py 在 --use_part_moe 时会自动建立自己的 part 日志。
# 这里把那些拆分日志放到临时目录，最终只保留上面的训练+测试总日志。
if [ "$tri_enabled" = "1" ] || [ "$tri_token_enabled" = "1" ]; then
    AUTO_PART_LOG_DIR="$TRI_LOG_DIR/.auto_${RUN_TIME}_gpu${GPU_id}_${experiment_name}"
elif [ "$time_enabled" = "1" ]; then
    AUTO_PART_LOG_DIR="$TIME_LOG_DIR/.auto_${RUN_TIME}_gpu${GPU_id}_${experiment_name}"
elif [ "$point_anchor_enabled" = "1" ]; then
    AUTO_PART_LOG_DIR="$POINT_ANCHOR_LOG_DIR/.auto_${RUN_TIME}_gpu${GPU_id}_${experiment_name}"
elif [ "$point_anchor_tb_enabled" = "1" ]; then
    if [ "$part_point_enabled" = "1" ]; then
        AUTO_PART_LOG_DIR="$PART_POINT_LOG_DIR/.auto_${RUN_TIME}_gpu${GPU_id}_${experiment_name}"
    else
        AUTO_PART_LOG_DIR="$POINT_ANCHOR_TB_LOG_DIR/.auto_${RUN_TIME}_gpu${GPU_id}_${experiment_name}"
    fi
elif [ "$part_point_enabled" = "1" ]; then
    AUTO_PART_LOG_DIR="$PART_POINT_LOG_DIR/.auto_${RUN_TIME}_gpu${GPU_id}_${experiment_name}"
elif [ "$point_cloth_budget_enabled" = "1" ]; then
    AUTO_PART_LOG_DIR="$POINT_CLOTH_LOG_DIR/.auto_${RUN_TIME}_gpu${GPU_id}_${experiment_name}"
elif [ "$point_depth_enabled" = "1" ]; then
    AUTO_PART_LOG_DIR="$POINT_DEPTH_LOG_DIR/.auto_${RUN_TIME}_gpu${GPU_id}_${experiment_name}"
elif [ "$dynomo_c_enabled" = "1" ]; then
    AUTO_PART_LOG_DIR="$REPO_ROOT/logs/dynomo_c/.auto_${RUN_TIME}_gpu${GPU_id}_${experiment_name}"
elif [ "$mapo_l2_wide_enabled" = "1" ]; then
    AUTO_PART_LOG_DIR="$REPO_ROOT/logs/mapo_all_dynamic_l2_wide/.auto_${RUN_TIME}_gpu${GPU_id}_${experiment_name}"
elif [ "$single_mlp_partial_match_enabled" = "1" ]; then
    AUTO_PART_LOG_DIR="$REPO_ROOT/logs/mapo_single_mlp_partial_match/.auto_${RUN_TIME}_gpu${GPU_id}_${experiment_name}"
elif [ "$mapo_all_dynamic_enabled" = "1" ]; then
    AUTO_PART_LOG_DIR="$REPO_ROOT/logs/mapo_all_dynamic/.auto_${RUN_TIME}_gpu${GPU_id}_${experiment_name}"
elif [ "$part_score_route_enabled" = "1" ]; then
    AUTO_PART_LOG_DIR="$PART_SCORE_ROUTE_LOG_DIR/.auto_${RUN_TIME}_gpu${GPU_id}_${experiment_name}"
elif [ "$point_enabled" = "1" ]; then
    AUTO_PART_LOG_DIR="$POINT_LOG_DIR/.auto_${RUN_TIME}_gpu${GPU_id}_${experiment_name}"
elif [ "$part_budget_enabled" = "1" ]; then
    AUTO_PART_LOG_DIR="$PART_BUDGET_LOG_DIR/.auto_${RUN_TIME}_gpu${GPU_id}_${experiment_name}"
else
    AUTO_PART_LOG_DIR="$PART_LOG_DIR/.auto_${RUN_TIME}_gpu${GPU_id}_${experiment_name}"
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
echo "[INFO] TRI_LOG_DIR: $TRI_LOG_DIR"
echo "[INFO] TIME_LOG_DIR: $TIME_LOG_DIR"
echo "[INFO] PART_BUDGET_LOG_DIR: $PART_BUDGET_LOG_DIR"
echo "[INFO] PART_SCORE_ROUTE_LOG_DIR: $PART_SCORE_ROUTE_LOG_DIR"
echo "[INFO] PART_LABEL_SCHEMA: $part_label_schema"
echo "[INFO] TRI_ENABLED: $tri_enabled"
echo "[INFO] TRI_PART_ENABLED: $tri_part_enabled"
echo "[INFO] TRI_GATE_ENABLED: $tri_gate_enabled"
echo "[INFO] TRI_TOKEN_ENABLED: $tri_token_enabled"
echo "[INFO] TIME_ENABLED: $time_enabled"
echo "[INFO] POINT_ENABLED: $point_enabled"
echo "[INFO] POINT_LOG_DIR: $POINT_LOG_DIR"
echo "[INFO] POINT_ANCHOR_ENABLED: $point_anchor_enabled"
echo "[INFO] POINT_ANCHOR_LOG_DIR: $POINT_ANCHOR_LOG_DIR"
echo "[INFO] POINT_ANCHOR_TB_ENABLED: $point_anchor_tb_enabled"
echo "[INFO] POINT_ANCHOR_TB_LOG_DIR: $POINT_ANCHOR_TB_LOG_DIR"
echo "[INFO] PART_POINT_ENABLED: $part_point_enabled"
echo "[INFO] PART_POINT_LOG_DIR: $PART_POINT_LOG_DIR"
echo "[INFO] POINT_UPDATE_ENABLED: $point_update_enabled"
echo "[INFO] POINT_CLOTH_BUDGET_ENABLED: $point_cloth_budget_enabled"
echo "[INFO] POINT_CLOTH_LOG_DIR: $POINT_CLOTH_LOG_DIR"
echo "[INFO] POINT_DEPTH_ENABLED: $point_depth_enabled"
echo "[INFO] POINT_DEPTH_LOG_DIR: $POINT_DEPTH_LOG_DIR"
echo "[INFO] DYNOMO_C_ENABLED: $dynomo_c_enabled"
echo "[INFO] DYNOMO_C_LABEL_ITER: $dynomo_c_label_iter"
echo "[INFO] DYNOMO_C_AFFINITY_DIM: $dynomo_c_affinity_dim"
echo "[INFO] DYNOMO_C_KNN: $dynomo_c_knn"
echo "[INFO] DYNOMO_C_PART_SAME_W: $dynomo_c_part_same_w"
echo "[INFO] DYNOMO_C_PART_ADJ_W: $dynomo_c_part_adj_w"
echo "[INFO] DYNOMO_C_PART_OTHER_W: $dynomo_c_part_other_w"
echo "[INFO] DYNOMO_C_MOTION_W: $dynomo_c_motion_w"
echo "[INFO] DYNOMO_C_ROTATION_W: $dynomo_c_rotation_w"
echo "[INFO] MAPO_ALL_DYNAMIC_ENABLED: $mapo_all_dynamic_enabled"
echo "[INFO] MAPO_L2_WIDE_ENABLED: $mapo_l2_wide_enabled"
echo "[INFO] SINGLE_MLP_PARTIAL_MATCH_ENABLED: $single_mlp_partial_match_enabled"
echo "[INFO] MAPO_MAX_PARTITION_LEVEL: $mapo_max_partition_level"
echo "[INFO] MAPO_PARTITION_LEVEL1_ITER: $mapo_partition_level1_iter"
echo "[INFO] MAPO_PARTITION_LEVEL2_ITER: $mapo_partition_level2_iter"
echo "[INFO] MAPO_PARTITION_LEVEL3_ITER: $mapo_partition_level3_iter"
echo "[INFO] MAPO_NUM_FRAMES: $mapo_num_frames"
echo "[INFO] MAPO_SOFT_ROUTING: $mapo_soft_routing_enabled"
echo "[INFO] MAPO_SOFT_BLEND_WIDTH: $mapo_soft_blend_width"
echo "[INFO] MAPO_SHARED_TRUNK: $mapo_shared_trunk_enabled"
echo "[INFO] MAPO_PARTIAL_SHARING: $mapo_partial_sharing_enabled"
echo "[INFO] MAPO_RESIDUAL_ALPHA: $mapo_residual_alpha"
echo "[INFO] MAPO_DYNAMIC_SCORE: $mapo_dynamic_score_enabled"
echo "[INFO] MOTION_TEMPERATURE_ENABLED: $motion_temperature_enabled"
echo "[INFO] MOTION_TEMPERATURE_MIN/MAX: $motion_temperature_min / $motion_temperature_max"
echo "[INFO] MOTION_VELOCITY_WEIGHT: $motion_velocity_weight"
echo "[INFO] MOTION_ACCELERATION_WEIGHT: $motion_acceleration_weight"
echo "[INFO] POINT_PATCH_SIZE: $point_patch_size"
echo "[INFO] POINT_START_ITER: $point_start_iter"
echo "[INFO] POINT_INTERVAL: $point_interval"
echo "[INFO] POINT_TOPK_PATCHES: $point_topk_patches"
echo "[INFO] POINT_ANCHOR_RADIUS: $point_anchor_radius"
echo "[INFO] POINT_MAX_ANCHORS: $point_max_anchors"
echo "[INFO] POINT_GRAD_BOOST: $point_grad_boost"
echo "[INFO] POINT_MIN_PATCH_COVERAGE: $point_min_patch_coverage"
echo "[INFO] POINT_ANCHOR_START_ITER: $point_anchor_start_iter"
echo "[INFO] POINT_ANCHOR_END_ITER: $point_anchor_end_iter"
echo "[INFO] POINT_ANCHOR_INTERVAL: $point_anchor_interval"
echo "[INFO] POINT_ANCHOR_PATCH_SIZE: $point_anchor_patch_size"
echo "[INFO] POINT_ANCHOR_TOPK: $point_anchor_topk"
echo "[INFO] POINT_ANCHOR_MIN_COVERAGE: $point_anchor_min_coverage"
echo "[INFO] POINT_ANCHOR_SPAWN_RADIUS: $point_anchor_spawn_radius"
echo "[INFO] POINT_ANCHOR_SPAWN_MAX_ANCHORS: $point_anchor_spawn_max_anchors"
echo "[INFO] POINT_ANCHOR_CHILDREN_PER_ANCHOR: $point_anchor_children_per_anchor"
echo "[INFO] POINT_ANCHOR_OFFSET_SCALE: $point_anchor_offset_scale"
echo "[INFO] POINT_ANCHOR_SCALE_RATIO: $point_anchor_scale_ratio"
echo "[INFO] POINT_ANCHOR_OPACITY_RATIO: $point_anchor_opacity_ratio"
echo "[INFO] POINT_ANCHOR_OPACITY_MIN: $point_anchor_opacity_min"
echo "[INFO] POINT_ANCHOR_MAX_POINTS: $point_anchor_max_points"
echo "[INFO] POINT_TB_START_ITER: $point_tb_start_iter"
echo "[INFO] POINT_TB_END_ITER: $point_tb_end_iter"
echo "[INFO] POINT_TB_INTERVAL: $point_tb_interval"
echo "[INFO] POINT_TB_PATCH_SIZE: $point_tb_patch_size"
echo "[INFO] POINT_TB_TOPK: $point_tb_topk"
echo "[INFO] POINT_TB_MIN_COVERAGE: $point_tb_min_coverage"
echo "[INFO] POINT_TB_ANCHOR_RADIUS: $point_tb_anchor_radius"
echo "[INFO] POINT_TB_MAX_ANCHORS: $point_tb_max_anchors"
echo "[INFO] POINT_TB_CHILDREN_PER_ANCHOR: $point_tb_children_per_anchor"
echo "[INFO] POINT_TB_OFFSET_SCALE: $point_tb_offset_scale"
echo "[INFO] POINT_TB_SCALE_RATIO: $point_tb_scale_ratio"
echo "[INFO] POINT_TB_OPACITY_RATIO: $point_tb_opacity_ratio"
echo "[INFO] POINT_TB_OPACITY_MIN: $point_tb_opacity_min"
echo "[INFO] POINT_TB_MAX_POINTS: $point_tb_max_points"
echo "[INFO] POINT_TB_TEMPORAL_ALPHA: $point_tb_temporal_alpha"
echo "[INFO] POINT_TB_HISTORY_MOMENTUM: $point_tb_history_momentum"
echo "[INFO] POINT_TB_MIN_PERSISTENCE: $point_tb_min_persistence"
echo "[INFO] POINT_TB_BOUNDARY_BETA: $point_tb_boundary_beta"
echo "[INFO] POINT_TB_BOUNDARY_KERNEL: $point_tb_boundary_kernel"
echo "[INFO] POINT_TB_TARGET_POINTS: $point_tb_target_points"
echo "[INFO] POINT_TB_REPLACE_AFTER_ITER: $point_tb_replace_after_iter"
echo "[INFO] POINT_TB_REPLACEMENT_RATIO: $point_tb_replacement_ratio"
echo "[INFO] POINT_TB_REPLACE_OPACITY_W: $point_tb_replace_opacity_w"
echo "[INFO] POINT_TB_REPLACE_GRADIENT_W: $point_tb_replace_gradient_w"
echo "[INFO] POINT_TB_REPLACE_VISIBILITY_W: $point_tb_replace_visibility_w"
echo "[INFO] POINT_UPDATE_EMA_MOMENTUM: $point_update_ema_momentum"
echo "[INFO] POINT_UPDATE_PROTECT_ANCHORS: $point_update_protect_anchors"
echo "[INFO] POINT_UPDATE_MIN_KEEP: $point_update_min_keep"
echo "[INFO] POINT_UPDATE_FINAL_CLAMP_ITER: $point_update_final_clamp_iter"
echo "[INFO] POINT_UPDATE_CLAMP_INTERVAL: $point_update_clamp_interval"
echo "[INFO] POINT_UPDATE_CLAMP_RATIO: $point_update_clamp_ratio"
echo "[INFO] POINT_UPDATE_EDGE_BETA: $point_update_edge_beta"
echo "[INFO] POINT_UPDATE_EDGE_KERNEL: $point_update_edge_kernel"
echo "[INFO] POINT_UPDATE_NONRIGID_BETA: $point_update_nonrigid_beta"
echo "[INFO] PART_SCORE_ROUTE_ENABLED: $part_score_route_enabled"
echo "[INFO] PART_SCORE_ROUTE_HIDDEN_DIM: $part_score_route_hidden_dim"
echo "[INFO] PART_SCORE_ROUTE_ALPHA: $part_score_route_alpha"
echo "[INFO] PART_SCORE_ROUTE_GATE_BIAS: $part_score_route_gate_bias"
echo "[INFO] PART_SCORE_ROUTE_MODE: $part_score_route_mode"
echo "[INFO] POINT_CLOTH_START_ITER: $point_cloth_start_iter"
echo "[INFO] POINT_CLOTH_END_ITER: $point_cloth_end_iter"
echo "[INFO] POINT_CLOTH_INTERVAL: $point_cloth_interval"
echo "[INFO] POINT_CLOTH_PATCH_SIZE: $point_cloth_patch_size"
echo "[INFO] POINT_CLOTH_TOPK: $point_cloth_topk"
echo "[INFO] POINT_CLOTH_MIN_COVERAGE: $point_cloth_min_coverage"
echo "[INFO] POINT_CLOTH_ANCHOR_RADIUS: $point_cloth_anchor_radius"
echo "[INFO] POINT_CLOTH_MAX_ANCHORS: $point_cloth_max_anchors"
echo "[INFO] POINT_CLOTH_CHILDREN_PER_ANCHOR: $point_cloth_children_per_anchor"
echo "[INFO] POINT_CLOTH_OFFSET_SCALE: $point_cloth_offset_scale"
echo "[INFO] POINT_CLOTH_SCALE_RATIO: $point_cloth_scale_ratio"
echo "[INFO] POINT_CLOTH_OPACITY_RATIO: $point_cloth_opacity_ratio"
echo "[INFO] POINT_CLOTH_OPACITY_MIN: $point_cloth_opacity_min"
echo "[INFO] POINT_CLOTH_MAX_POINTS: $point_cloth_max_points"
echo "[INFO] POINT_CLOTH_TEMPORAL_ALPHA: $point_cloth_temporal_alpha"
echo "[INFO] POINT_CLOTH_HISTORY_MOMENTUM: $point_cloth_history_momentum"
echo "[INFO] POINT_CLOTH_MIN_PERSISTENCE: $point_cloth_min_persistence"
echo "[INFO] POINT_CLOTH_BOUNDARY_BETA: $point_cloth_boundary_beta"
echo "[INFO] POINT_CLOTH_HF_BETA: $point_cloth_hf_beta"
echo "[INFO] POINT_CLOTH_NONRIGID_BETA: $point_cloth_nonrigid_beta"
echo "[INFO] POINT_CLOTH_BOUNDARY_KERNEL: $point_cloth_boundary_kernel"
echo "[INFO] POINT_CLOTH_PROTECT_THRESH: $point_cloth_protect_thresh"
echo "[INFO] POINT_CLOTH_REPLACE_AFTER_ITER: $point_cloth_replace_after_iter"
echo "[INFO] POINT_CLOTH_REPLACEMENT_RATIO: $point_cloth_replacement_ratio"
echo "[INFO] POINT_DEPTH_START_ITER: $point_depth_start_iter"
echo "[INFO] POINT_DEPTH_END_ITER: $point_depth_end_iter"
echo "[INFO] POINT_DEPTH_INTERVAL: $point_depth_interval"
echo "[INFO] POINT_DEPTH_PATCH_SIZE: $point_depth_patch_size"
echo "[INFO] POINT_DEPTH_TOPK: $point_depth_topk"
echo "[INFO] POINT_DEPTH_MIN_COVERAGE: $point_depth_min_coverage"
echo "[INFO] POINT_DEPTH_ANCHOR_RADIUS: $point_depth_anchor_radius"
echo "[INFO] POINT_DEPTH_MAX_ANCHORS: $point_depth_max_anchors"
echo "[INFO] POINT_DEPTH_SURFACE_THRESHOLD: $point_depth_surface_threshold"
echo "[INFO] POINT_DEPTH_DEPTH_THRESHOLD: $point_depth_depth_threshold"
echo "[INFO] POINT_DEPTH_DEPTH_RELATIVE: $point_depth_depth_relative"
echo "[INFO] POINT_DEPTH_CHILDREN_PER_ANCHOR: $point_depth_children_per_anchor"
echo "[INFO] POINT_DEPTH_OFFSET_SCALE: $point_depth_offset_scale"
echo "[INFO] POINT_DEPTH_SCALE_RATIO: $point_depth_scale_ratio"
echo "[INFO] POINT_DEPTH_OPACITY_RATIO: $point_depth_opacity_ratio"
echo "[INFO] POINT_DEPTH_OPACITY_MIN: $point_depth_opacity_min"
echo "[INFO] POINT_DEPTH_MAX_POINTS: $point_depth_max_points"
echo "[INFO] POINT_DEPTH_FIXED_BUDGET: $point_depth_fixed_budget"
echo "[INFO] POINT_DEPTH_REPLACE_OPACITY_W: $point_depth_replace_opacity_w"
echo "[INFO] POINT_DEPTH_REPLACE_GRADIENT_W: $point_depth_replace_gradient_w"
echo "[INFO] POINT_DEPTH_REPLACE_VISIBILITY_W: $point_depth_replace_visibility_w"
echo "[INFO] PART_BUDGET_ENABLED: $part_budget_enabled"
echo "[INFO] PART_BUDGET_ALPHA: $part_budget_alpha"
echo "[INFO] PART_BUDGET_START_ITER: $part_budget_start_iter"
echo "[INFO] PART_BUDGET_WARMUP: $part_budget_warmup"
echo "[INFO] PART_BUDGET_HIDDEN_DIM: $part_budget_hidden_dim"
echo "[INFO] PART_BUDGET_TOKEN_DIM: $part_budget_token_dim"
echo "[INFO] PART_BUDGET_MODE: $part_budget_mode"
echo "[INFO] PART_BUDGET_SUP_W: $part_budget_sup_w"
echo "[INFO] PART_BUDGET_BALANCE_W: $part_budget_balance_w"
echo "[INFO] PART_BUDGET_TARGET_MIX: $part_budget_target_mix"
echo "[INFO] PART_BUDGET_TARGET_SHARPNESS: $part_budget_target_sharpness"
echo "[INFO] PART_BUDGET_ROUTER_SHARPNESS: $part_budget_router_sharpness"
echo "[INFO] TRI_PLANE_DIM: $tri_plane_dim"
echo "[INFO] TRI_PLANE_RES: $tri_plane_res"
echo "[INFO] TRI_PLANE_EXTENT: $tri_plane_extent"
echo "[INFO] TOKEN_TRI_DIM: $token_tri_dim"
echo "[INFO] TOKEN_TRI_RES: $token_tri_res"
echo "[INFO] TOKEN_TRI_EXTENT: $token_tri_extent"
echo "[INFO] TOKEN_TRI_HEADS: $token_tri_heads"
echo "[INFO] TOKEN_TRI_LAYERS: $token_tri_layers"
echo "[INFO] TOKEN_TRI_HIDDEN_DIM: $token_tri_hidden_dim"
echo "[INFO] TOKEN_TRI_FUSION_MODE: $token_tri_fusion_mode"
echo "[INFO] TOKEN_TRI_FUSION_HIDDEN_DIM: $token_tri_fusion_hidden_dim"
echo "[INFO] TOKEN_TRI_ALPHA: $token_tri_alpha"
echo "[INFO] TOKEN_TRI_START_ITER: $token_tri_start_iter"
echo "[INFO] TOKEN_TRI_WARMUP: $token_tri_warmup"
echo "[INFO] TOKEN_TRI_ROUTE_BOUNDARY_W: $token_tri_route_boundary_w"
echo "[INFO] TOKEN_TRI_ROUTE_BOUNDARY_FLOOR: $token_tri_route_boundary_floor"
echo "[INFO] TOKEN_TRI_ROUTE_OUTPUT_ALPHA: $token_tri_route_output_alpha"
echo "[INFO] TOKEN_TRI_ROUTE_HARD_W: $token_tri_route_hard_w"
echo "[INFO] TOKEN_TRI_ROUTE_HARD_BOUNDARY_MIX: $token_tri_route_hard_boundary_mix"
echo "[INFO] TOKEN_TRI_ROUTE_HARD_MOTION_MIX: $token_tri_route_hard_motion_mix"
echo "[INFO] TOKEN_TRI_PART_FUSION_DELTA_SCALE: $token_tri_part_fusion_delta_scale"
echo "[INFO] TOKEN_TRI_PART_FUSION_SPATIAL_DELTA_SCALE: $token_tri_part_fusion_spatial_delta_scale"
echo "[INFO] TOKEN_TRI_PART_FUSION_SPATIAL_W: $token_tri_part_fusion_spatial_w"
echo "[INFO] TOKEN_TRI_PART_FUSION_SPATIAL_STD_FLOOR: $token_tri_part_fusion_spatial_std_floor"
echo "[INFO] TOKEN_TRI_PART_FUSION_SPATIAL_MOTION_MIX: $token_tri_part_fusion_spatial_motion_mix"
echo "[INFO] TOKEN_TRI_PART_FUSION_SPATIAL_BOUNDARY_MIX: $token_tri_part_fusion_spatial_boundary_mix"
echo "[INFO] TOKEN_TRI_PART_FUSION_SPATIAL_PART_MIX: $token_tri_part_fusion_spatial_part_mix"
echo "[INFO] TIME_SCALE_EMB_DIM: $time_scale_emb_dim"
echo "[INFO] TIME_SCALE_TEMPERATURE: $time_scale_temperature"
echo "[INFO] TRI_GATE_ALPHA: $tri_gate_alpha"
echo "[INFO] TRI_GATE_INIT: $tri_gate_init"
echo "[INFO] TRI_GATE_HIDDEN_DIM: $tri_gate_hidden_dim"
echo "[INFO] TRI_GATE_MODE: $tri_gate_mode"
echo "[INFO] TRI_GATE_START_ITER: $tri_gate_start_iter"
echo "[INFO] TRI_GATE_WARMUP: $tri_gate_warmup"
echo "[INFO] TRI_PART_ALPHA: $tri_part_alpha"
echo "[INFO] TRI_PART_MOTION_GAIN: $tri_part_motion_gain"
echo "[INFO] TRI_PART_BOUNDARY_GAIN: $tri_part_boundary_gain"
echo "[INFO] TRI_PART_HIDDEN_DIM: $tri_part_hidden_dim"
echo "[INFO] TRI_PART_REG_W: $tri_part_reg_w"
echo "[INFO] NUM_PARTS: $num_parts"
echo "[INFO] NON_RIGID_MLP_DEPTH: $non_rigid_mlp_depth"
echo "[INFO] NON_RIGID_MLP_WIDTH: $non_rigid_mlp_width"
echo "[INFO] FINAL_EVAL_ONLY: $final_eval_only"
echo "[INFO] SKIP_LOAD_TEST_CAMERAS: $skip_load_test_cameras"
echo "[INFO] IMAGE_DATA_DEVICE: $image_data_device"
echo "[INFO] DENSIFY_UNTIL_ITER: $densify_until_iter"
echo "[INFO] SEED: $seed"
echo "[INFO] VGGT_GARMENT_ENABLED: $vggt_garment_enabled"
echo "[INFO] VGGT_CANDIDATE_PATH: $vggt_candidate_path"
echo "[INFO] VGGT_TARGET_POINTS_FILE: $vggt_target_points_file"
echo "[INFO] VGGT_STRICT_BUDGET: $vggt_strict_budget_enabled"
echo "[INFO] VGGT_HIGH_ERROR: $vggt_high_error_enabled"
echo "[INFO] VGGT_HIGH_ERROR_QUANTILE: $vggt_high_error_quantile"
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
    --seed "$seed"
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

TIME_ARGS=()
if [ "$time_enabled" = "1" ]; then
    TIME_ARGS=(
        --use_time
        --time_scale_emb_dim "$time_scale_emb_dim"
        --time_scale_temperature "$time_scale_temperature"
    )
fi

POINT_ARGS=()
if [ "$point_enabled" = "1" ]; then
    POINT_ARGS=(
        --use_point
        --point_patch_size "$point_patch_size"
        --point_start_iter "$point_start_iter"
        --point_interval "$point_interval"
        --point_topk_patches "$point_topk_patches"
        --point_anchor_radius "$point_anchor_radius"
        --point_max_anchors "$point_max_anchors"
        --point_grad_boost "$point_grad_boost"
        --point_min_patch_coverage "$point_min_patch_coverage"
    )
fi

POINT_ANCHOR_ARGS=()
if [ "$point_anchor_enabled" = "1" ]; then
    POINT_ANCHOR_ARGS=(
        --use_point_anchor
        --point_anchor_start_iter "$point_anchor_start_iter"
        --point_anchor_end_iter "$point_anchor_end_iter"
        --point_anchor_interval "$point_anchor_interval"
        --point_anchor_patch_size "$point_anchor_patch_size"
        --point_anchor_topk "$point_anchor_topk"
        --point_anchor_min_coverage "$point_anchor_min_coverage"
        --point_anchor_radius "$point_anchor_spawn_radius"
        --point_anchor_max_anchors "$point_anchor_spawn_max_anchors"
        --point_anchor_children_per_anchor "$point_anchor_children_per_anchor"
        --point_anchor_offset_scale "$point_anchor_offset_scale"
        --point_anchor_scale_ratio "$point_anchor_scale_ratio"
        --point_anchor_opacity_ratio "$point_anchor_opacity_ratio"
        --point_anchor_opacity_min "$point_anchor_opacity_min"
        --point_anchor_max_points "$point_anchor_max_points"
    )
fi

POINT_ANCHOR_TB_ARGS=()
if [ "$point_anchor_tb_enabled" = "1" ]; then
    POINT_ANCHOR_TB_ARGS=(
        --use_point_anchor_tb
        --point_tb_start_iter "$point_tb_start_iter"
        --point_tb_end_iter "$point_tb_end_iter"
        --point_tb_interval "$point_tb_interval"
        --point_tb_patch_size "$point_tb_patch_size"
        --point_tb_topk "$point_tb_topk"
        --point_tb_min_coverage "$point_tb_min_coverage"
        --point_tb_anchor_radius "$point_tb_anchor_radius"
        --point_tb_max_anchors "$point_tb_max_anchors"
        --point_tb_children_per_anchor "$point_tb_children_per_anchor"
        --point_tb_offset_scale "$point_tb_offset_scale"
        --point_tb_scale_ratio "$point_tb_scale_ratio"
        --point_tb_opacity_ratio "$point_tb_opacity_ratio"
        --point_tb_opacity_min "$point_tb_opacity_min"
        --point_tb_max_points "$point_tb_max_points"
        --point_tb_temporal_alpha "$point_tb_temporal_alpha"
        --point_tb_history_momentum "$point_tb_history_momentum"
        --point_tb_min_persistence "$point_tb_min_persistence"
        --point_tb_boundary_beta "$point_tb_boundary_beta"
        --point_tb_boundary_kernel "$point_tb_boundary_kernel"
        --point_tb_target_points "$point_tb_target_points"
        --point_tb_replace_after_iter "$point_tb_replace_after_iter"
        --point_tb_replacement_ratio "$point_tb_replacement_ratio"
        --point_tb_replace_opacity_w "$point_tb_replace_opacity_w"
        --point_tb_replace_gradient_w "$point_tb_replace_gradient_w"
        --point_tb_replace_visibility_w "$point_tb_replace_visibility_w"
    )
    if [ "$point_update_enabled" = "1" ]; then
        POINT_ANCHOR_TB_ARGS+=(
            --use_point_update
            --point_update_ema_momentum "$point_update_ema_momentum"
            --point_update_min_keep "$point_update_min_keep"
            --point_update_final_clamp_iter "$point_update_final_clamp_iter"
            --point_update_clamp_interval "$point_update_clamp_interval"
            --point_update_clamp_ratio "$point_update_clamp_ratio"
            --point_update_edge_beta "$point_update_edge_beta"
            --point_update_edge_kernel "$point_update_edge_kernel"
            --point_update_nonrigid_beta "$point_update_nonrigid_beta"
        )
        if [ "$point_update_protect_anchors" = "1" ]; then
            POINT_ANCHOR_TB_ARGS+=(--point_update_protect_anchors)
        fi
    fi
fi

POINT_CLOTH_ARGS=()
if [ "$point_cloth_budget_enabled" = "1" ]; then
    POINT_CLOTH_ARGS=(
        --use_point_cloth_budget
        --point_cloth_start_iter "$point_cloth_start_iter"
        --point_cloth_end_iter "$point_cloth_end_iter"
        --point_cloth_interval "$point_cloth_interval"
        --point_cloth_patch_size "$point_cloth_patch_size"
        --point_cloth_topk "$point_cloth_topk"
        --point_cloth_min_coverage "$point_cloth_min_coverage"
        --point_cloth_anchor_radius "$point_cloth_anchor_radius"
        --point_cloth_max_anchors "$point_cloth_max_anchors"
        --point_cloth_children_per_anchor "$point_cloth_children_per_anchor"
        --point_cloth_offset_scale "$point_cloth_offset_scale"
        --point_cloth_scale_ratio "$point_cloth_scale_ratio"
        --point_cloth_opacity_ratio "$point_cloth_opacity_ratio"
        --point_cloth_opacity_min "$point_cloth_opacity_min"
        --point_cloth_max_points "$point_cloth_max_points"
        --point_cloth_temporal_alpha "$point_cloth_temporal_alpha"
        --point_cloth_history_momentum "$point_cloth_history_momentum"
        --point_cloth_min_persistence "$point_cloth_min_persistence"
        --point_cloth_boundary_beta "$point_cloth_boundary_beta"
        --point_cloth_hf_beta "$point_cloth_hf_beta"
        --point_cloth_nonrigid_beta "$point_cloth_nonrigid_beta"
        --point_cloth_boundary_kernel "$point_cloth_boundary_kernel"
        --point_cloth_protect_thresh "$point_cloth_protect_thresh"
        --point_cloth_replace_after_iter "$point_cloth_replace_after_iter"
        --point_cloth_replacement_ratio "$point_cloth_replacement_ratio"
        --point_update_ema_momentum "$point_update_ema_momentum"
        --point_update_min_keep "$point_update_min_keep"
        --point_update_final_clamp_iter "$point_update_final_clamp_iter"
        --point_update_clamp_interval "$point_update_clamp_interval"
        --point_update_clamp_ratio "$point_update_clamp_ratio"
    )
fi

POINT_DEPTH_ARGS=()
if [ "$point_depth_enabled" = "1" ]; then
    POINT_DEPTH_ARGS=(
        --use_point_depth
        --point_depth_start_iter "$point_depth_start_iter"
        --point_depth_end_iter "$point_depth_end_iter"
        --point_depth_interval "$point_depth_interval"
        --point_depth_patch_size "$point_depth_patch_size"
        --point_depth_topk "$point_depth_topk"
        --point_depth_min_coverage "$point_depth_min_coverage"
        --point_depth_anchor_radius "$point_depth_anchor_radius"
        --point_depth_max_anchors "$point_depth_max_anchors"
        --point_depth_surface_threshold "$point_depth_surface_threshold"
        --point_depth_depth_threshold "$point_depth_depth_threshold"
        --point_depth_depth_relative "$point_depth_depth_relative"
        --point_depth_children_per_anchor "$point_depth_children_per_anchor"
        --point_depth_offset_scale "$point_depth_offset_scale"
        --point_depth_scale_ratio "$point_depth_scale_ratio"
        --point_depth_opacity_ratio "$point_depth_opacity_ratio"
        --point_depth_opacity_min "$point_depth_opacity_min"
        --point_depth_max_points "$point_depth_max_points"
        --point_depth_replace_opacity_w "$point_depth_replace_opacity_w"
        --point_depth_replace_gradient_w "$point_depth_replace_gradient_w"
        --point_depth_replace_visibility_w "$point_depth_replace_visibility_w"
    )
    if [ "$point_depth_fixed_budget" = "1" ]; then
        POINT_DEPTH_ARGS+=(--point_depth_fixed_budget)
    fi
fi

VGGT_GARMENT_ARGS=()
if [ "$vggt_garment_enabled" = "1" ]; then
    VGGT_GARMENT_ARGS=(
        --use_vggt_garment
        --vggt_candidate_path "$vggt_candidate_path"
        --vggt_target_points_file "$vggt_target_points_file"
        --vggt_garment_start_iter "$vggt_garment_start_iter"
        --vggt_garment_end_iter "$vggt_garment_end_iter"
        --vggt_garment_interval "$vggt_garment_interval"
        --vggt_garment_max_spawn "$vggt_garment_max_spawn"
        --vggt_candidate_opacity "$vggt_candidate_opacity"
        --vggt_candidate_scale_ratio "$vggt_candidate_scale_ratio"
        --vggt_candidate_min_distance "$vggt_candidate_min_distance"
        --vggt_max_points "$vggt_max_points"
        --vggt_replace_opacity_w "$vggt_replace_opacity_w"
        --vggt_replace_gradient_w "$vggt_replace_gradient_w"
        --vggt_replace_visibility_w "$vggt_replace_visibility_w"
        --vggt_min_keep "$vggt_min_keep"
        --vggt_high_error_quantile "$vggt_high_error_quantile"
        --vggt_high_error_candidate_w "$vggt_high_error_candidate_w"
    )
    if [ "$vggt_strict_budget_enabled" = "1" ]; then
        VGGT_GARMENT_ARGS+=(--vggt_strict_budget)
    fi
    if [ "$vggt_high_error_enabled" = "1" ]; then
        VGGT_GARMENT_ARGS+=(--vggt_high_error_enabled)
    fi
fi

PART_MOE_ARGS=()
if [ "$part_moe_enabled" = "1" ]; then
    PART_MOE_ARGS=(
        --use_part_moe
        --part_moe_start_iter "$part_moe_start_iter"
        --part_moe_warmup "$part_moe_warmup"
        --part_moe_global_keep "$part_moe_global_keep"
        --part_label_knn "$part_label_knn"
        --part_label_vote_temperature "$part_label_vote_temperature"
        --part_label_refresh_interval "$part_label_refresh_interval"
        --part_moe_conf_threshold "$part_moe_conf_threshold"
        --num_parts "$num_parts"
        --part_label_schema "$part_label_schema"
        --part_log_dir "$AUTO_PART_LOG_DIR"
    )
    if [ "$part_label_robust_enabled" = "1" ]; then
        PART_MOE_ARGS+=(--part_label_robust)
    fi
    if [ "$part_confidence_route_enabled" = "1" ]; then
        PART_MOE_ARGS+=(--part_confidence_route)
    fi
    if [ "$part_point_enabled" = "1" ]; then
        PART_MOE_ARGS+=(--use_part_point)
    fi
    if [ "$part_score_route_enabled" = "1" ]; then
        PART_MOE_ARGS+=(
            --use_part_score_route
            --part_score_route_hidden_dim "$part_score_route_hidden_dim"
            --part_score_route_alpha "$part_score_route_alpha"
            --part_score_route_gate_bias "$part_score_route_gate_bias"
            --part_score_route_mode "$part_score_route_mode"
        )
    fi
    if [ "$temporal_conditioned_part_enabled" = "1" ]; then
        PART_MOE_ARGS+=(
            --use_temporal_conditioned_part_moe
            --temporal_conditioned_part_fusion_mode "$temporal_conditioned_part_fusion_mode"
            --temporal_conditioned_part_conf_threshold "$temporal_conditioned_part_conf_threshold"
            --temporal_conditioned_part_max_mix "$temporal_conditioned_part_max_mix"
        )
    fi
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
                --tri_gate_mode "$tri_gate_mode"
                --tri_gate_start_iter "$tri_gate_start_iter"
                --tri_gate_warmup "$tri_gate_warmup"
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
            --part_budget_mode "$part_budget_mode"
            --part_budget_sup_w "$part_budget_sup_w"
            --part_budget_balance_w "$part_budget_balance_w"
            --part_budget_target_mix "$part_budget_target_mix"
            --part_budget_target_sharpness "$part_budget_target_sharpness"
            --part_budget_router_sharpness "$part_budget_router_sharpness"
        )
    fi
fi

TRI_TOKEN_ARGS=()
if [ "$tri_token_enabled" = "1" ]; then
    TRI_TOKEN_ARGS=(
        --use_tri_token
        --num_parts "$num_parts"
        --part_label_schema "$part_label_schema"
        --token_tri_dim "$token_tri_dim"
        --token_tri_res "$token_tri_res"
        --token_tri_extent "$token_tri_extent"
        --token_tri_heads "$token_tri_heads"
        --token_tri_layers "$token_tri_layers"
        --token_tri_hidden_dim "$token_tri_hidden_dim"
        --token_tri_fusion_mode "$token_tri_fusion_mode"
        --token_tri_fusion_hidden_dim "$token_tri_fusion_hidden_dim"
        --token_tri_alpha "$token_tri_alpha"
        --token_tri_start_iter "$token_tri_start_iter"
        --token_tri_warmup "$token_tri_warmup"
        --token_tri_route_boundary_w "$token_tri_route_boundary_w"
        --token_tri_route_boundary_floor "$token_tri_route_boundary_floor"
        --token_tri_route_output_alpha "$token_tri_route_output_alpha"
        --token_tri_route_hard_w "$token_tri_route_hard_w"
        --token_tri_route_hard_boundary_mix "$token_tri_route_hard_boundary_mix"
        --token_tri_route_hard_motion_mix "$token_tri_route_hard_motion_mix"
        --token_tri_part_fusion_delta_scale "$token_tri_part_fusion_delta_scale"
        --token_tri_part_fusion_spatial_delta_scale "$token_tri_part_fusion_spatial_delta_scale"
        --token_tri_part_fusion_spatial_w "$token_tri_part_fusion_spatial_w"
        --token_tri_part_fusion_spatial_std_floor "$token_tri_part_fusion_spatial_std_floor"
        --token_tri_part_fusion_spatial_motion_mix "$token_tri_part_fusion_spatial_motion_mix"
        --token_tri_part_fusion_spatial_boundary_mix "$token_tri_part_fusion_spatial_boundary_mix"
        --token_tri_part_fusion_spatial_part_mix "$token_tri_part_fusion_spatial_part_mix"
    )
fi

DYNOMO_C_ARGS=()
if [ "$dynomo_c_enabled" = "1" ]; then
    DYNOMO_C_ARGS=(
        --use_dynomo_c
        --dynomo_c_label_iter "$dynomo_c_label_iter"
        --dynomo_c_affinity_dim "$dynomo_c_affinity_dim"
        --dynomo_c_knn "$dynomo_c_knn"
        --dynomo_c_part_same_w "$dynomo_c_part_same_w"
        --dynomo_c_part_adj_w "$dynomo_c_part_adj_w"
        --dynomo_c_part_other_w "$dynomo_c_part_other_w"
        --dynomo_c_motion_w "$dynomo_c_motion_w"
        --dynomo_c_rotation_w "$dynomo_c_rotation_w"
    )
fi

MAPO_ARGS=()
if [ "$mapo_all_dynamic_enabled" = "1" ]; then
    MAPO_ARGS=(
        --use_mapo_all_dynamic
        --mapo_max_partition_level "$mapo_max_partition_level"
        --mapo_partition_level1_iter "$mapo_partition_level1_iter"
        --mapo_partition_level2_iter "$mapo_partition_level2_iter"
        --mapo_partition_level3_iter "$mapo_partition_level3_iter"
        --mapo_num_frames "$mapo_num_frames"
        --mapo_soft_blend_width "$mapo_soft_blend_width"
        --mapo_residual_alpha "$mapo_residual_alpha"
        --mapo_dynamic_score_momentum "$mapo_dynamic_score_momentum"
        --mapo_dynamic_score_alpha "$mapo_dynamic_score_alpha"
    )
    if [ "$mapo_soft_routing_enabled" = "1" ]; then
        MAPO_ARGS+=(--mapo_soft_routing)
    fi
    if [ "$mapo_shared_trunk_enabled" = "1" ]; then
        MAPO_ARGS+=(--mapo_shared_trunk)
    fi
    if [ "$mapo_partial_sharing_enabled" = "1" ]; then
        MAPO_ARGS+=(--mapo_partial_sharing)
    fi
    if [ "$mapo_dynamic_score_enabled" = "1" ]; then
        MAPO_ARGS+=(--mapo_dynamic_score_enabled)
    fi
fi

MOTION_TEMPERATURE_ARGS=()
if [ "$motion_temperature_enabled" = "1" ]; then
    MOTION_TEMPERATURE_ARGS=(
        --use_motion_temporal_temperature
        --motion_temperature_min "$motion_temperature_min"
        --motion_temperature_max "$motion_temperature_max"
        --motion_velocity_weight "$motion_velocity_weight"
        --motion_acceleration_weight "$motion_acceleration_weight"
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
        "${TIME_ARGS[@]}" \
        "${POINT_ARGS[@]}" \
        "${POINT_ANCHOR_ARGS[@]}" \
        "${POINT_ANCHOR_TB_ARGS[@]}" \
        "${POINT_CLOTH_ARGS[@]}" \
        "${POINT_DEPTH_ARGS[@]}" \
        "${VGGT_GARMENT_ARGS[@]}" \
        "${DYNOMO_C_ARGS[@]}" \
        "${MAPO_ARGS[@]}" \
        "${MOTION_TEMPERATURE_ARGS[@]}" \
        "${PART_MOE_ARGS[@]}" \
        "${TRI_TOKEN_ARGS[@]}" \
        2>&1 | tee "$model_path/logs/train_${SEQUENCE}_${experiment_name}.log"

    echo "[INFO] Evaluating on GPU $GPU_id for sequence $SEQUENCE"
    RENDER_ENV=(CUDA_VISIBLE_DEVICES="$GPU_id")
    if [ "$image_data_device" != "cuda" ]; then
        RENDER_ENV+=(SEQAVATAR_IMAGE_DATA_DEVICE="$image_data_device")
    fi
    env "${RENDER_ENV[@]}" "$PYTHON_BIN" render.py \
        -s "$dataset_path" -m "$model_path" \
        "${COMMON_RENDER_ARGS[@]}" \
        "${TIME_ARGS[@]}" \
        "${POINT_ARGS[@]}" \
        "${POINT_ANCHOR_ARGS[@]}" \
        "${POINT_ANCHOR_TB_ARGS[@]}" \
        "${POINT_CLOTH_ARGS[@]}" \
        "${POINT_DEPTH_ARGS[@]}" \
        "${MAPO_ARGS[@]}" \
        "${MOTION_TEMPERATURE_ARGS[@]}" \
        "${PART_MOE_ARGS[@]}" \
        "${TRI_TOKEN_ARGS[@]}" \
        2>&1 | tee "$model_path/logs/render_${SEQUENCE}_${experiment_name}.log"

    echo "[INFO] Finished sequence: $SEQUENCE"
done

echo "================================================="
echo "[INFO] End time: $(date)"
echo "[INFO] All sequences finished."
echo "[INFO] Global log saved to: $GLOBAL_LOG_FILE"
echo "================================================="
