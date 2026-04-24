import os
import re
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from utils.general_utils import inverse_sigmoid


class DepthPriorPruner:
    def __init__(self, opt, train_cameras, gaussian_device):
        self.enabled = bool(getattr(opt, "depth_prior_enable", False))
        self.depth_dir = getattr(opt, "depth_prior_dir", "")
        self.interval = max(1, int(getattr(opt, "depth_prior_interval", 100)))
        self.start_iter = int(getattr(opt, "depth_prior_start_iter", 0))
        self.end_iter = int(getattr(opt, "depth_prior_end_iter", -1))
        self.tau_z = float(getattr(opt, "depth_prior_tau_z", 0.05))
        self.tau_alpha = float(getattr(opt, "depth_prior_tau_alpha", 0.005))
        self.soft_factor = float(getattr(opt, "depth_prior_soft_factor", 0.7))
        self.hard_prune = bool(getattr(opt, "depth_prior_hard_prune", False))
        self.max_prune_ratio = float(getattr(opt, "depth_prior_max_prune_ratio", 0.1))
        self.views_per_iter = max(1, int(getattr(opt, "depth_prior_views_per_iter", 1)))
        self.min_valid_views = max(1, int(getattr(opt, "depth_prior_min_valid_views", 1)))
        self.temporal_momentum = float(getattr(opt, "depth_prior_temporal_momentum", 0.0))
        self.depth_scale = float(getattr(opt, "depth_prior_depth_scale", 1.0))
        self.depth_min = 1e-6
        self.min_valid_ratio = float(getattr(opt, "depth_prior_min_valid_ratio", 0.0))
        self.rel_tau = float(getattr(opt, "depth_prior_rel_tau", 0.0))
        self.use_low_alpha = bool(getattr(opt, "depth_prior_use_low_alpha", False))
        self.hard_prune_start_iter = int(getattr(opt, "depth_prior_hard_prune_start_iter", self.start_iter))
        self.use_all_pose_views = bool(getattr(opt, "depth_prior_use_all_pose_views", True))
        self.view_fusion_mode = str(getattr(opt, "depth_prior_view_fusion", "min")).strip().lower()
        if self.view_fusion_mode not in {"min", "mean"}:
            self.view_fusion_mode = "min"
        self.temporal_window = max(1, int(getattr(opt, "depth_prior_temporal_window", 4)))
        self.temporal_mode = str(getattr(opt, "depth_prior_temporal_mode", "mean")).strip().lower()
        if self.temporal_mode not in {"mean", "last"}:
            self.temporal_mode = "mean"
        self.use_exp_alpha_decay = bool(getattr(opt, "depth_prior_exp_alpha_decay", True))
        self.k_alpha = float(getattr(opt, "depth_prior_k_alpha", 0.15))
        self.dynamic_tau_strength = float(getattr(opt, "depth_prior_dynamic_tau_strength", 0.9))
        self.low_quality_moderate_disable = float(getattr(opt, "depth_prior_low_quality_moderate_disable", 0.30))
        self.hard_prune_min_quality = float(getattr(opt, "depth_prior_hard_prune_min_quality", 0.45))
        self.decay_min_quality_scale = float(getattr(opt, "depth_prior_decay_min_quality_scale", 0.35))
        self.min_quality_score = float(getattr(opt, "depth_prior_min_quality_score", 0.05))
        self.front_consensus_ratio = 0.55
        self.severe_consensus_ratio = 0.25
        self.profile = os.environ.get("DEPTH_PRIOR_PROFILE", "adaptive_v1").strip().lower()
        self._apply_profile_overrides()

        self.compute_device = self._resolve_device(getattr(opt, "depth_prior_device", "cuda"), gaussian_device)
        self.gaussian_device = gaussian_device

        if not self.depth_dir or not os.path.isdir(self.depth_dir):
            self.enabled = False

        self.depth_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.image_to_depth_path: Dict[str, Optional[str]] = {}
        self.pose_to_cameras: Dict[int, List] = {}
        for cam in train_cameras:
            self.pose_to_cameras.setdefault(cam.pose_id, []).append(cam)
        for pose_id in self.pose_to_cameras:
            self.pose_to_cameras[pose_id] = sorted(self.pose_to_cameras[pose_id], key=lambda c: c.image_name)

        self._ema_delta: Optional[torch.Tensor] = None
        self._dz_history = deque(maxlen=self.temporal_window)

    @staticmethod
    def _resolve_device(device_str, fallback_device):
        try:
            device = torch.device(device_str)
            if device.type == "cuda" and not torch.cuda.is_available():
                return fallback_device
            if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
                return fallback_device
            return device
        except Exception:
            return fallback_device

    def _apply_profile_overrides(self):
        # Keep the training loop arguments fixed and only adjust behavior inside this module.
        # Profiles are selected by environment variable DEPTH_PRIOR_PROFILE.
        profile = self.profile
        if profile == "conservative_v1":
            self.interval = 500
            self.start_iter = 12000
            self.end_iter = -1
            self.tau_z = 0.30
            self.rel_tau = 0.05
            self.soft_factor = 0.999
            self.hard_prune = False
            self.views_per_iter = max(self.views_per_iter, 1)
            self.min_valid_views = max(self.min_valid_views, 1)
            self.min_valid_ratio = max(self.min_valid_ratio, 0.05)
            self.temporal_momentum = max(self.temporal_momentum, 0.6)
            self.front_consensus_ratio = 0.70
            self.severe_consensus_ratio = 0.40
        elif profile == "balanced_v1":
            self.interval = 350
            self.start_iter = 8000
            self.end_iter = -1
            self.tau_z = 0.22
            self.rel_tau = 0.05
            self.soft_factor = 0.996
            self.hard_prune = False
            self.views_per_iter = max(self.views_per_iter, 2)
            self.min_valid_views = max(self.min_valid_views, 1)
            self.min_valid_ratio = max(self.min_valid_ratio, 0.05)
            self.temporal_momentum = max(self.temporal_momentum, 0.7)
            self.front_consensus_ratio = 0.65
            self.severe_consensus_ratio = 0.35
        elif profile == "balanced_v2":
            self.interval = 250
            self.start_iter = 6000
            self.end_iter = -1
            self.tau_z = 0.18
            self.rel_tau = 0.04
            self.soft_factor = 0.994
            self.hard_prune = False
            self.views_per_iter = max(self.views_per_iter, 2)
            self.min_valid_views = max(self.min_valid_views, 1)
            self.min_valid_ratio = max(self.min_valid_ratio, 0.04)
            self.temporal_momentum = max(self.temporal_momentum, 0.75)
            self.front_consensus_ratio = 0.62
            self.severe_consensus_ratio = 0.32
        elif profile == "aggressive_v1":
            self.interval = 220
            self.start_iter = 5000
            self.end_iter = -1
            self.tau_z = 0.14
            self.rel_tau = 0.03
            self.soft_factor = 0.992
            self.hard_prune = False
            self.views_per_iter = max(self.views_per_iter, 2)
            self.min_valid_views = max(self.min_valid_views, 1)
            self.min_valid_ratio = max(self.min_valid_ratio, 0.03)
            self.temporal_momentum = max(self.temporal_momentum, 0.75)
            self.front_consensus_ratio = 0.58
            self.severe_consensus_ratio = 0.28
        else:
            # adaptive_v1 default: safe early/late gating and mild soft decay.
            self.interval = 300
            self.start_iter = 7000
            self.end_iter = -1
            self.tau_z = 0.20
            self.rel_tau = 0.04
            self.soft_factor = 0.996
            self.hard_prune = False
            self.views_per_iter = max(self.views_per_iter, 2)
            self.min_valid_views = max(self.min_valid_views, 1)
            self.min_valid_ratio = max(self.min_valid_ratio, 0.04)
            self.temporal_momentum = max(self.temporal_momentum, 0.7)
            self.front_consensus_ratio = 0.64
            self.severe_consensus_ratio = 0.34

    def should_apply(self, iteration: int) -> bool:
        if not self.enabled:
            return False
        if iteration < self.start_iter:
            return False
        if self.end_iter >= 0 and iteration > self.end_iter:
            return False
        if iteration % self.interval != 0:
            return False
        return True

    def apply(self, iteration: int, gaussians, viewpoint_cam):
        if not self.should_apply(iteration):
            return {"applied": False}

        selected_views = self._select_views(viewpoint_cam)
        if len(selected_views) == 0:
            return {"applied": False}

        depth_result = self._evaluate_depth_mismatch(
            gaussians.get_xyz.detach(),
            selected_views,
            gaussians.get_opacity.detach().squeeze(-1),
        )
        if depth_result is None:
            return {"applied": False}

        mismatch_mask = depth_result["mismatch_mask"].to(self.gaussian_device)
        severe_mask = depth_result["severe_mask"].to(self.gaussian_device)
        valid_mask = depth_result["valid_mask"].to(self.gaussian_device)
        delta_z = depth_result["delta_z"].to(self.gaussian_device)
        valid_count = depth_result["valid_count"].to(self.gaussian_device)
        used_views = int(depth_result["used_views"])
        quality_score = float(depth_result.get("quality_score", 1.0))
        adaptive_scale = float(depth_result.get("adaptive_scale", 1.0))
        abs_bad_ratio = depth_result.get("abs_bad_ratio", None)
        if isinstance(abs_bad_ratio, torch.Tensor):
            abs_bad_ratio = abs_bad_ratio.to(self.gaussian_device)
        else:
            abs_bad_ratio = None

        valid_ratio = float(valid_mask.float().mean().item()) if valid_mask.numel() > 0 else 0.0
        if valid_ratio < self.min_valid_ratio:
            return {
                "applied": False,
                "views_used": used_views,
                "valid_ratio": valid_ratio,
                "mean_delta_z": float(delta_z[valid_mask].mean().item()) if valid_mask.any() else 0.0,
                "depth_bad_count": int(mismatch_mask.sum().item()),
                "low_alpha_count": 0,
                "soft_count": 0,
                "hard_count": 0,
                "mean_valid_views": float(valid_count.float()[valid_mask].mean().item()) if valid_mask.any() else 0.0,
                "skipped_low_valid_ratio": 1,
                "depth_quality_score": quality_score,
                "adaptive_scale": adaptive_scale,
            }

        opacity = gaussians.get_opacity.detach().squeeze(-1)
        low_alpha_mask = opacity < self.tau_alpha if self.use_low_alpha else torch.zeros_like(opacity, dtype=torch.bool)

        soft_count = 0
        moderate_mask = mismatch_mask & (~severe_mask)
        if quality_score < self.low_quality_moderate_disable:
            if abs_bad_ratio is not None:
                strict_gate = abs_bad_ratio >= (1.25 * adaptive_scale * 1.25)
                moderate_mask = moderate_mask & strict_gate
            else:
                moderate_mask = torch.zeros_like(moderate_mask)
        if self.use_exp_alpha_decay:
            soft_count = self._apply_exp_decay(
                gaussians=gaussians,
                delta_z=delta_z,
                moderate_mask=moderate_mask,
                severe_mask=severe_mask,
                valid_mask=valid_mask,
                low_alpha_mask=low_alpha_mask,
                quality_score=quality_score,
                abs_bad_ratio=abs_bad_ratio,
            )
        else:
            # Keep soft pruning conservative: remap aggressive factors to milder decay.
            eff_soft = 1.0 - (1.0 - self.soft_factor) * 0.2
            severe_soft = max(0.85, eff_soft - 0.05)
            if severe_soft < 1.0:
                soft_count += gaussians.apply_opacity_decay(severe_mask, severe_soft)
            if eff_soft < 1.0:
                soft_count += gaussians.apply_opacity_decay(moderate_mask, eff_soft)
            if self.use_low_alpha and eff_soft < 1.0:
                soft_count += gaussians.apply_opacity_decay(low_alpha_mask & severe_mask, eff_soft)

        hard_count = 0
        if self.hard_prune and iteration >= self.hard_prune_start_iter and quality_score >= self.hard_prune_min_quality:
            # Hard stage: only prune severe front-outliers with multi-view agreement.
            hard_mask = severe_mask
            hard_mask = self._limit_hard_prune(hard_mask, delta_z, low_alpha_mask)
            if hard_mask.any():
                hard_count = int(hard_mask.sum().item())
                gaussians.prune_points(hard_mask)
                self._ema_delta = None
                self._dz_history.clear()

        if valid_mask.any():
            mean_delta_z = float(delta_z[valid_mask].mean().item())
        else:
            mean_delta_z = 0.0

        return {
            "applied": True,
            "views_used": used_views,
            "valid_ratio": valid_ratio,
            "mean_delta_z": mean_delta_z,
            "depth_bad_count": int(mismatch_mask.sum().item()),
            "low_alpha_count": int(low_alpha_mask.sum().item()) if self.use_low_alpha else 0,
            "soft_count": soft_count,
            "hard_count": hard_count,
            "mean_valid_views": float(valid_count.float()[valid_mask].mean().item()) if valid_mask.any() else 0.0,
            "skipped_low_valid_ratio": 0,
            "depth_quality_score": quality_score,
            "adaptive_scale": adaptive_scale,
        }

    def _select_views(self, viewpoint_cam):
        pose_cams = self.pose_to_cameras.get(viewpoint_cam.pose_id, [])
        if len(pose_cams) == 0:
            return [viewpoint_cam]

        selected = [viewpoint_cam]
        if self.views_per_iter <= 1:
            return selected
        if self.use_all_pose_views:
            return pose_cams

        for cam in pose_cams:
            if cam.image_name == viewpoint_cam.image_name:
                continue
            selected.append(cam)
            if len(selected) >= self.views_per_iter:
                break
        return selected

    def _evaluate_depth_mismatch(self, xyz_world: torch.Tensor, views, opacity: Optional[torch.Tensor]) -> Optional[dict]:
        if xyz_world.numel() == 0:
            return None

        xyz = xyz_world.to(self.compute_device)
        n_points = xyz.shape[0]
        delta_sum = torch.zeros(n_points, dtype=torch.float32, device=self.compute_device)
        delta_min = torch.full((n_points,), float("inf"), dtype=torch.float32, device=self.compute_device)
        depth_sum = torch.zeros(n_points, dtype=torch.float32, device=self.compute_device)
        valid_count = torch.zeros(n_points, dtype=torch.int32, device=self.compute_device)
        front_bad_count = torch.zeros(n_points, dtype=torch.int32, device=self.compute_device)
        severe_front_count = torch.zeros(n_points, dtype=torch.int32, device=self.compute_device)
        used_views = 0
        if opacity is None:
            active_mask = torch.ones(n_points, dtype=torch.bool, device=self.compute_device)
        else:
            active_mask = opacity.to(self.compute_device) > max(self.tau_alpha * 0.5, 1e-3)

        for cam in views:
            depth_bundle = self._load_depth_map(cam.image_name)
            if depth_bundle is None:
                continue

            used_views += 1
            depth = depth_bundle["depth"].to(self.compute_device)
            depth_valid = depth_bundle["valid_mask"].to(self.compute_device)
            if depth.dim() == 3:
                depth = depth.squeeze()
            if depth_valid.dim() == 3:
                depth_valid = depth_valid.squeeze()
            if depth.dim() != 2:
                continue
            if depth_valid.shape != depth.shape:
                continue

            h, w = depth.shape
            k = torch.as_tensor(cam.K, dtype=torch.float32, device=self.compute_device)
            r = torch.as_tensor(cam.R, dtype=torch.float32, device=self.compute_device)
            t = torch.as_tensor(cam.T, dtype=torch.float32, device=self.compute_device)

            xyz_cam = xyz @ r + t.unsqueeze(0)
            z = xyz_cam[:, 2]
            z_safe = torch.clamp(z, min=1e-6)

            u = (k[0, 0] * xyz_cam[:, 0] + k[0, 1] * xyz_cam[:, 1]) / z_safe + k[0, 2]
            v = (k[1, 1] * xyz_cam[:, 1]) / z_safe + k[1, 2]

            if cam.image_width != w:
                u = u * (float(w) / float(cam.image_width))
            if cam.image_height != h:
                v = v * (float(h) / float(cam.image_height))

            u_idx = torch.round(u).long()
            v_idx = torch.round(v).long()
            in_img = (u_idx >= 0) & (u_idx < w) & (v_idx >= 0) & (v_idx < h)
            valid = active_mask & (z > self.depth_min) & in_img
            if not valid.any():
                continue

            sampled_depth = torch.zeros_like(z)
            sampled_depth[valid] = depth[v_idx[valid], u_idx[valid]]
            sampled_depth_valid = torch.zeros_like(valid)
            sampled_depth_valid[valid] = depth_valid[v_idx[valid], u_idx[valid]]
            valid = valid & torch.isfinite(sampled_depth) & (sampled_depth > self.depth_min)
            valid = valid & sampled_depth_valid
            if not valid.any():
                continue

            depth_scale = self._estimate_view_scale(z[valid], sampled_depth[valid])
            sampled_depth = sampled_depth * depth_scale
            signed_delta = z - sampled_depth
            delta_z = torch.abs(signed_delta)
            rel_delta = delta_z / torch.clamp(sampled_depth, min=self.depth_min)
            abs_tau = self.tau_z + self.rel_tau * sampled_depth
            rel_gate = rel_delta > max(0.01, self.rel_tau * 0.5)
            front_bad = valid & (signed_delta < -abs_tau) & rel_gate
            severe_front_bad = valid & (signed_delta < -(abs_tau * 1.8)) & rel_gate

            delta_sum[valid] += delta_z[valid]
            delta_min[valid] = torch.minimum(delta_min[valid], delta_z[valid])
            depth_sum[valid] += sampled_depth[valid]
            valid_count[valid] += 1
            front_bad_count[front_bad] += 1
            severe_front_count[severe_front_bad] += 1

        if used_views == 0:
            return None

        enough_views = valid_count >= self.min_valid_views
        mean_delta = torch.zeros_like(delta_sum)
        min_delta = torch.zeros_like(delta_sum)
        mean_depth = torch.zeros_like(depth_sum)
        mean_delta[enough_views] = delta_sum[enough_views] / valid_count[enough_views].float()
        finite_min = torch.isfinite(delta_min)
        min_delta[enough_views & finite_min] = delta_min[enough_views & finite_min]
        mean_depth[enough_views] = depth_sum[enough_views] / valid_count[enough_views].float()
        fused_delta = min_delta if self.view_fusion_mode == "min" else mean_delta

        if self.temporal_momentum > 0.0:
            momentum = max(0.0, min(self.temporal_momentum, 0.999))
            if self._ema_delta is None or self._ema_delta.shape[0] != n_points:
                self._ema_delta = fused_delta.clone()
            else:
                update_mask = enough_views
                self._ema_delta[update_mask] = (
                    momentum * self._ema_delta[update_mask]
                    + (1.0 - momentum) * fused_delta[update_mask]
                )
            ema_delta = self._ema_delta
        else:
            ema_delta = fused_delta
        used_delta = self._smooth_temporal_delta(ema_delta, enough_views)

        denom = torch.clamp(valid_count.float(), min=1.0)
        abs_bad_ratio = torch.zeros_like(denom)
        abs_bad_ratio[enough_views] = torch.clamp(
            used_delta[enough_views] / torch.clamp(self.tau_z + self.rel_tau * mean_depth[enough_views], min=1e-4),
            min=0.0,
            max=4.0,
        )
        front_bad_ratio = front_bad_count.float() / denom
        severe_bad_ratio = severe_front_count.float() / denom
        quality_score = self._estimate_depth_quality(used_delta, enough_views, used_views)
        adaptive_scale = self._adaptive_threshold_scale(quality_score)
        front_ratio_thresh = min(0.92, self.front_consensus_ratio + (adaptive_scale - 1.0) * 0.18)
        severe_ratio_thresh = min(0.90, self.severe_consensus_ratio + (adaptive_scale - 1.0) * 0.15)
        mismatch_abs_thresh = 1.25 * adaptive_scale
        severe_abs_thresh = 1.8 * adaptive_scale

        mismatch_mask = enough_views & (
            (front_bad_ratio >= front_ratio_thresh)
            | (severe_bad_ratio >= severe_ratio_thresh)
            | (abs_bad_ratio >= mismatch_abs_thresh)
        )
        severe_mask = enough_views & (
            (severe_bad_ratio >= severe_ratio_thresh)
            | (abs_bad_ratio >= severe_abs_thresh)
        )
        return {
            "mismatch_mask": mismatch_mask,
            "severe_mask": severe_mask,
            "valid_mask": enough_views,
            "delta_z": used_delta,
            "valid_count": valid_count,
            "used_views": used_views,
            "quality_score": quality_score,
            "adaptive_scale": adaptive_scale,
            "abs_bad_ratio": abs_bad_ratio,
        }

    def _estimate_depth_quality(self, used_delta: torch.Tensor, enough_views: torch.Tensor, used_views: int) -> float:
        if used_delta.numel() == 0 or not enough_views.any():
            return self.min_quality_score
        mean_delta = float(used_delta[enough_views].mean().item())
        valid_ratio = float(enough_views.float().mean().item())
        q_delta = float(np.exp(-mean_delta / max(self.tau_z * 12.0, 0.6)))
        q_valid = float(np.clip(valid_ratio / 0.35, 0.0, 1.0))
        q_views = float(np.clip(float(used_views) / max(float(self.views_per_iter), 1.0), 0.0, 1.0))
        score = q_delta * (0.25 + 0.75 * q_valid) * (0.4 + 0.6 * q_views)
        return float(np.clip(score, self.min_quality_score, 1.0))

    def _adaptive_threshold_scale(self, quality_score: float) -> float:
        quality = float(np.clip(quality_score, self.min_quality_score, 1.0))
        scale = 1.0 + self.dynamic_tau_strength * (1.0 - quality)
        return float(np.clip(scale, 1.0, 2.0))

    def _smooth_temporal_delta(self, delta_t: torch.Tensor, valid_t: torch.Tensor) -> torch.Tensor:
        if self.temporal_window <= 1:
            return delta_t
        if delta_t.numel() == 0:
            return delta_t

        if len(self._dz_history) > 0:
            hist_n = self._dz_history[0][0].shape[0]
            if hist_n != delta_t.shape[0]:
                self._dz_history.clear()

        self._dz_history.append((delta_t.detach().clone(), valid_t.detach().clone()))
        if self.temporal_mode == "last":
            return delta_t

        delta_sum = torch.zeros_like(delta_t)
        cnt = torch.zeros_like(delta_t)
        for hist_delta, hist_valid in self._dz_history:
            delta_sum[hist_valid] += hist_delta[hist_valid]
            cnt[hist_valid] += 1.0

        out = delta_t.clone()
        smooth_valid = cnt > 0
        out[smooth_valid] = delta_sum[smooth_valid] / cnt[smooth_valid]
        return out

    def _apply_exp_decay(
        self,
        gaussians,
        delta_z: torch.Tensor,
        moderate_mask: torch.Tensor,
        severe_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        low_alpha_mask: torch.Tensor,
        quality_score: float,
        abs_bad_ratio: Optional[torch.Tensor],
    ) -> int:
        if delta_z.numel() == 0:
            return 0
        target_mask = valid_mask & (moderate_mask | severe_mask)
        if self.use_low_alpha:
            target_mask = target_mask | (low_alpha_mask & severe_mask)
        if not target_mask.any():
            return 0

        with torch.no_grad():
            cur_opacity = gaussians.get_opacity.squeeze(-1)
            if cur_opacity.shape[0] != delta_z.shape[0]:
                return 0

            quality = float(np.clip(quality_score, self.min_quality_score, 1.0))
            quality_gain = self.decay_min_quality_scale + (1.0 - self.decay_min_quality_scale) * quality
            safe_delta = torch.clamp(delta_z, min=0.0, max=5.0)
            if abs_bad_ratio is not None and abs_bad_ratio.shape[0] == safe_delta.shape[0]:
                ratio_score = torch.clamp(abs_bad_ratio, min=0.0, max=4.0) / 4.0
            else:
                ratio_score = torch.full_like(safe_delta, 0.5)
            point_conf = 0.4 + 0.6 * ratio_score
            base_decay = torch.exp(-self.k_alpha * quality_gain * safe_delta * point_conf)
            base_decay = 1.0 - quality_gain * (1.0 - base_decay)
            base_decay = torch.clamp(base_decay, min=0.85, max=0.99995)

            decay = torch.ones_like(cur_opacity)
            if moderate_mask.any():
                decay[moderate_mask] = base_decay[moderate_mask]
            if severe_mask.any():
                severe_decay = torch.pow(base_decay[severe_mask], 1.0 + 0.8 * quality_gain)
                severe_floor = 0.80 + 0.15 * (1.0 - quality_gain)
                severe_decay = torch.clamp(severe_decay, min=severe_floor, max=0.997)
                decay[severe_mask] = severe_decay

            new_opacity = cur_opacity.clone()
            new_opacity[target_mask] = new_opacity[target_mask] * decay[target_mask]
            new_opacity = torch.clamp(new_opacity, min=1e-6, max=1.0 - 1e-6)
            gaussians._opacity.data.copy_(inverse_sigmoid(new_opacity.unsqueeze(-1)))
        return int(target_mask.sum().item())

    def _estimate_view_scale(self, z_vals: torch.Tensor, depth_vals: torch.Tensor) -> float:
        if z_vals.numel() < 64:
            return 1.0
        ratio = z_vals / torch.clamp(depth_vals, min=self.depth_min)
        ratio = ratio[torch.isfinite(ratio)]
        if ratio.numel() < 64:
            return 1.0
        q10 = torch.quantile(ratio, 0.1)
        q90 = torch.quantile(ratio, 0.9)
        inlier = (ratio >= q10) & (ratio <= q90)
        if inlier.sum() < 32:
            return 1.0
        scale = float(torch.median(ratio[inlier]).item())
        if not np.isfinite(scale):
            return 1.0
        return float(np.clip(scale, 0.5, 2.0))

    def _load_depth_map(self, image_name: str) -> Optional[Dict[str, torch.Tensor]]:
        if image_name in self.image_to_depth_path:
            path = self.image_to_depth_path[image_name]
            if path is None:
                return None
            return self.depth_cache.get(path, None)

        path = self._find_depth_path(image_name)
        self.image_to_depth_path[image_name] = path
        if path is None:
            return None
        depth_bundle = self._read_depth_file(path)
        if depth_bundle is None:
            return None
        self.depth_cache[path] = depth_bundle
        return depth_bundle

    def _find_depth_path(self, image_name: str) -> Optional[str]:
        extensions = [".npy", ".npz", ".png", ".tif", ".tiff"]
        candidates = []
        for ext in extensions:
            candidates.append(os.path.join(self.depth_dir, image_name + ext))

        match = re.match(r"frame_(\d+)_view_(\d+)", image_name)
        if match:
            frame_id = int(match.group(1))
            view_id = int(match.group(2))
            frame = f"{frame_id:06d}"
            view = f"{view_id:02d}"
            for ext in extensions:
                candidates.extend(
                    [
                        os.path.join(self.depth_dir, view, frame + ext),
                        os.path.join(self.depth_dir, f"view_{view}", frame + ext),
                        os.path.join(self.depth_dir, f"frame_{frame}_view_{view}" + ext),
                        os.path.join(self.depth_dir, f"{view}_{frame}" + ext),
                        os.path.join(self.depth_dir, frame, view + ext),
                    ]
                )

        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    def _read_depth_file(self, path: str) -> Optional[Dict[str, torch.Tensor]]:
        ext = os.path.splitext(path)[1].lower()
        conf_arr = None
        try:
            if ext == ".npy":
                arr = np.load(path)
            elif ext == ".npz":
                data = np.load(path)
                key = None
                priority = ["depth", "depth_map", "render_depth", "metric_depth", "d"]
                for k in priority:
                    if k in data.files:
                        key = k
                        break
                if key is None:
                    for k in data.files:
                        if "depth" in k.lower():
                            key = k
                            break
                if key is None and len(data.files) == 1:
                    key = data.files[0]
                if key is None:
                    return None
                arr = data[key]
                conf_key = None
                for k in ["confidence", "conf", "confidence_map", "valid_mask", "mask"]:
                    if k in data.files:
                        conf_key = k
                        break
                if conf_key is not None:
                    conf_arr = data[conf_key]
            else:
                arr = np.array(Image.open(path))
        except Exception:
            return None

        if arr is None:
            return None
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.ndim != 2:
            return None

        arr = arr.astype(np.float32) * self.depth_scale
        valid_mask = self._build_valid_mask(arr, conf_arr)
        return {
            "depth": torch.from_numpy(arr),
            "valid_mask": torch.from_numpy(valid_mask),
        }

    def _build_valid_mask(self, depth_arr: np.ndarray, conf_arr: Optional[np.ndarray]) -> np.ndarray:
        valid = np.isfinite(depth_arr) & (depth_arr > self.depth_min)
        if valid.sum() == 0:
            return valid.astype(np.bool_)

        if valid.sum() >= 64:
            vals = depth_arr[valid]
            lo = float(np.percentile(vals, 0.5))
            hi = float(np.percentile(vals, 99.5))
            valid = valid & (depth_arr >= lo) & (depth_arr <= hi)

        if conf_arr is not None:
            if conf_arr.ndim == 3:
                conf_arr = conf_arr[..., 0]
            if conf_arr.shape == depth_arr.shape:
                conf = conf_arr.astype(np.float32)
                conf_valid = np.isfinite(conf)
                if conf_valid.any():
                    conf_vals = conf[conf_valid]
                    c_lo = float(np.percentile(conf_vals, 1.0))
                    c_hi = float(np.percentile(conf_vals, 99.0))
                    if c_hi > c_lo:
                        conf = np.clip((conf - c_lo) / (c_hi - c_lo + 1e-6), 0.0, 1.0)
                        valid = valid & (conf >= 0.2)

        return valid.astype(np.bool_)

    def _limit_hard_prune(self, hard_mask: torch.Tensor, delta_z: torch.Tensor, low_alpha_mask: torch.Tensor):
        if not hard_mask.any():
            return hard_mask
        if self.max_prune_ratio >= 1.0:
            return hard_mask
        if self.max_prune_ratio <= 0.0:
            return torch.zeros_like(hard_mask)

        n_points = hard_mask.shape[0]
        max_prune = int(n_points * self.max_prune_ratio)
        max_prune = max(max_prune, 1)
        candidate_count = int(hard_mask.sum().item())
        if candidate_count <= max_prune:
            return hard_mask

        candidate_idx = torch.nonzero(hard_mask, as_tuple=False).squeeze(1)
        scores = delta_z[candidate_idx].clone()
        scores[low_alpha_mask[candidate_idx]] = scores[low_alpha_mask[candidate_idx]] + 1e6
        keep_idx = torch.topk(scores, k=max_prune, largest=True).indices
        pruned_idx = candidate_idx[keep_idx]
        out = torch.zeros_like(hard_mask)
        out[pruned_idx] = True
        return out
