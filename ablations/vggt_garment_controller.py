"""VGGT garment proposals with isolated fixed-budget reallocation modes."""

from pathlib import Path

import numpy as np
import torch


class VggtGarmentController:
    def __init__(self, args):
        self.args = args
        self.done = False
        self.candidates = None
        self.next_candidate = 0
        self.last_protect_mask = None

    def _candidate_path(self, scene):
        configured = str(getattr(self.args, "vggt_candidate_path", ""))
        sequence = Path(scene.model_path).parts[-3]
        if configured:
            return Path(configured.format(sequence=sequence))
        return Path(self.args.model_path).parents[3] / "vggt_garment_candidates" / sequence / "candidates.npz"

    def _target_points(self, scene):
        configured = str(getattr(self.args, "vggt_target_points_file", ""))
        if configured:
            import json

            with open(configured) as handle:
                return int(json.load(handle).get(Path(scene.model_path).parts[-3], 0))
        return int(getattr(self.args, "vggt_target_points", 0))

    def _load(self, scene, gaussians):
        path = self._candidate_path(scene)
        if not path.exists():
            raise FileNotFoundError(
                f"[VGGT_GARMENT] candidate cache not found: {path}. "
                "Run scripts/vggt_generate_clothing_candidates.py first."
            )
        data = np.load(path, allow_pickle=False)
        required = {"canonical_xyz", "score"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"[VGGT_GARMENT] cache missing fields: {sorted(missing)}")
        device = gaussians.get_xyz.device
        self.candidates = {
            "canonical_xyz": torch.as_tensor(data["canonical_xyz"], device=device, dtype=gaussians.get_xyz.dtype),
            "score": torch.as_tensor(data["score"], device=device, dtype=gaussians.get_xyz.dtype),
        }
        if "colors" in data.files:
            self.candidates["colors"] = torch.as_tensor(
                data["colors"], device=device, dtype=gaussians.get_xyz.dtype
            ).clamp(0.0, 1.0)
        else:
            self.candidates["colors"] = None
        order = torch.argsort(self.candidates["score"], descending=True)
        for key, value in self.candidates.items():
            if value is not None:
                self.candidates[key] = value[order]
        print(f"[VGGT_GARMENT] loaded={path} candidates={len(self.candidates['score'])}", flush=True)

    @torch.no_grad()
    def _high_error_context(self, context, gaussians):
        if not context:
            return None
        values = [
            context.get("viewpoint_camera"),
            context.get("deformed_means3D"),
            context.get("visibility_filter"),
            context.get("image"),
            context.get("gt_image"),
            context.get("bound_mask"),
        ]
        if any(value is None for value in values):
            return None
        camera, deformed, visibility, image, gt_image, bound_mask = values
        if deformed.shape[0] != gaussians.get_xyz.shape[0]:
            return None
        error_map = torch.abs(image.detach() - gt_image.detach()).mean(dim=0)
        mask = bound_mask[0] if bound_mask.ndim == 3 else bound_mask
        error_map = error_map * mask.to(device=error_map.device, dtype=error_map.dtype)
        height, width = int(error_map.shape[-2]), int(error_map.shape[-1])
        ones = torch.ones((deformed.shape[0], 1), device=deformed.device, dtype=deformed.dtype)
        clip = torch.cat([deformed.detach(), ones], dim=-1) @ camera.full_proj_transform.to(deformed.device)
        w = clip[:, 3]
        safe_w = torch.where(w.abs() > 1e-8, w, torch.full_like(w, 1e-8))
        ndc = clip[:, :3] / safe_w[:, None]
        xy = torch.stack([
            ((ndc[:, 0] + 1.0) * 0.5 * max(width - 1, 1)).round().long(),
            ((ndc[:, 1] + 1.0) * 0.5 * max(height - 1, 1)).round().long(),
        ], dim=-1)
        valid = visibility.to(device=deformed.device, dtype=torch.bool)
        valid &= w > 1e-6
        valid &= (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        point_error = torch.zeros((deformed.shape[0],), device=deformed.device, dtype=error_map.dtype)
        point_error[valid] = error_map[xy[valid, 1], xy[valid, 0]]
        valid_values = point_error[valid]
        if valid_values.numel() == 0:
            threshold = 0.0
            high_error = torch.zeros_like(valid)
        else:
            quantile = float(getattr(self.args, "vggt_high_error_quantile", 0.75))
            threshold = float(torch.quantile(valid_values, min(max(quantile, 0.0), 1.0)).item())
            high_error = valid & (point_error >= threshold)
        return {"point_error": point_error, "high_error": high_error, "threshold": threshold}

    @torch.no_grad()
    def _select_candidates(self, gaussians, max_spawn, high_error_context):
        xyz = self.candidates["canonical_xyz"]
        colors = self.candidates["colors"]
        total = int(xyz.shape[0])
        if self.next_candidate >= total:
            empty = torch.zeros((gaussians.get_xyz.shape[0],), dtype=torch.bool, device=xyz.device)
            return xyz[:0], None, empty, 0, 0

        candidate_ids = torch.arange(self.next_candidate, total, device=xyz.device, dtype=torch.long)
        candidate_xyz = xyz[candidate_ids]
        nearest_dist = torch.cdist(candidate_xyz, gaussians.get_xyz.detach()).min(dim=1).values
        valid = nearest_dist >= float(getattr(self.args, "vggt_candidate_min_distance", 0.004))
        candidate_ids = candidate_ids[valid]
        candidate_xyz = candidate_xyz[valid]
        if candidate_xyz.numel() == 0:
            self.next_candidate = total
            empty = torch.zeros((gaussians.get_xyz.shape[0],), dtype=torch.bool, device=xyz.device)
            return xyz[:0], None, empty, 0, 0

        parent_ids = torch.cdist(candidate_xyz, gaussians.get_xyz.detach()).argmin(dim=1)
        base_score = self.candidates["score"][candidate_ids]
        high_error_parent = torch.zeros_like(base_score, dtype=torch.bool)
        high_error_score = torch.zeros_like(base_score)
        if high_error_context is not None:
            high_error_parent = high_error_context["high_error"][parent_ids]
            high_error_score = high_error_context["point_error"][parent_ids]
            high_error_score = high_error_score / high_error_score.max().clamp_min(1e-6)
            priority = base_score + float(getattr(self.args, "vggt_high_error_candidate_w", 1.0)) * high_error_score
        else:
            priority = base_score

        order = torch.argsort(priority, descending=True)
        candidate_ids = candidate_ids[order]
        candidate_xyz = candidate_xyz[order]
        parent_ids = parent_ids[order]
        high_error_parent = high_error_parent[order]
        min_distance = float(getattr(self.args, "vggt_candidate_min_distance", 0.004))
        selected_xyz = []
        selected_ids = []
        selected_parent_ids = []
        selected_high_error = []
        for index in range(int(candidate_xyz.shape[0])):
            point = candidate_xyz[index:index + 1]
            if selected_xyz and bool((torch.cdist(point, torch.cat(selected_xyz, dim=0)).min() < min_distance).item()):
                continue
            selected_xyz.append(point)
            selected_ids.append(candidate_ids[index:index + 1])
            selected_parent_ids.append(parent_ids[index:index + 1])
            selected_high_error.append(high_error_parent[index:index + 1])
            if len(selected_xyz) >= int(max_spawn):
                break
        if not selected_xyz:
            self.next_candidate = total
            empty = torch.zeros((gaussians.get_xyz.shape[0],), dtype=torch.bool, device=xyz.device)
            return xyz[:0], None, empty, 0, 0

        selected_xyz = torch.cat(selected_xyz, dim=0)
        selected_ids = torch.cat(selected_ids, dim=0)
        selected_parent_ids = torch.cat(selected_parent_ids, dim=0)
        selected_high_error = torch.cat(selected_high_error, dim=0)
        # Consume all candidates examined through the last selected proposal.
        self.next_candidate = int(candidate_ids[:index + 1].max().item()) + 1
        selected_colors = None if colors is None else colors[selected_ids]
        protect = torch.zeros((gaussians.get_xyz.shape[0],), dtype=torch.bool, device=xyz.device)
        protect[selected_parent_ids] = True
        return selected_xyz, selected_colors, protect, int(selected_high_error.sum().item()), int(high_error_parent.sum().item())

    @torch.no_grad()
    def after_iteration(self, iteration, scene, gaussians, context=None):
        if not getattr(self.args, "use_vggt_garment", False):
            return
        start = int(getattr(self.args, "vggt_garment_start_iter", 800))
        end = int(getattr(self.args, "vggt_garment_end_iter", 1500))
        interval = max(1, int(getattr(self.args, "vggt_garment_interval", 100)))
        if iteration < start or iteration > end or iteration % interval != 0:
            return
        if self.candidates is None:
            self._load(scene, gaussians)
        if self.done and not bool(getattr(self.args, "vggt_garment_repeat", False)):
            return

        max_spawn = int(getattr(self.args, "vggt_garment_max_spawn", 512))
        high_error_context = (
            self._high_error_context(context, gaussians)
            if bool(getattr(self.args, "vggt_high_error_enabled", False))
            else None
        )
        candidate_xyz, candidate_colors, candidate_parent_protect, selected_high_error, filtered_high_error = self._select_candidates(
            gaussians, max_spawn, high_error_context
        )
        requested = int(candidate_xyz.shape[0])
        if requested <= 0:
            self.done = True
            return

        protect_mask = None
        high_error_points = 0
        if high_error_context is not None:
            protect_mask = candidate_parent_protect | high_error_context["high_error"]
            high_error_points = int(high_error_context["high_error"].sum().item())
        replaced = gaussians.replace_low_value_points(
            requested,
            opacity_w=getattr(self.args, "vggt_replace_opacity_w", 1.0),
            gradient_w=getattr(self.args, "vggt_replace_gradient_w", 0.25),
            visibility_w=getattr(self.args, "vggt_replace_visibility_w", 0.10),
            use_long_term=True,
            protect_mask=protect_mask,
            min_keep=int(getattr(self.args, "vggt_min_keep", 1024)),
        )
        spawned = gaussians.spawn_from_explicit_candidates(
            candidate_xyz[:replaced],
            colors=None if candidate_colors is None else candidate_colors[:replaced],
            opacity=float(getattr(self.args, "vggt_candidate_opacity", 0.04)),
            scale_ratio=float(getattr(self.args, "vggt_candidate_scale_ratio", 0.65)),
            min_distance=0.0,
            max_points=int(getattr(self.args, "vggt_max_points", 120000)),
        )
        if spawned != replaced:
            raise RuntimeError(
                f"[VGGT_GARMENT] strict replacement invariant failed: replaced={replaced} spawned={spawned}"
            )
        self.last_protect_mask = None if protect_mask is None else protect_mask.detach().clone()
        self.done = not bool(getattr(self.args, "vggt_garment_repeat", False))
        print(
            "[VGGT_GARMENT] "
            f"iter={iteration} proposals={requested} replaced={replaced} spawned={spawned} "
            f"high_error_points={high_error_points} selected_high_error={selected_high_error} "
            f"filtered_high_error={filtered_high_error} target={self._target_points(scene)} "
            f"total_points={len(gaussians.get_xyz)}",
            flush=True,
        )

    @torch.no_grad()
    def finalize_budget(self, scene, gaussians):
        if not bool(getattr(self.args, "vggt_strict_budget", False)):
            return
        target = self._target_points(scene)
        if target <= 0:
            return
        before = int(len(gaussians.get_xyz))
        pruned = 0
        padded = 0
        if before > target:
            protect = self.last_protect_mask if (
                bool(getattr(self.args, "vggt_high_error_enabled", False))
                and self.last_protect_mask is not None
                and self.last_protect_mask.shape[0] == before
            ) else None
            pruned = gaussians.replace_low_value_points(
                before - target, use_long_term=True, protect_mask=protect, min_keep=0
            )
            remaining = before - target - pruned
            if remaining > 0:
                pruned += gaussians.replace_low_value_points(
                    remaining, use_long_term=True, protect_mask=None, min_keep=0
                )
        elif before < target:
            gaussians._ensure_point_value_buffers()
            source_ids = torch.argsort(gaussians.point_value_opacity_ema, descending=True)
            padded = gaussians.append_cloned_points(target - before, source_ids=source_ids)
        final = int(len(gaussians.get_xyz))
        if final != target:
            raise RuntimeError(f"[VGGT_GARMENT] final point count mismatch: final={final} target={target}")
        print(
            "[VGGT_GARMENT_FINAL] "
            f"before={before} pruned={pruned} padded={padded} target={target} final={final}",
            flush=True,
        )


def build_vggt_garment_controller(args):
    if not getattr(args, "use_vggt_garment", False):
        return None
    return VggtGarmentController(args)
