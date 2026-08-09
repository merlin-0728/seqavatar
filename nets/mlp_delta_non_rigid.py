import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _append_mlp_input_dim(mlp, extra_input_dim):
    extra_input_dim = int(extra_input_dim)
    if extra_input_dim <= 0:
        return mlp

    for idx, layer in enumerate(mlp):
        if isinstance(layer, nn.Linear):
            old_layer = layer
            new_layer = nn.Linear(
                old_layer.in_features + extra_input_dim,
                old_layer.out_features,
                bias=old_layer.bias is not None,
            )
            with torch.no_grad():
                new_layer.weight[:, :old_layer.in_features].copy_(old_layer.weight)
                if old_layer.bias is not None:
                    new_layer.bias.copy_(old_layer.bias)
            mlp[idx] = new_layer
            return mlp

    raise RuntimeError("[TRI] Cannot append tri-plane features: no Linear layer found.")


class TriPlaneFeature(nn.Module):
    def __init__(self, feature_dim=32, resolution=64, extent=1.2):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.resolution = int(resolution)
        self.extent = float(extent)
        if self.feature_dim <= 0:
            raise ValueError("[TRI] tri_plane_dim must be positive.")
        if self.resolution <= 1:
            raise ValueError("[TRI] tri_plane_res must be greater than 1.")
        if self.extent <= 0:
            raise ValueError("[TRI] tri_plane_extent must be positive.")
        self.planes = nn.Parameter(torch.zeros(3, self.feature_dim, self.resolution, self.resolution))

    def _normalize_xyz(self, query_xyz):
        return (query_xyz / self.extent).clamp(-1.0, 1.0)

    def _sample_plane(self, plane, grid):
        batch_size = grid.shape[0]
        plane = plane.unsqueeze(0).expand(batch_size, -1, -1, -1)
        sampled = F.grid_sample(
            plane,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(-1).transpose(1, 2).contiguous()

    def forward(self, query_xyz):
        if query_xyz.dim() == 2:
            query_xyz = query_xyz.unsqueeze(0)
        coords = self._normalize_xyz(query_xyz)

        xy_grid = coords[..., [0, 1]].unsqueeze(2)
        xz_grid = coords[..., [0, 2]].unsqueeze(2)
        yz_grid = coords[..., [1, 2]].unsqueeze(2)

        xy_feat = self._sample_plane(self.planes[0], xy_grid)
        xz_feat = self._sample_plane(self.planes[1], xz_grid)
        yz_feat = self._sample_plane(self.planes[2], yz_grid)
        return (xy_feat + xz_feat + yz_feat) / 3.0


class PartTriFeatureFiLM(nn.Module):
    def __init__(self, num_parts, feature_dim, resolution=64, extent=1.0,
                 alpha=1.0, motion_gain=0.5, boundary_gain=0.5, hidden_dim=64):
        super().__init__()
        self.num_parts = int(num_parts)
        self.feature_dim = int(feature_dim)
        self.resolution = int(resolution)
        self.extent = float(extent)
        self.alpha = float(alpha)
        self.motion_gain = float(motion_gain)
        self.boundary_gain = float(boundary_gain)
        self.hidden_dim = int(hidden_dim)
        if self.num_parts <= 0:
            raise ValueError("[TRI_PART] num_parts must be positive.")
        if self.feature_dim <= 0:
            raise ValueError("[TRI_PART] feature_dim must be positive.")
        if self.resolution <= 1:
            raise ValueError("[TRI_PART] resolution must be greater than 1.")
        if self.extent <= 0:
            raise ValueError("[TRI_PART] extent must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("[TRI_PART] hidden_dim must be positive.")

        self.part_delta_planes = nn.Parameter(
            torch.zeros(self.num_parts, 3, self.feature_dim, self.resolution, self.resolution)
        )
        self.part_embedding = nn.Embedding(self.num_parts, self.hidden_dim)
        self.gate_delta = nn.Sequential(
            nn.Linear(self.feature_dim + self.hidden_dim + 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.part_gate_logit = nn.Embedding(self.num_parts, 1)
        nn.init.normal_(self.part_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate_delta[-1].weight)
        nn.init.zeros_(self.gate_delta[-1].bias)
        self._init_part_gate_logits()

    def _init_part_gate_logits(self):
        # part_moe_leg schema: 0 unknown, 1 body, 2/3 hands, 4 face, 5/6 legs.
        priors = [0.35, 0.25, 0.60, 0.60, 0.25, 0.80, 0.80]
        if self.num_parts != len(priors):
            priors = [0.5 for _ in range(self.num_parts)]
        prior_tensor = torch.tensor(priors, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
        with torch.no_grad():
            self.part_gate_logit.weight.copy_(torch.logit(prior_tensor).view(self.num_parts, 1))

    def _normalize_labels(self, part_label, batch_size, num_points, device):
        part_label = part_label.long().to(device)
        part_label = torch.clamp(part_label, min=0, max=self.num_parts - 1)
        if part_label.dim() == 1:
            part_label = part_label.unsqueeze(0).expand(batch_size, -1)
        elif part_label.shape[0] == 1 and batch_size > 1:
            part_label = part_label.expand(batch_size, -1)
        if part_label.shape[1] != num_points:
            raise RuntimeError(
                f"[TRI_PART] part_label points {part_label.shape[1]} != tri feature points {num_points}."
            )
        return part_label

    def _normalize_conf(self, part_conf, part_label, dtype):
        if part_conf is None:
            return torch.ones(*part_label.shape, 1, device=part_label.device, dtype=dtype)
        part_conf = part_conf.to(device=part_label.device, dtype=dtype)
        if part_conf.dim() == 1:
            part_conf = part_conf.unsqueeze(0).expand(part_label.shape[0], -1)
        elif part_conf.dim() == 2 and part_conf.shape[0] == 1 and part_label.shape[0] > 1:
            part_conf = part_conf.expand(part_label.shape[0], -1)
        if part_conf.dim() == 2:
            part_conf = part_conf.unsqueeze(-1)
        return part_conf.clamp(0.0, 1.0)

    def _normalize_motion(self, motion_strength, tri_features):
        if motion_strength is None:
            return torch.zeros(*tri_features.shape[:2], 1, device=tri_features.device, dtype=tri_features.dtype)
        motion_strength = motion_strength.to(device=tri_features.device, dtype=tri_features.dtype)
        if motion_strength.dim() == 2:
            motion_strength = motion_strength.unsqueeze(-1)
        motion_strength = torch.log1p(motion_strength.clamp_min(0.0))
        mean = motion_strength.detach().mean(dim=1, keepdim=True)
        std = motion_strength.detach().std(dim=1, keepdim=True).clamp_min(1e-6)
        return ((motion_strength - mean) / std).clamp(-3.0, 3.0)

    def _normalize_xyz(self, query_xyz):
        return (query_xyz / self.extent).clamp(-1.0, 1.0)

    def _sample_plane(self, plane, grid):
        sampled = F.grid_sample(
            plane.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(-1).transpose(1, 2).contiguous().squeeze(0)

    def _sample_part_delta(self, query_xyz, part_label):
        if query_xyz.dim() == 2:
            query_xyz = query_xyz.unsqueeze(0)
        if query_xyz.shape[0] == 1 and part_label.shape[0] > 1:
            query_xyz = query_xyz.expand(part_label.shape[0], -1, -1)
        coords = self._normalize_xyz(query_xyz.to(device=self.part_delta_planes.device))
        out = torch.zeros(
            coords.shape[0],
            coords.shape[1],
            self.feature_dim,
            device=coords.device,
            dtype=coords.dtype,
        )
        plane_axes = ((0, 1), (0, 2), (1, 2))
        flat_out = out.reshape(-1, self.feature_dim)
        flat_coords = coords.reshape(-1, 3)
        flat_labels = part_label.reshape(-1)

        for pid in range(self.num_parts):
            idx = torch.nonzero(flat_labels == pid, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            part_coords = flat_coords.index_select(0, idx).unsqueeze(0)
            part_feat = 0.0
            for plane_id, axes in enumerate(plane_axes):
                grid = part_coords[..., list(axes)].unsqueeze(2)
                part_feat = part_feat + self._sample_plane(self.part_delta_planes[pid, plane_id], grid)
            flat_out = torch.index_copy(flat_out, 0, idx, part_feat / 3.0)

        return flat_out.reshape_as(out)

    def forward(self, tri_features, query_xyz, part_label, motion_strength=None, part_conf=None):
        batch_size, num_points = tri_features.shape[:2]
        part_label = self._normalize_labels(part_label, batch_size, num_points, tri_features.device)
        part_conf = self._normalize_conf(part_conf, part_label, tri_features.dtype)
        motion_norm = self._normalize_motion(motion_strength, tri_features)
        boundary_score = 1.0 - part_conf

        part_delta = self._sample_part_delta(query_xyz, part_label).to(
            device=tri_features.device,
            dtype=tri_features.dtype,
        )
        part_emb = self.part_embedding(part_label)
        gate_delta = self.gate_delta(torch.cat([tri_features, part_emb, motion_norm, boundary_score], dim=-1))
        gate_logits = (
            self.part_gate_logit(part_label)
            + self.motion_gain * motion_norm
            + self.boundary_gain * boundary_score
            + gate_delta
        )
        gate = torch.sigmoid(gate_logits)
        residual = self.alpha * gate * part_delta
        reg_weight = (1.0 - torch.sigmoid(motion_norm)) * part_conf
        reg_loss = (reg_weight * residual.pow(2).mean(dim=-1, keepdim=True)).mean()
        stats = {
            "gate_mean": gate.detach().mean(),
            "gate_std": gate.detach().std(),
            "motion_mean": motion_norm.detach().mean(),
            "boundary_mean": boundary_score.detach().mean(),
            "residual_norm": residual.detach().norm(dim=-1).mean(),
            "reg_loss": reg_loss,
        }
        return tri_features + residual, stats


class TriGateAdapter(nn.Module):
    def __init__(self, tri_dim, feature_dim, hidden_dim=128, gate_init=0.5):
        super().__init__()
        self.tri_dim = int(tri_dim)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        gate_init = float(gate_init)
        if self.tri_dim <= 0:
            raise ValueError("[TRI_GATE] tri_dim must be positive.")
        if self.feature_dim <= 0:
            raise ValueError("[TRI_GATE] feature_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("[TRI_GATE] hidden_dim must be positive.")
        if not 0.0 < gate_init < 1.0:
            raise ValueError("[TRI_GATE] gate_init must be in (0, 1).")

        self.adapter = nn.Sequential(
            nn.Linear(self.tri_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(self.feature_dim + self.tri_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        gate_bias = torch.logit(torch.tensor(gate_init, dtype=torch.float32)).item()
        nn.init.constant_(self.gate[-1].bias, gate_bias)

    def forward(self, base_features, tri_features):
        tri_res = self.adapter(tri_features)
        gate_input = torch.cat([base_features, tri_features], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        return gate * tri_res, gate


class TriFeatureGate(nn.Module):
    def __init__(self, tri_dim, feature_dim, hidden_dim=128, gate_init=0.5):
        super().__init__()
        self.tri_dim = int(tri_dim)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        gate_init = float(gate_init)
        self.gate_init = gate_init
        if self.tri_dim <= 0:
            raise ValueError("[TRI_GATE] tri_dim must be positive.")
        if self.feature_dim <= 0:
            raise ValueError("[TRI_GATE] feature_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("[TRI_GATE] hidden_dim must be positive.")
        if not 0.0 < gate_init < 1.0:
            raise ValueError("[TRI_GATE] gate_init must be in (0, 1).")

        self.gate = nn.Sequential(
            nn.Linear(self.feature_dim + self.tri_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        gate_bias = torch.logit(torch.tensor(gate_init, dtype=torch.float32)).item()
        nn.init.constant_(self.gate[-1].bias, gate_bias)

    def forward(self, base_features, tri_features):
        gate_input = torch.cat([base_features, tri_features], dim=-1)
        return torch.sigmoid(self.gate(gate_input))


class PartStatsEncoder(nn.Module):
    def __init__(self, num_parts, token_dim=32, hidden_dim=128, stat_dim=7):
        super().__init__()
        self.num_parts = int(num_parts)
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.stat_dim = int(stat_dim)
        if self.num_parts <= 0:
            raise ValueError("[PART_BUDGET] num_parts must be positive.")
        if self.token_dim <= 0:
            raise ValueError("[PART_BUDGET] token_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("[PART_BUDGET] hidden_dim must be positive.")
        if self.stat_dim <= 0:
            raise ValueError("[PART_BUDGET] stat_dim must be positive.")

        self.part_embedding = nn.Embedding(self.num_parts, self.token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.stat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.token_dim),
        )
        nn.init.normal_(self.part_embedding.weight, mean=0.0, std=0.02)

    def _normalize_inputs(self, part_label, motion_strength, part_conf):
        if part_label.dim() == 1:
            part_label = part_label.unsqueeze(0)
        batch_size, num_points = part_label.shape[:2]
        device = part_label.device
        dtype = motion_strength.dtype if motion_strength is not None else (
            part_conf.dtype if part_conf is not None else torch.float32
        )

        if motion_strength is None:
            motion_strength = torch.zeros(batch_size, num_points, 1, device=device, dtype=dtype)
        else:
            motion_strength = motion_strength.to(device=device, dtype=dtype)
            if motion_strength.dim() == 1:
                motion_strength = motion_strength.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)
            elif motion_strength.dim() == 2:
                if motion_strength.shape[0] == 1 and batch_size > 1:
                    motion_strength = motion_strength.expand(batch_size, -1)
                motion_strength = motion_strength.unsqueeze(-1)
            elif motion_strength.shape[0] == 1 and batch_size > 1:
                motion_strength = motion_strength.expand(batch_size, -1, -1)
        motion_strength = motion_strength.to(device=device, dtype=dtype)

        if part_conf is None:
            part_conf = torch.ones(batch_size, num_points, 1, device=device, dtype=dtype)
        else:
            part_conf = part_conf.to(device=device, dtype=dtype)
            if part_conf.dim() == 1:
                part_conf = part_conf.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)
            elif part_conf.dim() == 2:
                if part_conf.shape[0] == 1 and batch_size > 1:
                    part_conf = part_conf.expand(batch_size, -1)
                part_conf = part_conf.unsqueeze(-1)
            elif part_conf.shape[0] == 1 and batch_size > 1:
                part_conf = part_conf.expand(batch_size, -1, -1)
        part_conf = part_conf.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        return part_label.long().to(device=device), motion_strength, part_conf

    def forward(self, part_label, motion_strength=None, part_conf=None):
        part_label, motion_strength, part_conf = self._normalize_inputs(
            part_label, motion_strength, part_conf
        )
        batch_size, num_points = part_label.shape[:2]
        boundary_score = 1.0 - part_conf

        global_motion_mean = motion_strength.mean(dim=1, keepdim=True)
        global_motion_std = motion_strength.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        global_boundary_mean = boundary_score.mean(dim=1, keepdim=True)
        global_boundary_std = boundary_score.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        global_conf_mean = part_conf.mean(dim=1, keepdim=True)
        global_count_ratio = torch.ones(batch_size, 1, 1, device=part_label.device, dtype=motion_strength.dtype)
        global_rigidity = 1.0 / (1.0 + global_motion_mean.abs() + global_motion_std + global_boundary_mean)
        global_stats = torch.cat(
            [
                global_motion_mean,
                global_motion_std,
                global_boundary_mean,
                global_boundary_std,
                global_conf_mean,
                global_count_ratio,
                global_rigidity,
            ],
            dim=-1,
        )

        part_stats = []
        for pid in range(self.num_parts):
            mask = (part_label == pid).unsqueeze(-1).to(dtype=motion_strength.dtype)
            count = mask.sum(dim=1, keepdim=True)
            count_safe = count.clamp_min(1.0)

            motion_mean = (motion_strength * mask).sum(dim=1, keepdim=True) / count_safe
            motion_var = ((motion_strength - motion_mean) ** 2 * mask).sum(dim=1, keepdim=True) / count_safe
            motion_std = motion_var.clamp_min(1e-6).sqrt()

            boundary_mean = (boundary_score * mask).sum(dim=1, keepdim=True) / count_safe
            boundary_var = ((boundary_score - boundary_mean) ** 2 * mask).sum(dim=1, keepdim=True) / count_safe
            boundary_std = boundary_var.clamp_min(1e-6).sqrt()

            conf_mean = (part_conf * mask).sum(dim=1, keepdim=True) / count_safe
            count_ratio = count / float(num_points)
            rigidity = 1.0 / (1.0 + motion_mean.abs() + motion_std + boundary_mean)
            stats = torch.cat(
                [
                    motion_mean,
                    motion_std,
                    boundary_mean,
                    boundary_std,
                    conf_mean,
                    count_ratio,
                    rigidity,
                ],
                dim=-1,
            )
            valid = (count > 0).to(dtype=stats.dtype)
            stats = valid * stats + (1.0 - valid) * global_stats
            part_stats.append(stats)

        part_stats = torch.cat(part_stats, dim=1)
        part_ids = torch.arange(self.num_parts, device=part_label.device)
        part_token = self.mlp(part_stats) + self.part_embedding(part_ids)[None, :, :]
        return part_token, part_stats


class PartBudgetRouter(nn.Module):
    def __init__(self, feature_dim, token_dim, hidden_dim=128):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        if self.feature_dim <= 0:
            raise ValueError("[PART_BUDGET] feature_dim must be positive.")
        if self.token_dim <= 0:
            raise ValueError("[PART_BUDGET] token_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("[PART_BUDGET] hidden_dim must be positive.")

        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim + self.token_dim + 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, base_features, part_token, motion_strength, boundary_score):
        router_input = torch.cat([base_features, part_token, motion_strength, boundary_score], dim=-1)
        logits = self.mlp(router_input)
        budget = torch.softmax(logits, dim=-1)
        return budget, logits


class PartBudgetAdapter(nn.Module):
    def __init__(self, feature_dim, token_dim, hidden_dim=128):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        if self.feature_dim <= 0:
            raise ValueError("[PART_BUDGET] feature_dim must be positive.")
        if self.token_dim <= 0:
            raise ValueError("[PART_BUDGET] token_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("[PART_BUDGET] hidden_dim must be positive.")

        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim + self.token_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, base_features, part_token):
        adapter_input = torch.cat([base_features, part_token], dim=-1)
        return self.mlp(adapter_input)


class PartNonrigidExpert(nn.Module):
    def __init__(self, mlp, gaussian_warp, gaussian_rotation, gaussian_scaling):
        super().__init__()
        self.mlp = copy.deepcopy(mlp)
        self.gaussian_warp = copy.deepcopy(gaussian_warp)
        self.gaussian_rotation = copy.deepcopy(gaussian_rotation)
        self.gaussian_scaling = copy.deepcopy(gaussian_scaling)

    def forward(self, features):
        h = self.mlp(features)
        return self.gaussian_warp(h), self.gaussian_rotation(h), self.gaussian_scaling(h)


class NonrigidDeformer(nn.Module):
    def __init__(self, D=3, W=512, use_pose_cond=0, use_seq_pose_cond=0, use_seq_xyz_cond=0, 
                 pos_input_dim=63, pose_cond_dim=32, seq_pose_cond_dim=32, seq_xyz_cond_dim=96,
                 seq_len=6, seq_xyz_knn=1, time_step_num=1, smpl_type='smpl',
                 use_part_moe=False, num_parts=5, part_moe_global_keep=0.1,
                 use_tri=False, tri_plane_dim=32, tri_plane_res=64, tri_plane_extent=1.2,
                 use_tri_part=False, use_tri_gate=False, tri_gate_alpha=0.2,
                 tri_gate_init=0.5, tri_gate_hidden_dim=128, tri_gate_mode="additive",
                 tri_part_alpha=1.0, tri_part_motion_gain=0.5,
                 tri_part_boundary_gain=0.5, tri_part_hidden_dim=64,
                 part_label_schema="anatomy5", use_part_budget=False, part_budget_alpha=1.0,
                 part_budget_start_iter=16000, part_budget_warmup=1000,
                 part_budget_hidden_dim=128, part_budget_token_dim=32):
        super(NonrigidDeformer, self).__init__()

        self.use_pose_cond = use_pose_cond
        self.use_seq_pose_cond = use_seq_pose_cond
        self.use_seq_xyz_cond = use_seq_xyz_cond
        self.use_part_moe = use_part_moe
        self.use_tri = bool(use_tri and use_part_moe)
        self.use_tri_part = bool(use_tri_part and self.use_tri)
        self.use_tri_gate = bool(use_tri_gate and self.use_tri)
        self.tri_plane_dim = int(tri_plane_dim)
        self.tri_plane_res = int(tri_plane_res)
        self.tri_plane_extent = float(tri_plane_extent)
        self.tri_gate_alpha = float(tri_gate_alpha)
        self.tri_gate_init = float(tri_gate_init)
        self.tri_gate_hidden_dim = int(tri_gate_hidden_dim)
        self.tri_gate_mode = str(tri_gate_mode).lower()
        self.tri_part_alpha = float(tri_part_alpha)
        self.tri_part_motion_gain = float(tri_part_motion_gain)
        self.tri_part_boundary_gain = float(tri_part_boundary_gain)
        self.tri_part_hidden_dim = int(tri_part_hidden_dim)
        self.part_label_schema = str(part_label_schema)
        self.use_part_budget = bool(use_part_budget)
        self.part_budget_alpha = float(part_budget_alpha)
        self.part_budget_start_iter = int(part_budget_start_iter)
        self.part_budget_warmup = int(part_budget_warmup)
        self.part_budget_hidden_dim = int(part_budget_hidden_dim)
        self.part_budget_token_dim = int(part_budget_token_dim)
        if self.tri_gate_mode not in ("additive", "concat", "scale"):
            raise ValueError("[TRI_GATE] tri_gate_mode must be 'additive', 'concat', or 'scale'.")
        self.num_parts = num_parts
        self.part_moe_global_keep = part_moe_global_keep
        self.pos_input_dim = pos_input_dim
        self.part_moe_active = False
        self.part_experts = None
        self.last_tri_part_stats = None
        self.last_tri_part_reg_loss = None
        self.last_part_budget_stats = None

        self.input_ch = pos_input_dim
        self.pose_cond_dim, self.seq_pose_cond_dim, self.seq_xyz_cond_dim = 0, 0, 0

        if self.use_part_budget and self.use_tri:
            raise ValueError("[PART_BUDGET] part_budget is defined on top of part_moe_leg only; do not combine it with tri ablations.")
        if self.use_part_budget and not self.use_part_moe:
            raise ValueError("[PART_BUDGET] --use_part_budget must be used with --use_part_moe.")
        if self.use_part_budget:
            if self.part_label_schema != "part_moe_leg" or int(self.num_parts) != 7:
                raise ValueError(
                    "[PART_BUDGET] part_budget is defined on top of part_moe_leg: "
                    "use --part_label_schema part_moe_leg --num_parts 7."
                )
            print(
                "[PART_BUDGET] enabled=True; part-aware deformation router active. "
                f"alpha={self.part_budget_alpha} start={self.part_budget_start_iter} "
                f"warmup={self.part_budget_warmup} hidden={self.part_budget_hidden_dim} "
                f"token_dim={self.part_budget_token_dim}"
            )

        if self.use_pose_cond:
            self.PoseEncoder = PoseEncoder(32, pose_cond_dim, smpl_type)
            self.input_ch += pose_cond_dim
            
        if self.use_seq_pose_cond:
            self.SeqPoseEncoder = SeqPoseEncoder(seq_len, 16, seq_pose_cond_dim, time_step_num, smpl_type)
            self.input_ch += seq_pose_cond_dim

        if self.use_seq_xyz_cond:
            self.SeqXYZEncoder = SeqXYZEncoder(pos_emb_dim=pos_input_dim, hidden_dim1=96, hidden_dim2=256, output_dim=seq_xyz_cond_dim, 
                                        time_step_num=time_step_num, seq_len=seq_len, seq_xyz_knn=seq_xyz_knn)
            self.input_ch += seq_xyz_cond_dim
        if self.use_tri:
            self.TriPlaneFeature = TriPlaneFeature(
                feature_dim=self.tri_plane_dim,
                resolution=self.tri_plane_res,
                extent=self.tri_plane_extent,
            )
            self.tri_gate_base_input_ch = self.input_ch
            if self.use_tri_part and not self.use_tri_gate:
                self.PartTriFeatureFiLM = PartTriFeatureFiLM(
                    num_parts=self.num_parts,
                    feature_dim=self.tri_plane_dim,
                    resolution=self.tri_plane_res,
                    extent=self.tri_plane_extent,
                    alpha=self.tri_part_alpha,
                    motion_gain=self.tri_part_motion_gain,
                    boundary_gain=self.tri_part_boundary_gain,
                    hidden_dim=self.tri_part_hidden_dim,
                )
        
        layers = []
        in_dim = self.input_ch
        for _ in range(D):
            layers.append(nn.Linear(in_dim, W))
            layers.append(nn.ReLU())
            in_dim = W
        self.mlp = nn.Sequential(*layers)
        if self.use_tri and (not self.use_tri_gate or self.tri_gate_mode in ("concat", "scale")):
            self.mlp = _append_mlp_input_dim(self.mlp, self.tri_plane_dim)
            self.input_ch += self.tri_plane_dim
            if self.use_tri_gate:
                print(
                    f"[TRI_GATE] Tri-plane gated {self.tri_gate_mode} enabled: "
                    f"dim={self.tri_plane_dim} res={self.tri_plane_res} extent={self.tri_plane_extent} "
                    f"alpha={self.tri_gate_alpha} gate_init={self.tri_gate_init} "
                    f"hidden={self.tri_gate_hidden_dim}"
                )
            else:
                print(
                    "[TRI] Tri-plane feature enabled: "
                    f"dim={self.tri_plane_dim} res={self.tri_plane_res} extent={self.tri_plane_extent}"
                )
                if self.use_tri_part:
                    print(
                        "[TRI_PART] Part/motion-aware residual tri feature enabled: "
                        f"alpha={self.tri_part_alpha} motion_gain={self.tri_part_motion_gain} "
                        f"boundary_gain={self.tri_part_boundary_gain} hidden={self.tri_part_hidden_dim}"
                    )
        elif self.use_tri_gate:
            print(
                "[TRI_GATE] Tri-plane gated adapter enabled: "
                f"dim={self.tri_plane_dim} res={self.tri_plane_res} extent={self.tri_plane_extent} "
                f"alpha={self.tri_gate_alpha} gate_init={self.tri_gate_init} "
                f"hidden={self.tri_gate_hidden_dim}"
            )

        self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)
        if self.use_tri_gate:
            if self.tri_gate_mode in ("concat", "scale"):
                self.TriFeatureGate = TriFeatureGate(
                    tri_dim=self.tri_plane_dim,
                    feature_dim=self.tri_gate_base_input_ch,
                    hidden_dim=self.tri_gate_hidden_dim,
                    gate_init=self.tri_gate_init,
                )
            else:
                self.TriGateAdapter = TriGateAdapter(
                    tri_dim=self.tri_plane_dim,
                    feature_dim=self.tri_gate_base_input_ch,
                    hidden_dim=self.tri_gate_hidden_dim,
                    gate_init=self.tri_gate_init,
                )
        if self.use_part_budget:
            self.part_budget_stats_encoder = PartStatsEncoder(
                num_parts=self.num_parts,
                token_dim=self.part_budget_token_dim,
                hidden_dim=self.part_budget_hidden_dim,
            )
            self.part_budget_router = PartBudgetRouter(
                feature_dim=self.input_ch,
                token_dim=self.part_budget_token_dim,
                hidden_dim=self.part_budget_hidden_dim,
            )
            self.part_budget_adapter = PartBudgetAdapter(
                feature_dim=self.input_ch,
                token_dim=self.part_budget_token_dim,
                hidden_dim=self.part_budget_hidden_dim,
            )

    def init_part_moe_from_shared(self, num_parts=None):
        if self.part_moe_active:
            print("[PartMoE] Experts already initialized. Skip.")
            return False

        num_parts = int(num_parts or self.num_parts)
        self.num_parts = num_parts
        print(f"[PartMoE] Initializing {num_parts} experts from the shared non-rigid MLP.")
        self.part_experts = nn.ModuleList([
            PartNonrigidExpert(
                self.mlp,
                self.gaussian_warp,
                self.gaussian_rotation,
                self.gaussian_scaling,
            )
            for _ in range(num_parts)
        ])
        self.part_moe_active = True
        print(f"[PartMoE] expert_0: global/unknown; expert_1-{num_parts - 1}: routed part experts.")
        return True

    def freeze_shared_after_part_moe(self):
        for module in (self.mlp, self.gaussian_warp, self.gaussian_rotation, self.gaussian_scaling):
            for param in module.parameters():
                param.requires_grad_(False)

    def _normalize_budget_motion(self, motion_strength, features):
        if motion_strength is None:
            return torch.zeros(*features.shape[:2], 1, device=features.device, dtype=features.dtype)
        motion_strength = motion_strength.to(device=features.device, dtype=features.dtype)
        if motion_strength.dim() == 1:
            motion_strength = motion_strength.unsqueeze(0).expand(features.shape[0], -1).unsqueeze(-1)
        elif motion_strength.dim() == 2:
            if motion_strength.shape[0] == 1 and features.shape[0] > 1:
                motion_strength = motion_strength.expand(features.shape[0], -1)
            motion_strength = motion_strength.unsqueeze(-1)
        elif motion_strength.shape[0] == 1 and features.shape[0] > 1:
            motion_strength = motion_strength.expand(features.shape[0], -1, -1)
        motion_strength = torch.log1p(motion_strength.clamp_min(0.0))
        mean = motion_strength.detach().mean(dim=1, keepdim=True)
        std = motion_strength.detach().std(dim=1, keepdim=True).clamp_min(1e-6)
        return ((motion_strength - mean) / std).clamp(-3.0, 3.0)

    def _normalize_budget_conf(self, part_conf, features):
        if part_conf is None:
            return torch.ones(*features.shape[:2], 1, device=features.device, dtype=features.dtype)
        part_conf = part_conf.to(device=features.device, dtype=features.dtype)
        if part_conf.dim() == 1:
            part_conf = part_conf.unsqueeze(0).expand(features.shape[0], -1).unsqueeze(-1)
        elif part_conf.dim() == 2:
            if part_conf.shape[0] == 1 and features.shape[0] > 1:
                part_conf = part_conf.expand(features.shape[0], -1)
            part_conf = part_conf.unsqueeze(-1)
        elif part_conf.shape[0] == 1 and features.shape[0] > 1:
            part_conf = part_conf.expand(features.shape[0], -1, -1)
        return part_conf.clamp(0.0, 1.0)

    def apply_part_budget(self, features, part_label, query_xyz=None, motion_strength=None, part_conf=None,
                          part_budget_alpha_scale=1.0):
        if not self.use_part_budget:
            return features, None
        if part_label is None:
            return features, None

        part_label = part_label.long().to(features.device)
        if part_label.dim() == 1:
            part_label = part_label.unsqueeze(0).expand(features.shape[0], -1)
        elif part_label.shape[0] == 1 and features.shape[0] > 1:
            part_label = part_label.expand(features.shape[0], -1)
        part_label = torch.clamp(part_label, min=0, max=self.num_parts - 1)

        part_token, part_stats = self.part_budget_stats_encoder(
            part_label,
            motion_strength=motion_strength,
            part_conf=part_conf,
        )

        motion_norm = self._normalize_budget_motion(motion_strength, features)
        boundary_score = 1.0 - self._normalize_budget_conf(part_conf, features)
        token_idx = part_label.unsqueeze(-1).expand(-1, -1, self.part_budget_token_dim)
        point_token = torch.gather(part_token, 1, token_idx)

        budget, logits = self.part_budget_router(features, point_token, motion_norm, boundary_score)
        adapter_res = self.part_budget_adapter(features, point_token)

        alpha_scale = max(0.0, min(float(part_budget_alpha_scale), 1.0))
        alpha = self.part_budget_alpha * alpha_scale
        if alpha <= 0.0:
            stats = {
                "budget_mean": budget.detach().mean(dim=(0, 1)),
                "budget_std": budget.detach().std(dim=(0, 1), unbiased=False),
                "budget_min": budget.detach().amin(dim=(0, 1)),
                "budget_max": budget.detach().amax(dim=(0, 1)),
                "per_part_budget_mean": self._collect_per_part_budget_stats(budget.detach(), part_label, reduce="mean"),
                "per_part_budget_std": self._collect_per_part_budget_stats(budget.detach(), part_label, reduce="std"),
                "feature_delta_norm": torch.zeros((), device=features.device, dtype=features.dtype),
                "adapter_norm": adapter_res.detach().norm(dim=-1).mean(),
                "motion_mean": motion_norm.detach().mean(),
                "boundary_mean": boundary_score.detach().mean(),
                "entropy": self._budget_entropy(budget.detach()),
                "logits_mean": logits.detach().mean(),
            }
            return features, stats

        rigid_delta = budget[..., 0:1] - 1.0 / 3.0
        boundary_delta = budget[..., 2:3] - 1.0 / 3.0
        capacity_scale = 1.0 + alpha * (rigid_delta - boundary_delta)
        budgeted_features = features * capacity_scale + alpha * budget[..., 1:2] * adapter_res
        stats = {
            "budget_mean": budget.detach().mean(dim=(0, 1)),
            "budget_std": budget.detach().std(dim=(0, 1), unbiased=False),
            "budget_min": budget.detach().amin(dim=(0, 1)),
            "budget_max": budget.detach().amax(dim=(0, 1)),
            "per_part_budget_mean": self._collect_per_part_budget_stats(budget.detach(), part_label, reduce="mean"),
            "per_part_budget_std": self._collect_per_part_budget_stats(budget.detach(), part_label, reduce="std"),
            "feature_delta_norm": (budgeted_features - features).detach().norm(dim=-1).mean(),
            "adapter_norm": adapter_res.detach().norm(dim=-1).mean(),
            "motion_mean": motion_norm.detach().mean(),
            "boundary_mean": boundary_score.detach().mean(),
            "entropy": self._budget_entropy(budget.detach()),
            "logits_mean": logits.detach().mean(),
        }
        return budgeted_features, stats

    def _budget_entropy(self, budget):
        budget = budget.clamp_min(1e-8)
        return (-budget * budget.log()).sum(dim=-1).mean()

    def _collect_per_part_budget_stats(self, budget, part_label, reduce="mean"):
        if part_label.dim() == 1:
            part_label = part_label.unsqueeze(0).expand(budget.shape[0], -1)
        elif part_label.shape[0] == 1 and budget.shape[0] > 1:
            part_label = part_label.expand(budget.shape[0], -1)
        part_label = torch.clamp(part_label.long().to(budget.device), min=0, max=self.num_parts - 1)
        collected = []
        for pid in range(self.num_parts):
            mask = (part_label == pid).unsqueeze(-1).to(dtype=budget.dtype)
            count = mask.sum(dim=1, keepdim=True)
            count_safe = count.clamp_min(1.0)
            if reduce == "mean":
                value = (budget * mask).sum(dim=1, keepdim=True) / count_safe
            elif reduce == "std":
                mean = (budget * mask).sum(dim=1, keepdim=True) / count_safe
                var = ((budget - mean) ** 2 * mask).sum(dim=1, keepdim=True) / count_safe
                value = var.clamp_min(1e-6).sqrt()
            else:
                raise ValueError(f"Unknown reduce mode: {reduce}")
            valid = (count > 0).to(dtype=value.dtype)
            global_value = budget.mean(dim=1, keepdim=True) if reduce == "mean" else budget.std(dim=1, keepdim=True, unbiased=False)
            value = valid * value + (1.0 - valid) * global_value
            collected.append(value.squeeze(1))
        return torch.stack(collected, dim=1).detach()

    def forward_part_moe(self, features, part_label, part_moe_alpha=0.0, part_moe_global_keep=None):
        if not self.part_moe_active or self.part_experts is None:
            raise RuntimeError("[PartMoE] Experts have not been initialized.")

        part_label = part_label.long().to(features.device)
        part_label = torch.clamp(part_label, min=0, max=self.num_parts - 1)
        if part_label.dim() == 1:
            part_label = part_label.unsqueeze(0).expand(features.shape[0], -1)
        elif part_label.shape[0] == 1 and features.shape[0] > 1:
            part_label = part_label.expand(features.shape[0], -1)

        global_keep = self.part_moe_global_keep if part_moe_global_keep is None else float(part_moe_global_keep)
        global_keep = max(0.0, min(1.0, global_keep))
        max_part_weight = 1.0 - global_keep
        part_weight = max(0.0, min(float(part_moe_alpha), max_part_weight))
        global_weight = 1.0 - part_weight

        global_xyz, global_rotation, global_scaling = self.part_experts[0](features)

        if part_weight <= 0.0:
            return global_xyz, global_rotation, global_scaling

        feature_shape = features.shape
        flat_features = features.reshape(-1, feature_shape[-1])
        flat_labels = part_label.reshape(-1)

        d_xyz = global_xyz.reshape(-1, global_xyz.shape[-1])
        d_rotation = global_rotation.reshape(-1, global_rotation.shape[-1])
        d_scaling = global_scaling.reshape(-1, global_scaling.shape[-1])
        flat_global_xyz = d_xyz
        flat_global_rotation = d_rotation
        flat_global_scaling = d_scaling

        for pid in range(1, self.num_parts):
            idx = torch.nonzero(flat_labels == pid, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            part_xyz, part_rotation, part_scaling = self.part_experts[pid](
                flat_features.index_select(0, idx),
            )
            d_xyz = torch.index_copy(
                d_xyz,
                0,
                idx,
                global_weight * flat_global_xyz.index_select(0, idx) + part_weight * part_xyz,
            )
            d_rotation = torch.index_copy(
                d_rotation,
                0,
                idx,
                global_weight * flat_global_rotation.index_select(0, idx) + part_weight * part_rotation,
            )
            d_scaling = torch.index_copy(
                d_scaling,
                0,
                idx,
                global_weight * flat_global_scaling.index_select(0, idx) + part_weight * part_scaling,
            )

        return (
            d_xyz.reshape_as(global_xyz).contiguous(),
            d_rotation.reshape_as(global_rotation).contiguous(),
            d_scaling.reshape_as(global_scaling).contiguous(),
        )

    def forward_tri(self, features, part_label, part_moe_alpha=0.0, part_moe_global_keep=None):
        return self.forward_part_moe(
            features,
            part_label,
            part_moe_alpha=part_moe_alpha,
            part_moe_global_keep=part_moe_global_keep,
        )

    def sample_tri_features(self, query_xyz, batch_size, dtype):
        if query_xyz is None:
            raise RuntimeError("[TRI] query_xyz is required for tri-plane feature sampling.")
        if query_xyz.dim() == 2:
            query_xyz = query_xyz.unsqueeze(0)
        if query_xyz.shape[0] == 1 and batch_size > 1:
            query_xyz = query_xyz.expand(batch_size, -1, -1)
        return self.TriPlaneFeature(query_xyz.to(device=self.TriPlaneFeature.planes.device, dtype=dtype))

    def apply_tri_part_film(self, tri_features, query_xyz, part_label, motion_strength=None, part_conf=None):
        if not self.use_tri_part:
            return tri_features, None
        if part_label is None:
            return tri_features, None
        return self.PartTriFeatureFiLM(
            tri_features,
            query_xyz,
            part_label,
            motion_strength=motion_strength,
            part_conf=part_conf,
        )

    def apply_tri_gate_adapter(self, base_features, tri_features, tri_gate_alpha_scale=1.0):
        alpha_scale = max(0.0, min(float(tri_gate_alpha_scale), 1.0))
        alpha = self.tri_gate_alpha * alpha_scale
        if self.tri_gate_mode == "concat":
            gate = self.TriFeatureGate(base_features, tri_features)
            return torch.cat([base_features, alpha * gate * tri_features], dim=-1)
        if self.tri_gate_mode == "scale":
            gate = self.TriFeatureGate(base_features, tri_features)
            tri_scale = 1.0 + alpha * (gate - self.tri_gate_init)
            return torch.cat([base_features, tri_scale * tri_features], dim=-1)
        tri_res, _ = self.TriGateAdapter(base_features, tri_features)
        return base_features + alpha * tri_res

    def forward(self, x_emb, pose_conds=None, seq_pose_conds=None, seq_xyz_conds=None,
                part_label=None, part_enabled=False,
                query_xyz=None, part_moe_alpha=0.0, part_moe_global_keep=None,
                tri_gate_alpha_scale=1.0, part_conf=None, part_budget_alpha_scale=1.0):
        feats = []
        feats.append(x_emb)

        # single frame pose condition
        if self.use_pose_cond: 
            pose_feats = self.PoseEncoder(pose_conds)
            pose_feats = pose_feats.unsqueeze(1).expand(-1, x_emb.shape[1], -1)
            feats.append(pose_feats)
        
        # sequential pose condition
        if self.use_seq_pose_cond:
            seq_pose_feats = self.SeqPoseEncoder(seq_pose_conds)
            seq_pose_feats = seq_pose_feats.unsqueeze(1).expand(-1, x_emb.shape[1], -1)
            feats.append(seq_pose_feats)
        
        # sequential point-wise delta xyz condition
        seq_xyz_feats = None
        if self.use_seq_xyz_cond: 
            seq_xyz_feats = self.SeqXYZEncoder(seq_xyz_conds, x_emb)
            feats.append(seq_xyz_feats)

        features = torch.cat(feats, dim=-1)
        self.last_part_budget_stats = None
        if self.use_part_budget and self.part_moe_active and part_enabled and part_label is not None:
            motion_strength = None
            if seq_xyz_conds is not None:
                motion_strength = seq_xyz_conds.detach().norm(dim=-1)
                reduce_dims = tuple(range(2, motion_strength.dim()))
                motion_strength = motion_strength.mean(dim=reduce_dims).unsqueeze(-1)
            features, part_budget_stats = self.apply_part_budget(
                features,
                part_label,
                query_xyz=query_xyz,
                motion_strength=motion_strength,
                part_conf=part_conf,
                part_budget_alpha_scale=part_budget_alpha_scale,
            )
            self.last_part_budget_stats = part_budget_stats

        if self.use_tri:
            tri_features = self.sample_tri_features(query_xyz, x_emb.shape[0], x_emb.dtype)
            self.last_tri_part_stats = None
            self.last_tri_part_reg_loss = None
            if self.use_tri_gate:
                features = self.apply_tri_gate_adapter(
                    features,
                    tri_features,
                    tri_gate_alpha_scale=tri_gate_alpha_scale,
                )
            elif self.use_tri_part and part_enabled and part_label is not None:
                motion_strength = None
                if seq_xyz_conds is not None:
                    motion_strength = seq_xyz_conds.detach().norm(dim=-1)
                    reduce_dims = tuple(range(2, motion_strength.dim()))
                    motion_strength = motion_strength.mean(dim=reduce_dims).unsqueeze(-1)
                tri_features, tri_part_stats = self.apply_tri_part_film(
                    tri_features,
                    query_xyz,
                    part_label,
                    motion_strength=motion_strength,
                    part_conf=part_conf,
                )
                self.last_tri_part_stats = tri_part_stats
                self.last_tri_part_reg_loss = (
                    tri_part_stats["reg_loss"] if tri_part_stats is not None else None
                )
                features = torch.cat([features, tri_features], dim=-1)
            else:
                features = torch.cat([features, tri_features], dim=-1)

        if (
            self.use_part_moe
            and self.part_moe_active
            and part_enabled
            and part_label is not None
        ):
            if self.use_tri:
                return self.forward_tri(
                    features,
                    part_label,
                    part_moe_alpha=part_moe_alpha,
                    part_moe_global_keep=part_moe_global_keep,
                )
            return self.forward_part_moe(
                features,
                part_label,
                part_moe_alpha=part_moe_alpha,
                part_moe_global_keep=part_moe_global_keep,
            )

        h = self.mlp(features)
        d_xyz, d_scaling, d_rotation = self.gaussian_warp(h), self.gaussian_scaling(h), self.gaussian_rotation(h)
        
        return d_xyz, d_rotation, d_scaling

N_JOINT = {'smpl': 23, 'smplx': 54}

class PoseEncoder(nn.Module):
    def __init__(self, D1, D2, smpl_type):
        super(PoseEncoder, self).__init__()
        
        self.input_dim = 3 * N_JOINT[smpl_type] # axis-angle form
        self.mlp = nn.Sequential(nn.Linear(self.input_dim,D1), nn.ReLU(),
                                 nn.Linear(D1, D2), nn.ReLU())
    def forward(self, x):
        '''
        x: (B,J,3) Axis Angele
        return output (N,self.output_dim)
        '''
        bs = x.shape[0]
        x_joint_flat = x.view(bs, -1)

        return self.mlp(x_joint_flat)

class SeqPoseEncoder(nn.Module):
    def __init__(self, length, D1, D2, time_step_num, smpl_type):
        super(SeqPoseEncoder, self).__init__()

        self.input_dim = 3 * (N_JOINT[smpl_type] + 1) # axis-angle form, + global orientation
        self.mlp1 = nn.Sequential(nn.Linear(self.input_dim*time_step_num,D1), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Linear(D1*length, D2), nn.ReLU())

    def forward(self, x):
        # x: (B, N, T, J, DeltaStep, C)

        bs, T = x.shape[0], x.shape[1]
        x = self.mlp1(x.view(bs, T, -1))
        x = self.mlp2(x.view(bs, -1))

        return x

class SeqXYZEncoder(nn.Module):
    def __init__(self, vel_dim=3, pos_emb_dim=63, vel_emb_dim=64, pos_emb_proj_dim=32, 
                 hidden_dim1=96, hidden_dim2=256, output_dim=128, 
                 time_step_num=1, seq_len=6, seq_xyz_knn=5):
        super(SeqXYZEncoder, self).__init__()

        self.vel_encoder = nn.Sequential(nn.Linear(vel_dim*seq_xyz_knn*time_step_num, vel_emb_dim), nn.ReLU())
        self.pos_emb_proj = nn.Sequential(nn.Linear(pos_emb_dim, pos_emb_proj_dim), nn.ReLU())
        
        self.mlp1 = nn.Sequential(nn.Linear(vel_emb_dim+pos_emb_proj_dim, hidden_dim1), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Linear(hidden_dim1*seq_len, hidden_dim2), nn.ReLU(),
                                  nn.Linear(hidden_dim2, output_dim), nn.ReLU())
    def forward(self, x, x_emb):
        # x -> B, N, T, KNN, DeltaStep, C
        B, N, T = x.shape[0], x.shape[1], x.shape[2]

        pos_feat = self.pos_emb_proj(x_emb)
        pos_feat = pos_feat.unsqueeze(2).expand(-1, -1, T, -1)
        vel_emb = self.vel_encoder(x.view(B, N, T, -1))

        h = torch.concat([vel_emb, pos_feat], dim=-1)
        h = self.mlp1(h)
        h = self.mlp2(h.view(B, N, -1))

        return h
