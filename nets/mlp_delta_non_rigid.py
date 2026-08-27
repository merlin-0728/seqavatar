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


def _normalize_budget_query_xyz(query_xyz, features):
    if query_xyz is None:
        return torch.zeros(*features.shape[:2], 4, device=features.device, dtype=features.dtype)
    query_xyz = query_xyz.to(device=features.device, dtype=features.dtype)
    if query_xyz.dim() == 2:
        query_xyz = query_xyz.unsqueeze(0)
    if query_xyz.shape[0] == 1 and features.shape[0] > 1:
        query_xyz = query_xyz.expand(features.shape[0], -1, -1)
    radius = query_xyz.norm(dim=-1, keepdim=True)
    scale = radius.detach().mean(dim=1, keepdim=True).clamp_min(1e-3)
    xyz_norm = (query_xyz / scale).clamp(-3.0, 3.0)
    radius_norm = (radius / scale).clamp(0.0, 3.0)
    return torch.cat([xyz_norm, radius_norm], dim=-1)


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


class TokenTriPlaneBlock(nn.Module):
    def __init__(self, feature_dim, num_heads, hidden_dim):
        super().__init__()
        self.attn_norm = nn.LayerNorm(feature_dim)
        self.ffn_norm = nn.LayerNorm(feature_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, plane_tokens, cond_tokens):
        attn_out, _ = self.cross_attn(
            self.attn_norm(plane_tokens),
            cond_tokens,
            cond_tokens,
            need_weights=False,
        )
        plane_tokens = plane_tokens + attn_out
        plane_tokens = plane_tokens + self.ffn(self.ffn_norm(plane_tokens))
        return plane_tokens


class TokenConditionedTriPlane(nn.Module):
    def __init__(self, pose_dim, seq_pose_dim, num_parts, feature_dim=32, resolution=16,
                 extent=1.0, num_heads=4, num_layers=2, hidden_dim=128):
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.seq_pose_dim = int(seq_pose_dim)
        self.num_parts = int(num_parts)
        self.feature_dim = int(feature_dim)
        self.resolution = int(resolution)
        self.extent = float(extent)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)

        if self.num_parts <= 0:
            raise ValueError("[TRI_TOKEN] num_parts must be positive.")
        if self.feature_dim <= 0:
            raise ValueError("[TRI_TOKEN] token_tri_dim must be positive.")
        if self.resolution <= 1:
            raise ValueError("[TRI_TOKEN] token_tri_res must be greater than 1.")
        if self.extent <= 0:
            raise ValueError("[TRI_TOKEN] token_tri_extent must be positive.")
        if self.num_heads <= 0 or self.feature_dim % self.num_heads != 0:
            raise ValueError("[TRI_TOKEN] token_tri_dim must be divisible by token_tri_heads.")
        if self.num_layers <= 0:
            raise ValueError("[TRI_TOKEN] token_tri_layers must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("[TRI_TOKEN] token_tri_hidden_dim must be positive.")

        self.base_planes = nn.Parameter(torch.zeros(3, self.feature_dim, self.resolution, self.resolution))
        self.plane_query = nn.Parameter(torch.empty(3 * self.resolution * self.resolution, self.feature_dim))
        nn.init.normal_(self.plane_query, mean=0.0, std=0.02)

        self.pose_proj = self._make_token_mlp(self.pose_dim) if self.pose_dim > 0 else None
        self.seq_pose_proj = self._make_token_mlp(self.seq_pose_dim) if self.seq_pose_dim > 0 else None
        self.motion_proj = self._make_token_mlp(4)
        self.part_stats_encoder = PartStatsEncoder(
            num_parts=self.num_parts,
            token_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
        )
        self.fallback_part_tokens = nn.Parameter(torch.zeros(self.num_parts, self.feature_dim))
        nn.init.normal_(self.fallback_part_tokens, mean=0.0, std=0.02)

        self.cond_norm = nn.LayerNorm(self.feature_dim)
        self.blocks = nn.ModuleList([
            TokenTriPlaneBlock(self.feature_dim, self.num_heads, self.hidden_dim)
            for _ in range(self.num_layers)
        ])
        self.plane_norm = nn.LayerNorm(self.feature_dim)
        self.delta_head = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature_proj = nn.Linear(self.feature_dim, self.feature_dim)
        nn.init.zeros_(self.feature_proj.weight)
        nn.init.zeros_(self.feature_proj.bias)

    def _make_token_mlp(self, input_dim):
        return nn.Sequential(
            nn.Linear(int(input_dim), self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )

    def _normalize_xyz(self, query_xyz):
        return (query_xyz / self.extent).clamp(-1.0, 1.0)

    def _motion_stats(self, seq_xyz_conds, batch_size, device, dtype):
        if seq_xyz_conds is None:
            return torch.zeros(batch_size, 4, device=device, dtype=dtype)
        motion = seq_xyz_conds.detach().to(device=device, dtype=dtype).norm(dim=-1)
        motion = torch.log1p(motion.clamp_min(0.0)).reshape(batch_size, -1)
        return torch.stack(
            [
                motion.mean(dim=1),
                motion.std(dim=1, unbiased=False),
                motion.amax(dim=1),
                motion.pow(2).mean(dim=1).sqrt(),
            ],
            dim=-1,
        )

    def _point_motion(self, seq_xyz_conds, batch_size, num_points, device, dtype):
        if seq_xyz_conds is None:
            return None
        motion = seq_xyz_conds.detach().to(device=device, dtype=dtype).norm(dim=-1)
        reduce_dims = tuple(range(2, motion.dim()))
        motion = motion.mean(dim=reduce_dims).unsqueeze(-1)
        if motion.shape[0] == 1 and batch_size > 1:
            motion = motion.expand(batch_size, -1, -1)
        if motion.shape[1] != num_points:
            return None
        return motion

    def _sample_plane(self, planes, plane_id, grid):
        sampled = F.grid_sample(
            planes[:, plane_id],
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(-1).transpose(1, 2).contiguous()

    def _sample_planes(self, planes, query_xyz):
        coords = self._normalize_xyz(query_xyz)
        xy_grid = coords[..., [0, 1]].unsqueeze(2)
        xz_grid = coords[..., [0, 2]].unsqueeze(2)
        yz_grid = coords[..., [1, 2]].unsqueeze(2)
        xy_feat = self._sample_plane(planes, 0, xy_grid)
        xz_feat = self._sample_plane(planes, 1, xz_grid)
        yz_feat = self._sample_plane(planes, 2, yz_grid)
        return (xy_feat + xz_feat + yz_feat) / 3.0

    def forward(self, query_xyz, batch_size, num_points, dtype, pose_feats=None,
                seq_pose_feats=None, seq_xyz_conds=None, part_label=None, part_conf=None,
                alpha_scale=1.0):
        device = self.base_planes.device
        if query_xyz is None:
            zero_feat = torch.zeros(batch_size, num_points, self.feature_dim, device=device, dtype=dtype)
            return zero_feat, None

        if query_xyz.dim() == 2:
            query_xyz = query_xyz.unsqueeze(0)
        query_xyz = query_xyz.to(device=device, dtype=dtype)
        if query_xyz.shape[0] == 1 and batch_size > 1:
            query_xyz = query_xyz.expand(batch_size, -1, -1)
        if query_xyz.shape[1] != num_points:
            raise RuntimeError(
                f"[TRI_TOKEN] query_xyz points {query_xyz.shape[1]} != feature points {num_points}."
            )

        tokens = []
        if self.pose_proj is not None and pose_feats is not None:
            tokens.append(self.pose_proj(pose_feats.to(device=device, dtype=dtype)).unsqueeze(1))
        if self.seq_pose_proj is not None and seq_pose_feats is not None:
            tokens.append(self.seq_pose_proj(seq_pose_feats.to(device=device, dtype=dtype)).unsqueeze(1))

        motion_stats = self._motion_stats(seq_xyz_conds, batch_size, device, dtype)
        tokens.append(self.motion_proj(motion_stats).unsqueeze(1))

        point_motion = self._point_motion(seq_xyz_conds, batch_size, num_points, device, dtype)
        if part_label is not None:
            part_label = part_label.long().to(device=device)
            if part_label.dim() == 1:
                part_label = part_label.unsqueeze(0).expand(batch_size, -1)
            elif part_label.shape[0] == 1 and batch_size > 1:
                part_label = part_label.expand(batch_size, -1)
            part_label = torch.clamp(part_label, min=0, max=self.num_parts - 1)
            part_tokens, part_stats = self.part_stats_encoder(
                part_label,
                motion_strength=point_motion,
                part_conf=part_conf,
            )
        else:
            part_tokens = self.fallback_part_tokens.unsqueeze(0).expand(batch_size, -1, -1)
            part_stats = None
        tokens.append(part_tokens.to(device=device, dtype=dtype))

        cond_tokens = self.cond_norm(torch.cat(tokens, dim=1))
        plane_tokens = self.plane_query.to(dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        for block in self.blocks:
            plane_tokens = block(plane_tokens, cond_tokens)

        plane_delta = self.delta_head(self.plane_norm(plane_tokens))
        plane_delta = plane_delta.view(batch_size, 3, self.resolution, self.resolution, self.feature_dim)
        plane_delta = plane_delta.permute(0, 1, 4, 2, 3).contiguous()
        planes = self.base_planes.to(dtype=dtype).unsqueeze(0) + plane_delta
        raw_features = self._sample_planes(planes, query_xyz)

        alpha_scale = max(0.0, min(float(alpha_scale), 1.0))
        projected = self.feature_proj(raw_features) * alpha_scale
        stats = {
            "alpha_scale": torch.tensor(alpha_scale, device=device, dtype=dtype),
            "plane_mean": planes.detach().mean(),
            "plane_std": planes.detach().std(unbiased=False),
            "plane_delta_norm": plane_delta.detach().norm(dim=2).mean(),
            "raw_feature_norm": raw_features.detach().norm(dim=-1).mean(),
            "projected_norm": projected.detach().norm(dim=-1).mean(),
            "token_std": cond_tokens.detach().std(unbiased=False),
            "part_stats_mean": part_stats.detach().mean() if part_stats is not None else torch.zeros((), device=device, dtype=dtype),
        }
        return projected, stats


class TriTokenResidualAdapter(nn.Module):
    def __init__(self, base_feature_dim, token_dim, hidden_dim=128):
        super().__init__()
        self.base_feature_dim = int(base_feature_dim)
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        if self.base_feature_dim <= 0:
            raise ValueError("[TRI_TOKEN] base_feature_dim must be positive.")
        if self.token_dim <= 0:
            raise ValueError("[TRI_TOKEN] token_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("[TRI_TOKEN] hidden_dim must be positive.")

        self.mlp = nn.Sequential(
            nn.Linear(self.base_feature_dim + self.token_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.base_feature_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, base_features, token_features):
        return self.mlp(torch.cat([base_features, token_features], dim=-1))


class TriTokenRouteAdapter(nn.Module):
    def __init__(self, base_feature_dim, token_dim, hidden_dim=128):
        super().__init__()
        self.base_feature_dim = int(base_feature_dim)
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        if self.base_feature_dim <= 0:
            raise ValueError("[TRI_TOKEN] base_feature_dim must be positive.")
        if self.token_dim <= 0:
            raise ValueError("[TRI_TOKEN] token_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("[TRI_TOKEN] hidden_dim must be positive.")

        router_in_dim = self.base_feature_dim + self.token_dim + 6
        self.adapter = nn.Sequential(
            nn.Linear(router_in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.base_feature_dim),
        )
        self.router = nn.Sequential(
            nn.Linear(router_in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, base_features, token_features, motion_strength, boundary_score, query_feat=None):
        if query_feat is None:
            query_feat = torch.zeros(*base_features.shape[:2], 4, device=base_features.device, dtype=base_features.dtype)
        router_input = torch.cat([base_features, token_features, motion_strength, boundary_score, query_feat], dim=-1)
        routed_delta = self.adapter(router_input)
        route_gate = torch.sigmoid(self.router(router_input))
        return route_gate * routed_delta, route_gate


class TriTokenPartFusionGate(nn.Module):
    def __init__(self, base_feature_dim, token_dim, hidden_dim=128):
        super().__init__()
        self.base_feature_dim = int(base_feature_dim)
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        if self.base_feature_dim <= 0:
            raise ValueError("[TRI_TOKEN] base_feature_dim must be positive.")
        if self.token_dim <= 0:
            raise ValueError("[TRI_TOKEN] token_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("[TRI_TOKEN] hidden_dim must be positive.")

        fusion_in_dim = self.base_feature_dim + self.token_dim + 6
        self.gate = nn.Sequential(
            nn.Linear(fusion_in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, base_features, token_features, motion_strength, boundary_score, query_feat=None):
        if query_feat is None:
            query_feat = torch.zeros(*base_features.shape[:2], 4, device=base_features.device, dtype=base_features.dtype)
        fusion_input = torch.cat([base_features, token_features, motion_strength, boundary_score, query_feat], dim=-1)
        return torch.tanh(self.gate(fusion_input))


class TriTokenPartFusionSpatialGate(nn.Module):
    def __init__(self, base_feature_dim, token_dim, num_parts, hidden_dim=128):
        super().__init__()
        self.base_feature_dim = int(base_feature_dim)
        self.token_dim = int(token_dim)
        self.num_parts = int(num_parts)
        self.hidden_dim = int(hidden_dim)
        if self.base_feature_dim <= 0:
            raise ValueError("[TRI_TOKEN] base_feature_dim must be positive.")
        if self.token_dim <= 0:
            raise ValueError("[TRI_TOKEN] token_dim must be positive.")
        if self.num_parts <= 0:
            raise ValueError("[TRI_TOKEN] num_parts must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("[TRI_TOKEN] hidden_dim must be positive.")

        router_in_dim = self.base_feature_dim + self.token_dim + 6
        self.adapter = nn.Sequential(
            nn.Linear(router_in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.base_feature_dim),
        )
        self.route_gate = nn.Sequential(
            nn.Linear(router_in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.spatial_gate = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.part_gate_logit = nn.Embedding(self.num_parts, 1)

        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        nn.init.zeros_(self.route_gate[-1].weight)
        nn.init.zeros_(self.route_gate[-1].bias)
        nn.init.zeros_(self.spatial_gate[-1].weight)
        nn.init.zeros_(self.spatial_gate[-1].bias)

        priors = [0.35, 0.25, 0.60, 0.60, 0.25, 0.80, 0.80]
        if self.num_parts != len(priors):
            priors = [0.5 for _ in range(self.num_parts)]
        prior_tensor = torch.tensor(priors, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
        with torch.no_grad():
            self.part_gate_logit.weight.copy_(torch.logit(prior_tensor).view(self.num_parts, 1))

    def forward(self, base_features, token_features, part_label, motion_strength, boundary_score, query_feat=None):
        if query_feat is None:
            query_feat = torch.zeros(*base_features.shape[:2], 4, device=base_features.device, dtype=base_features.dtype)
        if part_label.dim() == 1:
            part_label = part_label.unsqueeze(0).expand(base_features.shape[0], -1)
        elif part_label.shape[0] == 1 and base_features.shape[0] > 1:
            part_label = part_label.expand(base_features.shape[0], -1)
        part_label = part_label.long().to(device=base_features.device)
        part_label = torch.clamp(part_label, min=0, max=self.num_parts - 1)

        router_input = torch.cat([base_features, token_features, motion_strength, boundary_score, query_feat], dim=-1)
        routed_delta = torch.tanh(self.adapter(router_input))
        route_logit = self.route_gate(router_input)
        spatial_center = query_feat - query_feat.mean(dim=1, keepdim=True)
        spatial_logit = self.spatial_gate(spatial_center)
        part_logit = self.part_gate_logit(part_label)
        gate_logits = route_logit + spatial_logit + part_logit
        route_gate = torch.sigmoid(gate_logits)
        return routed_delta, route_gate, route_logit, spatial_logit, part_logit


class TriTokenHardRouteAdapter(nn.Module):
    def __init__(self, base_feature_dim, token_dim, hidden_dim=128):
        super().__init__()
        self.base_feature_dim = int(base_feature_dim)
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        if self.base_feature_dim <= 0:
            raise ValueError("[TRI_TOKEN] base_feature_dim must be positive.")
        if self.token_dim <= 0:
            raise ValueError("[TRI_TOKEN] token_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("[TRI_TOKEN] hidden_dim must be positive.")

        router_in_dim = self.base_feature_dim + self.token_dim + 7
        self.adapter = nn.Sequential(
            nn.Linear(router_in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.base_feature_dim),
        )
        self.router = nn.Sequential(
            nn.Linear(router_in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.expert_router = nn.Sequential(
            nn.Linear(router_in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        nn.init.zeros_(self.expert_router[-1].weight)
        nn.init.zeros_(self.expert_router[-1].bias)

    def forward(self, base_features, token_features, motion_strength, boundary_score, hard_focus, query_feat=None):
        if query_feat is None:
            query_feat = torch.zeros(*base_features.shape[:2], 4, device=base_features.device, dtype=base_features.dtype)
        router_input = torch.cat(
            [base_features, token_features, motion_strength, boundary_score, hard_focus, query_feat],
            dim=-1,
        )
        routed_delta = self.adapter(router_input)
        route_gate = torch.sigmoid(self.router(router_input))
        expert_gate = torch.sigmoid(self.expert_router(router_input))
        return route_gate * routed_delta * hard_focus, route_gate, expert_gate


class TriTokenOutputRouteAdapter(nn.Module):
    def __init__(self, base_feature_dim, token_dim, output_dim=10, hidden_dim=128):
        super().__init__()
        self.base_feature_dim = int(base_feature_dim)
        self.token_dim = int(token_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        if self.base_feature_dim <= 0:
            raise ValueError("[TRI_TOKEN] base_feature_dim must be positive.")
        if self.token_dim <= 0:
            raise ValueError("[TRI_TOKEN] token_dim must be positive.")
        if self.output_dim <= 0:
            raise ValueError("[TRI_TOKEN] output_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("[TRI_TOKEN] hidden_dim must be positive.")

        router_in_dim = self.base_feature_dim + self.token_dim + self.output_dim + 6
        self.adapter = nn.Sequential(
            nn.Linear(router_in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        self.router = nn.Sequential(
            nn.Linear(router_in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, base_features, token_features, output_features, motion_strength, boundary_score, query_feat=None):
        if query_feat is None:
            query_feat = torch.zeros(*base_features.shape[:2], 4, device=base_features.device, dtype=base_features.dtype)
        router_input = torch.cat([base_features, token_features, output_features, motion_strength, boundary_score, query_feat], dim=-1)
        routed_delta = self.adapter(router_input)
        route_gate = torch.sigmoid(self.router(router_input))
        return route_gate * routed_delta, route_gate


class PartScoreRouteAdapter(nn.Module):
    def __init__(self, base_feature_dim, num_routed_parts, hidden_dim=128, gate_bias=-2.0):
        super().__init__()
        self.base_feature_dim = int(base_feature_dim)
        self.num_routed_parts = int(num_routed_parts)
        self.hidden_dim = int(hidden_dim)
        self.gate_bias = float(gate_bias)
        if self.base_feature_dim <= 0:
            raise ValueError("[PART_SCORE_ROUTE] base_feature_dim must be positive.")
        if self.num_routed_parts <= 0:
            raise ValueError("[PART_SCORE_ROUTE] num_routed_parts must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("[PART_SCORE_ROUTE] hidden_dim must be positive.")

        router_in_dim = self.base_feature_dim + 8
        self.shared = nn.Sequential(
            nn.Linear(router_in_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.weight_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.expert_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_routed_parts),
        )
        nn.init.zeros_(self.weight_head[-1].weight)
        nn.init.constant_(self.weight_head[-1].bias, self.gate_bias)
        nn.init.zeros_(self.expert_head[-1].weight)
        nn.init.zeros_(self.expert_head[-1].bias)

    def forward(self, base_features, score_focus, motion_strength, boundary_score, unknown_score, query_feat=None):
        if query_feat is None:
            query_feat = torch.zeros(*base_features.shape[:2], 4, device=base_features.device, dtype=base_features.dtype)
        if score_focus.dim() == 2:
            score_focus = score_focus.unsqueeze(-1)
        if motion_strength.dim() == 2:
            motion_strength = motion_strength.unsqueeze(-1)
        if boundary_score.dim() == 2:
            boundary_score = boundary_score.unsqueeze(-1)
        if unknown_score.dim() == 2:
            unknown_score = unknown_score.unsqueeze(-1)
        router_input = torch.cat(
            [base_features, score_focus, motion_strength, boundary_score, unknown_score, query_feat],
            dim=-1,
        )
        hidden = self.shared(router_input)
        route_logit = self.weight_head(hidden)
        expert_logits = self.expert_head(hidden)
        return route_logit, expert_logits


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
    def __init__(self, feature_dim, token_dim, hidden_dim=128, logit_scale=1.0):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.logit_scale = float(logit_scale)
        if self.feature_dim <= 0:
            raise ValueError("[PART_BUDGET] feature_dim must be positive.")
        if self.token_dim <= 0:
            raise ValueError("[PART_BUDGET] token_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("[PART_BUDGET] hidden_dim must be positive.")

        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim + self.token_dim + 6, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, base_features, part_token, motion_strength, boundary_score, query_feat=None):
        if query_feat is None:
            query_feat = torch.zeros(*base_features.shape[:2], 4, device=base_features.device, dtype=base_features.dtype)
        router_input = torch.cat([base_features, part_token, motion_strength, boundary_score, query_feat], dim=-1)
        logits = self.mlp(router_input)
        logits = logits * self.logit_scale
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


class PartBudgetOutputRouter(nn.Module):
    def __init__(self, feature_dim, token_dim, hidden_dim=128, logit_scale=1.0):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.logit_scale = float(logit_scale)
        if self.feature_dim <= 0:
            raise ValueError("[PART_BUDGET] feature_dim must be positive.")
        if self.token_dim <= 0:
            raise ValueError("[PART_BUDGET] token_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("[PART_BUDGET] hidden_dim must be positive.")

        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim + self.token_dim + 6, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, base_features, part_token, motion_strength, boundary_score, query_feat=None):
        if query_feat is None:
            query_feat = torch.zeros(*base_features.shape[:2], 4, device=base_features.device, dtype=base_features.dtype)
        router_input = torch.cat([base_features, part_token, motion_strength, boundary_score, query_feat], dim=-1)
        logits = self.mlp(router_input)
        logits = logits * self.logit_scale
        budget = torch.softmax(logits, dim=-1)
        return budget, logits


class PartBudgetFeatureFiLM(nn.Module):
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
            nn.Linear(self.hidden_dim, self.feature_dim * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, base_features, part_token):
        film_input = torch.cat([base_features, part_token], dim=-1)
        film = self.mlp(film_input)
        gamma, beta = film.chunk(2, dim=-1)
        return gamma, beta


class PartBudgetOutputFiLM(nn.Module):
    def __init__(self, feature_dim, token_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.token_dim = int(token_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        if self.feature_dim <= 0:
            raise ValueError("[PART_BUDGET] feature_dim must be positive.")
        if self.token_dim <= 0:
            raise ValueError("[PART_BUDGET] token_dim must be positive.")
        if self.output_dim <= 0:
            raise ValueError("[PART_BUDGET] output_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("[PART_BUDGET] hidden_dim must be positive.")

        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim + self.token_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, base_features, part_token):
        film_input = torch.cat([base_features, part_token], dim=-1)
        film = self.mlp(film_input)
        gamma, beta = film.chunk(2, dim=-1)
        return gamma, beta


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
                 use_tri_token=False, token_tri_dim=32, token_tri_res=16, token_tri_extent=1.0,
                 token_tri_heads=4, token_tri_layers=2, token_tri_hidden_dim=128,
                 token_tri_fusion_mode="concat", token_tri_fusion_hidden_dim=128,
                 token_tri_alpha=1.0, token_tri_start_iter=10000, token_tri_warmup=1000,
                 token_tri_route_boundary_w=0.0, token_tri_route_boundary_floor=0.15,
                 token_tri_route_output_alpha=0.2, token_tri_route_hard_w=0.0,
                 token_tri_route_hard_boundary_mix=0.65, token_tri_route_hard_motion_mix=0.35,
                 token_tri_part_fusion_delta_scale=0.35,
                 token_tri_part_fusion_spatial_delta_scale=0.45,
                 token_tri_part_fusion_spatial_w=0.01,
                 token_tri_part_fusion_spatial_std_floor=0.08,
                 token_tri_part_fusion_spatial_motion_mix=0.5,
                 token_tri_part_fusion_spatial_boundary_mix=0.5,
                 token_tri_part_fusion_spatial_part_mix=0.5,
                 use_time=False, time_scale_emb_dim=16, time_scale_temperature=1.5,
                 use_tri_part=False, use_tri_gate=False, tri_gate_alpha=0.2,
                 tri_gate_init=0.5, tri_gate_hidden_dim=128, tri_gate_mode="additive",
                 tri_part_alpha=1.0, tri_part_motion_gain=0.5,
                 tri_part_boundary_gain=0.5, tri_part_hidden_dim=64,
                 part_label_schema="anatomy5", use_part_budget=False, part_budget_alpha=1.0,
                 part_budget_start_iter=16000, part_budget_warmup=1000,
                 part_budget_hidden_dim=128, part_budget_token_dim=32,
                 part_budget_mode="base", part_budget_sup_w=0.02,
                 part_budget_balance_w=0.005, part_budget_target_mix=0.6,
                 part_budget_target_sharpness=2.0, part_budget_router_sharpness=1.0,
                 use_part_score_route=False, part_score_route_hidden_dim=128,
                 part_score_route_alpha=1.0, part_score_route_gate_bias=-2.0,
                 part_score_route_mode="boost",
                 part_score_route_use_route_gate=1,
                 part_score_route_use_unknown_mix=1,
                 part_score_route_signal_mode="full"):
        super(NonrigidDeformer, self).__init__()

        self.use_pose_cond = use_pose_cond
        self.use_seq_pose_cond = use_seq_pose_cond
        self.use_seq_xyz_cond = use_seq_xyz_cond
        self.use_part_moe = use_part_moe
        self.use_time = bool(use_time)
        self.use_tri = bool(use_tri and use_part_moe)
        self.use_tri_part = bool(use_tri_part and self.use_tri)
        self.use_tri_gate = bool(use_tri_gate and self.use_tri)
        self.use_tri_token = bool(use_tri_token)
        self.tri_plane_dim = int(tri_plane_dim)
        self.tri_plane_res = int(tri_plane_res)
        self.tri_plane_extent = float(tri_plane_extent)
        self.token_tri_dim = int(token_tri_dim)
        self.token_tri_res = int(token_tri_res)
        self.token_tri_extent = float(token_tri_extent)
        self.token_tri_heads = int(token_tri_heads)
        self.token_tri_layers = int(token_tri_layers)
        self.token_tri_hidden_dim = int(token_tri_hidden_dim)
        self.token_tri_fusion_mode = str(token_tri_fusion_mode).lower()
        self.token_tri_fusion_hidden_dim = int(token_tri_fusion_hidden_dim)
        self.token_tri_alpha = float(token_tri_alpha)
        self.token_tri_start_iter = int(token_tri_start_iter)
        self.token_tri_warmup = int(token_tri_warmup)
        self.token_tri_route_boundary_w = float(token_tri_route_boundary_w)
        self.token_tri_route_boundary_floor = float(token_tri_route_boundary_floor)
        self.token_tri_route_output_alpha = float(token_tri_route_output_alpha)
        self.token_tri_route_hard_w = float(token_tri_route_hard_w)
        self.token_tri_route_hard_boundary_mix = float(token_tri_route_hard_boundary_mix)
        self.token_tri_route_hard_motion_mix = float(token_tri_route_hard_motion_mix)
        self.token_tri_part_fusion_delta_scale = float(token_tri_part_fusion_delta_scale)
        self.token_tri_part_fusion_spatial_delta_scale = float(token_tri_part_fusion_spatial_delta_scale)
        self.token_tri_part_fusion_spatial_w = float(token_tri_part_fusion_spatial_w)
        self.token_tri_part_fusion_spatial_std_floor = float(token_tri_part_fusion_spatial_std_floor)
        self.token_tri_part_fusion_spatial_motion_mix = float(token_tri_part_fusion_spatial_motion_mix)
        self.token_tri_part_fusion_spatial_boundary_mix = float(token_tri_part_fusion_spatial_boundary_mix)
        self.token_tri_part_fusion_spatial_part_mix = float(token_tri_part_fusion_spatial_part_mix)
        self.time_scale_emb_dim = int(time_scale_emb_dim)
        self.time_scale_temperature = float(time_scale_temperature)
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
        self.part_budget_mode = str(part_budget_mode).lower()
        self.part_budget_sup_w = float(part_budget_sup_w)
        self.part_budget_balance_w = float(part_budget_balance_w)
        self.part_budget_target_mix = float(part_budget_target_mix)
        self.part_budget_target_sharpness = float(part_budget_target_sharpness)
        self.part_budget_router_sharpness = float(part_budget_router_sharpness)
        self.use_part_score_route = bool(use_part_score_route)
        self.part_score_route_hidden_dim = int(part_score_route_hidden_dim)
        self.part_score_route_alpha = float(part_score_route_alpha)
        self.part_score_route_gate_bias = float(part_score_route_gate_bias)
        self.part_score_route_mode = str(part_score_route_mode).lower()
        self.part_score_route_use_route_gate = bool(int(part_score_route_use_route_gate))
        self.part_score_route_use_unknown_mix = bool(int(part_score_route_use_unknown_mix))
        self.part_score_route_signal_mode = str(part_score_route_signal_mode).lower()
        if self.tri_gate_mode not in ("additive", "concat", "scale"):
            raise ValueError("[TRI_GATE] tri_gate_mode must be 'additive', 'concat', or 'scale'.")
        self.num_parts = num_parts
        self.part_moe_global_keep = part_moe_global_keep
        self.pos_input_dim = pos_input_dim
        self.part_moe_active = False
        self.part_experts = None
        self.last_tri_part_stats = None
        self.last_tri_part_reg_loss = None
        self.last_tri_token_stats = None
        self.last_tri_token_loss = None
        self.last_tri_token_part_alpha = None
        self.last_part_budget_stats = None
        self.last_part_budget_loss = None
        self.last_part_budget_ctx = None
        self.last_part_score_route_stats = None
        self.last_time_stats = None

        self.input_ch = pos_input_dim
        self.pose_cond_dim, self.seq_pose_cond_dim, self.seq_xyz_cond_dim = 0, 0, 0

        if self.use_part_budget and self.use_tri:
            raise ValueError("[PART_BUDGET] part_budget is defined on top of part_moe_leg only; do not combine it with tri ablations.")
        if self.use_time and (self.use_part_moe or self.use_tri or self.use_tri_token or self.use_part_budget):
            raise ValueError("[TIME] time is an original-baseline ablation; do not combine it with part_moe/tri/tri_token/part_budget.")
        if self.use_time and not self.use_seq_xyz_cond:
            raise ValueError("[TIME] --use_time requires sequential xyz conditions.")
        if self.use_tri_token and self.use_tri:
            raise ValueError("[TRI_TOKEN] tri_token is a separate ablation; do not combine it with --use_tri.")
        if self.use_tri_token and self.use_part_budget:
            raise ValueError("[TRI_TOKEN] tri_token is a separate ablation; do not combine it with part_budget.")
        if self.use_part_budget and not self.use_part_moe:
            raise ValueError("[PART_BUDGET] --use_part_budget must be used with --use_part_moe.")
        if self.use_part_score_route and not self.use_part_moe:
            raise ValueError("[PART_SCORE_ROUTE] --use_part_score_route must be used with --use_part_moe.")
        if self.use_part_score_route and (self.use_tri or self.use_tri_token or self.use_part_budget):
            raise ValueError("[PART_SCORE_ROUTE] do not combine score routing with tri/tri_token/part_budget ablations.")
        if self.use_part_score_route and (self.part_label_schema != "part_moe_leg" or int(self.num_parts) != 7):
            raise ValueError("[PART_SCORE_ROUTE] score routing is defined on top of part_moe_leg: use --part_label_schema part_moe_leg --num_parts 7.")
        if self.use_part_score_route and self.part_score_route_mode not in ("boost", "delta"):
            raise ValueError("[PART_SCORE_ROUTE] part_score_route_mode must be 'boost' or 'delta'.")
        if self.use_part_score_route and self.part_score_route_signal_mode not in ("full", "unknown_only"):
            raise ValueError("[PART_SCORE_ROUTE] part_score_route_signal_mode must be 'full' or 'unknown_only'.")
        if self.use_tri_token:
            if self.token_tri_fusion_mode in ("route_hard", "part_fusion", "part_fusion_spatial") and not self.use_part_moe:
                raise ValueError("[TRI_TOKEN] route_hard/part_fusion/part_fusion_spatial requires --use_part_moe.")
            if self.use_part_moe and (self.part_label_schema != "part_moe_leg" or int(self.num_parts) != 7):
                raise ValueError(
                    "[TRI_TOKEN] tri_token is defined on top of part_moe_leg: "
                    "use --part_label_schema part_moe_leg --num_parts 7."
                )
            if self.token_tri_fusion_mode not in ("concat", "residual", "route", "route_output", "route_hard", "part_fusion", "part_fusion_spatial"):
                raise ValueError("[TRI_TOKEN] token_tri_fusion_mode must be 'concat', 'residual', 'route', 'route_output', 'route_hard', 'part_fusion', or 'part_fusion_spatial'.")
            print(
                "[TRI_TOKEN] enabled=True; token-conditioned tri-plane memory active. "
                f"dim={self.token_tri_dim} res={self.token_tri_res} extent={self.token_tri_extent} "
                f"heads={self.token_tri_heads} layers={self.token_tri_layers} "
                f"hidden={self.token_tri_hidden_dim} fusion={self.token_tri_fusion_mode} "
                f"fusion_hidden={self.token_tri_fusion_hidden_dim} alpha={self.token_tri_alpha} "
                f"start={self.token_tri_start_iter} warmup={self.token_tri_warmup} "
                f"use_part_moe={self.use_part_moe}"
            )
            if self.token_tri_fusion_mode in ("route", "route_output"):
                print(
                    "[TRI_TOKEN] route fusion active; boundary-aware query routing enabled. "
                    f"start={self.token_tri_start_iter} warmup={self.token_tri_warmup}"
                )
                if self.token_tri_route_boundary_w > 0.0:
                    print(
                        "[TRI_TOKEN] route boundary supervision enabled. "
                        f"weight={self.token_tri_route_boundary_w} floor={self.token_tri_route_boundary_floor}"
                    )
            if self.token_tri_fusion_mode == "route_hard":
                print(
                    "[TRI_TOKEN] hard-route fusion active; route affects part expert blend and hard points only. "
                    f"boundary_mix={self.token_tri_route_hard_boundary_mix} "
                    f"motion_mix={self.token_tri_route_hard_motion_mix} "
                    f"supervision_w={self.token_tri_route_hard_w}"
                )
            if self.token_tri_fusion_mode == "route_output":
                print(
                    "[TRI_TOKEN] route output fusion active; direct output modulation enabled. "
                    f"output_alpha={self.token_tri_route_output_alpha}"
                )
            if self.token_tri_fusion_mode == "part_fusion":
                print(
                    "[TRI_TOKEN] part-fusion active; token-conditioned tri-plane controls "
                    "global/part expert fusion in forward_part_moe. "
                    f"delta_scale={self.token_tri_part_fusion_delta_scale}"
                )
            if self.token_tri_fusion_mode == "part_fusion_spatial":
                print(
                    "[TRI_TOKEN] part-fusion-spatial active; token-conditioned tri-plane controls "
                    "global/part fusion with explicit spatial/part/motion bias. "
                    f"delta_scale={self.token_tri_part_fusion_spatial_delta_scale} "
                    f"loss_w={self.token_tri_part_fusion_spatial_w} "
                    f"std_floor={self.token_tri_part_fusion_spatial_std_floor}"
                )
        if self.use_part_budget:
            if self.part_budget_mode not in ("base", "sup", "route", "full", "v2_base", "v2_sup", "v2_route", "v2_full"):
                raise ValueError("[PART_BUDGET] part_budget_mode must be one of base, sup, route, full, v2_base, v2_sup, v2_route, v2_full.")
            if self.part_label_schema != "part_moe_leg" or int(self.num_parts) != 7:
                raise ValueError(
                    "[PART_BUDGET] part_budget is defined on top of part_moe_leg: "
                    "use --part_label_schema part_moe_leg --num_parts 7."
                )
            if self.part_budget_mode.startswith("v2_"):
                print(
                    "[PART_BUDGET_V2] enabled=True; deep output modulation active. "
                    f"router_sharpness={self.part_budget_router_sharpness}"
                )
            print(
                "[PART_BUDGET] enabled=True; part-aware deformation router active. "
                f"alpha={self.part_budget_alpha} start={self.part_budget_start_iter} "
                f"warmup={self.part_budget_warmup} hidden={self.part_budget_hidden_dim} "
                f"token_dim={self.part_budget_token_dim} mode={self.part_budget_mode} "
                f"router_sharpness={self.part_budget_router_sharpness} "
                f"sup_w={self.part_budget_sup_w} balance_w={self.part_budget_balance_w} "
                f"target_mix={self.part_budget_target_mix} target_sharpness={self.part_budget_target_sharpness}"
            )

        if self.use_pose_cond:
            self.PoseEncoder = PoseEncoder(32, pose_cond_dim, smpl_type)
            self.input_ch += pose_cond_dim
            
        if self.use_seq_pose_cond:
            self.SeqPoseEncoder = SeqPoseEncoder(seq_len, 16, seq_pose_cond_dim, time_step_num, smpl_type)
            self.input_ch += seq_pose_cond_dim

        if self.use_seq_xyz_cond:
            self.SeqXYZEncoder = SeqXYZEncoder(pos_emb_dim=pos_input_dim, hidden_dim1=96, hidden_dim2=256, output_dim=seq_xyz_cond_dim, 
                                        time_step_num=time_step_num, seq_len=seq_len, seq_xyz_knn=seq_xyz_knn,
                                        use_time_scale_fusion=self.use_time,
                                        scale_emb_dim=self.time_scale_emb_dim,
                                        temperature=self.time_scale_temperature)
            self.input_ch += seq_xyz_cond_dim
        if self.use_part_score_route:
            self.PartScoreRouteAdapter = PartScoreRouteAdapter(
                base_feature_dim=self.input_ch,
                num_routed_parts=max(1, self.num_parts - 1),
                hidden_dim=self.part_score_route_hidden_dim,
                gate_bias=self.part_score_route_gate_bias,
            )
            print(
                "[PART_SCORE_ROUTE] enabled=True; score-based unknown routing active. "
                f"hidden={self.part_score_route_hidden_dim} alpha={self.part_score_route_alpha} "
                f"gate_bias={self.part_score_route_gate_bias} mode={self.part_score_route_mode}"
                f" route_gate={self.part_score_route_use_route_gate}"
                f" unknown_mix={self.part_score_route_use_unknown_mix}"
                f" signal_mode={self.part_score_route_signal_mode}"
            )
        if self.use_tri_token:
            cpu_rng_state = torch.get_rng_state()
            self.TokenTriPlaneFeature = TokenConditionedTriPlane(
                pose_dim=pose_cond_dim if self.use_pose_cond else 0,
                seq_pose_dim=seq_pose_cond_dim if self.use_seq_pose_cond else 0,
                num_parts=self.num_parts,
                feature_dim=self.token_tri_dim,
                resolution=self.token_tri_res,
                extent=self.token_tri_extent,
                num_heads=self.token_tri_heads,
                num_layers=self.token_tri_layers,
                hidden_dim=self.token_tri_hidden_dim,
            )
            self.tri_token_base_input_ch = self.input_ch
            if self.token_tri_fusion_mode == "residual":
                self.TokenTriResidualAdapter = TriTokenResidualAdapter(
                    base_feature_dim=self.tri_token_base_input_ch,
                    token_dim=self.token_tri_dim,
                    hidden_dim=self.token_tri_fusion_hidden_dim,
                )
            elif self.token_tri_fusion_mode == "route":
                self.TokenTriRouteAdapter = TriTokenRouteAdapter(
                    base_feature_dim=self.tri_token_base_input_ch,
                    token_dim=self.token_tri_dim,
                    hidden_dim=self.token_tri_fusion_hidden_dim,
                )
            elif self.token_tri_fusion_mode == "part_fusion":
                self.TokenTriPartFusionGate = TriTokenPartFusionGate(
                    base_feature_dim=self.tri_token_base_input_ch,
                    token_dim=self.token_tri_dim,
                    hidden_dim=self.token_tri_fusion_hidden_dim,
                )
            elif self.token_tri_fusion_mode == "part_fusion_spatial":
                self.TokenTriPartFusionSpatialGate = TriTokenPartFusionSpatialGate(
                    base_feature_dim=self.tri_token_base_input_ch,
                    token_dim=self.token_tri_dim,
                    num_parts=self.num_parts,
                    hidden_dim=self.token_tri_fusion_hidden_dim,
                )
            torch.set_rng_state(cpu_rng_state)
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
        if self.use_tri_token:
            cpu_rng_state = torch.get_rng_state()
            if self.token_tri_fusion_mode == "concat":
                self.mlp = _append_mlp_input_dim(self.mlp, self.token_tri_dim)
                self.input_ch += self.token_tri_dim
                print(
                    "[TRI_TOKEN] Token-conditioned tri-plane concat enabled: "
                    f"dim={self.token_tri_dim} res={self.token_tri_res} extent={self.token_tri_extent}"
                )
            elif self.token_tri_fusion_mode == "residual":
                print(
                    "[TRI_TOKEN] Token-conditioned tri-plane residual enabled: "
                    f"base_dim={self.tri_token_base_input_ch} token_dim={self.token_tri_dim} "
                    f"fusion_hidden={self.token_tri_fusion_hidden_dim}"
                )
            elif self.token_tri_fusion_mode == "route":
                print(
                    "[TRI_TOKEN] Token-conditioned tri-plane route enabled: "
                    f"base_dim={self.tri_token_base_input_ch} token_dim={self.token_tri_dim} "
                    f"fusion_hidden={self.token_tri_fusion_hidden_dim}"
                )
            elif self.token_tri_fusion_mode == "route_hard":
                print(
                    "[TRI_TOKEN] Token-conditioned tri-plane hard-route enabled: "
                    f"base_dim={self.tri_token_base_input_ch} token_dim={self.token_tri_dim} "
                    f"fusion_hidden={self.token_tri_fusion_hidden_dim}"
                )
            elif self.token_tri_fusion_mode == "route_output":
                print(
                    "[TRI_TOKEN] Token-conditioned tri-plane route-output enabled: "
                    f"base_dim={self.tri_token_base_input_ch} token_dim={self.token_tri_dim} "
                    f"fusion_hidden={self.token_tri_fusion_hidden_dim} "
                    f"output_alpha={self.token_tri_route_output_alpha}"
                )
            elif self.token_tri_fusion_mode == "part_fusion":
                print(
                    "[TRI_TOKEN] Token-conditioned tri-plane part-fusion enabled: "
                    f"base_dim={self.tri_token_base_input_ch} token_dim={self.token_tri_dim} "
                    f"fusion_hidden={self.token_tri_fusion_hidden_dim} "
                    f"delta_scale={self.token_tri_part_fusion_delta_scale}"
                )
            elif self.token_tri_fusion_mode == "part_fusion_spatial":
                print(
                    "[TRI_TOKEN] Token-conditioned tri-plane part-fusion-spatial enabled: "
                    f"base_dim={self.tri_token_base_input_ch} token_dim={self.token_tri_dim} "
                    f"fusion_hidden={self.token_tri_fusion_hidden_dim} "
                    f"delta_scale={self.token_tri_part_fusion_spatial_delta_scale} "
                    f"loss_w={self.token_tri_part_fusion_spatial_w}"
                )
            torch.set_rng_state(cpu_rng_state)

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
                logit_scale=self.part_budget_router_sharpness,
            )
            self.part_budget_adapter = PartBudgetAdapter(
                feature_dim=self.input_ch,
                token_dim=self.part_budget_token_dim,
                hidden_dim=self.part_budget_hidden_dim,
            )
        if self.use_tri_token and self.token_tri_fusion_mode in ("route", "route_output"):
            self.TokenTriRouteAdapter = TriTokenRouteAdapter(
                base_feature_dim=self.tri_token_base_input_ch,
                token_dim=self.token_tri_dim,
                hidden_dim=self.token_tri_fusion_hidden_dim,
            )
        if self.use_tri_token and self.token_tri_fusion_mode == "part_fusion":
            self.TokenTriPartFusionGate = TriTokenPartFusionGate(
                base_feature_dim=self.tri_token_base_input_ch,
                token_dim=self.token_tri_dim,
                hidden_dim=self.token_tri_fusion_hidden_dim,
            )
        if self.use_tri_token and self.token_tri_fusion_mode == "route_hard":
            self.TokenTriHardRouteAdapter = TriTokenHardRouteAdapter(
                base_feature_dim=self.tri_token_base_input_ch,
                token_dim=self.token_tri_dim,
                hidden_dim=self.token_tri_fusion_hidden_dim,
            )
        if self.use_tri_token and self.token_tri_fusion_mode == "route_output":
            self.TokenTriOutputRouteAdapter = TriTokenOutputRouteAdapter(
                base_feature_dim=self.tri_token_base_input_ch,
                token_dim=self.token_tri_dim,
                output_dim=10,
                hidden_dim=self.token_tri_fusion_hidden_dim,
            )
            self.part_budget_film = PartBudgetFeatureFiLM(
                feature_dim=self.input_ch,
                token_dim=self.part_budget_token_dim,
                hidden_dim=self.part_budget_hidden_dim,
            )
            self.part_budget_output_film = PartBudgetOutputFiLM(
                feature_dim=self.input_ch,
                token_dim=self.part_budget_token_dim,
                output_dim=10,
                hidden_dim=self.part_budget_hidden_dim,
            )
            self.part_budget_output_router = PartBudgetOutputRouter(
                feature_dim=self.input_ch,
                token_dim=self.part_budget_token_dim,
                hidden_dim=self.part_budget_hidden_dim,
                logit_scale=self.part_budget_router_sharpness,
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
            self.last_part_budget_loss = None
            self.last_part_budget_ctx = None
            return features, None
        if part_label is None:
            self.last_part_budget_loss = None
            self.last_part_budget_ctx = None
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
        query_feat = _normalize_budget_query_xyz(query_xyz, features)
        zero_query_feat = torch.zeros_like(query_feat)
        token_idx = part_label.unsqueeze(-1).expand(-1, -1, self.part_budget_token_dim)
        point_token = torch.gather(part_token, 1, token_idx)

        use_query_feat = self.part_budget_mode in ("route", "full") or self.part_budget_mode.startswith("v2_")
        route_query_feat = query_feat if use_query_feat else zero_query_feat
        budget, logits = self.part_budget_router(features, point_token, motion_norm, boundary_score, query_feat=route_query_feat)
        adapter_res = self.part_budget_adapter(features, point_token)
        gamma, beta = self.part_budget_film(features, point_token)
        routed_budget, routed_logits = self.part_budget_output_router(
            features,
            point_token,
            motion_norm,
            boundary_score,
            query_feat=route_query_feat,
        )

        alpha_scale = max(0.0, min(float(part_budget_alpha_scale), 1.0))
        alpha = self.part_budget_alpha * alpha_scale
        mode = self.part_budget_mode
        if mode in ("route", "full") or mode.startswith("v2_"):
            budget = routed_budget
            logits = routed_logits
        if mode == "base":
            rigid_delta = budget[..., 0:1] - 1.0 / 3.0
            boundary_delta = budget[..., 2:3] - 1.0 / 3.0
            capacity_scale = 1.0 + alpha * (rigid_delta - boundary_delta)
            budgeted_features = features * capacity_scale + alpha * budget[..., 1:2] * adapter_res
            sup_loss = None
        elif mode == "sup":
            rigid_delta = budget[..., 0:1] - 1.0 / 3.0
            boundary_delta = budget[..., 2:3] - 1.0 / 3.0
            capacity_scale = 1.0 + alpha * (rigid_delta - boundary_delta)
            budgeted_features = features * capacity_scale + alpha * budget[..., 1:2] * adapter_res
            sup_loss = self._budget_sup_loss(budget, motion_norm, boundary_score)
        elif mode == "v2_base":
            rigid_delta = budget[..., 0:1] - 1.0 / 3.0
            boundary_delta = budget[..., 2:3] - 1.0 / 3.0
            capacity_scale = 1.0 + alpha * (rigid_delta - boundary_delta)
            budgeted_features = features * capacity_scale + alpha * budget[..., 1:2] * adapter_res
            sup_loss = None
        elif mode == "v2_sup":
            rigid_delta = budget[..., 0:1] - 1.0 / 3.0
            boundary_delta = budget[..., 2:3] - 1.0 / 3.0
            capacity_scale = 1.0 + alpha * (rigid_delta - boundary_delta)
            budgeted_features = features * capacity_scale + alpha * budget[..., 1:2] * adapter_res
            sup_loss = self._budget_sup_loss(budget, motion_norm, boundary_score)
        elif mode == "v2_route":
            budgeted_features = self._apply_budget_route(features, budget, adapter_res, gamma, beta, alpha)
            sup_loss = None
        elif mode == "v2_full":
            budgeted_features = self._apply_budget_full(features, budget, adapter_res, gamma, beta, alpha)
            sup_loss = self._budget_sup_loss(budget, motion_norm, boundary_score)
        elif mode == "route":
            budgeted_features = self._apply_budget_route(features, budget, adapter_res, gamma, beta, alpha)
            sup_loss = None
        elif mode == "full":
            budgeted_features = self._apply_budget_full(features, budget, adapter_res, gamma, beta, alpha)
            sup_loss = self._budget_sup_loss(budget, motion_norm, boundary_score)
        else:
            raise ValueError(f"Unknown part_budget_mode: {self.part_budget_mode}")
        stats = {
            "budget_mean": budget.detach().mean(dim=(0, 1)),
            "budget_std": budget.detach().std(dim=(0, 1), unbiased=False),
            "budget_min": budget.detach().amin(dim=(0, 1)),
            "budget_max": budget.detach().amax(dim=(0, 1)),
            "budget_entropy": self._budget_entropy(budget.detach()),
            "budget_kl_uniform": self._budget_kl_uniform(budget.detach()),
            "routed_budget_mean": routed_budget.detach().mean(dim=(0, 1)),
            "routed_budget_std": routed_budget.detach().std(dim=(0, 1), unbiased=False),
            "routed_budget_entropy": self._budget_entropy(routed_budget.detach()),
            "routed_budget_kl_uniform": self._budget_kl_uniform(routed_budget.detach()),
            "budget_target_loss": self._budget_target_loss(budget.detach(), motion_norm.detach(), boundary_score.detach()),
            "per_part_budget_mean": self._collect_per_part_budget_stats(budget.detach(), part_label, reduce="mean"),
            "per_part_budget_std": self._collect_per_part_budget_stats(budget.detach(), part_label, reduce="std"),
            "feature_delta_norm": (budgeted_features - features).detach().norm(dim=-1).mean(),
            "adapter_norm": adapter_res.detach().norm(dim=-1).mean(),
            "film_norm": gamma.detach().norm(dim=-1).mean() + beta.detach().norm(dim=-1).mean(),
            "motion_mean": motion_norm.detach().mean(),
            "boundary_mean": boundary_score.detach().mean(),
            "query_norm": query_feat.detach().norm(dim=-1).mean(),
            "entropy": self._budget_entropy(budget.detach()),
            "logits_mean": logits.detach().mean(),
        }
        self.last_part_budget_loss = sup_loss
        self.last_part_budget_ctx = {
            "part_token": point_token,
            "base_budget": routed_budget.detach() if routed_budget is not None else budget.detach(),
            "budget": budget,
            "routed_budget": routed_budget,
            "logits": logits,
            "routed_logits": routed_logits,
            "motion_norm": motion_norm,
            "boundary_score": boundary_score,
            "query_feat": query_feat,
            "features": budgeted_features,
            "alpha": alpha,
            "mode": mode,
        }
        return budgeted_features, stats

    def apply_part_budget_output(
        self,
        d_xyz,
        d_rotation,
        d_scaling,
        part_token=None,
        features=None,
        budget=None,
        routed_budget=None,
        part_budget_alpha_scale=1.0,
    ):
        if not self.use_part_budget:
            return d_xyz, d_rotation, d_scaling
        if part_token is None or features is None:
            return d_xyz, d_rotation, d_scaling
        alpha_scale = max(0.0, min(float(part_budget_alpha_scale), 1.0))
        alpha = self.part_budget_alpha * alpha_scale
        if alpha <= 0.0:
            return d_xyz, d_rotation, d_scaling
        mod_budget = routed_budget if routed_budget is not None else budget
        if mod_budget is None:
            return d_xyz, d_rotation, d_scaling

        delta = torch.cat([d_xyz, d_rotation, d_scaling], dim=-1)
        out_gamma, out_beta = self.part_budget_output_film(features, part_token)
        rigid_delta = mod_budget[..., 0:1] - 1.0 / 3.0
        boundary_delta = mod_budget[..., 2:3] - 1.0 / 3.0
        adapt_gate = mod_budget[..., 1:2]
        delta = delta * (1.0 + alpha * (rigid_delta - boundary_delta) * torch.tanh(out_gamma))
        delta = delta + alpha * adapt_gate * out_beta
        return delta[..., :3], delta[..., 3:7], delta[..., 7:10]

    def _apply_budget_route(self, features, budget, adapter_res, gamma, beta, alpha):
        rigid_delta = budget[..., 0:1] - 1.0 / 3.0
        boundary_delta = budget[..., 2:3] - 1.0 / 3.0
        capacity_scale = 1.0 + alpha * (rigid_delta - boundary_delta)
        return features * capacity_scale + alpha * budget[..., 1:2] * adapter_res

    def _apply_budget_full(self, features, budget, adapter_res, gamma, beta, alpha):
        routed = self._apply_budget_route(features, budget, adapter_res, gamma, beta, alpha)
        return routed * (1.0 + alpha * torch.tanh(gamma)) + alpha * beta

    def _budget_kl_uniform(self, budget):
        budget = budget.clamp_min(1e-8)
        num_classes = budget.shape[-1]
        return (budget * (budget.log() + torch.log(torch.tensor(float(num_classes), device=budget.device, dtype=budget.dtype)))).sum(dim=-1).mean()

    def _budget_target_distribution(self, motion_strength, boundary_score):
        motion = motion_strength.detach()
        boundary = boundary_score.detach()
        rigid = torch.sigmoid(self.part_budget_target_sharpness * (1.0 - motion - boundary))
        adaptive = torch.sigmoid(self.part_budget_target_sharpness * (motion + 0.5 * boundary))
        boundary_gate = torch.sigmoid(self.part_budget_target_sharpness * (boundary + 0.5 * motion))
        target = torch.cat([rigid, adaptive, boundary_gate], dim=-1)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        mix = max(0.0, min(1.0, self.part_budget_target_mix))
        uniform = torch.full_like(target, 1.0 / target.shape[-1])
        return mix * target + (1.0 - mix) * uniform

    def _budget_target_loss(self, budget, motion_strength, boundary_score):
        target = self._budget_target_distribution(motion_strength, boundary_score).clamp_min(1e-8)
        budget = budget.clamp_min(1e-8)
        return -(target * budget.log()).sum(dim=-1).mean()

    def _budget_sup_loss(self, budget, motion_strength, boundary_score):
        target = self._budget_target_distribution(motion_strength, boundary_score)
        budget = budget.clamp_min(1e-8)
        target = target.clamp_min(1e-8)
        kl = (budget * (budget.log() - target.log())).sum(dim=-1).mean()
        balance = ((budget.mean(dim=(0, 1)) - (1.0 / budget.shape[-1])) ** 2).mean()
        sparse_w = 0.0
        if self.part_budget_mode.startswith("v2_"):
            sparse_w = 0.001
        sparse = self._budget_entropy(budget)
        return self.part_budget_sup_w * kl + self.part_budget_balance_w * balance + sparse_w * sparse

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

    def _prepare_part_moe_alpha(self, part_moe_alpha, features, max_part_weight):
        if isinstance(part_moe_alpha, torch.Tensor):
            alpha = part_moe_alpha.to(device=features.device, dtype=features.dtype)
        else:
            alpha = torch.tensor(float(part_moe_alpha), device=features.device, dtype=features.dtype)

        if alpha.dim() == 0:
            alpha = alpha.view(1, 1, 1).expand(features.shape[0], features.shape[1], 1)
        elif alpha.dim() == 1:
            alpha = alpha.view(1, -1, 1)
            if alpha.shape[0] == 1 and features.shape[0] > 1:
                alpha = alpha.expand(features.shape[0], -1, -1)
        elif alpha.dim() == 2:
            if alpha.shape[0] == features.shape[0] and alpha.shape[1] == features.shape[1]:
                alpha = alpha.unsqueeze(-1)
            elif alpha.shape[0] == 1 and alpha.shape[1] == features.shape[1]:
                alpha = alpha.unsqueeze(-1).expand(features.shape[0], -1, -1)
            else:
                alpha = alpha.unsqueeze(0)
                if alpha.shape[0] == 1 and features.shape[0] > 1:
                    alpha = alpha.expand(features.shape[0], -1, -1)
        elif alpha.dim() != 3:
            raise RuntimeError(f"[PartMoE] Unsupported part_moe_alpha dim: {alpha.dim()}")

        if alpha.shape[0] == 1 and features.shape[0] > 1:
            alpha = alpha.expand(features.shape[0], -1, -1)
        if alpha.shape[1] == 1 and features.shape[1] > 1:
            alpha = alpha.expand(-1, features.shape[1], -1)
        if alpha.shape[:2] != features.shape[:2]:
            raise RuntimeError(
                f"[PartMoE] part_moe_alpha shape {tuple(alpha.shape)} does not match features {tuple(features.shape[:2])}."
            )
        return alpha.clamp(0.0, max_part_weight)

    def apply_tri_token_hard_route(
        self,
        features,
        token_features,
        query_xyz=None,
        motion_strength=None,
        part_conf=None,
        part_enabled=False,
        tri_token_alpha_scale=1.0,
    ):
        if not self.use_tri_token or self.token_tri_fusion_mode != "route_hard":
            self.last_tri_token_loss = None
            self.last_tri_token_part_alpha = None
            return features, None, None

        alpha_scale = max(0.0, min(float(tri_token_alpha_scale), 1.0))
        alpha = self.token_tri_alpha * alpha_scale
        if alpha <= 0.0:
            self.last_tri_token_loss = None
            self.last_tri_token_part_alpha = None
            return features, None, None

        motion_norm = self._normalize_budget_motion(motion_strength, features)
        motion_focus = torch.sigmoid(motion_norm)
        if part_enabled:
            boundary_score = 1.0 - self._normalize_budget_conf(part_conf, features)
        else:
            boundary_score = torch.zeros(*features.shape[:2], 1, device=features.device, dtype=features.dtype)
        hard_focus = (
            self.token_tri_route_hard_boundary_mix * boundary_score
            + self.token_tri_route_hard_motion_mix * motion_focus
        ).clamp(0.0, 1.0)
        query_feat = _normalize_budget_query_xyz(query_xyz, features)
        routed_delta, route_gate, expert_gate = self.TokenTriHardRouteAdapter(
            features,
            token_features * (1.0 + hard_focus),
            motion_norm,
            boundary_score,
            hard_focus,
            query_feat=query_feat,
        )
        routed_features = features + alpha * routed_delta

        part_alpha = (hard_focus * expert_gate).clamp(0.0, 1.0)
        route_weight = (0.5 + hard_focus.detach()).clamp(0.5, 1.5)
        route_gate_loss = ((route_gate - hard_focus.detach()) ** 2) * route_weight
        expert_gate_loss = ((expert_gate - hard_focus.detach()) ** 2) * route_weight
        route_hard_loss = None
        if self.token_tri_route_hard_w > 0.0:
            route_hard_loss = self.token_tri_route_hard_w * (route_gate_loss.mean() + 0.5 * expert_gate_loss.mean())

        stats = {
            "route_hard_gate_mean": route_gate.detach().mean(),
            "route_hard_gate_std": route_gate.detach().std(unbiased=False),
            "route_hard_expert_gate_mean": expert_gate.detach().mean(),
            "route_hard_expert_gate_std": expert_gate.detach().std(unbiased=False),
            "route_hard_delta_norm": routed_delta.detach().norm(dim=-1).mean(),
            "route_hard_part_alpha_mean": part_alpha.detach().mean(),
            "route_hard_motion_mean": motion_norm.detach().mean(),
            "route_hard_motion_focus_mean": motion_focus.detach().mean(),
            "route_hard_boundary_mean": boundary_score.detach().mean(),
            "route_hard_focus_mean": hard_focus.detach().mean(),
            "route_hard_query_norm": query_feat.detach().norm(dim=-1).mean(),
        }
        if route_hard_loss is not None:
            stats["route_hard_loss"] = route_hard_loss.detach()
        self.last_tri_token_loss = route_hard_loss
        self.last_tri_token_part_alpha = part_alpha
        return routed_features, stats, part_alpha

    def forward_part_moe(
        self,
        features,
        part_label,
        part_moe_alpha=0.0,
        part_moe_global_keep=None,
        token_features=None,
        query_xyz=None,
        motion_strength=None,
        part_conf=None,
        tri_token_alpha_scale=1.0,
    ):
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
        part_weight = self._prepare_part_moe_alpha(part_moe_alpha, features, max_part_weight)
        fusion_stats = None
        if (
            self.use_tri_token
            and self.token_tri_fusion_mode == "part_fusion"
            and token_features is not None
            and hasattr(self, "TokenTriPartFusionGate")
        ):
            alpha_scale = max(0.0, min(float(tri_token_alpha_scale), 1.0))
            if alpha_scale > 0.0 and max_part_weight > 0.0:
                motion_norm = self._normalize_budget_motion(motion_strength, features)
                boundary_score = 1.0 - self._normalize_budget_conf(part_conf, features)
                query_feat = _normalize_budget_query_xyz(query_xyz, features)
                boundary_focus = 1.0 + boundary_score
                delta_unit = self.TokenTriPartFusionGate(
                    features,
                    token_features * boundary_focus,
                    motion_norm,
                    boundary_score,
                    query_feat=query_feat,
                )
                delta_range = max_part_weight * self.token_tri_part_fusion_delta_scale
                delta = alpha_scale * delta_range * delta_unit
                base_part_weight = part_weight
                part_weight = (base_part_weight + delta).clamp(0.0, max_part_weight)
                fusion_stats = {
                    "fusion_part_weight_mean": part_weight.detach().mean(),
                    "fusion_part_weight_std": part_weight.detach().std(unbiased=False),
                    "fusion_base_part_weight_mean": base_part_weight.detach().mean(),
                    "fusion_delta_mean": delta.detach().mean(),
                    "fusion_delta_abs_mean": delta.detach().abs().mean(),
                    "fusion_delta_unit_std": delta_unit.detach().std(unbiased=False),
                    "fusion_global_weight_mean": (1.0 - part_weight).detach().mean(),
                    "fusion_boundary_mean": boundary_score.detach().mean(),
                    "fusion_motion_mean": motion_norm.detach().mean(),
                    "fusion_query_norm": query_feat.detach().norm(dim=-1).mean(),
                }
        elif (
            self.use_tri_token
            and self.token_tri_fusion_mode == "part_fusion_spatial"
            and token_features is not None
            and hasattr(self, "TokenTriPartFusionSpatialGate")
        ):
            alpha_scale = max(0.0, min(float(tri_token_alpha_scale), 1.0))
            if alpha_scale > 0.0 and max_part_weight > 0.0:
                motion_norm = self._normalize_budget_motion(motion_strength, features)
                motion_focus = torch.sigmoid(motion_norm)
                boundary_score = 1.0 - self._normalize_budget_conf(part_conf, features)
                query_feat = _normalize_budget_query_xyz(query_xyz, features)
                boundary_focus = 1.0 + self.token_tri_part_fusion_spatial_boundary_mix * boundary_score
                motion_focus_mix = 1.0 + self.token_tri_part_fusion_spatial_motion_mix * motion_focus
                routed_delta_unit, route_gate, route_logit, spatial_logit, part_logit = self.TokenTriPartFusionSpatialGate(
                    features,
                    token_features * boundary_focus * motion_focus_mix,
                    part_label,
                    motion_norm,
                    boundary_score,
                    query_feat=query_feat,
                )
                spatial_gate = torch.sigmoid(spatial_logit)
                part_gate = torch.sigmoid(part_logit)
                delta_range = max_part_weight * self.token_tri_part_fusion_spatial_delta_scale
                hard_focus = (
                    self.token_tri_part_fusion_spatial_boundary_mix * boundary_score
                    + self.token_tri_part_fusion_spatial_motion_mix * motion_focus
                ).clamp(0.0, 1.0)
                delta_boost = (
                    (1.0 + hard_focus)
                    * (1.0 + self.token_tri_part_fusion_spatial_part_mix * part_gate)
                    * (1.0 + 0.5 * spatial_gate)
                )
                delta = alpha_scale * delta_range * delta_boost * routed_delta_unit
                base_part_weight = part_weight
                part_weight = (base_part_weight + delta).clamp(0.0, max_part_weight)
                gate_std = route_gate.std(unbiased=False)
                spatial_gate_std = spatial_gate.std(unbiased=False)
                part_gate_std = part_gate.std(unbiased=False)
                spatial_loss = None
                if self.token_tri_part_fusion_spatial_w > 0.0:
                    align_loss = ((route_gate - hard_focus.detach()) ** 2) * (0.5 + hard_focus.detach())
                    std_loss = F.relu(self.token_tri_part_fusion_spatial_std_floor - gate_std).pow(2)
                    spatial_loss = self.token_tri_part_fusion_spatial_w * (align_loss.mean() + 0.5 * std_loss)
                fusion_stats = {
                    "spatial_part_weight_mean": part_weight.detach().mean(),
                    "spatial_part_weight_std": part_weight.detach().std(unbiased=False),
                    "spatial_base_part_weight_mean": base_part_weight.detach().mean(),
                    "spatial_delta_mean": delta.detach().mean(),
                    "spatial_delta_abs_mean": delta.detach().abs().mean(),
                    "spatial_delta_unit_std": routed_delta_unit.detach().std(unbiased=False),
                    "spatial_route_gate_mean": route_gate.detach().mean(),
                    "spatial_route_gate_std": gate_std.detach(),
                    "spatial_spatial_gate_mean": spatial_gate.detach().mean(),
                    "spatial_spatial_gate_std": spatial_gate_std.detach(),
                    "spatial_part_gate_mean": part_gate.detach().mean(),
                    "spatial_part_gate_std": part_gate_std.detach(),
                    "spatial_global_weight_mean": (1.0 - part_weight).detach().mean(),
                    "spatial_boundary_mean": boundary_score.detach().mean(),
                    "spatial_motion_mean": motion_norm.detach().mean(),
                    "spatial_focus_mean": hard_focus.detach().mean(),
                    "spatial_query_norm": query_feat.detach().norm(dim=-1).mean(),
                }
                if spatial_loss is not None:
                    fusion_stats["spatial_loss"] = spatial_loss.detach()
                self.last_tri_token_loss = spatial_loss
                self.last_tri_token_part_alpha = part_weight
        score_route_stats = None
        score_route_gate = None
        score_route_focus = None
        score_route_unknown_score = None
        score_route_expert_logits = None
        if self.use_part_score_route and max_part_weight > 0.0:
            motion_norm = self._normalize_budget_motion(motion_strength, features)
            motion_focus = torch.sigmoid(motion_norm)
            boundary_score = 1.0 - self._normalize_budget_conf(part_conf, features)
            query_feat = _normalize_budget_query_xyz(query_xyz, features)
            score_focus = (0.7 * boundary_score + 0.3 * motion_focus).clamp(0.0, 1.0)
            unknown_score = (part_label == 0).to(device=features.device, dtype=features.dtype).unsqueeze(-1)
            if self.part_score_route_signal_mode == "unknown_only":
                motion_norm = torch.zeros_like(motion_norm)
                motion_focus = torch.zeros_like(motion_focus)
                boundary_score = torch.zeros_like(boundary_score)
                query_feat = torch.zeros_like(query_feat)
                score_focus = unknown_score.clone()
            score_focus = (score_focus * (1.0 + 0.5 * unknown_score)).clamp(0.0, 1.0)
            route_logit, expert_logits = self.PartScoreRouteAdapter(
                features,
                score_focus,
                motion_norm,
                boundary_score,
                unknown_score,
                query_feat=query_feat,
            )
            route_gate = torch.sigmoid(route_logit)
            base_part_weight = part_weight
            if self.part_score_route_use_route_gate:
                if self.part_score_route_mode == "delta":
                    route_delta = self.part_score_route_alpha * score_focus * (2.0 * route_gate - 1.0)
                    part_weight = (base_part_weight + route_delta * max_part_weight).clamp(0.0, max_part_weight)
                else:
                    route_scale = self.part_score_route_alpha * route_gate * score_focus
                    part_weight = (base_part_weight + (max_part_weight - base_part_weight) * route_scale).clamp(0.0, max_part_weight)
            score_route_stats = {
                "score_route_part_weight_mean": part_weight.detach().mean(),
                "score_route_part_weight_std": part_weight.detach().std(unbiased=False),
                "score_route_base_part_weight_mean": base_part_weight.detach().mean(),
                "score_route_gate_mean": route_gate.detach().mean(),
                "score_route_gate_std": route_gate.detach().std(unbiased=False),
                "score_route_focus_mean": score_focus.detach().mean(),
                "score_route_motion_mean": motion_norm.detach().mean(),
                "score_route_boundary_mean": boundary_score.detach().mean(),
                "score_route_unknown_mean": unknown_score.detach().mean(),
                "score_route_query_norm": query_feat.detach().norm(dim=-1).mean(),
                "score_route_mode_delta": torch.tensor(1.0 if self.part_score_route_mode == "delta" else 0.0, device=features.device, dtype=features.dtype),
                "score_route_use_route_gate": torch.tensor(1.0 if self.part_score_route_use_route_gate else 0.0, device=features.device, dtype=features.dtype),
                "score_route_use_unknown_mix": torch.tensor(1.0 if self.part_score_route_use_unknown_mix else 0.0, device=features.device, dtype=features.dtype),
                "score_route_signal_mode_unknown_only": torch.tensor(1.0 if self.part_score_route_signal_mode == "unknown_only" else 0.0, device=features.device, dtype=features.dtype),
            }
            score_route_gate = route_gate
            score_route_focus = score_focus
            score_route_unknown_score = unknown_score
            score_route_expert_logits = expert_logits
        if fusion_stats is not None:
            tri_token_stats = self.last_tri_token_stats or {}
            tri_token_stats.update(fusion_stats)
            self.last_tri_token_stats = tri_token_stats
        if score_route_stats is not None:
            tri_token_stats = self.last_tri_token_stats or {}
            tri_token_stats.update(score_route_stats)
            self.last_tri_token_stats = tri_token_stats
            self.last_part_score_route_stats = score_route_stats
        global_weight = 1.0 - part_weight

        global_xyz, global_rotation, global_scaling = self.part_experts[0](features)

        if float(part_weight.detach().max().item()) <= 0.0:
            return global_xyz, global_rotation, global_scaling

        feature_shape = features.shape
        flat_features = features.reshape(-1, feature_shape[-1])
        flat_labels = part_label.reshape(-1)
        flat_part_weight = part_weight.reshape(-1, 1)
        flat_global_weight = global_weight.reshape(-1, 1)

        d_xyz = global_xyz.reshape(-1, global_xyz.shape[-1])
        d_rotation = global_rotation.reshape(-1, global_rotation.shape[-1])
        d_scaling = global_scaling.reshape(-1, global_scaling.shape[-1])
        flat_global_xyz = d_xyz
        flat_global_rotation = d_rotation
        flat_global_scaling = d_scaling

        if self.use_part_score_route and self.part_score_route_use_unknown_mix and score_route_expert_logits is not None:
            unknown_idx = torch.nonzero(flat_labels == 0, as_tuple=False).flatten()
            if unknown_idx.numel() > 0:
                unknown_features = flat_features.index_select(0, unknown_idx)
                unknown_route_logits = score_route_expert_logits.reshape(-1, score_route_expert_logits.shape[-1]).index_select(0, unknown_idx)
                unknown_route_weights = torch.softmax(unknown_route_logits, dim=-1)
                unknown_part_weight = flat_part_weight.index_select(0, unknown_idx)
                unknown_global_weight = flat_global_weight.index_select(0, unknown_idx)

                routed_parts = []
                for pid in range(1, self.num_parts):
                    part_xyz, part_rotation, part_scaling = self.part_experts[pid](unknown_features)
                    routed_parts.append(torch.cat([part_xyz, part_rotation, part_scaling], dim=-1))
                routed_stack = torch.stack(routed_parts, dim=1)
                routed_mix = (unknown_route_weights.unsqueeze(-1) * routed_stack).sum(dim=1)
                global_mix = torch.cat(
                    [
                        flat_global_xyz.index_select(0, unknown_idx),
                        flat_global_rotation.index_select(0, unknown_idx),
                        flat_global_scaling.index_select(0, unknown_idx),
                    ],
                    dim=-1,
                )
                unknown_mix = unknown_global_weight * global_mix + unknown_part_weight * routed_mix
                d_xyz = torch.index_copy(d_xyz, 0, unknown_idx, unknown_mix[..., :3])
                d_rotation = torch.index_copy(d_rotation, 0, unknown_idx, unknown_mix[..., 3:7])
                d_scaling = torch.index_copy(d_scaling, 0, unknown_idx, unknown_mix[..., 7:10])
                if score_route_stats is not None:
                    score_route_stats["score_route_unknown_weight_mean"] = unknown_part_weight.detach().mean()
                    score_route_stats["score_route_unknown_weight_std"] = unknown_part_weight.detach().std(unbiased=False)
                    score_route_stats["score_route_unknown_route_entropy"] = (
                        -(unknown_route_weights * unknown_route_weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
                    )
                    score_route_stats["score_route_unknown_count"] = torch.tensor(
                        float(unknown_idx.numel()),
                        device=features.device,
                        dtype=features.dtype,
                    )

        for pid in range(1, self.num_parts):
            idx = torch.nonzero(flat_labels == pid, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            part_xyz, part_rotation, part_scaling = self.part_experts[pid](
                flat_features.index_select(0, idx),
            )
            part_weight_sel = flat_part_weight.index_select(0, idx)
            global_weight_sel = flat_global_weight.index_select(0, idx)
            d_xyz = torch.index_copy(
                d_xyz,
                0,
                idx,
                global_weight_sel * flat_global_xyz.index_select(0, idx) + part_weight_sel * part_xyz,
            )
            d_rotation = torch.index_copy(
                d_rotation,
                0,
                idx,
                global_weight_sel * flat_global_rotation.index_select(0, idx) + part_weight_sel * part_rotation,
            )
            d_scaling = torch.index_copy(
                d_scaling,
                0,
                idx,
                global_weight_sel * flat_global_scaling.index_select(0, idx) + part_weight_sel * part_scaling,
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

    def sample_tri_token_features(self, query_xyz, x_emb, pose_feats=None, seq_pose_feats=None,
                                  seq_xyz_conds=None, part_label=None, part_conf=None,
                                  tri_token_alpha_scale=1.0):
        token_features, stats = self.TokenTriPlaneFeature(
            query_xyz,
            batch_size=x_emb.shape[0],
            num_points=x_emb.shape[1],
            dtype=x_emb.dtype,
            pose_feats=pose_feats,
            seq_pose_feats=seq_pose_feats,
            seq_xyz_conds=seq_xyz_conds,
            part_label=part_label,
            part_conf=part_conf,
            alpha_scale=self.token_tri_alpha * tri_token_alpha_scale,
        )
        self.last_tri_token_stats = stats
        return token_features

    def apply_tri_token_route(self, features, token_features, query_xyz=None, motion_strength=None,
                              part_conf=None, part_enabled=False, tri_token_alpha_scale=1.0):
        if not self.use_tri_token or self.token_tri_fusion_mode not in ("route", "route_output"):
            self.last_tri_token_loss = None
            return features, None

        alpha_scale = max(0.0, min(float(tri_token_alpha_scale), 1.0))
        alpha = self.token_tri_alpha * alpha_scale
        if alpha <= 0.0:
            self.last_tri_token_loss = None
            return features, None

        motion_norm = self._normalize_budget_motion(motion_strength, features)
        if part_enabled:
            boundary_score = 1.0 - self._normalize_budget_conf(part_conf, features)
        else:
            boundary_score = torch.zeros(*features.shape[:2], 1, device=features.device, dtype=features.dtype)
        query_feat = _normalize_budget_query_xyz(query_xyz, features)
        boundary_focus = 1.0 + boundary_score
        routed_delta, route_gate = self.TokenTriRouteAdapter(
            features,
            token_features * boundary_focus,
            motion_norm,
            boundary_score,
            query_feat=query_feat,
        )
        routed_features = features + alpha * routed_delta
        route_boundary_loss = None
        route_boundary_target = None
        if self.token_tri_route_boundary_w > 0.0 and part_enabled:
            motion_focus = torch.sigmoid(motion_norm)
            route_boundary_target = (
                self.token_tri_route_boundary_floor
                + (1.0 - self.token_tri_route_boundary_floor)
                * (0.7 * boundary_score + 0.3 * motion_focus)
            ).clamp(0.0, 1.0)
            route_boundary_loss = F.mse_loss(route_gate, route_boundary_target)
            route_boundary_loss = self.token_tri_route_boundary_w * route_boundary_loss
        stats = {
            "route_gate_mean": route_gate.detach().mean(),
            "route_gate_std": route_gate.detach().std(unbiased=False),
            "route_delta_norm": routed_delta.detach().norm(dim=-1).mean(),
            "routed_token_norm": (token_features * boundary_focus).detach().norm(dim=-1).mean(),
            "motion_mean": motion_norm.detach().mean(),
            "boundary_mean": boundary_score.detach().mean(),
            "boundary_focus_mean": boundary_focus.detach().mean(),
            "query_norm": query_feat.detach().norm(dim=-1).mean(),
        }
        if route_boundary_target is not None:
            stats["route_boundary_target_mean"] = route_boundary_target.detach().mean()
        if route_boundary_loss is not None:
            stats["route_boundary_loss"] = route_boundary_loss.detach()
        self.last_tri_token_loss = route_boundary_loss
        return routed_features, stats

    def apply_tri_token_output_route(
        self,
        d_xyz,
        d_rotation,
        d_scaling,
        features,
        token_features,
        query_xyz=None,
        motion_strength=None,
        part_conf=None,
        part_enabled=False,
        tri_token_alpha_scale=1.0,
    ):
        if not self.use_tri_token or self.token_tri_fusion_mode != "route_output":
            return d_xyz, d_rotation, d_scaling, None

        alpha_scale = max(0.0, min(float(tri_token_alpha_scale), 1.0))
        alpha = self.token_tri_route_output_alpha * alpha_scale
        if alpha <= 0.0:
            return d_xyz, d_rotation, d_scaling, None

        motion_norm = self._normalize_budget_motion(motion_strength, features)
        if part_enabled:
            boundary_score = 1.0 - self._normalize_budget_conf(part_conf, features)
        else:
            boundary_score = torch.zeros(*features.shape[:2], 1, device=features.device, dtype=features.dtype)
        query_feat = _normalize_budget_query_xyz(query_xyz, features)
        boundary_focus = 1.0 + boundary_score
        output_features = torch.cat([d_xyz, d_rotation, d_scaling], dim=-1)
        routed_delta, route_gate = self.TokenTriOutputRouteAdapter(
            features,
            token_features * boundary_focus,
            output_features,
            motion_norm,
            boundary_score,
            query_feat=query_feat,
        )
        output_features = output_features + alpha * routed_delta
        stats = {
            "route_output_gate_mean": route_gate.detach().mean(),
            "route_output_gate_std": route_gate.detach().std(unbiased=False),
            "route_output_delta_norm": routed_delta.detach().norm(dim=-1).mean(),
            "route_output_token_norm": (token_features * boundary_focus).detach().norm(dim=-1).mean(),
            "route_output_mean": output_features.detach().mean(),
            "route_output_std": output_features.detach().std(unbiased=False),
            "route_output_boundary_mean": boundary_score.detach().mean(),
            "route_output_focus_mean": boundary_focus.detach().mean(),
        }
        return output_features[..., :3], output_features[..., 3:7], output_features[..., 7:10], stats

    def forward(self, x_emb, pose_conds=None, seq_pose_conds=None, seq_xyz_conds=None,
                part_label=None, part_enabled=False,
                query_xyz=None, part_moe_alpha=0.0, part_moe_global_keep=None,
                tri_gate_alpha_scale=1.0, part_conf=None, part_budget_alpha_scale=1.0,
                tri_token_alpha_scale=1.0):
        self.last_part_budget_ctx = None
        self.last_tri_token_stats = None
        self.last_tri_token_loss = None
        self.last_tri_token_part_alpha = None
        self.last_time_stats = None
        self.last_part_score_route_stats = None
        feats = []
        feats.append(x_emb)

        # single frame pose condition
        pose_frame_feats = None
        if self.use_pose_cond: 
            pose_frame_feats = self.PoseEncoder(pose_conds)
            pose_feats = pose_frame_feats.unsqueeze(1).expand(-1, x_emb.shape[1], -1)
            feats.append(pose_feats)
        
        # sequential pose condition
        seq_pose_frame_feats = None
        if self.use_seq_pose_cond:
            seq_pose_frame_feats = self.SeqPoseEncoder(seq_pose_conds)
            seq_pose_feats = seq_pose_frame_feats.unsqueeze(1).expand(-1, x_emb.shape[1], -1)
            feats.append(seq_pose_feats)
        
        # sequential point-wise delta xyz condition
        seq_xyz_feats = None
        if self.use_seq_xyz_cond: 
            seq_xyz_feats = self.SeqXYZEncoder(seq_xyz_conds, x_emb)
            feats.append(seq_xyz_feats)
            if self.use_time:
                alpha = getattr(self.SeqXYZEncoder, "last_attention", None)
                if alpha is not None:
                    alpha_detached = alpha.detach()
                    entropy = -(alpha_detached * torch.log(alpha_detached.clamp_min(1e-8))).sum(dim=-1).mean()
                    self.last_time_stats = {
                        "alpha_mean": alpha_detached.mean(),
                        "alpha_std": alpha_detached.std(unbiased=False),
                        "alpha_min": alpha_detached.min(),
                        "alpha_max": alpha_detached.max(),
                        "alpha_entropy": entropy,
                        "scale_mean": alpha_detached.mean(dim=(0, 1, 2)),
                    }

        features = torch.cat(feats, dim=-1)
        self.last_part_budget_stats = None
        self.last_part_budget_loss = None
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

        if self.use_tri_token:
            active_part_label = part_label if (self.use_part_moe and part_enabled) else None
            active_part_conf = part_conf if (self.use_part_moe and part_enabled) else None
            route_motion_strength = None
            if seq_xyz_conds is not None:
                route_motion_strength = seq_xyz_conds.detach().norm(dim=-1)
                reduce_dims = tuple(range(2, route_motion_strength.dim()))
                route_motion_strength = route_motion_strength.mean(dim=reduce_dims).unsqueeze(-1)
            token_features = self.sample_tri_token_features(
                query_xyz,
                x_emb,
                pose_feats=pose_frame_feats,
                seq_pose_feats=seq_pose_frame_feats,
                seq_xyz_conds=seq_xyz_conds,
                part_label=active_part_label,
                part_conf=active_part_conf,
                tri_token_alpha_scale=tri_token_alpha_scale,
            )
            tri_token_stats = self.last_tri_token_stats or {}
            if self.token_tri_fusion_mode == "concat":
                features = torch.cat([features, token_features], dim=-1)
            elif self.token_tri_fusion_mode == "residual":
                features = features + self.TokenTriResidualAdapter(features, token_features)
            elif self.token_tri_fusion_mode == "route_hard":
                features, route_stats, part_alpha_scale = self.apply_tri_token_hard_route(
                    features,
                    token_features,
                    query_xyz=query_xyz,
                    motion_strength=route_motion_strength,
                    part_conf=active_part_conf,
                    part_enabled=self.use_part_moe and part_enabled and part_label is not None,
                    tri_token_alpha_scale=tri_token_alpha_scale,
                )
                self.last_tri_token_part_alpha = part_alpha_scale
                if route_stats is not None:
                    tri_token_stats.update(route_stats)
                    self.last_tri_token_stats = tri_token_stats
            elif self.token_tri_fusion_mode == "part_fusion":
                self.last_tri_token_stats = tri_token_stats
            elif self.token_tri_fusion_mode == "part_fusion_spatial":
                self.last_tri_token_stats = tri_token_stats
            else:
                features, route_stats = self.apply_tri_token_route(
                    features,
                    token_features,
                    query_xyz=query_xyz,
                    motion_strength=route_motion_strength,
                    part_conf=active_part_conf,
                    part_enabled=self.use_part_moe and part_enabled and part_label is not None,
                    tri_token_alpha_scale=tri_token_alpha_scale,
                )
                if route_stats is not None:
                    tri_token_stats.update(route_stats)
                self.last_tri_token_stats = tri_token_stats

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

        effective_part_moe_alpha = part_moe_alpha
        if self.use_tri_token and self.token_tri_fusion_mode == "route_hard" and self.last_tri_token_part_alpha is not None:
            effective_part_moe_alpha = part_moe_alpha * self.last_tri_token_part_alpha

        if (
            self.use_part_moe
            and self.part_moe_active
            and part_enabled
            and part_label is not None
        ):
            if self.use_tri:
                d_xyz, d_rotation, d_scaling = self.forward_tri(
                    features,
                    part_label,
                    part_moe_alpha=effective_part_moe_alpha,
                    part_moe_global_keep=part_moe_global_keep,
                )
            elif self.use_tri_token and self.token_tri_fusion_mode == "part_fusion":
                d_xyz, d_rotation, d_scaling = self.forward_part_moe(
                    features,
                    part_label,
                    part_moe_alpha=effective_part_moe_alpha,
                    part_moe_global_keep=part_moe_global_keep,
                    token_features=token_features,
                    query_xyz=query_xyz,
                    motion_strength=route_motion_strength,
                    part_conf=active_part_conf,
                    tri_token_alpha_scale=tri_token_alpha_scale,
                )
            elif self.use_tri_token and self.token_tri_fusion_mode == "part_fusion_spatial":
                d_xyz, d_rotation, d_scaling = self.forward_part_moe(
                    features,
                    part_label,
                    part_moe_alpha=effective_part_moe_alpha,
                    part_moe_global_keep=part_moe_global_keep,
                    token_features=token_features,
                    query_xyz=query_xyz,
                    motion_strength=route_motion_strength,
                    part_conf=active_part_conf,
                    tri_token_alpha_scale=tri_token_alpha_scale,
                )
            else:
                d_xyz, d_rotation, d_scaling = self.forward_part_moe(
                    features,
                    part_label,
                    part_moe_alpha=effective_part_moe_alpha,
                    part_moe_global_keep=part_moe_global_keep,
                )

            if self.use_part_budget and self.part_budget_mode.startswith("v2_"):
                part_budget_ctx = self.last_part_budget_ctx or {}
                d_xyz, d_rotation, d_scaling = self.apply_part_budget_output(
                    d_xyz,
                    d_rotation,
                    d_scaling,
                    part_token=part_budget_ctx.get("part_token"),
                    features=part_budget_ctx.get("features"),
                    budget=part_budget_ctx.get("budget"),
                    routed_budget=part_budget_ctx.get("routed_budget"),
                    part_budget_alpha_scale=part_budget_alpha_scale,
                )
                self.last_part_budget_ctx = None
            if self.use_tri_token and self.token_tri_fusion_mode == "route_output":
                d_xyz, d_rotation, d_scaling, output_route_stats = self.apply_tri_token_output_route(
                    d_xyz,
                    d_rotation,
                    d_scaling,
                    features,
                    token_features,
                    query_xyz=query_xyz,
                    motion_strength=route_motion_strength,
                    part_conf=active_part_conf,
                    part_enabled=self.use_part_moe and part_enabled and part_label is not None,
                    tri_token_alpha_scale=tri_token_alpha_scale,
                )
                if output_route_stats is not None:
                    tri_token_stats.update(output_route_stats)
                    self.last_tri_token_stats = tri_token_stats
            return d_xyz, d_rotation, d_scaling

        h = self.mlp(features)
        d_xyz, d_scaling, d_rotation = self.gaussian_warp(h), self.gaussian_scaling(h), self.gaussian_rotation(h)
        if self.use_tri_token and self.token_tri_fusion_mode == "route_output":
            d_xyz, d_rotation, d_scaling, output_route_stats = self.apply_tri_token_output_route(
                d_xyz,
                d_rotation,
                d_scaling,
                features,
                token_features,
                query_xyz=query_xyz,
                motion_strength=route_motion_strength,
                part_conf=active_part_conf,
                part_enabled=self.use_part_moe and part_enabled and part_label is not None,
                tri_token_alpha_scale=tri_token_alpha_scale,
            )
            if output_route_stats is not None:
                tri_token_stats.update(output_route_stats)
                self.last_tri_token_stats = tri_token_stats
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
                 time_step_num=1, seq_len=6, seq_xyz_knn=5,
                 use_time_scale_fusion=False, scale_emb_dim=16, temperature=1.5):
        super(SeqXYZEncoder, self).__init__()

        self.seq_len = seq_len
        self.time_step_num = time_step_num
        self.seq_xyz_knn = seq_xyz_knn
        self.use_time_scale_fusion = bool(use_time_scale_fusion)
        self.temperature = float(temperature)
        self.last_attention = None

        if not self.use_time_scale_fusion:
            self.vel_encoder = nn.Sequential(nn.Linear(vel_dim * seq_xyz_knn * time_step_num, vel_emb_dim), nn.ReLU())
            self.pos_emb_proj = nn.Sequential(nn.Linear(pos_emb_dim, pos_emb_proj_dim), nn.ReLU())
            self.mlp1 = nn.Sequential(nn.Linear(vel_emb_dim + pos_emb_proj_dim, hidden_dim1), nn.ReLU())
            self.mlp2 = nn.Sequential(nn.Linear(hidden_dim1 * seq_len, hidden_dim2), nn.ReLU(),
                                      nn.Linear(hidden_dim2, output_dim), nn.ReLU())
        else:
            self.vel_encoder = nn.Sequential(nn.Linear(vel_dim * seq_xyz_knn, vel_emb_dim), nn.ReLU())
            self.pos_emb_proj = nn.Sequential(nn.Linear(pos_emb_dim, pos_emb_proj_dim), nn.ReLU())
            self.scale_embedding = nn.Parameter(torch.zeros(1, 1, 1, time_step_num, scale_emb_dim))
            self.scale_feature = nn.Sequential(
                nn.Linear(vel_emb_dim + pos_emb_proj_dim + scale_emb_dim, hidden_dim1),
                nn.ReLU()
            )
            self.scale_score = nn.Sequential(
                nn.Linear(hidden_dim1, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            nn.init.zeros_(self.scale_score[-1].weight)
            nn.init.zeros_(self.scale_score[-1].bias)
            self.temporal_mlp = nn.Sequential(
                nn.Linear(hidden_dim1 * seq_len, hidden_dim2),
                nn.ReLU(),
                nn.Linear(hidden_dim2, output_dim),
                nn.ReLU()
            )

    def forward(self, x, x_emb):
        # x -> [B, N, L, K, S, C] or [B, N, L, S, K, C]
        self.last_attention = None
        if not self.use_time_scale_fusion:
            B, N, T = x.shape[0], x.shape[1], x.shape[2]
            pos_feat = self.pos_emb_proj(x_emb)
            pos_feat = pos_feat.unsqueeze(2).expand(-1, -1, T, -1)
            vel_emb = self.vel_encoder(x.reshape(B, N, T, -1))

            h = torch.concat([vel_emb, pos_feat], dim=-1)
            h = self.mlp1(h)
            h = self.mlp2(h.reshape(B, N, -1))
            return h

        if x.dim() != 6:
            raise ValueError(f"[TIME] SeqXYZEncoder expects 6D input, got shape={tuple(x.shape)}")
        B, N, L, D3, D4, C = x.shape
        if L != self.seq_len:
            raise ValueError(f"[TIME] SeqXYZEncoder expects seq_len={self.seq_len}, got L={L}")
        if C != 3:
            raise ValueError(f"[TIME] SeqXYZEncoder expects last dim=3, got shape={tuple(x.shape)}")
        if D3 == self.seq_xyz_knn and D4 == self.time_step_num:
            x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
            D3, D4 = D4, D3
        elif not (D3 == self.time_step_num and D4 == self.seq_xyz_knn):
            raise ValueError(
                f"[TIME] Unsupported seq_xyz layout {tuple(x.shape)}; "
                f"expected dims to match knn={self.seq_xyz_knn} and time_step_num={self.time_step_num}."
            )

        S = D3
        K = D4
        vel_emb = self.vel_encoder(x.reshape(B, N, L, S, K * C))
        pos_feat = self.pos_emb_proj(x_emb).unsqueeze(2).unsqueeze(3).expand(-1, -1, L, S, -1)
        scale_feat = self.scale_embedding.expand(B, N, L, S, -1)
        h_scale = self.scale_feature(torch.cat([vel_emb, pos_feat, scale_feat], dim=-1))
        logits = self.scale_score(h_scale).squeeze(-1)
        alpha = torch.softmax(logits / self.temperature, dim=-1)
        self.last_attention = alpha.detach()
        h = torch.sum(alpha.unsqueeze(-1) * h_scale, dim=3)
        h = self.temporal_mlp(h.reshape(B, N, -1))
        return h
