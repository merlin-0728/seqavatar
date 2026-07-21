import copy

import torch
import torch.nn as nn


def _clone_mlp_with_extra_input(mlp, extra_input_dim):
    cloned = copy.deepcopy(mlp)
    if extra_input_dim <= 0:
        return cloned

    for idx, layer in enumerate(cloned):
        if isinstance(layer, nn.Linear):
            old_layer = layer
            new_layer = nn.Linear(
                old_layer.in_features + extra_input_dim,
                old_layer.out_features,
                bias=old_layer.bias is not None,
            )
            with torch.no_grad():
                new_layer.weight[:, :old_layer.in_features].copy_(old_layer.weight)
                new_layer.weight[:, old_layer.in_features:].zero_()
                if old_layer.bias is not None:
                    new_layer.bias.copy_(old_layer.bias)
            cloned[idx] = new_layer
            return cloned

    raise RuntimeError("[PartPAMO] Cannot widen MLP: no Linear layer found.")


class PartNonrigidExpert(nn.Module):
    def __init__(self, mlp, gaussian_warp, gaussian_rotation, gaussian_scaling, extra_input_dim=0):
        super().__init__()
        self.mlp = _clone_mlp_with_extra_input(mlp, int(extra_input_dim))
        self.gaussian_warp = copy.deepcopy(gaussian_warp)
        self.gaussian_rotation = copy.deepcopy(gaussian_rotation)
        self.gaussian_scaling = copy.deepcopy(gaussian_scaling)

    def forward(self, features, film=None):
        if film is None:
            h = self.mlp(features)
        else:
            gamma, beta = film
            h = features
            film_idx = 0
            for layer in self.mlp:
                h = layer(h)
                if isinstance(layer, nn.ReLU) and film_idx < gamma.shape[-2]:
                    h = gamma[..., film_idx, :] * h + beta[..., film_idx, :]
                    film_idx += 1
        return self.gaussian_warp(h), self.gaussian_rotation(h), self.gaussian_scaling(h)


class PartMotionEncoder(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=64, output_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


class PartMotionFiLM(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, num_layers=3, width=512):
        super().__init__()
        self.num_layers = int(num_layers)
        self.width = int(width)
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.out = nn.Linear(hidden_dim, self.num_layers * self.width * 2)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z_point):
        film = self.out(self.hidden(z_point))
        film = film.view(z_point.shape[:-1] + (self.num_layers, 2, self.width))
        gamma = 1.0 + film[..., 0, :]
        beta = film[..., 1, :]
        return gamma, beta


class PartRigidHead(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.out = nn.Linear(hidden_dim, 6)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z_part):
        rigid = self.out(self.hidden(z_part))
        return rigid[..., :3], rigid[..., 3:]


class PartRigidityMLP(nn.Module):
    def __init__(self, pos_input_dim, part_motion_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pos_input_dim + part_motion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_emb, z_point):
        return torch.sigmoid(self.mlp(torch.cat([x_emb, z_point], dim=-1)))


def axis_angle_to_matrix(axis_angle):
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z = axis_angle.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        dim=-1,
    ).reshape(axis_angle.shape[:-1] + (3, 3))

    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    eye = eye.expand(axis_angle.shape[:-1] + (3, 3))
    sin_term = torch.sin(angle)[..., None] / angle[..., None]
    cos_term = (1.0 - torch.cos(angle))[..., None] / (angle[..., None] * angle[..., None])
    return eye + sin_term * skew + cos_term * torch.matmul(skew, skew)


class NonrigidDeformer(nn.Module):
    def __init__(self, D=3, W=512, use_pose_cond=0, use_seq_pose_cond=0, use_seq_xyz_cond=0, 
                 pos_input_dim=63, pose_cond_dim=32, seq_pose_cond_dim=32, seq_xyz_cond_dim=96,
                 seq_len=6, seq_xyz_knn=1, time_step_num=1, smpl_type='smpl',
                 use_part_moe=False, num_parts=5, part_moe_global_keep=0.1,
                 use_part_pamo=False, part_pamo_dim=32, part_pamo_rigidity_min=0.0,
                 part_pamo_step1_only=False, part_pamo_fixed_rigidity=-1.0,
                 part_pamo_motion_film=False, part_pamo_motion_feat_mode="mean"):
        super(NonrigidDeformer, self).__init__()

        self.use_pose_cond = use_pose_cond
        self.use_seq_pose_cond = use_seq_pose_cond
        self.use_seq_xyz_cond = use_seq_xyz_cond
        self.use_part_moe = use_part_moe
        self.num_parts = num_parts
        self.part_moe_global_keep = part_moe_global_keep
        self.pos_input_dim = pos_input_dim
        self.use_part_pamo = bool(use_part_moe and use_part_pamo)
        self.part_pamo_dim = int(part_pamo_dim)
        self.part_pamo_rigidity_min = max(0.0, min(1.0, float(part_pamo_rigidity_min)))
        self.part_pamo_step1_only = bool(part_pamo_step1_only)
        self.part_pamo_motion_film = bool(part_pamo_motion_film)
        self.part_pamo_motion_feat_mode = part_pamo_motion_feat_mode
        fixed_rigidity = float(part_pamo_fixed_rigidity)
        self.part_pamo_fixed_rigidity = None if fixed_rigidity < 0.0 else max(0.0, min(1.0, fixed_rigidity))
        if self.part_pamo_motion_feat_mode == "mean":
            self.part_motion_feat_dim = 15
        elif self.part_pamo_motion_feat_mode == "rich":
            self.part_motion_feat_dim = 28
        else:
            raise ValueError(f"Unknown part_pamo_motion_feat_mode: {self.part_pamo_motion_feat_mode}")
        self.part_moe_active = False
        self.part_experts = None
        self.part_pamo_last_stats = {}
        if self.use_part_pamo:
            self.PartMotionEncoder = PartMotionEncoder(
                input_dim=self.part_motion_feat_dim,
                hidden_dim=max(64, self.part_pamo_dim * 2),
                output_dim=self.part_pamo_dim,
            )
            self.PartMotionFiLM = None
            if self.part_pamo_motion_film:
                self.PartMotionFiLM = PartMotionFiLM(
                    input_dim=self.part_pamo_dim,
                    hidden_dim=max(64, self.part_pamo_dim * 2),
                    num_layers=D,
                    width=W,
                )
            self.PartRigidHead = None
            self.PartRigidityMLP = None
            if not self.part_pamo_step1_only:
                self.PartRigidHead = PartRigidHead(
                    input_dim=self.part_pamo_dim,
                    hidden_dim=max(64, self.part_pamo_dim * 2),
                )
                if self.part_pamo_fixed_rigidity is None:
                    self.PartRigidityMLP = PartRigidityMLP(
                        pos_input_dim=self.pos_input_dim,
                        part_motion_dim=self.part_pamo_dim,
                        hidden_dim=max(64, self.part_pamo_dim * 2),
                    )
            modules = ["PartMotionEncoder"]
            if self.PartMotionFiLM is not None:
                modules.append("PartMotionFiLM")
            if self.PartRigidHead is not None:
                modules.append("PartRigidHead")
            if self.PartRigidityMLP is not None:
                modules.append("PartRigidityMLP")
            print(
                "[PartPAMO] enabled=True "
                f"modules={','.join(modules)} "
                f"part_pamo_dim={self.part_pamo_dim} "
                f"motion_feat_mode={self.part_pamo_motion_feat_mode} "
                f"motion_film={self.part_pamo_motion_film} "
                f"rigidity_min={self.part_pamo_rigidity_min} "
                f"step1_only={self.part_pamo_step1_only} "
                f"fixed_rigidity={self.part_pamo_fixed_rigidity}"
            )
        elif self.use_part_moe:
            print("[PartPAMO] enabled=False; PartPAMO modules are not constructed.")

        self.input_ch = pos_input_dim
        self.pose_cond_dim, self.seq_pose_cond_dim, self.seq_xyz_cond_dim = 0, 0, 0

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
        
        layers = []
        in_dim = self.input_ch
        for _ in range(D):
            layers.append(nn.Linear(in_dim, W))
            layers.append(nn.ReLU())
            in_dim = W
        self.mlp = nn.Sequential(*layers)

        self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)

    def init_part_moe_from_shared(self, num_parts=None):
        if self.part_moe_active:
            print("[PartMoE] Experts already initialized. Skip.")
            return False

        num_parts = int(num_parts or self.num_parts)
        self.num_parts = num_parts
        extra_input_dim = self.part_pamo_dim if self.use_part_pamo else 0
        print(f"[PartMoE] Initializing {num_parts} experts from the shared non-rigid MLP.")
        if self.use_part_pamo:
            print(f"[PartPAMO] Appending per-part motion code dim={self.part_pamo_dim} to each expert input.")
        self.part_experts = nn.ModuleList([
            PartNonrigidExpert(
                self.mlp,
                self.gaussian_warp,
                self.gaussian_rotation,
                self.gaussian_scaling,
                extra_input_dim=extra_input_dim,
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

    def encode_part_motion(self, part_motion_conds, features, part_label):
        if not self.use_part_pamo:
            return features, None, None
        if part_motion_conds is None:
            raise RuntimeError("[PartPAMO] part_motion_conds is required when --use_part_pamo is enabled.")

        part_motion_conds = part_motion_conds.to(device=features.device, dtype=features.dtype)
        if part_motion_conds.dim() == 2:
            part_motion_conds = part_motion_conds.unsqueeze(0)
        if part_motion_conds.shape[0] == 1 and features.shape[0] > 1:
            part_motion_conds = part_motion_conds.expand(features.shape[0], -1, -1)

        if part_motion_conds.shape[1] < self.num_parts:
            pad_shape = (
                part_motion_conds.shape[0],
                self.num_parts - part_motion_conds.shape[1],
                part_motion_conds.shape[2],
            )
            part_motion_conds = torch.cat(
                [part_motion_conds, torch.zeros(pad_shape, device=features.device, dtype=features.dtype)],
                dim=1,
            )
        elif part_motion_conds.shape[1] > self.num_parts:
            part_motion_conds = part_motion_conds[:, :self.num_parts]

        z_part = self.PartMotionEncoder(part_motion_conds)
        gather_idx = part_label.unsqueeze(-1).expand(-1, -1, self.part_pamo_dim)
        z_point = torch.gather(z_part, 1, gather_idx)
        return torch.cat([features, z_point], dim=-1), z_part, z_point

    def encode_part_motion_film(self, z_point):
        if not (self.use_part_pamo and self.part_pamo_motion_film):
            return None
        if getattr(self, "PartMotionFiLM", None) is None:
            raise RuntimeError("[PartPAMO] PartMotionFiLM is required when --part_pamo_motion_film is enabled.")
        return self.PartMotionFiLM(z_point)

    def get_part_centers(self, query_xyz, part_label):
        query_xyz = query_xyz.to(device=part_label.device)
        if query_xyz.dim() == 2:
            query_xyz = query_xyz.unsqueeze(0)
        if query_xyz.shape[0] == 1 and part_label.shape[0] > 1:
            query_xyz = query_xyz.expand(part_label.shape[0], -1, -1)

        B, N = part_label.shape
        centers = torch.zeros(B, self.num_parts, 3, device=query_xyz.device, dtype=query_xyz.dtype)
        counts = torch.zeros(B, self.num_parts, 1, device=query_xyz.device, dtype=query_xyz.dtype)
        centers.scatter_add_(1, part_label.unsqueeze(-1).expand(-1, -1, 3), query_xyz)
        counts.scatter_add_(1, part_label.unsqueeze(-1), torch.ones(B, N, 1, device=query_xyz.device, dtype=query_xyz.dtype))

        global_center = query_xyz.mean(dim=1, keepdim=True)
        centers = centers / counts.clamp_min(1.0)
        centers = torch.where(counts > 0, centers, global_center.expand(-1, self.num_parts, -1))
        return centers

    def compute_point_rigidity(self, x_emb, z_point, d_xyz):
        if self.part_pamo_fixed_rigidity is not None:
            return torch.full(
                d_xyz.shape[:-1] + (1,),
                self.part_pamo_fixed_rigidity,
                device=d_xyz.device,
                dtype=d_xyz.dtype,
            )
        if self.PartRigidityMLP is None:
            raise RuntimeError("[PartPAMO] PartRigidityMLP is required unless fixed rigidity is enabled.")
        x_emb = x_emb.to(device=d_xyz.device, dtype=d_xyz.dtype)
        if x_emb.dim() == 2:
            x_emb = x_emb.unsqueeze(0)
        if x_emb.shape[0] == 1 and d_xyz.shape[0] > 1:
            x_emb = x_emb.expand(d_xyz.shape[0], -1, -1)
        rigidity = self.PartRigidityMLP(x_emb, z_point.to(dtype=d_xyz.dtype))
        if self.part_pamo_rigidity_min > 0.0:
            rigidity = self.part_pamo_rigidity_min + (1.0 - self.part_pamo_rigidity_min) * rigidity
        return rigidity

    def update_part_motion_stats(self, part_label, z_part, z_point, film, part_weight):
        if z_part is None or z_point is None:
            return
        with torch.no_grad():
            stats = {
                "part_weight": float(part_weight),
                "z_norm_mean": float(z_point.detach().norm(dim=-1).mean().item()),
                "z_norm_std": float(z_point.detach().norm(dim=-1).std(unbiased=False).item()) if z_point.numel() > 0 else 0.0,
                "z_part_norm_mean": float(z_part.detach().norm(dim=-1).mean().item()),
                "motion_feat_mode": self.part_pamo_motion_feat_mode,
                "motion_film": bool(self.part_pamo_motion_film),
            }
            if film is not None:
                gamma, beta = film
                gamma_delta = gamma.detach() - 1.0
                beta = beta.detach()
                stats.update({
                    "film_gamma_delta_mean": float(gamma_delta.mean().item()),
                    "film_gamma_delta_std": float(gamma_delta.std(unbiased=False).item()) if gamma_delta.numel() > 1 else 0.0,
                    "film_gamma_delta_abs_mean": float(gamma_delta.abs().mean().item()),
                    "film_beta_mean": float(beta.mean().item()),
                    "film_beta_std": float(beta.std(unbiased=False).item()) if beta.numel() > 1 else 0.0,
                    "film_beta_abs_mean": float(beta.abs().mean().item()),
                })
            self.part_pamo_last_stats = stats

    def update_part_pamo_stats(self, part_label, point_rigidity, d_xyz, rigid_residual, rigid_contrib, part_weight):
        with torch.no_grad():
            labels = part_label.detach()
            rigidity = point_rigidity.detach().squeeze(-1)
            mlp_norm = d_xyz.detach().norm(dim=-1)
            rigid_norm = rigid_residual.detach().norm(dim=-1)
            contrib_norm = rigid_contrib.detach().norm(dim=-1)
            eps = 1e-8

            per_part = []
            for pid in range(self.num_parts):
                mask = labels == pid
                if mask.any():
                    values = rigidity[mask]
                    per_part.append({
                        "part": int(pid),
                        "count": int(mask.sum().item()),
                        "r_mean": float(values.mean().item()),
                        "r_std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
                    })

            stats = dict(getattr(self, "part_pamo_last_stats", {}))
            stats.update({
                "part_weight": float(part_weight),
                "r_mean": float(rigidity.mean().item()),
                "r_std": float(rigidity.std(unbiased=False).item()) if rigidity.numel() > 1 else 0.0,
                "r_min": float(rigidity.min().item()),
                "r_max": float(rigidity.max().item()),
                "mlp_norm_mean": float(mlp_norm.mean().item()),
                "rigid_norm_mean": float(rigid_norm.mean().item()),
                "rigid_contrib_norm_mean": float(contrib_norm.mean().item()),
                "ratio": float((contrib_norm.mean() / mlp_norm.mean().clamp_min(eps)).item()),
                "per_part": per_part,
            })
            self.part_pamo_last_stats = stats

    def apply_part_rigid_residual(self, d_xyz, z_part, z_point, x_emb, query_xyz, part_label, part_weight):
        if not self.use_part_pamo:
            return d_xyz
        if self.part_pamo_step1_only:
            return d_xyz
        if query_xyz is None:
            raise RuntimeError("[PartPAMO] query_xyz is required for the part-level rigid residual branch.")
        if z_part is None:
            raise RuntimeError("[PartPAMO] z_part is required for the part-level rigid residual branch.")
        if self.PartRigidHead is None:
            raise RuntimeError("[PartPAMO] PartRigidHead is required for the part-level rigid residual branch.")
        if z_point is None:
            raise RuntimeError("[PartPAMO] point-wise z_part is required for internal rigidity.")
        if x_emb is None:
            raise RuntimeError("[PartPAMO] x_emb is required for internal rigidity.")

        query_xyz = query_xyz.to(device=d_xyz.device, dtype=d_xyz.dtype)
        if query_xyz.dim() == 2:
            query_xyz = query_xyz.unsqueeze(0)
        if query_xyz.shape[0] == 1 and d_xyz.shape[0] > 1:
            query_xyz = query_xyz.expand(d_xyz.shape[0], -1, -1)

        rot_vec, trans = self.PartRigidHead(z_part)
        rot_mat = axis_angle_to_matrix(rot_vec)
        centers = self.get_part_centers(query_xyz, part_label).to(dtype=d_xyz.dtype)

        gather_xyz = part_label.unsqueeze(-1).expand(-1, -1, 3)
        point_centers = torch.gather(centers, 1, gather_xyz)
        point_trans = torch.gather(trans, 1, gather_xyz)
        point_rot = torch.gather(
            rot_mat,
            1,
            part_label.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3),
        )

        local_xyz = query_xyz - point_centers
        rotated_xyz = torch.matmul(point_rot, local_xyz.unsqueeze(-1)).squeeze(-1)
        rigid_residual = rotated_xyz + point_centers + point_trans - query_xyz
        point_rigidity = self.compute_point_rigidity(x_emb, z_point, d_xyz)
        rigid_contrib = float(part_weight) * point_rigidity * rigid_residual
        self.update_part_pamo_stats(part_label, point_rigidity, d_xyz, rigid_residual, rigid_contrib, part_weight)
        return d_xyz + rigid_contrib

    def forward_part_moe(self, features, part_label, part_motion_conds=None,
                         x_emb=None, query_xyz=None, part_moe_alpha=0.0, part_moe_global_keep=None):
        if not self.part_moe_active or self.part_experts is None:
            raise RuntimeError("[PartMoE] Experts have not been initialized.")
        self.part_pamo_last_stats = {}

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

        expert_features, z_part, z_point = self.encode_part_motion(features=features, part_label=part_label,
                                                                   part_motion_conds=part_motion_conds)
        motion_film = self.encode_part_motion_film(z_point)
        self.update_part_motion_stats(part_label, z_part, z_point, motion_film, part_weight)
        global_xyz, global_rotation, global_scaling = self.part_experts[0](expert_features, film=motion_film)

        if part_weight <= 0.0:
            return global_xyz, global_rotation, global_scaling

        feature_shape = expert_features.shape
        flat_features = expert_features.reshape(-1, feature_shape[-1])
        flat_labels = part_label.reshape(-1)
        flat_film = None
        if motion_film is not None:
            flat_film = (
                motion_film[0].reshape(-1, motion_film[0].shape[-2], motion_film[0].shape[-1]),
                motion_film[1].reshape(-1, motion_film[1].shape[-2], motion_film[1].shape[-1]),
            )

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
            part_film = None
            if flat_film is not None:
                part_film = (
                    flat_film[0].index_select(0, idx),
                    flat_film[1].index_select(0, idx),
                )
            part_xyz, part_rotation, part_scaling = self.part_experts[pid](
                flat_features.index_select(0, idx),
                film=part_film,
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

        d_xyz = self.apply_part_rigid_residual(
            d_xyz.reshape_as(global_xyz).contiguous(),
            z_part,
            z_point,
            x_emb,
            query_xyz,
            part_label,
            part_weight,
        )

        return (
            d_xyz,
            d_rotation.reshape_as(global_rotation).contiguous(),
            d_scaling.reshape_as(global_scaling).contiguous(),
        )

    def forward(self, x_emb, pose_conds=None, seq_pose_conds=None, seq_xyz_conds=None,
                part_label=None, part_enabled=False,
                part_motion_conds=None, query_xyz=None,
                part_moe_alpha=0.0, part_moe_global_keep=None):
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
        if self.use_seq_xyz_cond: 
            seq_xyz_feats = self.SeqXYZEncoder(seq_xyz_conds, x_emb)
            feats.append(seq_xyz_feats)

        features = torch.cat(feats, dim=-1)
        if (
            self.use_part_moe
            and self.part_moe_active
            and part_enabled
            and part_label is not None
        ):
            return self.forward_part_moe(
                features,
                part_label,
                part_motion_conds=part_motion_conds,
                x_emb=x_emb,
                query_xyz=query_xyz,
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
