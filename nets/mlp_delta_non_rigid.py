import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _adaptive_motion_gate(score, alpha=1.0, temp=0.5, reduce_dims=None):
    if reduce_dims is None:
        reduce_dims = tuple(range(1, score.dim()))
    alpha = float(alpha)
    temp = max(float(temp), 1e-6)
    score_ref = score.detach()
    mean = score_ref.mean(dim=reduce_dims, keepdim=True)
    std = score_ref.std(dim=reduce_dims, unbiased=False, keepdim=True)
    threshold = mean + alpha * std
    denom = (std * temp).clamp_min(1e-6)
    return torch.sigmoid((score - threshold) / denom)


def _init_last_linear_bias(module, bias):
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Linear):
            nn.init.constant_(layer.bias, float(bias))
            return


def _zero_init_linear(module):
    if isinstance(module, nn.Linear):
        nn.init.zeros_(module.weight)
        nn.init.zeros_(module.bias)


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


class MotionTokenEncoder(nn.Module):
    def __init__(self, in_dim, token_num=32, token_dim=64, hidden_dim=128):
        super().__init__()
        self.token_num = int(token_num)
        self.logit_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, token_num),
        )
        self.codebook = nn.Parameter(torch.randn(token_num, token_dim) * 0.02)
        self.last_weights = None

    def forward(self, x):
        weights = torch.softmax(self.logit_mlp(x), dim=-1)
        self.last_weights = weights.detach()
        return torch.matmul(weights, self.codebook), weights

    def pop_token_stats(self):
        if self.last_weights is None:
            return {}, []
        with torch.no_grad():
            weights = self.last_weights.reshape(-1, self.token_num)
            weights = weights.clamp_min(1e-8)
            entropy = -(weights * torch.log(weights)).sum(dim=-1)
            max_prob = weights.max(dim=-1).values
            top1 = weights.argmax(dim=-1)
            usage = torch.bincount(top1, minlength=self.token_num).float()
            usage = usage / usage.sum().clamp_min(1.0)
            used = usage > 0
            usage_entropy = -(usage[used] * torch.log(usage[used])).sum() if used.any() else usage.new_tensor(0.0)
            log_token_num = torch.log(weights.new_tensor(float(self.token_num))).clamp_min(1e-8)
            stats = {
                "entropy_mean": entropy.mean().item(),
                "entropy_norm": (entropy.mean() / log_token_num).item(),
                "entropy_min": entropy.min().item(),
                "entropy_max": entropy.max().item(),
                "max_prob_mean": max_prob.mean().item(),
                "max_prob_min": max_prob.min().item(),
                "max_prob_max": max_prob.max().item(),
                "usage_nonzero": used.float().sum().item(),
                "usage_max": usage.max().item(),
                "usage_entropy": usage_entropy.item(),
                "usage_entropy_norm": (usage_entropy / log_token_num).item(),
            }
            usage_list = usage.cpu().tolist()
        self.last_weights = None
        return stats, usage_list


class NonrigidDeformer(nn.Module):
    def __init__(self, D=3, W=512, use_pose_cond=0, use_seq_pose_cond=0, use_seq_xyz_cond=0, 
                 pos_input_dim=63, pose_cond_dim=32, seq_pose_cond_dim=32, seq_xyz_cond_dim=96,
                 seq_len=6, seq_xyz_knn=1, time_step_num=1, smpl_type='smpl',
                 use_part_moe=False, num_parts=5, part_moe_global_keep=0.1,
                 use_amc_causal=False, amc_causal_mode="gated_residual", amc_causal_window=3,
                 amc_motion_gate_alpha=1.0, amc_motion_gate_temp=0.5,
                 use_tdp_semantic_encoder=False, tdp_semantic_mode="gated_residual",
                 tdp_base_time_step_num=0, tdp_gate_init_bias=-4.0,
                 tdp_debug_stats=False,
                 use_dif=False, dif_mode="peak", dif_sigma_init=-7.0,
                 dif_sigma_min=1e-4, dif_sigma_max=0.05,
                 dif_residual_beta=1.0, dif_residual_warmup=0, dif_eps=1e-6,
                 use_acc_cond=False, seq_acc_cond_dim=64,
                 use_motion_token=False, motion_token_mode="none",
                 motion_token_use_acc=False, motion_token_use_part=False,
                 motion_token_use_codebook=False, motion_token_num=32,
                 motion_token_dim=64, motion_token_part_dim=16,
                 motion_token_acc_dim=64, motion_token_num_parts=55):
        super(NonrigidDeformer, self).__init__()

        self.use_pose_cond = use_pose_cond
        self.use_seq_pose_cond = use_seq_pose_cond
        self.use_seq_xyz_cond = use_seq_xyz_cond
        self.use_part_moe = use_part_moe
        self.num_parts = num_parts
        self.part_moe_global_keep = part_moe_global_keep
        self.part_moe_active = False
        self.part_experts = None
        self.use_amc_causal = bool(use_amc_causal)
        self.use_tdp_semantic_encoder = bool(use_tdp_semantic_encoder)
        self.tdp_debug_stats = bool(tdp_debug_stats)
        self.use_dif = bool(use_dif)
        self.dif_mode = str(dif_mode or "peak").lower()
        self.dif_eps = float(dif_eps)
        self.dif_sigma_min = float(dif_sigma_min)
        self.dif_sigma_max = float(dif_sigma_max)
        self.dif_residual_beta = float(dif_residual_beta)
        self.dif_residual_warmup = int(dif_residual_warmup or 0)
        self.dif_current_beta = self.dif_residual_beta
        self.last_dif_mu = None
        self.last_dif_sigma = None
        self.last_dif_residual = None
        self.use_acc_cond = bool(use_acc_cond) or (bool(use_motion_token) and bool(motion_token_use_acc))
        self.seq_acc_cond_dim = int(seq_acc_cond_dim)
        self.use_motion_token = bool(use_motion_token)
        self.motion_token_mode = str(motion_token_mode or "none").lower()
        self.motion_token_use_acc = bool(motion_token_use_acc)
        self.motion_token_use_part = bool(motion_token_use_part)
        self.motion_token_use_codebook = bool(motion_token_use_codebook)

        self.input_ch = pos_input_dim
        self.pose_cond_dim, self.seq_pose_cond_dim, self.seq_xyz_cond_dim = 0, 0, 0

        if self.use_pose_cond:
            self.PoseEncoder = PoseEncoder(32, pose_cond_dim, smpl_type)
            self.input_ch += pose_cond_dim
            
        if self.use_seq_pose_cond:
            self.SeqPoseEncoder = SeqPoseEncoder(
                seq_len,
                16,
                seq_pose_cond_dim,
                time_step_num,
                smpl_type,
                use_amc_causal=self.use_amc_causal,
                amc_causal_mode=amc_causal_mode,
                amc_causal_window=amc_causal_window,
                amc_motion_gate_alpha=amc_motion_gate_alpha,
                amc_motion_gate_temp=amc_motion_gate_temp,
                use_tdp_semantic_encoder=self.use_tdp_semantic_encoder,
                tdp_semantic_mode=tdp_semantic_mode,
                tdp_base_time_step_num=tdp_base_time_step_num,
                tdp_gate_init_bias=tdp_gate_init_bias,
                tdp_debug_stats=self.tdp_debug_stats,
            )
            self.input_ch += seq_pose_cond_dim

        if self.use_seq_xyz_cond:
            self.SeqXYZEncoder = SeqXYZEncoder(pos_emb_dim=pos_input_dim, hidden_dim1=96, hidden_dim2=256, output_dim=seq_xyz_cond_dim, 
                                        time_step_num=time_step_num, seq_len=seq_len, seq_xyz_knn=seq_xyz_knn,
                                        use_amc_causal=self.use_amc_causal,
                                        amc_causal_mode=amc_causal_mode,
                                        amc_causal_window=amc_causal_window,
                                        amc_motion_gate_alpha=amc_motion_gate_alpha,
                                        amc_motion_gate_temp=amc_motion_gate_temp,
                                        use_tdp_semantic_encoder=self.use_tdp_semantic_encoder,
                                        tdp_semantic_mode=tdp_semantic_mode,
                                        tdp_base_time_step_num=tdp_base_time_step_num,
                                        tdp_gate_init_bias=tdp_gate_init_bias,
                                            tdp_debug_stats=self.tdp_debug_stats)
            self.input_ch += seq_xyz_cond_dim

        token_input_dim = 0
        if self.use_acc_cond:
            if not self.use_seq_xyz_cond:
                raise ValueError("acceleration condition requires use_seq_xyz_cond=True.")
            acc_cond_dim = int(motion_token_acc_dim) if (self.use_motion_token and self.motion_token_use_acc) else self.seq_acc_cond_dim
            self.SeqAccEncoder = SeqXYZEncoder(
                pos_emb_dim=pos_input_dim,
                hidden_dim1=96,
                hidden_dim2=256,
                output_dim=acc_cond_dim,
                time_step_num=time_step_num,
                seq_len=seq_len,
                seq_xyz_knn=seq_xyz_knn,
            )
            self.input_ch += acc_cond_dim
            if self.use_motion_token and self.motion_token_use_acc:
                token_input_dim += acc_cond_dim

        if self.use_motion_token and self.motion_token_use_part:
            self.motion_token_part_embedding = nn.Embedding(
                int(motion_token_num_parts),
                int(motion_token_part_dim),
            )
            self.motion_token_num_parts = int(motion_token_num_parts)
            self.input_ch += int(motion_token_part_dim)
            token_input_dim += int(motion_token_part_dim)

        if self.use_motion_token and self.motion_token_use_codebook:
            if not self.use_seq_xyz_cond:
                raise ValueError("motion token codebook requires use_seq_xyz_cond=True.")
            token_input_dim += seq_xyz_cond_dim
            self.MotionTokenEncoder = MotionTokenEncoder(
                token_input_dim,
                token_num=int(motion_token_num),
                token_dim=int(motion_token_dim),
            )
            self.input_ch += int(motion_token_dim)
        
        layers = []
        in_dim = self.input_ch
        for _ in range(D):
            layers.append(nn.Linear(in_dim, W))
            layers.append(nn.ReLU())
            in_dim = W
        self.mlp = nn.Sequential(*layers)

        self.gaussian_warp = nn.Linear(W, 3)
        if self.use_dif:
            self.gaussian_warp_sigma = nn.Linear(W, 1)
            nn.init.zeros_(self.gaussian_warp_sigma.weight)
            nn.init.constant_(self.gaussian_warp_sigma.bias, float(dif_sigma_init))
            if self.dif_mode in {"sigma_rectifier", "sigma_rectifier_v2"}:
                self.dif_rectifier = nn.Sequential(
                    nn.Linear(W + 3 + 1, W),
                    nn.ReLU(),
                    nn.Linear(W, 3),
                )
                _zero_init_linear(self.dif_rectifier[-1])
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)

    def set_dif_iteration(self, iteration):
        if not self.use_dif:
            return
        if self.dif_residual_warmup <= 0:
            self.dif_current_beta = self.dif_residual_beta
            return
        progress = max(0.0, min(1.0, float(iteration) / float(self.dif_residual_warmup)))
        self.dif_current_beta = self.dif_residual_beta * progress

    def _record_dif_stats(self, mu, sigma, residual):
        self.last_dif_mu = mu
        self.last_dif_sigma = sigma
        self.last_dif_residual = residual

    def forward_delta_x(self, h):
        mu = self.gaussian_warp(h)
        if not self.use_dif:
            return mu

        sigma = F.softplus(self.gaussian_warp_sigma(h)) + self.dif_eps
        if self.dif_mode == "sigma_rectifier_v2":
            sigma = sigma.clamp(min=self.dif_sigma_min, max=self.dif_sigma_max)

        if self.dif_mode in {"peak", "uncert_loss"}:
            # For a unimodal Gaussian, argmax_x N(x | mu, sigma) is exactly mu.
            self._record_dif_stats(mu, sigma, torch.zeros_like(mu))
            return mu

        if self.dif_mode == "sigma_rectifier":
            rectifier_in = torch.cat([h, mu, torch.log(sigma)], dim=-1)
            residual = sigma * torch.tanh(self.dif_rectifier(rectifier_in))
            self._record_dif_stats(mu, sigma, residual)
            return mu + residual

        if self.dif_mode == "sigma_rectifier_v2":
            rectifier_in = torch.cat([h, mu, torch.log(sigma)], dim=-1)
            residual = float(self.dif_current_beta) * sigma * torch.tanh(self.dif_rectifier(rectifier_in))
            self._record_dif_stats(mu, sigma, residual)
            return mu + residual

        raise ValueError(f"Unsupported DIF mode: {self.dif_mode}")

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

    def pop_tdp_stats(self):
        stats = {}
        if hasattr(self, "SeqPoseEncoder") and hasattr(self.SeqPoseEncoder, "pop_tdp_stats"):
            for key, value in self.SeqPoseEncoder.pop_tdp_stats().items():
                stats[f"pose/{key}"] = value
        if hasattr(self, "SeqXYZEncoder") and hasattr(self.SeqXYZEncoder, "pop_tdp_stats"):
            for key, value in self.SeqXYZEncoder.pop_tdp_stats().items():
                stats[f"xyz/{key}"] = value
        return stats

    def pop_dif_stats(self):
        if not self.use_dif or self.last_dif_mu is None or self.last_dif_sigma is None:
            return {}
        with torch.no_grad():
            mu = self.last_dif_mu.detach()
            sigma = self.last_dif_sigma.detach()
            residual = (
                self.last_dif_residual.detach()
                if self.last_dif_residual is not None
                else torch.zeros_like(mu)
            )
            mu_norm = torch.norm(mu, dim=-1).mean()
            residual_norm = torch.norm(residual, dim=-1).mean()
            ratio = residual_norm / mu_norm.clamp_min(1e-8)
            active_ratio = (torch.norm(residual, dim=-1) > 1e-5).float().mean()
            stats = {
                "beta": float(self.dif_current_beta),
                "sigma_mean": sigma.mean().item(),
                "sigma_min": sigma.min().item(),
                "sigma_max": sigma.max().item(),
                "mu_norm": mu_norm.item(),
                "residual_norm": residual_norm.item(),
                "residual_mu_ratio": ratio.item(),
                "active_ratio": active_ratio.item(),
            }
        self.last_dif_mu = None
        self.last_dif_sigma = None
        self.last_dif_residual = None
        return stats

    def pop_motion_token_stats(self):
        if not self.use_motion_token:
            return {}, []
        if not self.motion_token_use_codebook or not hasattr(self, "MotionTokenEncoder"):
            return {"codebook_active": 0.0}, []
        stats, usage = self.MotionTokenEncoder.pop_token_stats()
        if stats:
            stats["codebook_active"] = 1.0
        return stats, usage

    def freeze_shared_after_part_moe(self):
        for module in (self.mlp, self.gaussian_warp, self.gaussian_rotation, self.gaussian_scaling):
            for param in module.parameters():
                param.requires_grad_(False)

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
            part_xyz, part_rotation, part_scaling = self.part_experts[pid](flat_features.index_select(0, idx))
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

    def forward(self, x_emb, pose_conds=None, seq_pose_conds=None, seq_xyz_conds=None,
                seq_acc_conds=None,
                part_label=None, part_enabled=False, part_moe_alpha=0.0, part_moe_global_keep=None):
        feats = []
        feats.append(x_emb)
        token_feats = []

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
            if self.use_motion_token and self.motion_token_use_codebook:
                token_feats.append(seq_xyz_feats)

        if self.use_acc_cond:
            if seq_acc_conds is None:
                raise RuntimeError("seq_acc_conds is required when use_acc_cond=True.")
            seq_acc_feats = self.SeqAccEncoder(seq_acc_conds, x_emb)
            feats.append(seq_acc_feats)
            if self.use_motion_token and self.motion_token_use_acc:
                token_feats.append(seq_acc_feats)

        if self.use_motion_token and self.motion_token_use_part:
            if part_label is None:
                raise RuntimeError("part_label is required when motion_token_use_part=True.")
            part_label = part_label.long().to(x_emb.device)
            part_label = torch.clamp(part_label, min=0, max=self.motion_token_num_parts - 1)
            if part_label.dim() == 1:
                part_label = part_label.unsqueeze(0).expand(x_emb.shape[0], -1)
            elif part_label.shape[0] == 1 and x_emb.shape[0] > 1:
                part_label = part_label.expand(x_emb.shape[0], -1)
            part_feats = self.motion_token_part_embedding(part_label)
            feats.append(part_feats)
            token_feats.append(part_feats)

        if self.use_motion_token and self.motion_token_use_codebook:
            if not token_feats:
                raise RuntimeError("Motion token codebook has no input features.")
            z_token, _ = self.MotionTokenEncoder(torch.cat(token_feats, dim=-1))
            feats.append(z_token)

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
                part_moe_alpha=part_moe_alpha,
                part_moe_global_keep=part_moe_global_keep,
            )

        h = self.mlp(features)
        d_xyz, d_scaling, d_rotation = self.forward_delta_x(h), self.gaussian_scaling(h), self.gaussian_rotation(h)
        
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
    def __init__(self, length, D1, D2, time_step_num, smpl_type,
                 use_amc_causal=False, amc_causal_mode="gated_residual", amc_causal_window=3,
                 amc_motion_gate_alpha=1.0, amc_motion_gate_temp=0.5,
                 use_tdp_semantic_encoder=False, tdp_semantic_mode="gated_residual",
                 tdp_base_time_step_num=0, tdp_gate_init_bias=-4.0,
                 tdp_debug_stats=False):
        super(SeqPoseEncoder, self).__init__()

        self.input_dim = 3 * (N_JOINT[smpl_type] + 1) # axis-angle form, + global orientation
        self.time_step_num = time_step_num
        self.use_amc_causal = bool(use_amc_causal)
        self.use_tdp_semantic_encoder = bool(use_tdp_semantic_encoder)
        self.tdp_semantic_mode = tdp_semantic_mode
        self.tdp_debug_stats = bool(tdp_debug_stats)
        self._tdp_stats = {}
        self.amc_causal_mode = amc_causal_mode
        self.amc_causal_window = max(1, int(amc_causal_window))
        self.amc_motion_gate_alpha = float(amc_motion_gate_alpha)
        self.amc_motion_gate_temp = float(amc_motion_gate_temp)
        self.mlp2 = nn.Sequential(nn.Linear(D1*length, D2), nn.ReLU())

        if self.use_tdp_semantic_encoder:
            if tdp_semantic_mode not in {"gated_residual", "baseline_residual"}:
                raise ValueError(f"Unsupported TDP semantic mode: {tdp_semantic_mode}")
            self.tdp_base_channels = int(tdp_base_time_step_num)
            if self.tdp_base_channels <= 1:
                raise ValueError(f"tdp_base_time_step_num must be > 1, got {tdp_base_time_step_num}")
            expected_channels = self.tdp_base_channels + self.tdp_base_channels + self.tdp_base_channels - 1
            if self.time_step_num != expected_channels:
                raise ValueError(
                    f"TDP semantic pose encoder expected {expected_channels} channels "
                    f"from base={self.tdp_base_channels}, got {self.time_step_num}"
                )
            hidden_gate_dim = max(1, D1 // 2)
            if self.tdp_semantic_mode == "baseline_residual":
                self.mlp1 = nn.Sequential(nn.Linear(self.input_dim * self.tdp_base_channels, D1), nn.ReLU())
                self.tdp_vel_encoder = nn.Sequential(
                    nn.Linear(self.input_dim * self.tdp_base_channels, D1),
                    nn.ReLU(),
                )
                self.tdp_acc_encoder = nn.Sequential(
                    nn.Linear(self.input_dim * (self.tdp_base_channels - 1), D1),
                    nn.ReLU(),
                )
                self.tdp_residual_out = nn.Linear(D1 * length, D2)
                _zero_init_linear(self.tdp_residual_out)
            else:
                self.tdp_full_encoder = nn.Sequential(
                    nn.Linear(self.input_dim * self.tdp_base_channels, D1),
                    nn.ReLU(),
                    nn.Linear(D1, D1),
                    nn.ReLU(),
                )
                self.tdp_vel_encoder = nn.Sequential(
                    nn.Linear(self.input_dim * self.tdp_base_channels, D1),
                    nn.ReLU(),
                    nn.Linear(D1, D1),
                    nn.ReLU(),
                )
                self.tdp_acc_encoder = nn.Sequential(
                    nn.Linear(self.input_dim * (self.tdp_base_channels - 1), D1),
                    nn.ReLU(),
                    nn.Linear(D1, D1),
                    nn.ReLU(),
                )
            self.tdp_gate_v = nn.Sequential(
                nn.Linear(D1, hidden_gate_dim),
                nn.ReLU(),
                nn.Linear(hidden_gate_dim, 1),
                nn.Sigmoid(),
            )
            self.tdp_gate_a = nn.Sequential(
                nn.Linear(D1, hidden_gate_dim),
                nn.ReLU(),
                nn.Linear(hidden_gate_dim, 1),
                nn.Sigmoid(),
            )
            _init_last_linear_bias(self.tdp_gate_v, tdp_gate_init_bias)
            _init_last_linear_bias(self.tdp_gate_a, tdp_gate_init_bias)
        else:
            self.mlp1 = nn.Sequential(nn.Linear(self.input_dim * time_step_num, D1), nn.ReLU())

        if self.use_amc_causal:
            if self.amc_causal_mode != "gated_residual":
                raise ValueError(f"Unsupported AMC causal mode: {self.amc_causal_mode}")
            if self.use_tdp_semantic_encoder:
                raise ValueError("AMC causal and TDP semantic encoder are mutually exclusive.")
            self.amc_step_encoder = nn.Sequential(nn.Linear(self.input_dim, D1), nn.ReLU())
            self.amc_gru = nn.GRU(D1, D1, batch_first=True)
            self.amc_out = nn.Linear(D1, D2)
            nn.init.zeros_(self.amc_out.weight)
            nn.init.zeros_(self.amc_out.bias)

    def _record_tdp_stats(self, values):
        if not self.tdp_debug_stats:
            return
        with torch.no_grad():
            self._tdp_stats["count"] = self._tdp_stats.get("count", 0) + 1
            for key, value in values.items():
                if torch.is_tensor(value):
                    value = value.detach().float().mean().item()
                self._tdp_stats[key] = self._tdp_stats.get(key, 0.0) + float(value)

    def pop_tdp_stats(self):
        count = int(self._tdp_stats.get("count", 0))
        if count <= 0:
            return {}
        stats = {key: value / count for key, value in self._tdp_stats.items() if key != "count"}
        self._tdp_stats = {}
        return stats

    def forward(self, x):
        # x: (B, N, T, J, DeltaStep, C)

        bs, T = x.shape[0], x.shape[1]
        if x.shape[2] != self.time_step_num:
            raise RuntimeError(
                f"SeqPoseEncoder expected {self.time_step_num} motion channels, "
                f"got {x.shape[2]} with shape {tuple(x.shape)}"
            )
        if self.use_tdp_semantic_encoder:
            base = self.tdp_base_channels
            full = x[:, :, :base].reshape(bs, T, -1)
            vel = x[:, :, base:base * 2].reshape(bs, T, -1)
            acc = x[:, :, base * 2:].reshape(bs, T, -1)
            f_vel = self.tdp_vel_encoder(vel)
            f_acc = self.tdp_acc_encoder(acc)
            if self.tdp_semantic_mode == "baseline_residual":
                f_full = self.mlp1(full)
            else:
                f_full = self.tdp_full_encoder(full)
            gate_v = self.tdp_gate_v(f_full)
            gate_a = self.tdp_gate_a(f_full)
            residual_x = gate_v * f_vel + gate_a * f_acc
            self._record_tdp_stats({
                "gate_v_mean": gate_v.mean(),
                "gate_v_max": gate_v.max(),
                "gate_v_active_0p1": (gate_v > 0.1).float().mean(),
                "gate_a_mean": gate_a.mean(),
                "gate_a_max": gate_a.max(),
                "gate_a_active_0p1": (gate_a > 0.1).float().mean(),
                "vel_full_norm_ratio": f_vel.norm() / f_full.norm().clamp_min(1e-6),
                "acc_full_norm_ratio": f_acc.norm() / f_full.norm().clamp_min(1e-6),
                "residual_full_norm_ratio": residual_x.norm() / f_full.norm().clamp_min(1e-6),
            })
            if self.tdp_semantic_mode == "baseline_residual":
                base_feat = self.mlp2(f_full.reshape(bs, -1))
                residual_feat = self.tdp_residual_out(residual_x.reshape(bs, -1))
                self._record_tdp_stats({
                    "residual_feat_base_feat_norm_ratio": residual_feat.norm() / base_feat.norm().clamp_min(1e-6),
                })
                return base_feat + residual_feat
            base_x = f_full + residual_x
            return self.mlp2(base_x.reshape(bs, -1))

        base_x = self.mlp1(x.reshape(bs, T, -1))
        base_feat = self.mlp2(base_x.reshape(bs, -1))

        if not self.use_amc_causal:
            return base_feat

        causal_window = min(self.amc_causal_window, T)
        # The last channel is the shortest time step because generate_time_steps
        # keeps channels ordered from large step to small step.
        recent_short_step = x[:, :causal_window, -1].contiguous()
        causal_chain = torch.flip(recent_short_step, dims=[1]).contiguous()
        motion_score = torch.linalg.norm(causal_chain, dim=-1)
        joint_gate = _adaptive_motion_gate(
            motion_score,
            alpha=self.amc_motion_gate_alpha,
            temp=self.amc_motion_gate_temp,
            reduce_dims=(1, 2),
        )
        gated_chain = causal_chain * joint_gate.unsqueeze(-1)
        causal_in = gated_chain.reshape(bs, causal_window, -1)
        causal_step_feat = self.amc_step_encoder(causal_in)
        _, causal_hidden = self.amc_gru(causal_step_feat)
        causal_delta = self.amc_out(causal_hidden.squeeze(0))
        motion_gate = joint_gate.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)

        return base_feat + motion_gate * causal_delta

class SeqXYZEncoder(nn.Module):
    def __init__(self, vel_dim=3, pos_emb_dim=63, vel_emb_dim=64, pos_emb_proj_dim=32, 
                 hidden_dim1=96, hidden_dim2=256, output_dim=128, 
                 time_step_num=1, seq_len=6, seq_xyz_knn=5,
                 use_amc_causal=False, amc_causal_mode="gated_residual", amc_causal_window=3,
                 amc_motion_gate_alpha=1.0, amc_motion_gate_temp=0.5,
                 use_tdp_semantic_encoder=False, tdp_semantic_mode="gated_residual",
                 tdp_base_time_step_num=0, tdp_gate_init_bias=-4.0,
                 tdp_debug_stats=False):
        super(SeqXYZEncoder, self).__init__()

        self.time_step_num = time_step_num
        self.use_amc_causal = bool(use_amc_causal)
        self.use_tdp_semantic_encoder = bool(use_tdp_semantic_encoder)
        self.tdp_semantic_mode = tdp_semantic_mode
        self.tdp_debug_stats = bool(tdp_debug_stats)
        self._tdp_stats = {}
        self.amc_causal_mode = amc_causal_mode
        self.amc_causal_window = max(1, int(amc_causal_window))
        self.amc_motion_gate_alpha = float(amc_motion_gate_alpha)
        self.amc_motion_gate_temp = float(amc_motion_gate_temp)
        self.pos_emb_proj = nn.Sequential(nn.Linear(pos_emb_dim, pos_emb_proj_dim), nn.ReLU())

        if self.use_tdp_semantic_encoder:
            if tdp_semantic_mode not in {"gated_residual", "baseline_residual"}:
                raise ValueError(f"Unsupported TDP semantic mode: {tdp_semantic_mode}")
            self.tdp_base_channels = int(tdp_base_time_step_num)
            if self.tdp_base_channels <= 1:
                raise ValueError(f"tdp_base_time_step_num must be > 1, got {tdp_base_time_step_num}")
            expected_channels = self.tdp_base_channels + self.tdp_base_channels + self.tdp_base_channels - 1
            if self.time_step_num != expected_channels:
                raise ValueError(
                    f"TDP semantic xyz encoder expected {expected_channels} channels "
                    f"from base={self.tdp_base_channels}, got {self.time_step_num}"
                )
            hidden_gate_dim = max(1, (vel_emb_dim + pos_emb_proj_dim) // 2)
            if self.tdp_semantic_mode == "baseline_residual":
                self.vel_encoder = nn.Sequential(
                    nn.Linear(vel_dim * seq_xyz_knn * self.tdp_base_channels, vel_emb_dim),
                    nn.ReLU(),
                )
                self.tdp_vel_encoder = nn.Sequential(
                    nn.Linear(vel_dim * seq_xyz_knn * self.tdp_base_channels, vel_emb_dim),
                    nn.ReLU(),
                )
                self.tdp_acc_encoder = nn.Sequential(
                    nn.Linear(vel_dim * seq_xyz_knn * (self.tdp_base_channels - 1), vel_emb_dim),
                    nn.ReLU(),
                )
                self.tdp_residual_mlp1 = nn.Sequential(nn.Linear(vel_emb_dim + pos_emb_proj_dim, hidden_dim1), nn.ReLU())
                self.tdp_residual_out = nn.Linear(hidden_dim1 * seq_len, output_dim)
                _zero_init_linear(self.tdp_residual_out)
            else:
                self.tdp_full_vel_encoder = nn.Sequential(
                    nn.Linear(vel_dim * seq_xyz_knn * self.tdp_base_channels, vel_emb_dim),
                    nn.ReLU(),
                    nn.Linear(vel_emb_dim, vel_emb_dim),
                    nn.ReLU(),
                )
                self.tdp_vel_encoder = nn.Sequential(
                    nn.Linear(vel_dim * seq_xyz_knn * self.tdp_base_channels, vel_emb_dim),
                    nn.ReLU(),
                    nn.Linear(vel_emb_dim, vel_emb_dim),
                    nn.ReLU(),
                )
                self.tdp_acc_encoder = nn.Sequential(
                    nn.Linear(vel_dim * seq_xyz_knn * (self.tdp_base_channels - 1), vel_emb_dim),
                    nn.ReLU(),
                    nn.Linear(vel_emb_dim, vel_emb_dim),
                    nn.ReLU(),
                )
            gate_input_dim = vel_emb_dim + pos_emb_proj_dim
            self.tdp_gate_v = nn.Sequential(
                nn.Linear(gate_input_dim, hidden_gate_dim),
                nn.ReLU(),
                nn.Linear(hidden_gate_dim, 1),
                nn.Sigmoid(),
            )
            self.tdp_gate_a = nn.Sequential(
                nn.Linear(gate_input_dim, hidden_gate_dim),
                nn.ReLU(),
                nn.Linear(hidden_gate_dim, 1),
                nn.Sigmoid(),
            )
            _init_last_linear_bias(self.tdp_gate_v, tdp_gate_init_bias)
            _init_last_linear_bias(self.tdp_gate_a, tdp_gate_init_bias)
        else:
            self.vel_encoder = nn.Sequential(nn.Linear(vel_dim * seq_xyz_knn * time_step_num, vel_emb_dim), nn.ReLU())

        self.mlp1 = nn.Sequential(nn.Linear(vel_emb_dim+pos_emb_proj_dim, hidden_dim1), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Linear(hidden_dim1*seq_len, hidden_dim2), nn.ReLU(),
                                  nn.Linear(hidden_dim2, output_dim), nn.ReLU())

        if self.use_amc_causal:
            if self.amc_causal_mode != "gated_residual":
                raise ValueError(f"Unsupported AMC causal mode: {self.amc_causal_mode}")
            if self.use_tdp_semantic_encoder:
                raise ValueError("AMC causal and TDP semantic encoder are mutually exclusive.")
            self.amc_vel_encoder = nn.Sequential(nn.Linear(vel_dim * seq_xyz_knn, vel_emb_dim), nn.ReLU())
            self.amc_mlp1 = nn.Sequential(nn.Linear(vel_emb_dim + pos_emb_proj_dim, hidden_dim1), nn.ReLU())
            self.amc_gru = nn.GRU(hidden_dim1, hidden_dim1, batch_first=True)
            self.amc_out = nn.Linear(hidden_dim1, output_dim)
            nn.init.zeros_(self.amc_out.weight)
            nn.init.zeros_(self.amc_out.bias)

    def _record_tdp_stats(self, values):
        if not self.tdp_debug_stats:
            return
        with torch.no_grad():
            self._tdp_stats["count"] = self._tdp_stats.get("count", 0) + 1
            for key, value in values.items():
                if torch.is_tensor(value):
                    value = value.detach().float().mean().item()
                self._tdp_stats[key] = self._tdp_stats.get(key, 0.0) + float(value)

    def pop_tdp_stats(self):
        count = int(self._tdp_stats.get("count", 0))
        if count <= 0:
            return {}
        stats = {key: value / count for key, value in self._tdp_stats.items() if key != "count"}
        self._tdp_stats = {}
        return stats

    def forward(self, x, x_emb):
        # x -> B, N, T, KNN, DeltaStep, C
        B, N, T = x.shape[0], x.shape[1], x.shape[2]
        if x.shape[4] != self.time_step_num:
            raise RuntimeError(
                f"SeqXYZEncoder expected {self.time_step_num} motion channels, "
                f"got {x.shape[4]} with shape {tuple(x.shape)}"
            )

        pos_feat = self.pos_emb_proj(x_emb)
        pos_feat = pos_feat.unsqueeze(2).expand(-1, -1, T, -1)
        if self.use_tdp_semantic_encoder:
            base = self.tdp_base_channels
            full = x[:, :, :, :, :base, :].reshape(B, N, T, -1)
            vel = x[:, :, :, :, base:base * 2, :].reshape(B, N, T, -1)
            acc = x[:, :, :, :, base * 2:, :].reshape(B, N, T, -1)
            if self.tdp_semantic_mode == "baseline_residual":
                f_full = self.vel_encoder(full)
            else:
                f_full = self.tdp_full_vel_encoder(full)
            f_vel = self.tdp_vel_encoder(vel)
            f_acc = self.tdp_acc_encoder(acc)
            gate_input = torch.concat([f_full, pos_feat], dim=-1)
            gate_v = self.tdp_gate_v(gate_input)
            gate_a = self.tdp_gate_a(gate_input)
            residual_emb = gate_v * f_vel + gate_a * f_acc
            if self.tdp_semantic_mode == "baseline_residual":
                vel_emb = f_full
            else:
                vel_emb = f_full + residual_emb
            self._record_tdp_stats({
                "gate_v_mean": gate_v.mean(),
                "gate_v_max": gate_v.max(),
                "gate_v_active_0p1": (gate_v > 0.1).float().mean(),
                "gate_a_mean": gate_a.mean(),
                "gate_a_max": gate_a.max(),
                "gate_a_active_0p1": (gate_a > 0.1).float().mean(),
                "vel_full_norm_ratio": f_vel.norm() / f_full.norm().clamp_min(1e-6),
                "acc_full_norm_ratio": f_acc.norm() / f_full.norm().clamp_min(1e-6),
                "residual_full_norm_ratio": residual_emb.norm() / f_full.norm().clamp_min(1e-6),
            })
        else:
            vel_emb = self.vel_encoder(x.reshape(B, N, T, -1))

        h = torch.concat([vel_emb, pos_feat], dim=-1)
        h = self.mlp1(h)
        base_feat = self.mlp2(h.reshape(B, N, -1))

        if self.use_tdp_semantic_encoder and self.tdp_semantic_mode == "baseline_residual":
            residual_h = torch.concat([residual_emb, pos_feat], dim=-1)
            residual_h = self.tdp_residual_mlp1(residual_h)
            residual_feat = self.tdp_residual_out(residual_h.reshape(B, N, -1))
            self._record_tdp_stats({
                "residual_feat_base_feat_norm_ratio": residual_feat.norm() / base_feat.norm().clamp_min(1e-6),
            })
            base_feat = base_feat + residual_feat

        if not self.use_amc_causal:
            return base_feat

        causal_window = min(self.amc_causal_window, T)
        recent_short_step = x[:, :, :causal_window, :, -1, :].contiguous()
        causal_chain = torch.flip(recent_short_step, dims=[2]).contiguous()
        motion_score = torch.linalg.norm(causal_chain, dim=-1).mean(dim=3)
        interval_gate = _adaptive_motion_gate(
            motion_score,
            alpha=self.amc_motion_gate_alpha,
            temp=self.amc_motion_gate_temp,
            reduce_dims=(1, 2),
        )
        gated_chain = causal_chain * interval_gate.unsqueeze(-1).unsqueeze(-1)
        causal_vel = self.amc_vel_encoder(gated_chain.reshape(B, N, causal_window, -1))
        causal_pos = pos_feat[:, :, :causal_window, :]
        causal_h = self.amc_mlp1(torch.concat([causal_vel, causal_pos], dim=-1))
        causal_h = causal_h.reshape(B * N, causal_window, -1)
        _, causal_hidden = self.amc_gru(causal_h)
        causal_delta = self.amc_out(causal_hidden.squeeze(0)).reshape(B, N, -1)
        motion_gate = interval_gate.mean(dim=2, keepdim=True)

        return base_feat + motion_gate * causal_delta
