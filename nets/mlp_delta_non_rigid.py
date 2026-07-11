import copy

import torch
import torch.nn as nn

from nets.mlp_temporal_state import TemporalStateEncoder


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
                 use_state=False, state_dim=64,
                 state_hidden_dim=128, state_layers=3,
                 state_film=True, use_state_warm=False,
                 state_start_iter=1500, state_ramp_iter=3000,
                 state_max_alpha=1.0,
                 state_identity_init=True,
                 use_state_gate=False, state_gate_hidden_dim=128,
                 state_gate_bias=-1.0):
        super(NonrigidDeformer, self).__init__()

        self.use_pose_cond = use_pose_cond
        self.use_seq_pose_cond = use_seq_pose_cond
        self.use_seq_xyz_cond = use_seq_xyz_cond
        self.use_state = bool(use_state)
        self.use_state_warm = bool(use_state_warm)
        self.state_film = bool(state_film)
        self.state_start_iter = int(state_start_iter)
        self.state_ramp_iter = int(state_ramp_iter)
        self.state_max_alpha = float(state_max_alpha)
        self.state_identity_init = bool(state_identity_init)
        self.use_state_gate = bool(use_state_gate)
        self.state_gate_hidden_dim = int(state_gate_hidden_dim)
        self.state_gate_bias = float(state_gate_bias)
        self.use_part_moe = use_part_moe
        self.num_parts = num_parts
        self.part_moe_global_keep = part_moe_global_keep
        self.part_moe_active = False
        self.part_experts = None

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

        if self.use_state:
            state_input_dim = 3 * (N_JOINT[smpl_type] + 1) * time_step_num
            self.StateEncoder = TemporalStateEncoder(
                input_dim=state_input_dim,
                hidden_dim=state_hidden_dim,
                output_dim=state_dim,
                num_layers=state_layers,
            )
            self.state_film_layer = nn.Linear(state_dim, D * W * 2)
            self.state_layers = nn.ModuleList()
            in_dim = self.input_ch
            for _ in range(D):
                self.state_layers.append(nn.Linear(in_dim, W))
                in_dim = W

        if self.use_state_warm:
            state_input_dim = 3 * (N_JOINT[smpl_type] + 1) * time_step_num
            self.StateWarmEncoder = TemporalStateEncoder(
                input_dim=state_input_dim,
                hidden_dim=state_hidden_dim,
                output_dim=state_dim,
                num_layers=state_layers,
            )

            baseline_linear_layers = [m for m in self.mlp if isinstance(m, nn.Linear)]
            self.state_warm_layers = nn.ModuleList()
            for layer in baseline_linear_layers:
                new_layer = nn.Linear(layer.in_features, layer.out_features)
                if self.state_identity_init:
                    new_layer.weight.data.copy_(layer.weight.data)
                    new_layer.bias.data.copy_(layer.bias.data)
                self.state_warm_layers.append(new_layer)

            self.state_warm_film = nn.Linear(state_dim, len(self.state_warm_layers) * W * 2)
            nn.init.zeros_(self.state_warm_film.weight)
            nn.init.zeros_(self.state_warm_film.bias)

            if self.use_state_gate:
                self.state_gate = nn.Sequential(
                    nn.Linear(self.input_ch + state_dim, self.state_gate_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.state_gate_hidden_dim, 1),
                )
                nn.init.zeros_(self.state_gate[-1].weight)
                nn.init.constant_(self.state_gate[-1].bias, self.state_gate_bias)

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
                state_conds=None, iteration=None, part_label=None, part_enabled=False,
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
                part_moe_alpha=part_moe_alpha,
                part_moe_global_keep=part_moe_global_keep,
            )

        if (
            self.use_state_warm
            and iteration is not None
            and int(iteration) >= self.state_start_iter
        ):
            h_base = self.mlp(features)
            if state_conds is None:
                state_conds = seq_pose_conds
            if state_conds is None:
                raise RuntimeError("state_conds or seq_pose_conds is required when use_state_warm=True.")
            B = state_conds.shape[0]
            L = state_conds.shape[1]
            state_seq = state_conds.reshape(B, L, -1)
            state = self.StateWarmEncoder(state_seq)
            film = self.state_warm_film(state)
            film = film.view(B, len(self.state_warm_layers), 2, -1)

            h_state = features
            for i, layer in enumerate(self.state_warm_layers):
                h_state = layer(h_state)
                gamma = 1.0 + film[:, i, 0].unsqueeze(1)
                beta = film[:, i, 1].unsqueeze(1)
                h_state = gamma * h_state + beta
                h_state = torch.relu(h_state)

            if self.state_ramp_iter > 0:
                alpha = float(int(iteration) - self.state_start_iter) / float(self.state_ramp_iter)
                alpha = max(0.0, min(self.state_max_alpha, alpha))
            else:
                alpha = self.state_max_alpha
            if self.use_state_gate:
                state_per_point = state.unsqueeze(1).expand(-1, features.shape[1], -1)
                gate = torch.sigmoid(self.state_gate(torch.cat([features, state_per_point], dim=-1)))
                h = h_base + alpha * gate * (h_state - h_base)
            else:
                h = (1.0 - alpha) * h_base + alpha * h_state
        elif self.use_state:
            if state_conds is None:
                state_conds = seq_pose_conds
            if state_conds is None:
                raise RuntimeError("state_conds or seq_pose_conds is required when use_state=True.")
            B = state_conds.shape[0]
            L = state_conds.shape[1]
            state_seq = state_conds.reshape(B, L, -1)
            state = self.StateEncoder(state_seq)
            film = self.state_film_layer(state)
            film = film.view(B, len(self.state_layers), 2, -1)

            h = features
            for i, layer in enumerate(self.state_layers):
                h = layer(h)
                gamma = film[:, i, 0].unsqueeze(1)
                beta = film[:, i, 1].unsqueeze(1)
                if self.state_film:
                    h = gamma * h + beta
                else:
                    h = h + beta
                h = torch.relu(h)
        else:
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
