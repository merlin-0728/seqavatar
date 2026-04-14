import torch
import torch.nn as nn

class NonrigidDeformer(nn.Module):
    def __init__(self, D=3, W=512, use_pose_cond=0, use_seq_pose_cond=0, use_seq_xyz_cond=0, 
                 pos_input_dim=63, pose_cond_dim=32, seq_pose_cond_dim=32, seq_xyz_cond_dim=96,
                 seq_len=6, seq_xyz_knn=1, time_step_num=1, smpl_type='smpl',
                 use_label_cond=1, label_emb_dim=8, vg_feat_dim=1,
                 use_dual_source_branch=0, source_branch_width=128):
        super(NonrigidDeformer, self).__init__()

        self.use_pose_cond = use_pose_cond
        self.use_seq_pose_cond = use_seq_pose_cond
        self.use_seq_xyz_cond = use_seq_xyz_cond
        self.use_label_cond = use_label_cond
        self.vg_feat_dim = vg_feat_dim
        self.use_dual_source_branch = use_dual_source_branch
        self._phase1_active = None

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

        if self.use_label_cond:
            self.label_embedding = nn.Embedding(2, label_emb_dim)
            self.input_ch += label_emb_dim

        if self.vg_feat_dim > 0:
            self.input_ch += vg_feat_dim

        if self.use_dual_source_branch:
            self.smpl_branch = nn.Sequential(nn.Linear(self.input_ch, source_branch_width), nn.ReLU())
            self.vggt_branch = nn.Sequential(nn.Linear(self.input_ch, source_branch_width), nn.ReLU())
            in_dim = source_branch_width
        else:
            in_dim = self.input_ch
        
        layers = []
        for _ in range(D):
            layers.append(nn.Linear(in_dim, W))
            layers.append(nn.ReLU())
            in_dim = W
        self.mlp = nn.Sequential(*layers)

        self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)

    def set_phase(self, phase1_active):
        """Phase-1: freeze SMPL branch, keep VGGT branch trainable."""
        if self._phase1_active == phase1_active:
            return

        if self.use_pose_cond:
            for param in self.PoseEncoder.parameters():
                param.requires_grad = not phase1_active
        if self.use_seq_pose_cond:
            for param in self.SeqPoseEncoder.parameters():
                param.requires_grad = not phase1_active

        if self.use_dual_source_branch:
            for param in self.smpl_branch.parameters():
                param.requires_grad = not phase1_active
            for param in self.vggt_branch.parameters():
                param.requires_grad = True
        self._phase1_active = phase1_active

    def forward(
        self,
        x_emb,
        pose_conds=None,
        seq_pose_conds=None,
        seq_xyz_conds=None,
        point_labels=None,
        vg_feat=None,
    ):
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

        if self.use_label_cond:
            if point_labels is None:
                point_labels = torch.zeros(
                    x_emb.shape[0], x_emb.shape[1], dtype=torch.long, device=x_emb.device
                )
            label_feats = self.label_embedding(point_labels.long().clamp(min=0, max=1))
            feats.append(label_feats)

        if self.vg_feat_dim > 0:
            if vg_feat is None:
                vg_feat = torch.zeros(
                    x_emb.shape[0], x_emb.shape[1], self.vg_feat_dim, dtype=x_emb.dtype, device=x_emb.device
                )
            feats.append(vg_feat.float())

        input_feat = torch.cat(feats, dim=-1)
        if self.use_dual_source_branch:
            if point_labels is None:
                point_labels = torch.zeros(
                    x_emb.shape[0], x_emb.shape[1], dtype=torch.long, device=x_emb.device
                )
            source_mask = point_labels.float().unsqueeze(-1)  # 0=SMPL, 1=VGGT
            h_smpl = self.smpl_branch(input_feat)
            h_vggt = self.vggt_branch(input_feat)
            h = h_smpl * (1.0 - source_mask) + h_vggt * source_mask
        else:
            h = input_feat

        h = self.mlp(h)
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
