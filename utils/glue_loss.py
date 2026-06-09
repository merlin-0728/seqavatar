import torch
import torch.nn as nn
import torch.nn.functional as F


def build_projected_coord_colors(viewpoint_camera, means3D, clamp=True):
    """Build per-Gaussian [u, v, 0] colors in normalized image coordinates."""
    ones = torch.ones((means3D.shape[0], 1), dtype=means3D.dtype, device=means3D.device)
    means_h = torch.cat([means3D, ones], dim=1)

    clip = means_h @ viewpoint_camera.full_proj_transform
    w = clip[:, 3:4]
    safe_w = torch.where(torch.abs(w) > 1e-8, w, torch.ones_like(w) * 1e-8)
    ndc = clip[:, :3] / safe_w

    u = (ndc[:, 0] + 1.0) * 0.5
    v = (ndc[:, 1] + 1.0) * 0.5
    z = torch.zeros_like(u)
    colors = torch.stack([u, v, z], dim=1)

    if clamp:
        colors = torch.clamp(colors, 0.0, 1.0)
    return colors


class FeatureMatchingLoss(nn.Module):
    def __init__(self, max_num_keypoints=1024, device="cuda", alpha_threshold=0.9):
        super().__init__()
        from lightglue import LightGlue, SuperPoint

        self.device = torch.device(device)
        self.alpha_threshold = alpha_threshold
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

        for module in (self.extractor, self.matcher):
            for param in module.parameters():
                param.requires_grad_(False)

    @staticmethod
    def _as_bchw(image):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return torch.clamp(image, 0.0, 1.0)

    @staticmethod
    def _parse_matches(matches_data, feats0):
        if "matches" in matches_data:
            matches = matches_data["matches"][0]
            scores = matches_data.get("scores", [None])[0]
            if scores is None:
                scores = torch.ones((matches.shape[0],), device=matches.device, dtype=torch.float32)
            return matches, scores

        matches0 = matches_data["matches0"][0]
        valid = matches0 > -1
        idx0 = torch.where(valid)[0]
        matches = torch.stack([idx0, matches0[valid]], dim=1)

        if "matching_scores0" in matches_data:
            scores = matches_data["matching_scores0"][0][valid]
        else:
            scores = torch.ones((matches.shape[0],), device=feats0["keypoints"].device, dtype=torch.float32)
        return matches, scores

    def forward(self, pred_rgb, gt_rgb, proj_rgb, proj_mask, fg_mask=None):
        pred_rgb = self._as_bchw(pred_rgb).to(self.device)
        gt_rgb = self._as_bchw(gt_rgb).to(self.device)

        if proj_rgb.ndim == 3:
            proj_rgb = proj_rgb.unsqueeze(0)
        if proj_mask.ndim == 2:
            proj_mask = proj_mask.unsqueeze(0).unsqueeze(0)
        elif proj_mask.ndim == 3:
            proj_mask = proj_mask.unsqueeze(0)

        proj_rgb = proj_rgb.to(self.device)
        proj_mask = proj_mask.to(self.device)
        _, _, height, width = proj_rgb.shape

        with torch.no_grad():
            feats_pred = self.extractor.extract(pred_rgb)
            feats_gt = self.extractor.extract(gt_rgb)
            matches_data = self.matcher({"image0": feats_pred, "image1": feats_gt})
            matches, scores = self._parse_matches(matches_data, feats_pred)
            kpts_pred = feats_pred["keypoints"][0]
            kpts_gt = feats_gt["keypoints"][0]

        if matches.shape[0] == 0:
            return proj_rgb.new_zeros(())

        matched_pred = kpts_pred[matches[:, 0]]
        grid_x = ((matched_pred[:, 0] + 0.5) / width) * 2.0 - 1.0
        grid_y = ((matched_pred[:, 1] + 0.5) / height) * 2.0 - 1.0
        sampling_grid = torch.stack([grid_x, grid_y], dim=1).unsqueeze(0).unsqueeze(0)

        sampled_premult = F.grid_sample(
            proj_rgb[:, :2],
            sampling_grid,
            mode="bilinear",
            align_corners=False,
            padding_mode="zeros",
        ).view(2, -1).T

        sampled_alpha = F.grid_sample(
            proj_mask[:, :1],
            sampling_grid,
            mode="bilinear",
            align_corners=False,
            padding_mode="zeros",
        ).view(1, -1).T

        sampled_coords = sampled_premult / (sampled_alpha + 1e-6)
        matched_gt = kpts_gt[matches[:, 1]]
        target_coords = torch.stack(
            [
                (matched_gt[:, 0] + 0.5) / width,
                (matched_gt[:, 1] + 0.5) / height,
            ],
            dim=1,
        )

        valid = (sampled_alpha[:, 0] > self.alpha_threshold)
        if fg_mask is not None:
            if fg_mask.ndim == 3:
                fg_mask = fg_mask[0]
            fg_mask = fg_mask.to(device=self.device, dtype=proj_rgb.dtype)
            sampled_fg = F.grid_sample(
                fg_mask[None, None],
                sampling_grid,
                mode="nearest",
                align_corners=False,
                padding_mode="zeros",
            ).view(-1)
            valid = valid & (sampled_fg > 0.5)

        if not torch.any(valid):
            return proj_rgb.new_zeros(())

        valid_scores = scores.to(self.device)[valid].unsqueeze(1)
        coord_loss = torch.abs(sampled_coords[valid] - target_coords[valid]) * valid_scores
        return coord_loss.sum() / (valid_scores.sum() + 1e-6)
