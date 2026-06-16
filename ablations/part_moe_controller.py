import numpy as np
import torch
from scipy.spatial import cKDTree

from part_label.common import (
    LABELS,
    SOURCE_LABELS,
    default_part_label_dir,
    label_counts,
    smplx_lbs_vertex_labels,
    write_colored_ply,
    write_json,
)


class NoOpPartMoeController:
    def after_iteration(self, iteration, scene, gaussians):
        return


class PartMoeController:
    def __init__(self, args):
        self.args = args
        self.loaded = False
        self.start_iter = int(getattr(args, "part_moe_start_iter", 15000))

    def after_iteration(self, iteration, scene, gaussians):
        if self.loaded:
            return
        if iteration != self.start_iter:
            return

        if self.args.part_grouping_mode != "prior_only":
            raise ValueError("The in-process Part-MoE controller currently supports only prior_only.")

        print(f"\n[PartMoE] Build labels at iteration {iteration}.")
        torch.cuda.synchronize()

        # Overwrite iteration_{start_iter} after the optimizer step so labels
        # match the exact Gaussian tensor that will keep training.
        scene.save(iteration)
        torch.cuda.synchronize()

        out_dir = default_part_label_dir(scene.model_path, iteration)
        out_dir.mkdir(parents=True, exist_ok=True)
        label_path, conf_path = self._build_prior_only_labels(out_dir, iteration, gaussians, scene.model_path)
        gaussians.load_part_labels(label_path, conf_path)
        self.loaded = True
        torch.cuda.synchronize()
        print("[PartMoE] Part labels loaded; continue training.\n")

    def _build_prior_only_labels(self, out_dir, iteration, gaussians, model_path):
        gaussian_xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
        canon_vertices = gaussians.canon_vertices.detach().cpu().numpy().reshape(-1, 3).astype(np.float32)

        vertex_labels, seg_meta = smplx_lbs_vertex_labels(gaussians.SMPL_NEUTRAL)
        tree = cKDTree(canon_vertices)
        smpl_dist, vert_ids = tree.query(gaussian_xyz, k=1)
        smpl_prior = vertex_labels[vert_ids].astype(np.uint8)
        smpl_dist = smpl_dist.astype(np.float32)

        max_dist = float(self.args.part_max_smpl_dist)
        final = np.zeros_like(smpl_prior, dtype=np.uint8)
        conf = np.zeros(smpl_prior.shape[0], dtype=np.float32)
        source = np.zeros(smpl_prior.shape[0], dtype=np.uint8)

        prior_valid = (smpl_prior > 0) & (smpl_prior <= 4) & (smpl_dist < max_dist)
        final[prior_valid] = smpl_prior[prior_valid]
        conf[prior_valid] = np.clip(1.0 - smpl_dist[prior_valid] / max_dist, 0.0, 1.0)
        source[prior_valid] = 1
        vote = np.zeros((smpl_prior.shape[0], 7), dtype=np.uint32)

        label_path = out_dir / "gaussian_part_label.npy"
        conf_path = out_dir / "gaussian_part_conf.npy"
        np.save(out_dir / "gaussian_smpl_prior_label.npy", smpl_prior)
        np.save(out_dir / "gaussian_smpl_dist.npy", smpl_dist)
        np.save(label_path, final)
        np.save(conf_path, conf)
        np.save(out_dir / "gaussian_part_vote.npy", vote)
        np.save(out_dir / "gaussian_part_source.npy", source)

        write_colored_ply(out_dir / "debug_smpl_prior.ply", gaussian_xyz, smpl_prior)
        write_colored_ply(out_dir / "gaussian_part_debug.ply", gaussian_xyz, final)

        common_meta = {
            "model_path": model_path,
            "iteration": iteration,
            "out_dir": str(out_dir),
            "num_gaussians": int(gaussian_xyz.shape[0]),
            "labels": LABELS,
            "segmentation": seg_meta,
        }
        write_json(
            out_dir / "smpl_prior_meta.json",
            {
                **common_meta,
                "label_counts": label_counts(smpl_prior),
                "distance_stats": {
                    "min": float(np.min(smpl_dist)),
                    "mean": float(np.mean(smpl_dist)),
                    "median": float(np.median(smpl_dist)),
                    "max": float(np.max(smpl_dist)),
                },
            },
        )
        write_json(
            out_dir / "gaussian_part_meta.json",
            {
                **common_meta,
                "source_labels": SOURCE_LABELS,
                "label_counts": label_counts(final),
                "source_counts": {SOURCE_LABELS[i]: int(np.sum(source == i)) for i in SOURCE_LABELS},
                "params": {
                    "prior_only": True,
                    "max_smpl_dist": max_dist,
                    "use_part_moe": True,
                    "part_moe_start_iter": int(getattr(self.args, "part_moe_start_iter", self.start_iter)),
                    "num_parts": int(self.args.num_parts),
                },
            },
        )

        print("[PartMoE] Saved Gaussian part labels:")
        print(f"  {label_path}")
        print(f"  counts: {label_counts(final)}")
        return str(label_path), str(conf_path)


def build_part_moe_controller(args):
    if not getattr(args, "use_part_moe", False):
        return NoOpPartMoeController()
    return PartMoeController(args)
