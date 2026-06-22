# 主要作用：
# 在 train.py 的同一个训练进程内，于指定迭代步生成高斯点 part label。
# 当前实现是 prior-only：用 canonical 高斯点到 SMPL-X canonical 顶点的最近邻关系分层。
# 生成标签后立即加载到 GaussianModel，随后 train.py 会复制 shared MLP 为 Part-MoE experts。

import numpy as np
import torch
from scipy.spatial import cKDTree

from part_label.common import (
    LABELS,
    SOURCE_LABELS,
    default_part_label_dir,
    label_counts,
    smpl_vertex_segmentation_labels,
    smplx_lbs_vertex_labels,
    write_colored_ply,
    write_json,
)


# 空控制器：未开启 --use_part_moe 时保持训练流程不变。
class NoOpPartMoeController:
    # 每轮训练后的回调入口；空实现表示不做任何额外操作。
    def after_iteration(self, iteration, scene, gaussians):
        return


# Part-MoE 分层控制器：只在 part_moe_start_iter 这一步生成并加载 part label。
class PartMoeController:
    # 保存运行参数和触发分层的迭代步。
    def __init__(self, args):
        self.args = args
        self.loaded = False
        self.start_iter = int(getattr(args, "part_moe_start_iter", 15000))

    # train.py 每轮 optimizer step 后调用；到达指定步数时执行分层。
    def after_iteration(self, iteration, scene, gaussians):
        if self.loaded:
            return
        if iteration != self.start_iter:
            return

        if self.args.part_grouping_mode not in ("prior_only", "smpl_vertex_seg"):
            raise ValueError(
                "The in-process Part-MoE controller currently supports "
                "part_grouping_mode=prior_only or smpl_vertex_seg."
            )

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

    # 根据 SMPL-X 最近邻顶点标签生成 gaussian_part_label.npy 等分层文件。
    def _build_prior_only_labels(self, out_dir, iteration, gaussians, model_path):
        gaussian_xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
        canon_vertices = gaussians.canon_vertices.detach().cpu().numpy().reshape(-1, 3).astype(np.float32)

        vertex_labels, seg_meta = self._build_vertex_labels(gaussians, canon_vertices.shape[0])
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
                    "part_grouping_mode": str(getattr(self.args, "part_grouping_mode", "prior_only")),
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

    # 根据当前数据集/命令参数选择顶点分层来源：DNA/SMPL-X 走旧 LBS 逻辑，I3D/SMPL 可走 JSON 顶点分区。
    def _build_vertex_labels(self, gaussians, num_vertices):
        mode = str(getattr(self.args, "part_grouping_mode", "prior_only"))
        smpl_type = str(getattr(self.args, "smpl_type", "smplx"))
        if mode == "smpl_vertex_seg":
            if smpl_type != "smpl":
                raise ValueError("part_grouping_mode=smpl_vertex_seg is intended for smpl models only.")
            seg_path = getattr(self.args, "smpl_vertex_seg_path", "")
            return smpl_vertex_segmentation_labels(seg_path, num_vertices=num_vertices)
        return smplx_lbs_vertex_labels(gaussians.SMPL_NEUTRAL)


# 工厂函数：train.py 只通过这个入口创建 Part-MoE 控制器。
def build_part_moe_controller(args):
    if not getattr(args, "use_part_moe", False):
        return NoOpPartMoeController()
    return PartMoeController(args)
