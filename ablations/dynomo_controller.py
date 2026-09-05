from pathlib import Path

from ablations.part_moe_controller import PartMoeController


class DynOMoController:
    def __init__(self, args):
        self.args = args
        self.last_iter = None
        self.last_num_gaussians = None

    def refresh(self, scene, gaussians, iteration):
        if not getattr(self.args, "use_dynomo_c", False):
            return
        iteration = int(iteration)
        label_iter = int(getattr(self.args, "dynomo_c_label_iter", 0))
        if iteration < label_iter:
            return

        num_gaussians = int(gaussians.get_xyz.shape[0])
        if self.last_iter == iteration and self.last_num_gaussians == num_gaussians:
            return
        if self.last_num_gaussians == num_gaussians and gaussians.part_label_enabled:
            self.last_iter = iteration
            return

        builder = PartMoeController(self.args)
        out_dir = Path(scene.model_path) / "dynomo_labels" / f"iteration_{iteration}"
        out_dir.mkdir(parents=True, exist_ok=True)
        label_path, conf_path = builder._build_prior_only_labels(out_dir, iteration, gaussians, scene.model_path)
        gaussians.load_part_labels(label_path, conf_path)
        self.last_iter = iteration
        self.last_num_gaussians = num_gaussians


def build_dynomo_controller(args):
    if not getattr(args, "use_dynomo_c", False):
        return None
    return DynOMoController(args)
