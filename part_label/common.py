import ast
import json
import os
import pickle
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


REPO_ROOT = Path(__file__).resolve().parents[1]

LABELS = {
    0: "unknown",
    1: "body",
    2: "left_hand",
    3: "right_hand",
    4: "face",
    5: "hair",
    6: "cloth",
}

LABEL_COLORS = {
    0: (120, 120, 120),
    1: (70, 130, 220),
    2: (40, 190, 90),
    3: (230, 70, 70),
    4: (245, 185, 130),
    5: (245, 220, 55),
    6: (70, 210, 210),
}

SOURCE_LABELS = {
    0: "unknown",
    1: "smpl_prior",
    2: "semantic_vote",
    3: "semantic_hair_override",
    4: "semantic_cloth_override",
    5: "hand_split_by_smpl",
}


def setup_repo(gpu=None):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    python_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.chdir(repo)


def _namespace_eval(text):
    text = text.strip()
    if not text:
        return Namespace()
    if text.startswith("Namespace(") and text.endswith(")"):
        expr = ast.parse(text, mode="eval")
        if not isinstance(expr.body, ast.Call) or getattr(expr.body.func, "id", "") != "Namespace":
            raise ValueError("cfg_args must be an argparse Namespace")
        values = {}
        for kw in expr.body.keywords:
            values[kw.arg] = ast.literal_eval(kw.value)
        return Namespace(**values)
    raise ValueError("Unsupported cfg_args format")


def load_cfg_args(model_path, source_path=None):
    cfg_path = Path(model_path) / "cfg_args"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing cfg_args: {cfg_path}")
    cfg = _namespace_eval(cfg_path.read_text())
    cfg.model_path = str(Path(model_path).resolve())
    if source_path is not None:
        cfg.source_path = str(Path(source_path).resolve())
    else:
        cfg.source_path = str(Path(cfg.source_path).resolve())
    if not hasattr(cfg, "data_device"):
        cfg.data_device = "cuda"
    return cfg


def namespace_to_dict(ns):
    return {k: _jsonable(v) for k, v in vars(ns).items()}


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, ensure_ascii=False))


def read_ply_xyz(path):
    ply = PlyData.read(str(path))
    vertex = ply.elements[0]
    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )
    return xyz


def write_colored_ply(path, xyz, labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    colors = np.array([LABEL_COLORS.get(int(l), LABEL_COLORS[0]) for l in labels], dtype=np.uint8)
    vertex = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"] = xyz[:, 0]
    vertex["y"] = xyz[:, 1]
    vertex["z"] = xyz[:, 2]
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))


def model_point_cloud_path(model_path, iteration):
    return Path(model_path) / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply"


def default_part_label_dir(model_path, iteration):
    return Path(model_path) / "part_labels" / f"iteration_{int(iteration)}"


def load_smpl_neutral(smpl_type="smplx", actor_gender="neutral"):
    if smpl_type != "smplx":
        raise ValueError("This part-label module currently expects SMPL-X")
    path = REPO_ROOT / "smpl_model" / "models" / f"SMPLX_{actor_gender.upper()}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def smplx_lbs_vertex_labels(smpl_neutral):
    weights_obj = smpl_neutral["weights"]
    if hasattr(weights_obj, "detach"):
        weights_obj = weights_obj.detach().cpu().numpy()
    weights = np.asarray(weights_obj)
    dominant = np.argmax(weights, axis=1)
    def as_dict(value):
        if value is None:
            return {}
        if hasattr(value, "detach"):
            return {}
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        return dict(value)

    joint2num = as_dict(smpl_neutral.get("joint2num", {}))
    part2num = as_dict(smpl_neutral.get("part2num", {}))

    def ids_with_prefix(prefixes):
        out = set()
        for table in (joint2num, part2num):
            for name, idx in table.items():
                if any(name.startswith(prefix) for prefix in prefixes):
                    out.add(int(idx))
        return out

    left_hand = ids_with_prefix(["L_Hand", "L_Index", "L_Middle", "L_Ring", "L_Pinky", "L_Thumb"])
    right_hand = ids_with_prefix(["R_Hand", "R_Index", "R_Middle", "R_Ring", "R_Pinky", "R_Thumb"])
    face = ids_with_prefix(["Head", "Jaw", "L_Eye", "R_Eye"])

    labels = np.ones(weights.shape[0], dtype=np.uint8)
    labels[np.isin(dominant, list(left_hand))] = 2
    labels[np.isin(dominant, list(right_hand))] = 3
    labels[np.isin(dominant, list(face))] = 4
    return labels, {
        "method": "smplx_lbs_dominant_part_fallback",
        "left_hand_ids": sorted(left_hand),
        "right_hand_ids": sorted(right_hand),
        "face_ids": sorted(face),
    }


def label_counts(labels):
    labels = np.asarray(labels).astype(np.int64)
    counts = {}
    for idx, name in LABELS.items():
        counts[name] = int(np.sum(labels == idx))
    return counts


def parse_camera_image_name(view):
    name = getattr(view, "image_name", "")
    # SeqAvatar DNA names are frame_000000_view_00.
    frame = getattr(view, "pose_id", None)
    camera_view = None
    parts = name.split("_")
    if len(parts) >= 4 and parts[0] == "frame" and parts[2] == "view":
        try:
            frame = int(parts[1])
            camera_view = int(parts[3])
        except ValueError:
            pass
    return camera_view, frame
