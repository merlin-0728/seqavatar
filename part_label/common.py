# 主要作用：
# 保存高斯点分层和 Part-MoE 日志所需的通用工具。
# 当前最终 use_part_moe 流程依赖这里的标签定义、日志重定向、JSON/PLY 写入、
# 默认 part label 输出目录，以及 SMPL-X LBS 权重到 body/hand/face 标签的映射。

import ast
import atexit
import datetime
import json
import os
import pickle
import re
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


REPO_ROOT = Path(__file__).resolve().parents[1]
PART_LOG_ROOT = REPO_ROOT / "logs" / "part"
TRI_LOG_ROOT = REPO_ROOT / "logs" / "tri"
PART_BUDGET_LOG_ROOT = REPO_ROOT / "logs" / "budget"

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

PART_LABEL_SCHEMAS = {
    "anatomy5": {
        0: "unknown",
        1: "body",
        2: "left_hand",
        3: "right_hand",
        4: "face",
    },
    "part_moe_leg": {
        0: "unknown",
        1: "body",
        2: "left_hand",
        3: "right_hand",
        4: "face",
        5: "left_leg_foot",
        6: "right_leg_foot",
    },
}

PART_LABEL_COLORS = {
    "anatomy5": {idx: LABEL_COLORS[idx] for idx in PART_LABEL_SCHEMAS["anatomy5"]},
    "part_moe_leg": {
        0: LABEL_COLORS[0],
        1: LABEL_COLORS[1],
        2: LABEL_COLORS[2],
        3: LABEL_COLORS[3],
        4: LABEL_COLORS[4],
        5: (255, 165, 0),
        6: (128, 0, 255),
    },
}

SOURCE_LABELS = {
    0: "unknown",
    1: "smpl_prior",
    2: "semantic_vote",
    3: "semantic_hair_override",
    4: "semantic_cloth_override",
    5: "hand_split_by_smpl",
}


# 返回当前消融 schema 的 part 名称。
def get_part_label_names(schema="anatomy5"):
    if schema not in PART_LABEL_SCHEMAS:
        raise ValueError(f"Unknown part_label_schema: {schema}")
    return PART_LABEL_SCHEMAS[schema]


# 返回当前消融 schema 的可视化颜色。
def get_part_label_colors(schema="anatomy5"):
    if schema not in PART_LABEL_COLORS:
        raise ValueError(f"Unknown part_label_schema: {schema}")
    return PART_LABEL_COLORS[schema]


# 返回当前 schema 应该使用的 expert 数量。
def get_num_parts_from_schema(schema="anatomy5"):
    return len(get_part_label_names(schema))


# 简单 tee 输出流：把 stdout/stderr 同时写到终端和 part 日志文件。
class _TeeStream:
    # 保存多个输出目标。
    def __init__(self, *streams):
        self.streams = streams

    # 写入一段日志文本到所有输出目标。
    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except ValueError:
                pass
        return len(data)

    # 刷新所有输出目标。
    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except ValueError:
                pass

    # 保留终端交互状态判断。
    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    # 返回底层文件描述符，兼容部分日志库。
    def fileno(self):
        return self.streams[0].fileno()


_ACTIVE_TEE_FILES = []


# 把路径片段转换成适合作为日志文件名的安全字符串。
def safe_log_component(value):
    text = str(value).strip().strip(os.sep)
    if not text:
        return "run"
    text = text.replace(os.sep, "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


# 解析 part 日志目录；默认写到 logs/part。
def resolve_part_log_dir(value=None):
    path = Path(value).expanduser() if value else PART_LOG_ROOT
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


# 根据 model_path/exp_name 和脚本名生成日志文件名前缀。
def infer_model_log_stem(args, script_name):
    model_path = getattr(args, "model_path", "") or ""
    exp_name = getattr(args, "exp_name", "") or ""
    if model_path:
        try:
            model_rel = Path(model_path).resolve().relative_to(REPO_ROOT / "output")
        except ValueError:
            model_rel = Path(model_path).resolve()
    elif exp_name:
        model_rel = Path(exp_name)
    else:
        model_rel = Path("manual")

    mode = "part"
    if getattr(args, "use_part_budget", False):
        budget_mode = str(getattr(args, "part_budget_mode", "base")).lower()
        mode = f"part_budget_{budget_mode}"
    elif getattr(args, "use_tri", False):
        mode = "tri"
    elif getattr(args, "use_part_moe", False):
        mode = "part_moe"

    bits = [script_name, mode, *model_rel.parts]
    return safe_log_component("_".join(str(bit) for bit in bits if str(bit)))


# 开启 part 相关日志重定向；只有 use_part_moe 或 force=True 时生效。
def enable_part_stdout_logging(args, script_name, force=False):
    if not force and not getattr(args, "use_part_moe", False):
        return None

    if getattr(args, "use_part_budget", False):
        default_log_dir = PART_BUDGET_LOG_ROOT
    elif getattr(args, "use_tri", False):
        default_log_dir = TRI_LOG_ROOT
    else:
        default_log_dir = PART_LOG_ROOT
    log_dir = resolve_part_log_dir(getattr(args, "part_log_dir", None) or default_log_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = infer_model_log_stem(args, script_name)
    log_path = log_dir / f"{stem}_{timestamp}.log"
    log_file = log_path.open("w", encoding="utf-8")
    _ACTIVE_TEE_FILES.append(log_file)

    sys.stdout = _TeeStream(sys.stdout, log_file)
    sys.stderr = _TeeStream(sys.stderr, log_file)

    def _close_log_file(file_obj=log_file):
        try:
            file_obj.flush()
            file_obj.close()
        except Exception:
            pass

    atexit.register(_close_log_file)
    print(f"[PART_LOG] {script_name} log: {log_path}", flush=True)
    return log_path


# 设置脚本运行环境：GPU、PATH、sys.path 和工作目录。
def setup_repo(gpu=None):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    python_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.chdir(repo)


# 解析 SeqAvatar 输出目录里的 cfg_args Namespace 文本。
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


# 读取训练输出目录中的 cfg_args，并补齐 source_path/data_device。
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


# 把 argparse Namespace 转成可写 JSON 的 dict。
def namespace_to_dict(ns):
    return {k: _jsonable(v) for k, v in vars(ns).items()}


# 把 Path、numpy 数组和 numpy 标量递归转换成 JSON 兼容对象。
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


# 写 JSON 文件，自动创建父目录。
def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, ensure_ascii=False))


# 从 PLY 文件读取 xyz 坐标。
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


# 按 part label 给点云上色并写出 PLY，方便检查分层结果。
def write_colored_ply(path, xyz, labels, schema="anatomy5"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    label_colors = get_part_label_colors(schema)
    colors = np.array([label_colors.get(int(l), label_colors[0]) for l in labels], dtype=np.uint8)
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


# 返回某个模型迭代步的 point_cloud.ply 路径。
def model_point_cloud_path(model_path, iteration):
    return Path(model_path) / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply"


# 返回默认的 part label 输出目录。
def default_part_label_dir(model_path, iteration):
    return Path(model_path) / "part_labels" / f"iteration_{int(iteration)}"


# 加载 SMPL-X neutral 模型参数。
def load_smpl_neutral(smpl_type="smplx", actor_gender="neutral"):
    if smpl_type != "smplx":
        raise ValueError("This part-label module currently expects SMPL-X")
    path = REPO_ROOT / "smpl_model" / "models" / f"SMPLX_{actor_gender.upper()}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


# 用 SMPL-X LBS 主导关节把每个 SMPL-X 顶点粗分为 body/hand/face，可选额外分左右腿脚。
def smplx_lbs_vertex_labels(smpl_neutral, schema="anatomy5"):
    get_part_label_names(schema)
    weights_obj = smpl_neutral["weights"]
    if hasattr(weights_obj, "detach"):
        weights_obj = weights_obj.detach().cpu().numpy()
    weights = np.asarray(weights_obj)
    dominant = np.argmax(weights, axis=1)
    # 兼容不同 SMPL-X pickle 中 joint2num/part2num 的保存格式。
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

    # 找出指定前缀对应的 SMPL-X 关节/部位 id。
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
    left_leg_foot = ids_with_prefix(["L_Hip", "L_Knee", "L_Ankle", "L_Foot", "L_Toes", "L_Thigh", "L_Calf"])
    right_leg_foot = ids_with_prefix(["R_Hip", "R_Knee", "R_Ankle", "R_Foot", "R_Toes", "R_Thigh", "R_Calf"])

    labels = np.ones(weights.shape[0], dtype=np.uint8)
    if schema == "part_moe_leg":
        labels[np.isin(dominant, list(left_leg_foot))] = 5
        labels[np.isin(dominant, list(right_leg_foot))] = 6
    labels[np.isin(dominant, list(left_hand))] = 2
    labels[np.isin(dominant, list(right_hand))] = 3
    labels[np.isin(dominant, list(face))] = 4
    return labels, {
        "method": "smplx_lbs_dominant_part_fallback",
        "part_label_schema": schema,
        "left_hand_ids": sorted(left_hand),
        "right_hand_ids": sorted(right_hand),
        "face_ids": sorted(face),
        "left_leg_foot_ids": sorted(left_leg_foot) if schema == "part_moe_leg" else [],
        "right_leg_foot_ids": sorted(right_leg_foot) if schema == "part_moe_leg" else [],
        "vertex_label_counts": label_counts(labels, schema=schema),
    }


# 把 SMPL 顶点分区名称映射成当前消融使用的 part id。
def map_smpl_seg_name_to_part_id(name, schema="anatomy5"):
    get_part_label_names(schema)
    key = str(name).lower()
    if key in {"lefthand", "lefthandindex1"}:
        return 2
    if key in {"righthand", "righthandindex1"}:
        return 3
    if key == "head":
        return 4
    if schema == "part_moe_leg":
        if key in {"leftupleg", "leftleg", "leftfoot", "lefttoebase"}:
            return 5
        if key in {"rightupleg", "rightleg", "rightfoot", "righttoebase"}:
            return 6
    return 1


# 读取 SMPL 6890 顶点分区 JSON，并映射到当前 Part-MoE schema 的标签。
def smpl_vertex_segmentation_labels(seg_path, num_vertices=6890, schema="anatomy5"):
    label_names = get_part_label_names(schema)
    seg_path = Path(seg_path).expanduser()
    if not seg_path.is_absolute():
        seg_path = REPO_ROOT / seg_path
    if not seg_path.exists():
        raise FileNotFoundError(f"Missing SMPL vertex segmentation file: {seg_path}")

    with seg_path.open("r", encoding="utf-8") as f:
        segmentation = json.load(f)
    if not isinstance(segmentation, dict):
        raise ValueError(f"SMPL vertex segmentation must be a dict: {seg_path}")

    labels = np.ones(int(num_vertices), dtype=np.uint8)

    listed_vids = []
    part_keys = {name: [] for name in label_names.values() if name != "unknown"}
    for name, vids in segmentation.items():
        vids = np.asarray(vids, dtype=np.int64)
        if vids.size == 0:
            continue
        if int(vids.min()) < 0 or int(vids.max()) >= int(num_vertices):
            raise ValueError(
                f"Invalid SMPL vertex id in {name}: "
                f"min={int(vids.min())}, max={int(vids.max())}, num_vertices={num_vertices}"
            )

        listed_vids.append(vids)
        pid = map_smpl_seg_name_to_part_id(name, schema=schema)
        labels[vids] = pid
        part_keys[label_names.get(pid, f"part_{pid}")].append(str(name))

    if listed_vids:
        all_vids = np.concatenate(listed_vids, axis=0)
        unique_vids = np.unique(all_vids)
        missing_count = int(num_vertices) - int(unique_vids.shape[0])
        duplicate_count = int(all_vids.shape[0]) - int(unique_vids.shape[0])
        listed_vertex_count = int(all_vids.shape[0])
        unique_vertex_count = int(unique_vids.shape[0])
    else:
        missing_count = int(num_vertices)
        duplicate_count = 0
        listed_vertex_count = 0
        unique_vertex_count = 0

    return labels, {
        "method": "smpl_vertex_segmentation_json",
        "path": str(seg_path),
        "part_label_schema": schema,
        "num_vertices": int(num_vertices),
        "num_keys": int(len(segmentation)),
        "part_keys": part_keys,
        "listed_vertex_count": listed_vertex_count,
        "unique_vertex_count": unique_vertex_count,
        "missing_vertex_count": missing_count,
        "duplicate_vertex_count": duplicate_count,
        "vertex_label_counts": label_counts(labels, schema=schema),
    }


# 统计每个 part label 的高斯点数量。
def label_counts(labels, schema="anatomy5"):
    labels = np.asarray(labels).astype(np.int64)
    label_names = get_part_label_names(schema)
    counts = {}
    for idx, name in label_names.items():
        counts[name] = int(np.sum(labels == idx))
    for idx in sorted(set(labels.tolist()) - set(label_names.keys())):
        counts[f"part_{idx}"] = int(np.sum(labels == idx))
    return counts


# 从 SeqAvatar 的 image_name 中解析相机 view id 和 frame id。
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
