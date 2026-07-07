#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import focal2fov, getNerfppNorm, BasicPointCloud
import numpy as np
import torch
import cv2
from utils.sh_utils import SH2RGB
from smpl_model.smpl.smpl_numpy import SMPL
from smpl_model.smplx.body_models import SMPLX
from utils.SMCReader import SMCReader
from utils.transformation_util import axis_angle_to_matrix, matrix_to_axis_angle
from utils.general_utils import storePly, fetchPly
from utils.smpl_utils import batch_rodrigues, create_canonical_vertices
from utils.image_utils import load_image_mask
import pickle

class CameraInfo(NamedTuple):
    uid: int
    pose_id: int
    R: np.array
    T: np.array
    K: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    bkgd_mask: np.array
    bound_mask: np.array
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    canon_params: dict
    canon_vertices: np.array
    smpl_params_dict: dict
    cond_dict: dict

##################################   ID3-Human   ##################################
ID3HUMAN_CFG = {
    'ID1_1': {'train_view': [1, 3, 6, 8], 'test_view': [2, 4, 7, 9], 'interval': 1},
    'ID1_2': {'train_view': [0, 2, 5, 7], 'test_view': [1, 6, 8, 9], 'interval': 1},
    'ID2_1': {'train_view': [1, 3, 5, 7, 9], 'test_view': [2, 4, 6, 10], 'interval': 3},
    'ID2_2': {'train_view': [1, 3, 5, 7, 9], 'test_view': [2, 4, 6, 8, 10], 'interval': 3},
    'ID3_1': {'train_view': [1, 3, 5, 7, 9], 'test_view': [2, 4, 6, 8, 10], 'interval': 3},
    'ID3_2': {'train_view': [1, 3, 5, 7, 9], 'test_view': [2, 4, 6, 8, 10], 'interval': 3},
}

def readID3HumanInfo(path, white_background, eval, time_steps, motion_cond_options=None):
    scene_name = os.path.basename(path).split('-')[0]
    if scene_name not in ID3HUMAN_CFG.keys():
        raise ValueError('Unknown dataset')
    
    # camera view splitting, follow Dyco's setting
    train_view, test_view, interval = ID3HUMAN_CFG[scene_name]['train_view'], ID3HUMAN_CFG[scene_name]['test_view'], ID3HUMAN_CFG[scene_name]['interval']
    
    # load SMPL model
    smpl_model = SMPL(sex='neutral', model_dir='smpl_model/models/')

    # build canonical space SMPL points
    canon_params, canon_vertices = create_canonical_vertices(smpl_model, 'smpl')

    # read cameras
    test_cam_infos = {}
    smpl_params_dict, cond_dict = {}, {} # observation space smpl params, conditions for non-rigid deformation
    delta_pose_xyz_cache = {} # cache for sequential condition calculation
    
    print("Reading Training Transforms")
    train_cam_infos = readCamerasI3DHuman(path, smpl_model, train_view, white_background, split='train', 
                        interval=interval, time_steps=time_steps, 
                        smpl_params_dict=smpl_params_dict, cond_dict=cond_dict,
                        delta_pose_xyz_cache=delta_pose_xyz_cache,
                        motion_cond_options=motion_cond_options)

    print("Reading Test Novelview Transforms")
    test_cam_infos['novelview'] = readCamerasI3DHuman(path, smpl_model, test_view, white_background, split='novelview', 
                                    interval=interval, time_steps=time_steps, 
                                    smpl_params_dict=smpl_params_dict, cond_dict=cond_dict,
                                    delta_pose_xyz_cache=delta_pose_xyz_cache,
                                    motion_cond_options=motion_cond_options)
    
    print("Reading Test Novelpose Transforms")
    test_cam_infos['novelpose'] = readCamerasI3DHuman(path, smpl_model, test_view, white_background, split='novelpose', 
                                    interval=interval, time_steps=time_steps, 
                                    smpl_params_dict=smpl_params_dict, cond_dict=cond_dict,
                                    delta_pose_xyz_cache=delta_pose_xyz_cache,
                                    motion_cond_options=motion_cond_options)

    if not eval:
        for key in test_cam_infos.keys():
            train_cam_infos.extend(test_cam_infos[key])
            test_cam_infos[key] = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if len(train_view) == 1:
        nerf_normalization['radius'] = 1

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = canon_vertices.shape[0]
        print(f"Generating random point cloud ({num_pts})...")
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=canon_vertices, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, canon_vertices, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos, test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization, ply_path=ply_path,
                           canon_params=canon_params, canon_vertices=canon_vertices,
                           smpl_params_dict=smpl_params_dict, cond_dict=cond_dict)
    return scene_info

def readCamerasI3DHuman(path, smpl_model, output_view, white_background, image_scaling=1.0, split='train', interval=1, time_steps=None, smpl_params_dict=None, cond_dict=None, delta_pose_xyz_cache=None, multi=300.0, motion_cond_options=None):
    cam_infos = []
    scene_name = os.path.basename(path).split('-')[0]
    scene_dir = os.path.join(os.path.dirname(path), scene_name + '-' + split)
    with open(os.path.join(scene_dir, 'cameras.pkl'), 'rb') as f: 
        cameras = pickle.load(f)
    with open(os.path.join(scene_dir, 'mesh_infos.pkl'), 'rb') as f:   
        mesh_infos = pickle.load(f)
    with open(os.path.join(scene_dir, 'frameid_pose.pkl'), 'rb') as f:   
        frameid_pose = pickle.load(f)

    frame_ids = sorted(list(set([int(img_name.split('_')[1]) for img_name in cameras.keys()])))

    # define how to read each pose_id's pose and xyz in I3D-Human dataset
    def get_pose_xyz_func(pose_id): 
        data = frameid_pose[pose_id]
        rh, poses_data, th = data['Rh'], data['poses'], data['Th']
        pose = np.concatenate([rh[None, :], poses_data], axis=0)
        pose_mat = axis_angle_to_matrix(torch.from_numpy(pose))

        full_poses = np.zeros(72, dtype=np.float32)
        full_poses[3:] = poses_data.flatten()
        R_mat = cv2.Rodrigues(rh[None, :])[0].astype(np.float32)

        shapes = np.zeros((1,10), dtype=np.float32)
        xyz, _ = smpl_model(full_poses[None, :].astype(np.float32), shapes.reshape(-1))
        xyz_transformed = (xyz @ R_mat.T + th[None, :]).astype(np.float32)
        
        return pose_mat, xyz_transformed
    
    for idx, pose_index in enumerate(frame_ids):
        frame_name = 'frame_{:06d}_view_{:02d}'.format(pose_index, output_view[0])

        # load SMPL data in observation space
        smpl_id = pose_index // interval
        if smpl_id not in smpl_params_dict.keys():
            smpl_param = {}
            Rh = mesh_infos[frame_name]['Rh'][None, :]
            smpl_param['Rh'] = Rh
            smpl_param['R'] = cv2.Rodrigues(Rh)[0].astype(np.float32)
            smpl_param['Th'] = mesh_infos[frame_name]['Th'][None, :]
            smpl_param['poses'] = mesh_infos[frame_name]['poses'][None, :]
            smpl_param['joints'] = mesh_infos[frame_name]['joints']
            smpl_param['shapes'] = np.zeros((1,10), dtype=np.float32)
            smpl_param['rot_mats'] = batch_rodrigues(torch.from_numpy(smpl_param['poses']).view(-1, 3)).view([1, -1, 3, 3])
            smpl_params_dict[smpl_id] = smpl_param
            
        smpl_param = smpl_params_dict[smpl_id]

        # load conditions
        if smpl_id not in cond_dict.keys():
            pose_conds = torch.from_numpy(frameid_pose[smpl_id]['poses']).unsqueeze(0)
            cond_result = get_seq_pose_xyz_cond(pose_index, time_steps, interval, get_pose_xyz_func, delta_pose_xyz_cache, multi=multi, motion_cond_options=motion_cond_options)
            if len(cond_result) == 3:
                seq_pose_conds, seq_xyz_conds, seq_acc_conds = cond_result
                cond_dict[smpl_id] = {'pose_conds': pose_conds, 'seq_pose_conds': seq_pose_conds, 'seq_xyz_conds': seq_xyz_conds, 'seq_acc_conds': seq_acc_conds}
            else:
                seq_pose_conds, seq_xyz_conds = cond_result
                cond_dict[smpl_id] = {'pose_conds': pose_conds, 'seq_pose_conds': seq_pose_conds, 'seq_xyz_conds': seq_xyz_conds}

        conds = cond_dict[smpl_id]
        pose_conds, seq_pose_conds, seq_xyz_conds = conds['pose_conds'], conds['seq_pose_conds'], conds['seq_xyz_conds']

        for view_index in output_view:
            cam_id = idx * len(output_view) + view_index
            image_name = f'frame_{pose_index:06d}_view_{view_index:02d}'

            # Load image, mask, K, D, R, T
            image_path = os.path.join(scene_dir, 'images', image_name + '.png')
            bkgd_mask_path = os.path.join(scene_dir, 'masks', image_name + '.png')

            # Load image, mask, K, D, R, T
            image_path = os.path.join(scene_dir, 'images', image_name + '.png')
            bkgd_mask_path = os.path.join(scene_dir, 'masks', image_name + '.png')

            # --- 修改开始：同时检查图片和Mask是否存在 ---
            if not os.path.exists(image_path):
                print(f"Warning: Image missing, skipping: {image_path}")
                continue
            if not os.path.exists(bkgd_mask_path):
                print(f"Warning: Mask missing, skipping: {bkgd_mask_path}")
                continue
            # --- 修改结束 ---------------------------

            K = cameras[image_name]['intrinsics']
            # -----------------------------------------------

            K = cameras[image_name]['intrinsics']
            R = cameras[image_name]['extrinsics'][:3, :3]
            T = cameras[image_name]['extrinsics'][:3, 3:4]

            image, bkgd_mask, bound_mask, K = load_image_mask(image_path, bkgd_mask_path, white_background, image_scaling, K)
            if bkgd_mask is None:
                continue

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            w2c = np.eye(4)
            w2c[:3,:3] = R
            w2c[:3,3:4] = T

            # get the world-to-camera transform and set R, T
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            focalX, focalY, width, height = K[0, 0], K[1, 1], image.size[0], image.size[1]
            FovX, FovY = focal2fov(focalX, width), focal2fov(focalY, height)

            cam_infos.append(CameraInfo(uid=cam_id, pose_id=smpl_id, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, bkgd_mask=bkgd_mask, 
                                        bound_mask=bound_mask, width=width, height=height))
    return cam_infos

##################################   DNA-Rendering   ##################################
def _project_points_to_image(points, K, c2w):
    w2c = np.linalg.inv(c2w)
    pts_cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = pts_cam[:, 2]
    safe_z = np.where(np.abs(z) < 1e-8, 1e-8, z)
    u = K[0, 0] * (pts_cam[:, 0] / safe_z) + K[0, 2]
    v = K[1, 1] * (pts_cam[:, 1] / safe_z) + K[1, 2]
    return u, v, z


def _load_gray_scaled(path, scale):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def _sample_flow_channels(flow, u, v):
    map_x = u.astype(np.float32).reshape(-1, 1)
    map_y = v.astype(np.float32).reshape(-1, 1)
    sampled = cv2.remap(flow, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return sampled.reshape(-1, flow.shape[-1])


def _build_dna_train_flow_view_token_features(
    path,
    train_view,
    pose_num=100,
    scale=0.25,
    mag_scale=1.0,
):
    """Build per-train-view flow tokens without averaging the view dimension.

    Returned features have shape [pose, vertex, train_view, 4], with channels:
    [flow_u, flow_v, flow_magnitude, forward_backward_confidence].
    """
    feature_dim = 4
    mag_scale = float(mag_scale)
    cache_dir = os.path.join(path, "flow_features")
    os.makedirs(cache_dir, exist_ok=True)
    scale_tag = f"{scale:.3f}".replace(".", "p")
    mag_tag = f"{mag_scale:g}".replace(".", "p")
    view_tag = f"v{len(train_view)}"
    cache_name = f"dna_train_farneback_viewtoken_s{scale_tag}_ms{mag_tag}_{view_tag}_fd{feature_dim}.npz"
    cache_path = os.path.join(cache_dir, cache_name)
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        features = cached["features"].astype(np.float32, copy=False)
        camera_centers = cached["camera_centers"].astype(np.float32, copy=False)
        print(
            f"[FlowCond] Loaded train-view flow-token cache: {cache_path}, "
            f"features={features.shape}, camera_centers={camera_centers.shape}"
        )
        return features, camera_centers

    first_model = np.load(os.path.join(path, "model", "000000.npz"), allow_pickle=True)
    vertex_num = int(first_model["obs_xyz"].shape[0])
    view_num = len(train_view)
    features = np.zeros((pose_num, vertex_num, view_num, feature_dim), dtype=np.float32)
    camera_centers = np.zeros((pose_num, view_num, 3), dtype=np.float32)
    diag_cache = {}
    print(f"[FlowCond] Building train-view flow-token cache with Farneback: {cache_path}")
    print(f"[FlowCond] Keeping view dimension. Train views: {train_view}")

    for pose_id in range(1, pose_num):
        prev_model = np.load(os.path.join(path, "model", f"{pose_id - 1:06d}.npz"), allow_pickle=True)
        prev_xyz = prev_model["obs_xyz"].astype(np.float32, copy=False)

        for view_slot, view_id in enumerate(train_view):
            prev_img_path = os.path.join(path, "images", f"{view_id:02d}", f"{pose_id - 1:06d}.png")
            curr_img_path = os.path.join(path, "images", f"{view_id:02d}", f"{pose_id:06d}.png")
            cam_path = os.path.join(path, "cameras", f"{view_id:02d}", f"{pose_id - 1:06d}.npz")
            if not (os.path.exists(prev_img_path) and os.path.exists(curr_img_path) and os.path.exists(cam_path)):
                continue

            prev_gray = _load_gray_scaled(prev_img_path, scale)
            curr_gray = _load_gray_scaled(curr_img_path, scale)
            flow_fwd = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0,
            )
            flow_bwd = cv2.calcOpticalFlowFarneback(
                curr_gray, prev_gray, None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0,
            )

            cam_params = np.load(cam_path, allow_pickle=True)
            K = cam_params["K"]
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = cam_params["RT"][:3, :3]
            c2w[:3, 3] = cam_params["RT"][:3, 3]
            camera_centers[pose_id, view_slot] = c2w[:3, 3].astype(np.float32)

            u, v, z = _project_points_to_image(prev_xyz, K, c2w)
            u_s, v_s = u * scale, v * scale
            h, w = flow_fwd.shape[:2]
            valid = (z > 1e-6) & (u_s >= 0) & (u_s <= w - 1) & (v_s >= 0) & (v_s <= h - 1)
            if not np.any(valid):
                continue

            sampled_fwd = _sample_flow_channels(flow_fwd, u_s, v_s)
            end_u = u_s + sampled_fwd[:, 0]
            end_v = v_s + sampled_fwd[:, 1]
            sampled_bwd = _sample_flow_channels(flow_bwd, end_u, end_v)
            fb_error = np.linalg.norm(sampled_fwd + sampled_bwd, axis=-1)
            conf = np.exp(-fb_error / 2.0).astype(np.float32)

            diag_key = (w, h)
            if diag_key not in diag_cache:
                diag_cache[diag_key] = float(np.sqrt(float(w * w + h * h)))
            diag = max(diag_cache[diag_key], 1e-6)
            flow_norm = sampled_fwd / diag
            mag = np.linalg.norm(flow_norm, axis=-1)
            valid_idx = np.where(valid)[0]
            features[pose_id, valid_idx, view_slot, 0] = flow_norm[valid_idx, 0].astype(np.float32) * mag_scale
            features[pose_id, valid_idx, view_slot, 1] = flow_norm[valid_idx, 1].astype(np.float32) * mag_scale
            features[pose_id, valid_idx, view_slot, 2] = mag[valid_idx].astype(np.float32) * mag_scale
            features[pose_id, valid_idx, view_slot, 3] = conf[valid_idx]

    np.savez_compressed(
        cache_path,
        features=features,
        camera_centers=camera_centers,
        train_view=np.asarray(train_view, dtype=np.int32),
        scale=np.asarray([scale], dtype=np.float32),
        mag_scale=np.asarray([mag_scale], dtype=np.float32),
    )
    print(
        f"[FlowCond] Saved train-view flow-token cache: {cache_path}, "
        f"features={features.shape}, camera_centers={camera_centers.shape}"
    )
    return features, camera_centers


def _build_dna_train_flow_features(
    path,
    train_view,
    pose_num=100,
    scale=0.25,
    feature_dim=3,
    feature_mode="mean_max_conf",
    mag_scale=1.0,
):
    feature_dim = int(feature_dim)
    feature_mode = str(feature_mode or "mean_max_conf").lower()
    if feature_mode not in {"mean_max_conf", "mean_max_std", "uv_mag_std", "reproj_residual"}:
        raise ValueError(f"Unsupported DNA flow feature_mode={feature_mode}")
    expected_dims = {
        "mean_max_conf": 3,
        "mean_max_std": 3,
        "uv_mag_std": 6,
        "reproj_residual": 8,
    }
    expected_dim = expected_dims[feature_mode]
    if feature_dim != expected_dim:
        raise ValueError(f"DNA flow condition expects feature_dim={expected_dim} for {feature_mode}, got {feature_dim}")
    mag_scale = float(mag_scale)

    cache_dir = os.path.join(path, "flow_features")
    os.makedirs(cache_dir, exist_ok=True)
    scale_tag = f"{scale:.3f}".replace(".", "p")
    mag_tag = f"{mag_scale:g}".replace(".", "p")
    cache_name = f"dna_train_farneback_s{scale_tag}_{feature_mode}_ms{mag_tag}_fd{feature_dim}.npz"
    cache_path = os.path.join(cache_dir, cache_name)
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        features = cached["features"].astype(np.float32, copy=False)
        print(f"[FlowCond] Loaded train-view flow cache: {cache_path}, shape={features.shape}")
        return features

    first_model = np.load(os.path.join(path, "model", "000000.npz"), allow_pickle=True)
    vertex_num = int(first_model["obs_xyz"].shape[0])
    features = np.zeros((pose_num, vertex_num, feature_dim), dtype=np.float32)
    diag_cache = {}
    print(f"[FlowCond] Building train-view flow cache with Farneback: {cache_path}")
    print(f"[FlowCond] Using train views only: {train_view}")

    for pose_id in range(1, pose_num):
        prev_model = np.load(os.path.join(path, "model", f"{pose_id - 1:06d}.npz"), allow_pickle=True)
        curr_model = np.load(os.path.join(path, "model", f"{pose_id:06d}.npz"), allow_pickle=True)
        prev_xyz = prev_model["obs_xyz"].astype(np.float32, copy=False)
        curr_xyz = curr_model["obs_xyz"].astype(np.float32, copy=False)

        u_sum = np.zeros((vertex_num,), dtype=np.float32)
        v_sum = np.zeros((vertex_num,), dtype=np.float32)
        mag_sum = np.zeros((vertex_num,), dtype=np.float32)
        mag_sq_sum = np.zeros((vertex_num,), dtype=np.float32)
        mag_max = np.zeros((vertex_num,), dtype=np.float32)
        res_u_sum = np.zeros((vertex_num,), dtype=np.float32)
        res_v_sum = np.zeros((vertex_num,), dtype=np.float32)
        res_mag_sum = np.zeros((vertex_num,), dtype=np.float32)
        res_mag_sq_sum = np.zeros((vertex_num,), dtype=np.float32)
        res_mag_max = np.zeros((vertex_num,), dtype=np.float32)
        conf_sum = np.zeros((vertex_num,), dtype=np.float32)
        count = np.zeros((vertex_num,), dtype=np.float32)

        for view_id in train_view:
            prev_img_path = os.path.join(path, "images", f"{view_id:02d}", f"{pose_id - 1:06d}.png")
            curr_img_path = os.path.join(path, "images", f"{view_id:02d}", f"{pose_id:06d}.png")
            cam_path = os.path.join(path, "cameras", f"{view_id:02d}", f"{pose_id - 1:06d}.npz")
            curr_cam_path = os.path.join(path, "cameras", f"{view_id:02d}", f"{pose_id:06d}.npz")
            need_curr_cam = feature_mode == "reproj_residual"
            if not (
                os.path.exists(prev_img_path)
                and os.path.exists(curr_img_path)
                and os.path.exists(cam_path)
                and (not need_curr_cam or os.path.exists(curr_cam_path))
            ):
                continue

            prev_gray = _load_gray_scaled(prev_img_path, scale)
            curr_gray = _load_gray_scaled(curr_img_path, scale)
            flow_fwd = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0,
            )
            flow_bwd = cv2.calcOpticalFlowFarneback(
                curr_gray, prev_gray, None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0,
            )

            cam_params = np.load(cam_path, allow_pickle=True)
            K = cam_params["K"]
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = cam_params["RT"][:3, :3]
            c2w[:3, 3] = cam_params["RT"][:3, 3]
            u, v, z = _project_points_to_image(prev_xyz, K, c2w)
            u_s, v_s = u * scale, v * scale
            h, w = flow_fwd.shape[:2]
            valid = (z > 1e-6) & (u_s >= 0) & (u_s <= w - 1) & (v_s >= 0) & (v_s <= h - 1)
            smpl_flow = None
            if feature_mode == "reproj_residual":
                curr_cam_params = np.load(curr_cam_path, allow_pickle=True)
                curr_K = curr_cam_params["K"]
                curr_c2w = np.eye(4, dtype=np.float32)
                curr_c2w[:3, :3] = curr_cam_params["RT"][:3, :3]
                curr_c2w[:3, 3] = curr_cam_params["RT"][:3, 3]
                curr_u, curr_v, curr_z = _project_points_to_image(curr_xyz, curr_K, curr_c2w)
                curr_u_s, curr_v_s = curr_u * scale, curr_v * scale
                valid = valid & (curr_z > 1e-6) & (curr_u_s >= 0) & (curr_u_s <= w - 1) & (curr_v_s >= 0) & (curr_v_s <= h - 1)
                smpl_flow = np.stack([curr_u_s - u_s, curr_v_s - v_s], axis=-1)
            if not np.any(valid):
                continue

            sampled_fwd = _sample_flow_channels(flow_fwd, u_s, v_s)
            end_u = u_s + sampled_fwd[:, 0]
            end_v = v_s + sampled_fwd[:, 1]
            sampled_bwd = _sample_flow_channels(flow_bwd, end_u, end_v)
            fb_error = np.linalg.norm(sampled_fwd + sampled_bwd, axis=-1)

            diag_key = (w, h)
            if diag_key not in diag_cache:
                diag_cache[diag_key] = float(np.sqrt(float(w * w + h * h)))
            diag = max(diag_cache[diag_key], 1e-6)
            flow_norm = sampled_fwd / diag
            mag = np.linalg.norm(flow_norm, axis=-1)
            if smpl_flow is not None:
                residual_norm = (sampled_fwd - smpl_flow) / diag
                res_mag = np.linalg.norm(residual_norm, axis=-1)
            conf = np.exp(-fb_error / 2.0).astype(np.float32)

            valid_idx = np.where(valid)[0]
            flow_valid = flow_norm[valid_idx].astype(np.float32)
            mag_valid = mag[valid_idx].astype(np.float32)
            conf_valid = conf[valid_idx].astype(np.float32)
            u_sum[valid_idx] += flow_valid[:, 0]
            v_sum[valid_idx] += flow_valid[:, 1]
            mag_sum[valid_idx] += mag_valid
            mag_sq_sum[valid_idx] += mag_valid * mag_valid
            mag_max[valid_idx] = np.maximum(mag_max[valid_idx], mag_valid)
            if smpl_flow is not None:
                res_valid = residual_norm[valid_idx].astype(np.float32)
                res_mag_valid = res_mag[valid_idx].astype(np.float32)
                res_u_sum[valid_idx] += res_valid[:, 0]
                res_v_sum[valid_idx] += res_valid[:, 1]
                res_mag_sum[valid_idx] += res_mag_valid
                res_mag_sq_sum[valid_idx] += res_mag_valid * res_mag_valid
                res_mag_max[valid_idx] = np.maximum(res_mag_max[valid_idx], res_mag_valid)
            conf_sum[valid_idx] += conf_valid
            count[valid_idx] += 1.0

        has_obs = count > 0
        if np.any(has_obs):
            mean_mag = mag_sum[has_obs] / count[has_obs]
            if feature_mode == "mean_max_conf":
                features[pose_id, has_obs, 0] = mean_mag * mag_scale
                features[pose_id, has_obs, 1] = mag_max[has_obs] * mag_scale
                features[pose_id, has_obs, 2] = conf_sum[has_obs] / count[has_obs]
            elif feature_mode == "mean_max_std":
                features[pose_id, has_obs, 0] = mean_mag * mag_scale
                features[pose_id, has_obs, 1] = mag_max[has_obs] * mag_scale
                var_mag = np.maximum((mag_sq_sum[has_obs] / count[has_obs]) - mean_mag * mean_mag, 0.0)
                features[pose_id, has_obs, 2] = np.sqrt(var_mag).astype(np.float32) * mag_scale
            elif feature_mode == "uv_mag_std":
                mean_u = u_sum[has_obs] / count[has_obs]
                mean_v = v_sum[has_obs] / count[has_obs]
                var_mag = np.maximum((mag_sq_sum[has_obs] / count[has_obs]) - mean_mag * mean_mag, 0.0)
                consistency = np.linalg.norm(np.stack([mean_u, mean_v], axis=-1), axis=-1) / np.maximum(mean_mag, 1e-6)
                features[pose_id, has_obs, 0] = mean_u * mag_scale
                features[pose_id, has_obs, 1] = mean_v * mag_scale
                features[pose_id, has_obs, 2] = mean_mag * mag_scale
                features[pose_id, has_obs, 3] = mag_max[has_obs] * mag_scale
                features[pose_id, has_obs, 4] = np.sqrt(var_mag).astype(np.float32) * mag_scale
                features[pose_id, has_obs, 5] = np.clip(consistency, 0.0, 1.0).astype(np.float32)
            elif feature_mode == "reproj_residual":
                mean_u = u_sum[has_obs] / count[has_obs]
                mean_v = v_sum[has_obs] / count[has_obs]
                mean_res_u = res_u_sum[has_obs] / count[has_obs]
                mean_res_v = res_v_sum[has_obs] / count[has_obs]
                mean_res_mag = res_mag_sum[has_obs] / count[has_obs]
                var_res_mag = np.maximum((res_mag_sq_sum[has_obs] / count[has_obs]) - mean_res_mag * mean_res_mag, 0.0)
                consistency = np.linalg.norm(np.stack([mean_u, mean_v], axis=-1), axis=-1) / np.maximum(mean_mag, 1e-6)
                features[pose_id, has_obs, 0] = mean_res_u * mag_scale
                features[pose_id, has_obs, 1] = mean_res_v * mag_scale
                features[pose_id, has_obs, 2] = mean_res_mag * mag_scale
                features[pose_id, has_obs, 3] = res_mag_max[has_obs] * mag_scale
                features[pose_id, has_obs, 4] = np.sqrt(var_res_mag).astype(np.float32) * mag_scale
                features[pose_id, has_obs, 5] = mean_u * mag_scale
                features[pose_id, has_obs, 6] = mean_v * mag_scale
                features[pose_id, has_obs, 7] = np.clip(consistency, 0.0, 1.0).astype(np.float32)

    np.savez_compressed(
        cache_path,
        features=features,
        train_view=np.asarray(train_view, dtype=np.int32),
        scale=np.asarray([scale], dtype=np.float32),
        feature_mode=np.asarray([feature_mode]),
        mag_scale=np.asarray([mag_scale], dtype=np.float32),
    )
    print(f"[FlowCond] Saved train-view flow cache: {cache_path}, shape={features.shape}")
    return features


def readDNARenderingInfo(path, white_background, eval, time_steps, motion_cond_options=None):
    motion_cond_options = dict(motion_cond_options or {})
    scene_name = os.path.basename(path)
    main_path = os.path.join(path, scene_name + '.smc')
    smc_reader = SMCReader(main_path)

    # camera view splitting
    train_view = [i for i in range(0, 48, 2)]
    test_view = [i for i in range(48, 60, 2)]

    # load SMPL-X model
    gender = smc_reader.actor_info['gender']
    smpl_model = SMPLX('smpl_model/models/', smpl_type='smplx', gender=gender, 
                        use_face_contour=True, flat_hand_mean=False, use_pca=False,
                        num_pca_comps=24, num_betas=10, num_expression_coeffs=10, ext='pkl')
    
    # build canonical space SMPLX points
    canon_params, canon_vertices = create_canonical_vertices(smpl_model, 'smplx')

    test_cam_infos = {}
    smpl_params_dict, cond_dict = {}, {} # observation space smpl params, conditions for non-rigid deformation
    delta_pose_xyz_cache = {} # cache for sequential condition calculation

    if bool(motion_cond_options.get("use_flow_cond", False)):
        flow_mode = str(motion_cond_options.get("flow_cond_mode", "flow")).lower()
        feature_dim = int(motion_cond_options.get("flow_feature_dim", 3))
        flow_knn_agg = str(motion_cond_options.get("flow_knn_agg", "mean")).lower()
        flow_view_token = bool(motion_cond_options.get("flow_view_token", False))
        if flow_knn_agg == "mean_max":
            if feature_dim % 2 != 0:
                raise ValueError(f"flow_feature_dim must be even when flow_knn_agg=mean_max, got {feature_dim}")
            vertex_feature_dim = feature_dim // 2
        else:
            vertex_feature_dim = feature_dim
        motion_cond_options["flow_vertex_feature_dim"] = int(vertex_feature_dim)
        motion_cond_options["flow_vertex_num"] = int(canon_vertices.shape[0])
        if flow_mode == "flow":
            if flow_view_token:
                if vertex_feature_dim != 4:
                    raise ValueError(f"flow_view_token expects flow_feature_dim=4, got {vertex_feature_dim}")
                flow_feature_map, flow_train_camera_centers = _build_dna_train_flow_view_token_features(
                    path,
                    train_view,
                    pose_num=100,
                    scale=float(motion_cond_options.get("flow_image_scale", 0.25)),
                    mag_scale=float(motion_cond_options.get("flow_mag_scale", 1.0)),
                )
                motion_cond_options["flow_feature_map"] = flow_feature_map
                motion_cond_options["flow_train_camera_centers"] = flow_train_camera_centers
                motion_cond_options["flow_train_view_num"] = int(len(train_view))
            else:
                motion_cond_options["flow_feature_map"] = _build_dna_train_flow_features(
                    path,
                    train_view,
                    pose_num=100,
                    scale=float(motion_cond_options.get("flow_image_scale", 0.25)),
                    feature_dim=vertex_feature_dim,
                    feature_mode=str(motion_cond_options.get("flow_feature_mode", "mean_max_conf")),
                    mag_scale=float(motion_cond_options.get("flow_mag_scale", 1.0)),
                )
        elif flow_mode == "zero":
            print("[FlowCond] flow_zero mode: using zero flow features with the same FlowEncoder.")
            if flow_view_token:
                motion_cond_options["flow_train_view_num"] = int(len(train_view))
        else:
            raise ValueError(f"Unsupported flow_cond_mode: {flow_mode}")

    # read cameras
    print("Reading Training Transforms")
    train_cam_infos = readCamerasDNARendering(path, train_view, white_background, split='train', time_steps=time_steps,
                        smpl_params_dict=smpl_params_dict, cond_dict=cond_dict, 
                        delta_pose_xyz_cache=delta_pose_xyz_cache,
                        motion_cond_options=motion_cond_options)

    print("Reading Novel View Transforms")
    test_cam_infos['novelview'] = readCamerasDNARendering(path, test_view, white_background, 
                                    split='novelview', time_steps=time_steps, 
                                    smpl_params_dict=smpl_params_dict, cond_dict=cond_dict,
                                    delta_pose_xyz_cache=delta_pose_xyz_cache,
                                    motion_cond_options=motion_cond_options)
    
    if not eval:
        for key in test_cam_infos.keys():
            train_cam_infos.extend(test_cam_infos[key])
            test_cam_infos[key] = []
    
    nerf_normalization = getNerfppNorm(train_cam_infos)
    if len(train_view) == 1:
        nerf_normalization['radius'] = 1
    
    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = canon_vertices.shape[0]
        print(f"Generating random point cloud ({num_pts})...")
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=canon_vertices, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, canon_vertices, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos, test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization, ply_path=ply_path,
                           canon_params=canon_params, canon_vertices=canon_vertices,
                           smpl_params_dict=smpl_params_dict, cond_dict=cond_dict)
    return scene_info

def readCamerasDNARendering(path, output_view, white_background, split='train', interval=1, time_steps=None, smpl_params_dict=None, cond_dict=None, delta_pose_xyz_cache=None, multi=300.0, motion_cond_options=None):
    cam_infos = []
    if split == 'train':
        pose_start, pose_interval, pose_num = 0, 1, 100
    elif split == 'novelview':
        pose_start, pose_interval, pose_num = 0, 5, 20
    else:
        raise ValueError('Unknown split type for DNA-Rendering dataset')
    
    # define how to read each pose_id's pose and xyz in DNA-Rendering dataset    
    def get_pose_xyz_func(pose_id):
        model_file = os.path.join(path, 'model', f'{pose_id:06d}.npz')
        smpl_param = np.load(model_file, allow_pickle=True)
        poses = smpl_param['poses'].reshape(-1, 3)
        pose_mat = axis_angle_to_matrix(torch.from_numpy(poses))
        return pose_mat, smpl_param['obs_xyz']
    
    for idx, pose_index in enumerate(range(pose_start, pose_start + pose_num * pose_interval, pose_interval)):
        # load smpl data 
        if pose_index not in smpl_params_dict.keys():
            model_file = os.path.join(path, 'model', f'{pose_index:06d}.npz')
            loaded_data = np.load(model_file, allow_pickle=True)
            smpl_param = {key: loaded_data[key] for key in loaded_data.files}
            smpl_param['rot_mats'] = batch_rodrigues(torch.from_numpy(smpl_param['poses']).view(-1, 3)).view([1, -1, 3, 3])
            smpl_params_dict[pose_index] = smpl_param

        smpl_param = smpl_params_dict[pose_index]

        # load condition
        if pose_index not in cond_dict.keys():
            pose_conds = smpl_param['poses'].reshape(-1,3)[1:]
            if not isinstance(pose_conds, torch.Tensor):
                pose_conds = torch.from_numpy(pose_conds).unsqueeze(0)

            cond_result = get_seq_pose_xyz_cond(pose_index, time_steps, interval, get_pose_xyz_func, delta_pose_xyz_cache, multi=multi, motion_cond_options=motion_cond_options)
            if len(cond_result) == 3:
                seq_pose_conds, seq_xyz_conds, seq_acc_conds = cond_result
                cond_dict[pose_index] = {'pose_conds': pose_conds, 'seq_pose_conds': seq_pose_conds, 'seq_xyz_conds': seq_xyz_conds, 'seq_acc_conds': seq_acc_conds}
            else:
                seq_pose_conds, seq_xyz_conds = cond_result
                cond_dict[pose_index] = {'pose_conds': pose_conds, 'seq_pose_conds': seq_pose_conds, 'seq_xyz_conds': seq_xyz_conds}
            if bool((motion_cond_options or {}).get("use_flow_cond", False)):
                flow_feature_dim = int((motion_cond_options or {}).get("flow_vertex_feature_dim", (motion_cond_options or {}).get("flow_feature_dim", 3)))
                flow_vertex_num = int((motion_cond_options or {}).get("flow_vertex_num", smpl_param['obs_xyz'].shape[0]))
                flow_mode = str((motion_cond_options or {}).get("flow_cond_mode", "flow")).lower()
                flow_view_token = bool((motion_cond_options or {}).get("flow_view_token", False))
                if flow_mode == "flow":
                    flow_map = (motion_cond_options or {}).get("flow_feature_map", None)
                    if flow_map is None:
                        raise RuntimeError("use_flow_cond=True and flow_cond_mode=flow, but flow_feature_map is missing.")
                    flow_pose_id = int(np.clip(pose_index, 0, flow_map.shape[0] - 1))
                    seq_flow_conds = torch.from_numpy(flow_map[flow_pose_id].astype(np.float32, copy=False))
                elif flow_mode == "zero":
                    if flow_view_token:
                        flow_train_view_num = int((motion_cond_options or {}).get("flow_train_view_num", 1))
                        seq_flow_conds = torch.zeros((flow_vertex_num, flow_train_view_num, flow_feature_dim), dtype=torch.float32)
                    else:
                        seq_flow_conds = torch.zeros((flow_vertex_num, flow_feature_dim), dtype=torch.float32)
                else:
                    raise ValueError(f"Unsupported flow_cond_mode: {flow_mode}")
                cond_dict[pose_index]['seq_flow_conds'] = seq_flow_conds
                if flow_view_token:
                    centers = (motion_cond_options or {}).get("flow_train_camera_centers", None)
                    if centers is not None:
                        flow_pose_id = int(np.clip(pose_index, 0, centers.shape[0] - 1))
                        cond_dict[pose_index]['flow_train_camera_centers'] = torch.from_numpy(
                            centers[flow_pose_id].astype(np.float32, copy=False)
                        )
                    else:
                        flow_train_view_num = int((motion_cond_options or {}).get("flow_train_view_num", 1))
                        cond_dict[pose_index]['flow_train_camera_centers'] = torch.zeros((flow_train_view_num, 3), dtype=torch.float32)

        conds = cond_dict[pose_index]
        pose_conds, seq_pose_conds, seq_xyz_conds = conds['pose_conds'], conds['seq_pose_conds'], conds['seq_xyz_conds']

        for view_index in output_view:
            cam_id = idx * len(output_view) + view_index
            image_name = f'frame_{pose_index:06d}_view_{view_index:02d}'

            # Load camera, image, mask
            cam_path = os.path.join(path, 'cameras', f'{view_index:02d}', f'{pose_index:06d}.npz')
            img_path = os.path.join(path, 'images', f'{view_index:02d}', f'{pose_index:06d}.png')
            bkgd_mask_path = os.path.join(path, 'bkgd_masks', f'{view_index:02d}', f'{pose_index:06d}.png')

            cam_params = np.load(cam_path, allow_pickle=True)
            # D = cam_params['D']
            K = cam_params['K']
            R, T = cam_params['RT'][:3, :3], cam_params['RT'][:3, 3]
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3:4] = T.reshape(-1, 1)

            image, bkgd_mask, bound_mask, _ = load_image_mask(img_path, bkgd_mask_path, white_background)

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            focalX, focalY, width, height = K[0, 0], K[1, 1], image.size[0], image.size[1]
            FovX, FovY = focal2fov(focalX, width), focal2fov(focalY, height)

            cam_infos.append(CameraInfo(uid=cam_id, pose_id=pose_index, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                                        image_path=img_path, image_name=image_name, bkgd_mask=bkgd_mask,
                                        bound_mask=bound_mask, width=width, height=height))
    
    return cam_infos

##################################   ZJU-Mocap   ##################################
ZJUMOCAP_CFG = {
    'CoreView_377': {'train': {'interval': 1, 'num': 570},
                     'test':  {'interval': 30, 'num': 19}},
    'CoreView_386': {'train': {'interval': 1, 'num': 540},
                     'test':  {'interval': 30, 'num': 18}},
    'CoreView_387': {'train': {'interval': 1, 'num': 540},
                     'test':  {'interval': 30, 'num': 18}},
    'CoreView_392': {'train': {'interval': 1, 'num': 556},
                     'test':  {'interval': 30, 'num': 19}},
    'CoreView_393': {'train': {'interval': 1, 'num': 658},
                     'test':  {'interval': 30, 'num': 22}},
    'CoreView_394': {'train': {'interval': 1, 'num': 475},
                     'test':  {'interval': 30, 'num': 16}}
    }
def readZJUMoCapInfo(path, white_background, eval, time_steps, motion_cond_options=None):
    # camera view splitting
    train_view = [0]
    test_view = [i for i in range(1, 23)]

    # load SMPL model
    smpl_model = SMPL(sex='neutral', model_dir='smpl_model/models/')

    # build canonical space SMPL points
    canon_params, canon_vertices = create_canonical_vertices(smpl_model, 'smpl')

    test_cam_infos = {}
    smpl_params_dict, cond_dict = {}, {} # observation space smpl params, conditions for non-rigid deformation
    delta_pose_xyz_cache = {} # cache for sequential condition calculation
    

    # read cameras
    print("Reading Training Transforms")
    train_cam_infos = readCamerasZJUMoCap(path, train_view, white_background, split='train', time_steps=time_steps, 
                                          smpl_params_dict=smpl_params_dict, cond_dict=cond_dict,
                                          delta_pose_xyz_cache=delta_pose_xyz_cache,
                                          motion_cond_options=motion_cond_options)
    
    print("Reading Test Transforms")
    test_cam_infos['test'] = readCamerasZJUMoCap(path, test_view, white_background, split='test', time_steps=time_steps, 
                                            smpl_params_dict=smpl_params_dict, cond_dict=cond_dict,
                                            delta_pose_xyz_cache=delta_pose_xyz_cache,
                                            motion_cond_options=motion_cond_options)
    
    if not eval:
        for key in test_cam_infos.keys():
            train_cam_infos.extend(test_cam_infos[key])
            test_cam_infos[key] = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if len(train_view) == 1:
        nerf_normalization['radius'] = 1

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = canon_vertices.shape[0]
        print(f"Generating random point cloud ({num_pts})...")
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=canon_vertices, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, canon_vertices, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos, test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization, ply_path=ply_path,
                           canon_params=canon_params, canon_vertices=canon_vertices,
                           smpl_params_dict=smpl_params_dict, cond_dict=cond_dict)
    return scene_info

def readCamerasZJUMoCap(path, output_view, white_background, image_scaling=0.5, split='train', interval=1, time_steps=None, smpl_params_dict=None, cond_dict=None, delta_pose_xyz_cache=None, motion_cond_options=None):
    cam_infos = []
    pose_start = 0
    scene_name = os.path.basename(path)
    pose_interval, pose_num = ZJUMOCAP_CFG[scene_name][split]['interval'], ZJUMOCAP_CFG[scene_name][split]['num']

    ann_file = os.path.join(path, 'annots.npy')
    annots = np.load(ann_file, allow_pickle=True).item()
    cams = annots['cams']
    ims = np.array([
        np.array(ims_data['ims'])[output_view]
        for ims_data in annots['ims'][pose_start:pose_start + pose_num * pose_interval]
    ])

    cam_inds = np.array([
        np.arange(len(ims_data['ims']))[output_view]
        for ims_data in annots['ims'][pose_start:pose_start + pose_num * pose_interval]
    ])

    # define how to read each pose_id's pose and xyz in ZJU-Mocap dataset
    def get_pose_xyz_func(pose_id):
        smpl_param_path = os.path.join(path, "new_params", '{}.npy'.format(pose_id))
        loaded_data = np.load(smpl_param_path, allow_pickle=True).item()
        pose = np.concatenate([loaded_data['Rh'], loaded_data['poses'][:, 3:].reshape(-1, 3)], axis=0)
        vertices_path = os.path.join(path, 'new_vertices', '{}.npy'.format(pose_id))
        xyz = np.load(vertices_path).astype(np.float32)
        pose_mat = axis_angle_to_matrix(torch.from_numpy(pose))
        return pose_mat, xyz
    
    for idx, pose_index in enumerate(range(pose_start, pose_start + pose_num * pose_interval, pose_interval)):
        # load smpl data 
        if pose_index not in smpl_params_dict.keys():
            smpl_param = {}
            smpl_param_path = os.path.join(path, "new_params", '{}.npy'.format(pose_index))
            loaded_smpl_data = np.load(smpl_param_path, allow_pickle=True).item()
            Rh = loaded_smpl_data['Rh']
            smpl_param['Rh'] = Rh
            smpl_param['R'] = cv2.Rodrigues(Rh)[0].astype(np.float32)
            smpl_param['Th'] = loaded_smpl_data['Th'].astype(np.float32)
            smpl_param['shapes'] = loaded_smpl_data['shapes'].astype(np.float32)
            smpl_param['poses'] = loaded_smpl_data['poses'].astype(np.float32)
            smpl_param['rot_mats'] = batch_rodrigues(torch.from_numpy(smpl_param['poses']).view(-1, 3)).view([1, -1, 3, 3])
            smpl_params_dict[pose_index] = smpl_param

        smpl_param = smpl_params_dict[pose_index]

        if pose_index not in cond_dict.keys():
            pose_conds = torch.from_numpy(smpl_param['poses'][:, 3:].reshape(-1, 3)).unsqueeze(0)
            cond_result = get_seq_pose_xyz_cond(pose_index, time_steps, interval, get_pose_xyz_func, delta_pose_xyz_cache, motion_cond_options=motion_cond_options)
            if len(cond_result) == 3:
                seq_pose_conds, seq_xyz_conds, seq_acc_conds = cond_result
                cond_dict[pose_index] = {'pose_conds': pose_conds, 'seq_pose_conds': seq_pose_conds, 'seq_xyz_conds': seq_xyz_conds, 'seq_acc_conds': seq_acc_conds}
            else:
                seq_pose_conds, seq_xyz_conds = cond_result
                cond_dict[pose_index] = {'pose_conds': pose_conds, 'seq_pose_conds': seq_pose_conds, 'seq_xyz_conds': seq_xyz_conds}

        conds = cond_dict[pose_index]
        pose_conds, seq_pose_conds, seq_xyz_conds = conds['pose_conds'], conds['seq_pose_conds'], conds['seq_xyz_conds']

        for view_index in range(len(output_view)):
            cam_id = idx * len(output_view) + view_index

            # Load image, mask, K, D, R, T
            image_path = os.path.join(path, ims[pose_index][view_index].replace('\\', '/'))
            bkgd_mask_path = os.path.join(path, 'mask_cihp', ims[pose_index][view_index].replace('\\', '/')).replace('jpg', 'png')
            image_name = ims[pose_index][view_index].split('.')[0]

            cam_ind = cam_inds[pose_index][view_index]
            K = np.array(cams['K'][cam_ind])
            D = np.array(cams['D'][cam_ind])
            R = np.array(cams['R'][cam_ind])
            T = np.array(cams['T'][cam_ind]) / 1000.

            image, bkgd_mask, bound_mask, K = load_image_mask(image_path, bkgd_mask_path, white_background, image_scaling, K, D)

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            w2c = np.eye(4)
            w2c[:3,:3] = R
            w2c[:3,3:4] = T
            # get the world-to-camera transform and set R, T
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            focalX, focalY, width, height = K[0, 0], K[1, 1], image.size[0], image.size[1]
            FovX, FovY = focal2fov(focalX, width), focal2fov(focalY, height)

            image_name = f'frame_{pose_index:06d}_view_{view_index:02d}'
            cam_infos.append(CameraInfo(uid=cam_id, pose_id=pose_index, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, bkgd_mask=bkgd_mask, 
                                        bound_mask=bound_mask, width=width, height=height))
    
    return cam_infos

def _relative_pose_delta(cur_pose_mat, former_pose_mat):
    delta_pose_mat = torch.matmul(cur_pose_mat, torch.linalg.inv(former_pose_mat))
    return matrix_to_axis_angle(delta_pose_mat)


def get_seq_pose_xyz_cond(pose_index, time_steps, interval, get_pose_xyz_func, delta_pose_xyz_cache, multi=1.0, motion_cond_options=None):
    motion_cond_options = motion_cond_options or {}
    use_msti = bool(motion_cond_options.get("use_msti", False))
    msti_mode = str(motion_cond_options.get("msti_mode", "none")).lower()
    msti_mid_type = str(motion_cond_options.get("msti_mid_type", "real")).lower()
    use_amc_pair = bool(motion_cond_options.get("use_amc_pair", False))
    amc_pair_mode = str(motion_cond_options.get("amc_pair_mode", "baseline_full")).lower()
    use_amc_causal = bool(motion_cond_options.get("use_amc_causal", False))
    use_tdp = bool(motion_cond_options.get("use_tdp", False))
    tdp_mode = str(motion_cond_options.get("tdp_mode", "keep_base")).lower()
    fix_stms = bool(motion_cond_options.get("fix_stms", False))
    use_acc_cond = bool(motion_cond_options.get("use_acc_cond", False))
    use_motion_token = bool(motion_cond_options.get("use_motion_token", False))
    motion_token_fix_stms = bool(motion_cond_options.get("motion_token_fix_stms", False))
    motion_token_use_acc = bool(motion_cond_options.get("motion_token_use_acc", False))
    fix_stms = fix_stms or (use_motion_token and motion_token_fix_stms)
    use_acc_cond = use_acc_cond or (use_motion_token and motion_token_use_acc)
    expected_channels = int(motion_cond_options.get("motion_cond_time_step_num", len(time_steps)))

    if sum([use_msti and msti_mode != "none", use_amc_pair, use_amc_causal, use_tdp]) > 1:
        raise ValueError("MSTI, AMC-pair, AMC-causal, and TDP are mutually exclusive in motion condition construction.")

    if use_amc_pair:
        return get_seq_pose_xyz_cond_amc_pair(
            pose_index,
            time_steps,
            interval,
            get_pose_xyz_func,
            delta_pose_xyz_cache,
            multi=multi,
            amc_pair_mode=amc_pair_mode,
            expected_channels=expected_channels,
        )

    if use_msti and msti_mode != "none":
        if msti_mid_type != "real":
            raise ValueError(f"Only real-mid MSTI is implemented, got msti_mid_type={msti_mid_type}")
        return get_seq_pose_xyz_cond_msti(
            pose_index,
            time_steps,
            interval,
            get_pose_xyz_func,
            delta_pose_xyz_cache,
            multi=multi,
            msti_mode=msti_mode,
            expected_channels=expected_channels,
        )

    if use_tdp:
        return get_seq_pose_xyz_cond_tdp(
            pose_index,
            time_steps,
            interval,
            get_pose_xyz_func,
            delta_pose_xyz_cache,
            multi=multi,
            tdp_mode=tdp_mode,
            expected_channels=expected_channels,
        )

    def get_delta(cur_id, former_id):
        delta_key = str(cur_id) + '-' + str(former_id)
        if delta_key not in delta_pose_xyz_cache.keys():
            cur_pose_mat, cur_obs_xyz = get_pose_xyz_func(cur_id)
            former_pose_mat, former_obs_xyz = get_pose_xyz_func(former_id)

            delta_pose_mat = torch.matmul(cur_pose_mat, torch.linalg.inv(former_pose_mat))
            posedelta = matrix_to_axis_angle(delta_pose_mat)
            xyz_delta = cur_obs_xyz - former_obs_xyz
            delta_pose_xyz_cache[delta_key] = {'pose_delta': posedelta, 'xyz_delta': xyz_delta}
        return delta_pose_xyz_cache[delta_key]['pose_delta'], delta_pose_xyz_cache[delta_key]['xyz_delta']

    seq_pose_conds, seq_xyz_conds, seq_acc_conds = {}, {}, {}
    for time_step, seq_len in time_steps.items():
        seq_pose_cond, seq_xyz_cond, seq_acc_cond = [], [], []
        for i in range(seq_len):
            cur_id = max((pose_index - i * time_step) // interval, 0)
            former_id = max((pose_index - (i + 1) * time_step) // interval, 0)
            posedelta, xyz_delta = get_delta(cur_id, former_id)
            seq_pose_cond.append(posedelta)
            seq_xyz_cond.append(xyz_delta)
            if use_acc_cond:
                older_id = max((pose_index - (i + 2) * time_step) // interval, 0)
                _, prev_xyz_delta = get_delta(former_id, older_id)
                dt = float(time_step / interval)
                seq_acc_cond.append((xyz_delta / dt) - (prev_xyz_delta / dt))

        seq_pose_conds[time_step] = torch.stack(seq_pose_cond, axis=0)
        seq_xyz_conds[time_step] = torch.from_numpy(np.stack(seq_xyz_cond, axis=1))
        if use_acc_cond:
            seq_acc_conds[time_step] = torch.from_numpy(np.stack(seq_acc_cond, axis=1))
    
    seq_pose_conds = torch.stack([seq_pose_conds[time_step] for time_step in time_steps], axis=1).unsqueeze(0)
    if fix_stms:
        seq_xyz_conds = torch.stack(
            [
                (seq_xyz_conds[time_step] * multi) / float(time_step / interval)
                for time_step in time_steps
            ],
            axis=2,
        )
    else:
        seq_xyz_conds = (torch.stack([seq_xyz_conds[time_step] for time_step in time_steps], axis=2) * multi) / float(time_step / interval)

    if seq_pose_conds.shape[2] != expected_channels:
        raise RuntimeError(f"seq_pose_conds channel dim {seq_pose_conds.shape[2]} != expected {expected_channels}")
    if seq_xyz_conds.shape[2] != expected_channels:
        raise RuntimeError(f"seq_xyz_conds channel dim {seq_xyz_conds.shape[2]} != expected {expected_channels}")

    if use_acc_cond:
        seq_acc_conds = torch.stack(
            [
                seq_acc_conds[time_step] * multi
                for time_step in time_steps
            ],
            axis=2,
        )
        if seq_acc_conds.shape[2] != expected_channels:
            raise RuntimeError(f"seq_acc_conds channel dim {seq_acc_conds.shape[2]} != expected {expected_channels}")
        return seq_pose_conds, seq_xyz_conds, seq_acc_conds

    return seq_pose_conds, seq_xyz_conds


def get_seq_pose_xyz_cond_tdp(
    pose_index,
    time_steps,
    interval,
    get_pose_xyz_func,
    delta_pose_xyz_cache,
    multi=1.0,
    tdp_mode="keep_base",
    expected_channels=None,
):
    if tdp_mode not in {"local", "keep_base"}:
        raise ValueError(f"TDP mode must be 'local' or 'keep_base', got {tdp_mode}")

    time_step_items = sorted(time_steps.items(), key=lambda item: item[0], reverse=True)
    if not time_step_items:
        raise ValueError("time_steps must not be empty")
    seq_lens = {int(seq_len) for _, seq_len in time_step_items}
    if len(seq_lens) != 1:
        raise ValueError(f"TDP expects a single seq_len across time steps, got {sorted(seq_lens)}")
    seq_len = seq_lens.pop()

    base_steps = [int(step) for step, _ in time_step_items]
    base_channels = len(base_steps)
    max_step = max(base_steps)
    expected = 2 * base_channels if tdp_mode == "local" else base_channels + 2 * base_channels - 1
    if expected_channels is not None and int(expected_channels) != expected:
        raise ValueError(
            f"TDP condition channel mismatch: expected_channels={expected_channels}, "
            f"computed={expected}, mode={tdp_mode}, base_steps={base_channels}"
        )

    def frame_to_id(frame_id):
        return max(int(frame_id) // int(interval), 0)

    def get_pair_delta(cur_id, former_id):
        delta_key = f"tdp_{tdp_mode}_dt_v1:{cur_id}-{former_id}"
        if delta_key not in delta_pose_xyz_cache:
            cur_pose_mat, cur_obs_xyz = get_pose_xyz_func(cur_id)
            former_pose_mat, former_obs_xyz = get_pose_xyz_func(former_id)
            posedelta = _relative_pose_delta(cur_pose_mat, former_pose_mat)
            dt = max(int(cur_id) - int(former_id), 1)
            xyz_delta = ((cur_obs_xyz - former_obs_xyz) * multi) / float(dt)
            delta_pose_xyz_cache[delta_key] = {
                "pose_delta": posedelta,
                "xyz_delta": xyz_delta.astype(np.float32, copy=False),
            }
        return (
            delta_pose_xyz_cache[delta_key]["pose_delta"],
            delta_pose_xyz_cache[delta_key]["xyz_delta"],
        )

    seq_pose_cond_by_i, seq_xyz_cond_by_i = [], []
    for i in range(seq_len):
        anchor_frame = pose_index - i * max_step
        anchor_id = frame_to_id(anchor_frame)
        point_ids = [frame_to_id(anchor_frame - step) for step in base_steps] + [anchor_id]

        pose_channels, xyz_channels = [], []

        if tdp_mode == "keep_base":
            for former_id in point_ids[:-1]:
                d_pose_full, d_xyz_full = get_pair_delta(anchor_id, former_id)
                pose_channels.append(d_pose_full)
                xyz_channels.append(d_xyz_full)
        else:
            d_pose, d_xyz = get_pair_delta(point_ids[-1], point_ids[0])
            pose_channels.append(d_pose)
            xyz_channels.append(d_xyz)

        vel_pose_channels, vel_xyz_channels = [], []
        for from_id, to_id in zip(point_ids[:-1], point_ids[1:]):
            d_pose_vel, d_xyz_vel = get_pair_delta(to_id, from_id)
            vel_pose_channels.append(d_pose_vel)
            vel_xyz_channels.append(d_xyz_vel)
            pose_channels.append(d_pose_vel)
            xyz_channels.append(d_xyz_vel)

        for vel_idx in range(len(vel_pose_channels) - 1):
            a_pose = vel_pose_channels[vel_idx + 1] - vel_pose_channels[vel_idx]
            a_xyz = vel_xyz_channels[vel_idx + 1] - vel_xyz_channels[vel_idx]
            pose_channels.append(a_pose)
            xyz_channels.append(a_xyz.astype(np.float32, copy=False))

        if len(pose_channels) != expected:
            raise RuntimeError(f"TDP produced {len(pose_channels)} pose channels, expected {expected}")
        seq_pose_cond_by_i.append(torch.stack(pose_channels, dim=0))
        seq_xyz_cond_by_i.append(torch.from_numpy(np.stack(xyz_channels, axis=1)))

    seq_pose_conds = torch.stack(seq_pose_cond_by_i, dim=0).unsqueeze(0)
    seq_xyz_conds = torch.stack(seq_xyz_cond_by_i, dim=1)

    if seq_pose_conds.shape[2] != expected:
        raise RuntimeError(f"seq_pose_conds channel dim {seq_pose_conds.shape[2]} != expected {expected}")
    if seq_xyz_conds.shape[2] != expected:
        raise RuntimeError(f"seq_xyz_conds channel dim {seq_xyz_conds.shape[2]} != expected {expected}")

    return seq_pose_conds, seq_xyz_conds


def get_seq_pose_xyz_cond_amc_pair(
    pose_index,
    time_steps,
    interval,
    get_pose_xyz_func,
    delta_pose_xyz_cache,
    multi=1.0,
    amc_pair_mode="baseline_full",
    expected_channels=None,
):
    if amc_pair_mode != "baseline_full":
        raise ValueError(f"AMC-pair mode must be 'baseline_full', got {amc_pair_mode}")

    time_step_items = list(time_steps.items())
    if not time_step_items:
        raise ValueError("time_steps must not be empty")
    seq_lens = {int(seq_len) for _, seq_len in time_step_items}
    if len(seq_lens) != 1:
        raise ValueError(f"AMC-pair expects a single seq_len across time steps, got {sorted(seq_lens)}")
    seq_len = seq_lens.pop()

    base_channels = len(time_step_items)
    expected = base_channels + base_channels * (base_channels - 1) // 2
    if expected_channels is not None and int(expected_channels) != expected:
        raise ValueError(
            f"AMC-pair condition channel mismatch: expected_channels={expected_channels}, "
            f"computed={expected}, base_steps={base_channels}"
        )

    # Scheme A keeps the original full-channel time grid. The first N channels
    # are generated by the same (time_step, seq_i) pairs as the baseline path.
    baseline_xyz_divisor = float(time_step_items[-1][0] / interval)

    def get_pair_delta(cur_id, former_id):
        delta_key = f"amc_pair_scheme_a_raw_v1:{cur_id}-{former_id}"
        if delta_key not in delta_pose_xyz_cache:
            cur_pose_mat, cur_obs_xyz = get_pose_xyz_func(cur_id)
            former_pose_mat, former_obs_xyz = get_pose_xyz_func(former_id)
            posedelta = _relative_pose_delta(cur_pose_mat, former_pose_mat)
            xyz_delta = cur_obs_xyz - former_obs_xyz
            delta_pose_xyz_cache[delta_key] = {
                "pose_delta": posedelta,
                "xyz_delta": xyz_delta,
            }
        return (
            delta_pose_xyz_cache[delta_key]["pose_delta"],
            delta_pose_xyz_cache[delta_key]["xyz_delta"],
        )

    seq_pose_cond_by_i, seq_xyz_cond_by_i = [], []
    for i in range(seq_len):
        pose_channels, xyz_channels = [], []
        former_ids = []

        for time_step, _ in time_step_items:
            cur_id = max((pose_index - i * time_step) // interval, 0)
            former_id = max((pose_index - (i + 1) * time_step) // interval, 0)
            d_pose_full, d_xyz_full = get_pair_delta(cur_id, former_id)
            pose_channels.append(d_pose_full)
            xyz_channels.append(d_xyz_full)
            former_ids.append(former_id)

        for src_idx in range(base_channels):
            for dst_idx in range(src_idx + 1, base_channels):
                from_id = former_ids[src_idx]
                to_id = former_ids[dst_idx]
                d_pose_pair, d_xyz_pair = get_pair_delta(to_id, from_id)
                pose_channels.append(d_pose_pair)
                xyz_channels.append(d_xyz_pair)

        if len(pose_channels) != expected:
            raise RuntimeError(f"AMC-pair produced {len(pose_channels)} pose channels, expected {expected}")
        seq_pose_cond_by_i.append(torch.stack(pose_channels, dim=0))
        seq_xyz_cond_by_i.append(torch.from_numpy(np.stack(xyz_channels, axis=1)))

    seq_pose_conds = torch.stack(seq_pose_cond_by_i, dim=0).unsqueeze(0)
    seq_xyz_conds = (torch.stack(seq_xyz_cond_by_i, dim=1) * multi) / baseline_xyz_divisor

    if seq_pose_conds.shape[2] != expected:
        raise RuntimeError(f"seq_pose_conds channel dim {seq_pose_conds.shape[2]} != expected {expected}")
    if seq_xyz_conds.shape[2] != expected:
        raise RuntimeError(f"seq_xyz_conds channel dim {seq_xyz_conds.shape[2]} != expected {expected}")

    return seq_pose_conds, seq_xyz_conds


def get_seq_pose_xyz_cond_msti(
    pose_index,
    time_steps,
    interval,
    get_pose_xyz_func,
    delta_pose_xyz_cache,
    multi=1.0,
    msti_mode="lite",
    expected_channels=None,
):
    if msti_mode not in {"lite", "full"}:
        raise ValueError(f"MSTI mode must be 'lite' or 'full', got {msti_mode}")

    time_step_items = list(time_steps.items())
    if not time_step_items:
        raise ValueError("time_steps must not be empty")
    seq_lens = {int(seq_len) for _, seq_len in time_step_items}
    if len(seq_lens) != 1:
        raise ValueError(f"MSTI expects a single seq_len across time steps, got {sorted(seq_lens)}")
    seq_len = seq_lens.pop()

    max_step = max(time_steps.keys())
    if msti_mode == "lite":
        ordered_steps = [(max_step, time_steps[max_step])]
        ordered_steps += [(step, step_len) for step, step_len in time_step_items if step != max_step]
        expected = len(time_step_items) + 2
    else:
        ordered_steps = time_step_items
        expected = len(time_step_items) * 3

    if expected_channels is not None and int(expected_channels) != expected:
        raise ValueError(
            f"MSTI condition channel mismatch: expected_channels={expected_channels}, "
            f"computed={expected}, mode={msti_mode}, base_steps={len(time_step_items)}"
        )

    def get_pair_delta(cur_id, former_id):
        delta_key = f"msti_real_dt_v1:{cur_id}-{former_id}"
        if delta_key not in delta_pose_xyz_cache:
            cur_pose_mat, cur_obs_xyz = get_pose_xyz_func(cur_id)
            former_pose_mat, former_obs_xyz = get_pose_xyz_func(former_id)
            posedelta = _relative_pose_delta(cur_pose_mat, former_pose_mat)
            dt = max(int(cur_id) - int(former_id), 1)
            xyz_delta = ((cur_obs_xyz - former_obs_xyz) * multi) / float(dt)
            delta_pose_xyz_cache[delta_key] = {
                "pose_delta": posedelta,
                "xyz_delta": xyz_delta.astype(np.float32, copy=False),
            }
        return (
            delta_pose_xyz_cache[delta_key]["pose_delta"],
            delta_pose_xyz_cache[delta_key]["xyz_delta"],
        )

    seq_pose_cond_by_i, seq_xyz_cond_by_i = [], []
    for i in range(seq_len):
        pose_channels, xyz_channels = [], []
        for time_step, _ in ordered_steps:
            cur_id = max((pose_index - i * time_step) // interval, 0)
            former_id = max((pose_index - (i + 1) * time_step) // interval, 0)

            d_pose_full, d_xyz_full = get_pair_delta(cur_id, former_id)
            expand_step = msti_mode == "full" or (msti_mode == "lite" and time_step == max_step)
            if expand_step:
                mid_id = former_id + (cur_id - former_id) // 2
                d_pose_sub1, d_xyz_sub1 = get_pair_delta(mid_id, former_id)
                d_pose_sub2, d_xyz_sub2 = get_pair_delta(cur_id, mid_id)
                pose_channels.extend([d_pose_full, d_pose_sub1, d_pose_sub2])
                xyz_channels.extend([d_xyz_full, d_xyz_sub1, d_xyz_sub2])
            else:
                pose_channels.append(d_pose_full)
                xyz_channels.append(d_xyz_full)

        if len(pose_channels) != expected:
            raise RuntimeError(f"MSTI produced {len(pose_channels)} pose channels, expected {expected}")
        seq_pose_cond_by_i.append(torch.stack(pose_channels, dim=0))
        seq_xyz_cond_by_i.append(torch.from_numpy(np.stack(xyz_channels, axis=1)))

    seq_pose_conds = torch.stack(seq_pose_cond_by_i, dim=0).unsqueeze(0)
    seq_xyz_conds = torch.stack(seq_xyz_cond_by_i, dim=1)

    if seq_pose_conds.shape[2] != expected:
        raise RuntimeError(f"seq_pose_conds channel dim {seq_pose_conds.shape[2]} != expected {expected}")
    if seq_xyz_conds.shape[2] != expected:
        raise RuntimeError(f"seq_xyz_conds channel dim {seq_xyz_conds.shape[2]} != expected {expected}")

    return seq_pose_conds, seq_xyz_conds

sceneLoadTypeCallbacks = {
    "ZJU_MoCap" : readZJUMoCapInfo,
    "DNARendering": readDNARenderingInfo,
    "I3DHuman": readID3HumanInfo,
}
