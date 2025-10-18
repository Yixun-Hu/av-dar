import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import pathlib

from PIL import Image
import open3d as o3d

from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

import argparse

import numpy_indexed as npi

def json_load(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def pcd_downsample(xyzs, voxel_size, res=1000, center_mode='random'):
    v_coods = np.floor(xyzs / voxel_size).astype(np.int64)

    voxel_id = v_coods[:, 0] * res * res * 4 + v_coods[:, 1] * res * 4 + v_coods[:, 2]

    unique_voxel_ids, unique_indices, rev_indices = np.unique(voxel_id, return_index=True, return_inverse=True)

    # center_mode \in ['random', 'mean']
    if center_mode == 'random':
        down_pcd = xyzs[unique_indices]
    elif center_mode == 'mean':
        down_pcd = np.zeros((len(unique_indices), 3))
        label_cnt = np.bincount(rev_indices)
        mask = label_cnt > 0

        print(f'Number of voxels: {mask.sum()}')
        print(f'% of voxels: {mask.sum() / len(unique_indices) * 100:.2f}%')
        
        down_pcd[mask, 0] = np.bincount(rev_indices, weights=xyzs[:, 0])[mask] / label_cnt[mask]
        down_pcd[mask, 1] = np.bincount(rev_indices, weights=xyzs[:, 1])[mask] / label_cnt[mask]
        down_pcd[mask, 2] = np.bincount(rev_indices, weights=xyzs[:, 2])[mask] / label_cnt[mask]
        # import IPython; IPython.embed(); exit(1)
        dist_to_center = np.linalg.norm(xyzs - down_pcd[rev_indices], axis=1)
        unique, amin = npi.group_by(rev_indices).argmin(dist_to_center)
        down_pcd[unique] = xyzs[amin]

    else:
        raise ValueError('center_mode should be in [random  mean]')

    return down_pcd, rev_indices

def to_pix_space(xyzs, camera_pose, intrinsics, width, height, transform=None):
    w2c = np.linalg.inv(camera_pose)

    xyzts = np.hstack([xyzs, np.ones((len(xyzs), 1))])

    if transform is not None:
        xyzts = np.dot(xyzts, transform.T)

    xyzts = np.dot(xyzts, w2c.T)
    xyzts = np.dot(xyzts, intrinsics.T)
    ndc = xyzts[:, :3] / xyzts[:, 3:]
    mask = (ndc[:, 0] >= -1) & (ndc[:, 0] <= 1) & (ndc[:, 1] >= -1) & (ndc[:, 1] <= 1) & (ndc[:, 2] <= 1) & (ndc[:, 2] >= -1)
    ndc = ndc[mask]

    x, y, z = ndc[:, 0], ndc[:, 1], ndc[:, 2]
    u = ((x + 1) * width / 2).astype(int)
    v = ((1 - y) * height / 2).astype(int)
    d = (z + 1) / 2
    return mask, u, v, d

class RayTracingScene:
    def __init__(self, mesh):
        mesh0 = o3d.t.geometry.TriangleMesh()
        mesh0.vertex['positions'] = o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.Dtype.Float32)
        mesh0.vertex['normals'] = o3d.core.Tensor(np.asarray(mesh.vertex_normals), dtype=o3d.core.Dtype.Float32)
        mesh0.triangle['indices'] = o3d.core.Tensor(np.asarray(mesh.triangles), dtype=o3d.core.Dtype.Int32)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh0)

        self.scene = scene

    def __call__(self, xyzs, width, height, intrinsics, camera_pose, transform=None):

        in_cam_mask, u, v, d = to_pix_space(xyzs, camera_pose, intrinsics, width, height, transform=transform)
        view_point = camera_pose[:4, 3]
        if transform is not None:
            view_point = np.dot(view_point, np.linalg.inv(transform).T)


        ray_d = xyzs[in_cam_mask]
        ray_o = np.zeros_like(ray_d)
        ray_o[:] = view_point[:3]
        ray_d = ray_d - ray_o
        dist = np.linalg.norm(ray_d, axis=1)
        ray_d = ray_d / dist[:, None]

        ray = o3d.core.Tensor(
            np.stack([ray_o, ray_d], axis=1).reshape(-1, 6).astype(np.float32),
            dtype=o3d.core.Dtype.Float32)
        result = self.scene.cast_rays(ray)

        depth_map = result["t_hit"].numpy()
        depth_map = np.clip(depth_map, 0, 1000)
        
        in_scene_mask = depth_map > dist - 0.01
        in_cam_indices = np.where(in_cam_mask)[0]
        mask_indices = in_cam_indices[in_scene_mask]

        mask = np.zeros(len(xyzs), dtype=bool)
        mask[mask_indices] = True

        return mask, u[in_scene_mask], v[in_scene_mask], d[in_scene_mask], dist[in_scene_mask]


class DinoFeatureExtractor:
    def __init__(self, model_id, img_h, img_w):
        self.model = torch.hub.load('facebookresearch/dinov2', model_id).to('cuda')
        self.model.eval()

        self.patch_size = self.model.patch_size

        self.transform = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
        ])

        self.n_patches_h = img_h // self.patch_size
        self.n_patches_w = img_w // self.patch_size

        self.feat_dim = 1024

    @torch.no_grad()
    def extract_features(self, img, img_h=None, img_w=None):
        # img = Image.fromarray(img_arr)
        img_t = self.transform(img).to('cuda')

        features = self.model.forward_features(img_t.unsqueeze(0))['x_norm_patchtokens']
        features = features.reshape(self.n_patches_h, self.n_patches_w, self.feat_dim)
        # return features.cpu().numpy()
        if img_h is None:
            img_h = self.n_patches_h

        if img_w is None:
            img_w = self.n_patches_w

        # bilinear interpolation
        features = torch.permute(features, (2, 0, 1))
        features = F.interpolate(features.unsqueeze(0), (img_h, img_w), mode='bilinear', align_corners=False)
        features = features.squeeze(0)
        features = torch.permute(features, (1, 2, 0))

        return features.cpu().numpy()

def extract_features(
        voxels,
        model_id, 
        img_h, img_w, 
        camera_meta_data,
        camera_data,
        geometry_path,
        output_path='temp',
    ):
    
    output_path = pathlib.Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    camera_data = json_load(camera_data)
    camera_meta_data = json_load(camera_meta_data)

    score_fn = lambda x: np.exp(- (x- 2)**2)

    mesh = o3d.io.read_triangle_mesh(str(geometry_path))
    mesh.compute_vertex_normals()
    scene = RayTracingScene(mesh)

    dino = DinoFeatureExtractor(model_id, img_h, img_w)
    buffer = np.zeros((len(voxels), 4, 1024), dtype=np.float32)
    scores = -np.ones((len(voxels), 4), dtype=np.float32)
    cam_ids = np.zeros((len(voxels), 4), dtype=np.int32)
    extrinsics = np.zeros((len(camera_data), 16), dtype=np.float32)

    width = int(camera_meta_data['width'])
    height = int(camera_meta_data['height'])

    intrinsics = np.array(camera_meta_data['intrinsics'])
    base_transform = np.array(camera_meta_data['geometry_to_visual'])

    for i, cam in enumerate(camera_data):
        extrinsics[i] = np.array(cam['pose']).reshape(-1)

    seen_mask = np.zeros(len(voxels), dtype=bool)
    progress = tqdm(camera_data, desc='Extracting features')

    for i, cam in enumerate(progress):
        extrinsic = np.array(cam['pose'])
        image_path = images_path / cam['file']

        image = Image.open(image_path)
        image_arr = np.array(image).astype(np.float32)

        mask, u, v, d, t = scene(
            voxels, width, height, intrinsics, extrinsic, transform=base_transform
        )

        feature = dino.extract_features(
            image, height, width
        )
        
        seen_mask = seen_mask | mask
        score = score_fn(t)
        cid = i

        min_score_id = np.argmin(scores[mask], axis=1)
        update_mask = scores[mask, min_score_id] < score
        update_bin = min_score_id[update_mask]
        update_indices = np.nonzero(mask)[0][update_mask]

        buffer[update_indices, update_bin] = feature[v[update_mask], u[update_mask]]
        scores[update_indices, update_bin] = score[update_mask]
        cam_ids[update_indices, update_bin] = cid

        if i % 5 == 0:
            base = output_path / 'temp-png'
            base.mkdir(exist_ok=True, parents=True)
            normalize = lambda x: (x - x.min()) / (x.max() - x.min())
            image.resize((width, height)).save(base / cam['file'])
            Image.fromarray(
                np.uint8(np.clip(normalize(feature[:, :, :3]), 0, 1) * 255)
            ).save(base / f'feat_3_{cam["file"]}')

            feature_0 = np.clip(normalize(feature[:, :, :3]), 0, 1)
            feature_0[v, u] = 0
            Image.fromarray(
                np.uint8(feature_0 * 255)
            ).save(base / f'feat_0_{cam["file"]}')
            Image.fromarray(
                np.uint8(np.clip(normalize(np.linalg.norm(feature, axis=-1)), 0, 1) * 255)
            ).save(base / f'feat_norm_{cam["file"]}')
        
        progress.set_postfix({'seen': f"{seen_mask.sum() / len(voxels) * 100: .2f}%", 'Image Shape': (width, height)})

    np.save(output_path / 'features.npy', buffer[seen_mask])
    np.save(output_path / 'scores.npy', scores[seen_mask])
    np.save(output_path / 'cam_ids.npy', cam_ids[seen_mask])
    np.save(output_path / 'voxels.npy', voxels[seen_mask])
    np.save(output_path / 'extrinsics.npy', extrinsics)

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument('--output_root', type=str, required=True)
parser.add_argument('--model_id', type=str, default='dinov2_vitl14')
parser.add_argument("--dino_img_h", type=int, default=518)
parser.add_argument("--dino_img_w", type=int, default=518)
parser.add_argument("--name", type=str, default="classroomBase", choices=["classroomBase", "dampenedBase", "hallwayBase", "complexBase"])
if __name__ == '__main__':
    args = parser.parse_args()

    base_path = pathlib.Path(args.data_root) / args.name
    output_path = pathlib.Path(args.output_root) / args.name

    images_path = base_path / 'images'
    camera_data = base_path / 'selected_camera_meta_data.json'
    camera_meta_data = base_path / 'camera_meta_data.json'
    geometry_path = base_path / f'{args.name}.obj'
    voxels_path = base_path / 'voxel_samples.npy'

    voxels = np.load(voxels_path)
    extract_features(
        voxels,
        args.model_id,
        args.dino_img_h, 
        args.dino_img_w,
        camera_meta_data,
        camera_data, 
        geometry_path,
        output_path,
    )