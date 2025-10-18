import os
import json
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import pathlib

from PIL import Image
import open3d as o3d

from tqdm import tqdm

import argparse

import numpy_indexed as npi


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

class RayTracingScene:
    def __init__(self, mesh):
        mesh0 = o3d.t.geometry.TriangleMesh()
        mesh0.vertex['positions'] = o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.Dtype.Float32)
        mesh0.vertex['normals'] = o3d.core.Tensor(np.asarray(mesh.vertex_normals), dtype=o3d.core.Dtype.Float32)
        mesh0.vertex['colors'] = o3d.core.Tensor(np.asarray(mesh.vertex_colors), dtype=o3d.core.Dtype.Float32)
        mesh0.triangle['indices'] = o3d.core.Tensor(np.asarray(mesh.triangles), dtype=o3d.core.Dtype.Int32)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh0)

        self.scene = scene

    def __call__(self, xyzs, width, height, intrinsics, extrinsics):
        icoor = np.zeros((width, height, 3), dtype=np.float32)
        icoor[..., 0] = np.arange(0, width)[:, None]
        icoor[..., 1] = np.arange(0, height)[None, :]
        icoor[..., 2] = 1
        icoor = icoor.reshape(-1, 3)
        icoor = icoor @ np.linalg.inv(intrinsics)

        ray_d = icoor
        ray_o = np.zeros_like(ray_d)

        cam2world = np.linalg.inv(extrinsics)
        ray_o = ray_o @ cam2world[:3, :3] + cam2world[3, :3]
        ray_d = ray_d @ cam2world[:3, :3]

        ray = o3d.core.Tensor(
            np.stack([ray_o, ray_d], axis=1).reshape(-1, 6),
            dtype=o3d.core.Dtype.Float32)
        result = self.scene.cast_rays(ray)

        depth_map = result["t_hit"].numpy()
        depth_map = depth_map.reshape(width, height).transpose(1, 0)

        cam_coords = (xyzs @ extrinsics[:3, :3] + extrinsics[3, :3]) @ intrinsics

        x = cam_coords[:, 0] / cam_coords[:, 2]
        y = cam_coords[:, 1] / cam_coords[:, 2]
        z = cam_coords[:, 2]

        mask = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (z > 0)
        mask_ind = np.nonzero(mask)[0]

        xx = x[mask].astype(int)
        yy = y[mask].astype(int)
        sample_depth = depth_map[yy, xx]
        # print(xx.shape, yy.shape, sample_depth.shape, z[mask].shape)

        # mask = mask & (np.abs(sample_depth - z[mask]) < 0.1)
        mask[mask_ind[np.abs(sample_depth - z[mask]) > 0.1]] = False
        # print(mask.sum(), mask_ind.shape, mask.shape)

        return x[mask], y[mask], z[mask], mask
    
    def world_space_color(self, image, width, height, intrinsics, extrinsics):
        icoor = np.zeros((width, height, 3), dtype=np.float32)
        icoor[..., 0] = np.arange(0, width)[:, None]
        icoor[..., 1] = np.arange(0, height)[None, :]
        icoor[..., 2] = 1
        icoor = icoor.reshape(-1, 3)
        icoor = icoor @ np.linalg.inv(intrinsics)

        ray_d = icoor
        ray_o = np.zeros_like(ray_d)

        cam2world = np.linalg.inv(extrinsics)
        ray_o = ray_o @ cam2world[:3, :3] + cam2world[3, :3]
        ray_d = ray_d @ cam2world[:3, :3]

        ray = o3d.core.Tensor(
            np.stack([ray_o, ray_d], axis=1).reshape(-1, 6),
            dtype=o3d.core.Dtype.Float32)
        result = self.scene.cast_rays(ray)

        depth_map = result["t_hit"].numpy()
        depth_map = depth_map.reshape(-1, 1)

        mask = np.isfinite(depth_map).reshape(-1)

        # xyz_0 = ray_o + ray_d * depth_map


        xyz_map = (ray_o + ray_d * depth_map)
        colors = image.transpose(1, 0, 2).reshape(-1, 3)
        # colors = image.reshape(-1, 3)[mask]

        return mask, xyz_map, colors

def sample_voxels(
        mesh_path: str,
        camera_path: str,
        n_points: int,
        image_scale: float,
):
    
    print('Loading mesh...')
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    if not mesh.has_vertex_colors() and not mesh.has_textures():
        print("The mesh does not have vertex colors or textures.")
        exit()

    print('Sampling points...')
    sampled_points = mesh.sample_points_uniformly(number_of_points=2 * n_points)

    colors = np.asarray(sampled_points.colors)
    points = np.asarray(sampled_points.points)

    np.save("temp/sampled_points.npy", np.hstack((points, colors)))

    print('Filtering points...')    
    scene = RayTracingScene(mesh)

    cameras = json.load(open(camera_path, 'r'))['KRT']
    seen_mask = np.zeros(len(points), dtype=bool)

    pbar = tqdm(range(len(cameras)), desc='Filtering points')
    for i in pbar:
        camera = cameras[i]
        width = int(camera['width'] // image_scale)
        height = int(camera['height'] // image_scale)
        intrinsics = np.array(camera['K'])
        extrinsics = np.array(camera['T'])
        intrinsics[:, :2] /= image_scale

        # mask, _, _ = scene.world_space_color(np.zeros((width, height, 3)), width, height, intrinsics, extrinsics)
        
        _, _, _, mask = scene(points, width, height, intrinsics, extrinsics)
        seen_mask = seen_mask | mask

        pbar.set_postfix({'seen': seen_mask.sum(), 'Image scale': image_scale, 'Image Shape': (width, height)})

    points = points[seen_mask]
    colors = colors[seen_mask]

    np.save("temp/seen_points.npy", np.hstack((points, colors)))

    print('Downsampling points...')
    random_indices = np.random.choice(len(points), n_points, replace=False)
    points = points[random_indices]
    colors = colors[random_indices]

    np.save("temp/final_points.npy", np.hstack((points, colors)))

    print('Saving voxels...')
    voxels, _ = pcd_downsample(points, 0.2, 2000, 'mean')
    np.save("temp/voxels.npy", voxels)
    
    seen_voxel = np.zeros(len(voxels), dtype=bool)
    pbar = tqdm(range(len(cameras)), desc='Filtering voxels')
    for i in pbar:
        camera = cameras[i]
        width = int(camera['width'] // image_scale)
        height = int(camera['height'] // image_scale)
        intrinsics = np.array(camera['K'])
        extrinsics = np.array(camera['T'])
        intrinsics[:, :2] /= image_scale

        _, _, _, mask = scene(voxels, width, height, intrinsics, extrinsics)
        seen_voxel = seen_voxel | mask

        pbar.set_postfix({'seen': f"{seen_voxel.sum() / len(voxels) * 100: .2f}%", 'Image scale': image_scale, 'Image Shape': (width, height)})

    print(f'Number of seen voxels: {seen_voxel.sum()}')
    print(f'% of seen voxels: {seen_voxel.sum() / len(voxels) * 100:.2f}%')

    return voxels

class DinoFeatureExtractor:
    def __init__(self, model_id, img_h, img_w, noise_level=0.0):
        self.model = torch.hub.load('facebookresearch/dinov2', model_id).to('cuda')
        self.model.eval()

        self.patch_size = self.model.patch_size

        self.transform = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda x: x + noise_level * torch.randn_like(x))
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
        voxels, img_path, 
        model_id, img_h, img_w, 
        cam_path, img_scale, 
        mesh_path, camera_ids_path=None, output_path='temp', noise_level=0.0):
    
    output_path = pathlib.Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    score_fn = lambda x: np.exp(- (x - 2)**2)

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    scene = RayTracingScene(mesh)

    dino = DinoFeatureExtractor(model_id, img_h, img_w)
    buffer = np.zeros((len(voxels), 8, 1024), dtype=np.float32)
    scores = -np.ones((len(voxels), 8), dtype=np.float32)
    cam_ids = np.zeros((len(voxels), 8), dtype=np.int32)

    cameras = json.load(open(cam_path, 'r'))['KRT']

    extrinsics = np.zeros((len(cameras), 16), dtype=np.float32)
    for i, cam in enumerate(cameras):
        extrinsics[i] = np.array(cam['T']).reshape(-1)
    np.save(output_path / 'extrinsics.npy', extrinsics)

    cam_ids_valid = range(len(cameras))
    if camera_ids_path is not None:
        cam_ids_valid = json.load(open(camera_ids_path, 'r'))['camera_idx']

    print(f'Number of cameras: {len(cam_ids_valid)}')

    pbar = tqdm(cam_ids_valid, desc='Extracting features')
    tot_mask = np.zeros(len(voxels), dtype=bool)
    for j, i in enumerate(pbar):
        camera = cameras[i]
        width = int(camera['width'] // img_scale)
        height = int(camera['height'] // img_scale)
        intrinsics = np.array(camera['K'])
        extrinsics = np.array(camera['T'])
        intrinsics[:, :2] /= img_scale

        image = Image.open(
            f'{img_path}/{camera["cameraId"]}.jpg'
        )
        image_arr = np.array(image).astype(np.float32)

        cx, cy, cz, mask = scene(voxels, width, height, intrinsics, extrinsics)

        feature = dino.extract_features(
            image, height, width
        )
        cx = np.round(cx).astype(np.int32)
        cy = np.round(cy).astype(np.int32)
        
        cx[cx < 0] = 0
        cy[cy < 0] = 0
        cx[cx >= width] = width - 1
        cy[cy >= height] = height - 1

        score = score_fn(cz)
        cid = i 

        min_score_id = np.argmin(scores[mask], axis=1)
        update_mask = scores[mask, min_score_id] < score
        update_bin = min_score_id[update_mask]
        update_indices = np.nonzero(mask)[0][update_mask]

        buffer[update_indices, update_bin] = feature[cy[update_mask], cx[update_mask]]
        scores[update_indices, update_bin] = score[update_mask]
        cam_ids[update_indices, update_bin] = cid

        if j % 20 == 0:
            normalize = lambda x: (x - x.min()) / (x.max() - x.min())
            image.resize((width, height)).save(f'temp-png/{camera["cameraId"].replace("/", "_")}.png')
            Image.fromarray(
                np.uint8(np.clip(normalize(feature[:, :, :3]), 0, 1) * 255)
            ).save(f'temp-png/{camera["cameraId"].replace("/", "_")}_feat.png')

            feature_0 = np.clip(normalize(feature[:, :, :3]), 0, 1)
            feature_0[cy, cx] = 0
            Image.fromarray(
                np.uint8(feature_0 * 255)
            ).save(f'temp-png/{camera["cameraId"].replace("/", "_")}_feat_0.png')
            Image.fromarray(
                np.uint8(np.clip(normalize(np.linalg.norm(feature, axis=-1)), 0, 1) * 255)
            ).save(f'temp-png/{camera["cameraId"].replace("/", "_")}_feat_norm.png')

        tot_mask = tot_mask | mask
        
        pbar.set_postfix({'seen': f"{tot_mask.sum() / len(voxels) * 100: .2f}%", 'Image scale': img_scale, 'Image Shape': (width, height)})

    np.save(output_path / 'features.npy', buffer[tot_mask])
    np.save(output_path / 'scores.npy', scores[tot_mask])
    np.save(output_path / 'cam_ids.npy', cam_ids[tot_mask])
    np.save(output_path / 'voxels.npy', voxels[tot_mask])


parser = argparse.ArgumentParser()
parser.add_argument("--mesh_path", type=str, required=True) # e.g. mesh/FurnishedRoom.obj
parser.add_argument("--camera_path", type=str, required=True) # e.g. <raf-path>/raf_furnishedroom/cameras.json
parser.add_argument("--image_path", type=str, required=True) # e.g. <raf-path>/raf_furnishedroom/cameras.json
parser.add_argument('--output_path', type=str, required=True) # e.g. image-features/raf/FurnishedRoom
parser.add_argument('--camera_ids_path', type=str, default=None) # please provide json file with {"camera_idx": [0, 1, 2, ...]} if you want to use a subset of cameras
parser.add_argument("--n_points", type=int, default=100000)
parser.add_argument("--image_scale", type=float, default=30)
parser.add_argument('--model_id', type=str, default='dinov2_vitl14')
parser.add_argument("--dino_img_h", type=int, default=756)
parser.add_argument("--dino_img_w", type=int, default=504)
parser.add_argument("--noise_level", type=float, default=0.0)
parser.add_argument('--voxel_path', type=str, default=None)
if __name__ == '__main__':
    args = parser.parse_args()
    if args.voxel_path is not None:
        voxels = np.load(args.voxel_path)
    else:
        voxels = sample_voxels(args.mesh_path, args.camera_path, args.n_points, args.image_scale)

    # voxels = np.load('temp/voxels.npy')
    extract_features(voxels, args.image_path,
                     args.model_id, args.dino_img_h, args.dino_img_w, 
                     args.camera_path, args.image_scale, args.mesh_path, args.camera_ids_path, args.output_path, args.noise_level)