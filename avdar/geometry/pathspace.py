'''
Discrete path space model
'''
import numpy as np
import numpy_indexed as npi

from .mesh_scene import RayTracingScene
from ..utils.sample_utils import sample_fibonacci_sphere

import logging


logger = logging.getLogger(__name__)

class SpecularPathSampler:
    def __init__(self, sampler, max_length: int):
        self.sampler = sampler
        self.max_path_length = max_length
        # self.cache = {}

    def reset_length(self, max_path_length):
        self.max_path_length = max_path_length
        # self.cache = {}

    def fast_sample(self, source_xyz, listener_xyz):
        return self.sampler.get_source_sampler(
            src_xyz=source_xyz, max_steps=self.max_path_length
        )(listener_xyz)
    
    def get_sampler(self, source_xyz):
        return self.sampler.get_source_sampler(
            src_xyz=source_xyz, max_steps=self.max_path_length
        )
    
    @classmethod
    def from_config(cls, config_dict, max_length, mesh_path):
        meta_sampler_cls = {
            'beam_tracing': BeamTracingPathSampler,
        }[config_dict.name]
        scene = RayTracingScene(mesh_path)
        _sampler = meta_sampler_cls(scene, **config_dict.options)
        return cls(_sampler, max_length)


class BeamTracingPathSampler(SpecularPathSampler):
    ASPECT = {
        512: 0.11962,
        1024: 0.08459,
        2048: 0.05935,
        4096: 0.04077,
        8192: 0.02736,
        16384: 0.01960,
    }
    def __init__(self, scene, num_sample_directions, dist_thresh=1000, deterministic=False):
        self.scene:RayTracingScene = scene
        self.num_sample_directions = num_sample_directions
        # self.aspect = aspect
        self.aspect = BeamTracingPathSampler.ASPECT.get(
            num_sample_directions, 0.0014 * np.log2(num_sample_directions)
        )
        self.dist_thresh = dist_thresh
        self.deterministic = deterministic

    def sample(self, src_xyz, dst_xyz, max_steps):
        return self.get_source_sampler(src_xyz, max_steps)(dst_xyz)
    
    def sample_reflect(self, rays_d, normals, return_local_transform=False):
        '''
        compute the specular reflection directions: d' = d - 2(d dot n)n
        '''

        cos_theta = np.sum(rays_d * normals, axis=-1, keepdims=True)
        proj_normal = cos_theta * normals
        specular = rays_d - 2 * proj_normal
        reflect_rays_d = specular


        if return_local_transform:
            local_transform = np.zeros((rays_d.shape[0], 3, 3), dtype=rays_d.dtype)
            tangent_0 = rays_d - proj_normal
            tangent_0 = tangent_0 / np.linalg.norm(tangent_0, axis=-1, keepdims=True)
            tangent_1 = np.cross(normals, tangent_0)
            local_transform[:, 2] = normals
            local_transform[:, 0] = tangent_0
            local_transform[:, 1] = tangent_1
            return reflect_rays_d, local_transform, np.abs(cos_theta[..., 0])


        return reflect_rays_d
    
    def beam_visible(self, rays_o, rays_d, dst_xyz, path_len):
        '''
        Check whether the beam from rays_o along rays_d can reach dst_xyz:
        true means the beam will reach dst_xyz without additional reflections
        ----------------------------------------
        logic: true if visible_mask & acute_mask & ratio_mask & dist_mask
            - visible_mask: no occlusion between rays_o and dst_xyz
            - acute_mask: angle between rays_d and direction to dst_xyz is acute
            - ratio_mask: ratio between distance to ray and distance along ray is small enough
            - dist_mask: distance to ray is small enough
        ----------------------------------------
        inputs:
            rays_o: (N, 3) ray origins
            rays_d: (N, 3) ray directions (normalized)
            dst_xyz: (3,) destination point
            path_len: (N,) accumulated path length from source to rays_o
        ----------------------------------------
        outputs:
            hit_mask: (N,) boolean mask indicating whether each ray hits the dst_xyz
            path_todst_len: (N,) total path length from source to dst_xyz along the ray
            sin_phi: (N,) see Eq (24) in Our Paper https://arxiv.org/pdf/2504.21847
            last_d: (N, 3) direction from *source* to *dst_xyz* at the last segment
        ----------------------------------------
        '''
        visible_mask = self.scene.visible(rays_o, dst_xyz)

        rays_d_deviate = dst_xyz[None] - rays_o
        alen_deviate = np.linalg.norm(rays_d_deviate, axis=-1)
        rays_d_deviate = rays_d_deviate / (alen_deviate[..., None] + 1e-7)

        cos_neariest = np.sum(rays_d * rays_d_deviate, axis=-1)
        sin_neariest = np.sqrt(1 - cos_neariest ** 2 + 1e-5)
        
        acute_mask = cos_neariest > 0

        dist_to_ray = alen_deviate * sin_neariest
        path_tot_len = path_len + alen_deviate * cos_neariest
        path_todst_len = np.sqrt(np.square(path_tot_len) + np.square(dist_to_ray))

        ratio_aspect = dist_to_ray / path_tot_len
        ratio_mask = ratio_aspect < self.aspect

        dist_mask = dist_to_ray < self.dist_thresh

        hit_mask = visible_mask & acute_mask & ratio_mask & dist_mask

        sin_phi = np.sqrt(1 / (1/ (ratio_aspect ** 2 + 1e-6) + 1))

        last_d = dst_xyz - (rays_o - rays_d * path_len[..., None])
        last_d = last_d / (np.linalg.norm(last_d, axis=-1, keepdims=True) + 1e-7)

        return hit_mask, path_todst_len, sin_phi, last_d

    def get_source_sampler(self, src_xyz, max_steps):

        src_xyz = src_xyz.astype(np.float32)
        rays_o = np.zeros((self.num_sample_directions, 3))
        rays_o[:, :] = src_xyz

        start_rays_d = rays_d = sample_fibonacci_sphere(self.num_sample_directions, random_rotation=(not self.deterministic))[..., [1, 2, 0]]

        # Prepare storage
        path_xyzs = np.full((self.num_sample_directions, max_steps, 3), np.nan, dtype=src_xyz.dtype) # hit points j in path i
        directions = np.zeros((self.num_sample_directions, max_steps, 3), dtype=src_xyz.dtype)  # directions after each hit
        cos_thetas = np.zeros((self.num_sample_directions, max_steps), dtype=src_xyz.dtype) # cos theta at each hit
        local_transforms = np.zeros((self.num_sample_directions, max_steps, 3, 3), dtype=src_xyz.dtype) # local transform at each hit normal-z
        lengths = np.zeros((self.num_sample_directions, max_steps), dtype=src_xyz.dtype) # accumulated lengths at each hit

        accum_lengths = np.zeros((self.num_sample_directions, 1), dtype=src_xyz.dtype)
        valid_indices = np.arange(self.num_sample_directions)
        for i in range(max_steps):

            # Calculate the ray-scene intersection by Open3D
            t_hit, normals = self.scene.calculate_ray_intersection(rays_o + rays_d * 1e-3, rays_d, return_normals=True)

            mask = np.isfinite(t_hit)[:, 0]
            valid_indices = valid_indices[mask]

            rays_o, rays_d, t_hit, normals = rays_o[mask], rays_d[mask], t_hit[mask], normals[mask]

            rays_o = rays_o + rays_d * (t_hit + 1e-3)
            rays_d, local_trans, cos_theta = self.sample_reflect(rays_d, normals, return_local_transform=True)

            accum_lengths = t_hit + accum_lengths[mask]
            path_xyzs[valid_indices, i] = rays_o
            directions[valid_indices, i] = rays_d
            cos_thetas[valid_indices, i] = cos_theta
            local_transforms[valid_indices, i] = local_trans
            lengths[valid_indices, i:i+1] = accum_lengths

        directions = np.concatenate((start_rays_d[:, np.newaxis], directions), axis=1)


        def filter_valid_paths(dst_xyz):
            dst_xyz = dst_xyz.astype(np.float32)
            
            sampled_vert_xyzs = []
            sampled_vert_cos_thetas = []
            sampled_vert_lengths = []
            sampled_vert_local_transforms = []
            sampled_directions = []
            sampled_last_directions = []
            sampled_lengths = []
            sampled_vert_dev_ratio = []
            valid_indices = np.arange(self.num_sample_directions)
            for i in range(max_steps):
                mask = np.isfinite(path_xyzs[valid_indices, i, 0])
                valid_indices = valid_indices[mask]

                rays_o = path_xyzs[valid_indices, i]
                rays_d = directions[valid_indices, i + 1]
                acc_len = lengths[valid_indices, i]

                # Compute visibility
                hit_mask, acc_len, ratio_, last_d = self.beam_visible(rays_o, rays_d, dst_xyz, acc_len)
                hit_indices = valid_indices[hit_mask]

                sampled_vert_xyzs.extend([p for p in path_xyzs[hit_indices, :i + 1]])
                sampled_vert_cos_thetas.extend([c for c in cos_thetas[hit_indices, :i + 1]])
                sampled_vert_lengths.extend([l for l in lengths[hit_indices, :i + 1]])
                sampled_vert_local_transforms.extend([t for t in local_transforms[hit_indices, :i + 1]])
                sampled_vert_dev_ratio.extend([r * np.ones(i + 1, dtype=lengths.dtype) for r in ratio_[hit_mask]])
                
                sampled_directions.extend([d for d in directions[hit_indices, :i + 2]])
                sampled_lengths.extend([l for l in acc_len[hit_mask]])
                sampled_last_directions.extend([d for d in last_d[hit_mask]])

            if self.scene.visible(src_xyz, dst_xyz):
                d = dst_xyz - src_xyz
                l = np.linalg.norm(d)
                d = d / (l + 1e-7)
                sampled_vert_xyzs.append(np.zeros((0, 3)))
                sampled_vert_cos_thetas.append(np.zeros((0)))
                sampled_vert_lengths.append(np.zeros((0)))
                sampled_vert_dev_ratio.append(np.zeros((0)))
                sampled_vert_local_transforms.append(np.zeros((0, 3, 3)))
                sampled_directions.append(d[None])
                sampled_lengths.append(l)
                sampled_last_directions.append(d)

            return {
                'path_indices': None,
                'path_xyzs': sampled_vert_xyzs,
                'path_cos_thetas': sampled_vert_cos_thetas,
                'path_local_transforms': sampled_vert_local_transforms,
                'path_deviate_sin_phi': sampled_vert_dev_ratio,
                'path_lengths': sampled_vert_lengths,
                'directions': sampled_directions,
                'lengths': sampled_lengths,
                'last_directions': sampled_last_directions,
            }
        
        return filter_valid_paths

    

def voxel_normal_filter(xyzs, normals, vox_size, opt):
    indices3d = np.round(xyzs / vox_size).astype(np.int64) # [N, 3]
    indices3d = indices3d - np.min(indices3d, axis=0)
    indices1d = indices3d[:, 0] + indices3d[:, 1] * 1000 + indices3d[:, 2] * 1000 * 1000

    _, indices = np.unique(indices1d, return_inverse=True)
    _, mean_xyzs = npi.group_by(indices).mean(xyzs)
    _, mean_normals = npi.group_by(indices).mean(normals)

    if opt == 'mean':
        logger.info('Using mean normal')
        mean_normals = mean_normals / np.linalg.norm(mean_normals, axis=1, keepdims=True)
        normals = mean_normals[indices]
    
    return normals
