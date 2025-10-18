import open3d as o3d
import numpy as np


class RayTracingScene:
    def __init__(self, file_name):
        raw_mesh = o3d.io.read_triangle_mesh(file_name)
        raw_mesh.compute_vertex_normals()
        tmesh = o3d.t.geometry.TriangleMesh()
        tmesh.vertex['positions'] = o3d.core.Tensor(np.array(raw_mesh.vertices), o3d.core.float32)
        tmesh.triangle["indices"] = o3d.core.Tensor(np.array(raw_mesh.triangles), o3d.core.int32)
        # import IPython; IPython.embed()

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(tmesh)

        self.mesh = raw_mesh
        self.sampled_xyzs = None

    def sample(self, n_points):
        sampled_points = self.mesh.sample_points_uniformly(number_of_points=n_points, use_triangle_normal=True)
        return sampled_points
    
    def compute_camera_visibility(self, xyzs, intrinsics, extrinsics, width, height):
        w2c = extrinsics
        c2w = np.linalg.inv(w2c)
        normalized = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
        cam_to_world = lambda x: (x @ np.linalg.inv(intrinsics)) @ c2w[:3, :3] + c2w[3, :3]
        world_to_cam = lambda x: (x @ w2c[:3, :3] + w2c[3, :3]) @ intrinsics

        # w2c = np.linalg.inv(extrinsics)
        
        # xyz1s = np.concatenate([xyzs, np.ones((xyzs.shape[0], 1))], axis=1)
        
        # cam_xyzs = np.dot(xyz1s, w2c)
        # cam_xyzs = cam_xyzs[:, :3]
        # # vis_3dpoints(cam_xyzs)

        # pix_xyzs = np.dot(cam_xyzs, intrinsics)
        pix_xyzs = world_to_cam(xyzs)
        cam_x = pix_xyzs[:, 0] / pix_xyzs[:, 2]
        cam_y = pix_xyzs[:, 1] / pix_xyzs[:, 2]
        cam_z = pix_xyzs[:, 2]

        mask = (cam_x >= -.5) & (cam_x < width-.5) & (cam_y >= -.5) & (cam_y < height-.5)
        mask = mask & (cam_z > 0)

        # return mask, cam_x, cam_y, cam_z

        rays_o = np.zeros((mask.sum(), 3))
        rays_d = np.zeros((mask.sum(), 3))

        rays_d[:, 0] = cam_x[mask]
        rays_d[:, 1] = cam_y[mask]
        rays_d[:, 2] = 1

        rays_o[:, 2] = 0

        # rays_o = rays_o
        
        rays_o = cam_to_world(rays_o)
        rays_d = cam_to_world(rays_d) - rays_o
        rays_d = normalized(rays_d)


        o3d_rays = o3d.core.Tensor(
            np.stack([
                rays_o, rays_d
            ], axis=1).reshape([-1, 6]),
            o3d.core.float32
        )
        rc_results = self.scene.cast_rays(o3d_rays)
        t = rc_results['t_hit'].numpy().reshape(-1, 1)
        # print(t.min(), t.max(), t.mean(), t.shape)
        # hit_xyz1s = np.concatenate((rays_o + rays_d * t, np.ones((t.shape[0], 1))), axis=1)
        # hit_cam_z = ((hit_xyz1s @ w2c)[:, :3] @ intrinsics)[:, 2]
        # import IPython; IPython.embed()

        hit_world_xyz = rays_o + rays_d * t
        hit_cam_xyz = world_to_cam(hit_world_xyz)
        hit_cam_z = hit_cam_xyz[:, 2]
        mask[np.nonzero(mask)[0][cam_z[mask] > hit_cam_z + 1e-3]] = False
        # print(hit_cam_z.min(), hit_cam_z.max(), hit_cam_z.mean(), hit_cam_z.shape)
        # print(cam_z[mask].min(), cam_z[mask].max(), cam_z[mask].mean(), cam_z[mask].shape)

        return mask, cam_x, cam_y, cam_z
    

    def visible(self, src, dst):
        if src.ndim == 1:
            src = src.reshape(1, 3)
        if dst.ndim == 1:
            dst = dst.reshape(1, 3)
        
        rays_d = dst - src
        rays_d = rays_d / np.linalg.norm(rays_d, axis=1, keepdims=True)

        t = self.calculate_ray_intersection(src + rays_d * 1e-3, rays_d) + 1e-3
        distance = np.linalg.norm(dst - src, axis=1, keepdims=True)
        return (t > distance - 1e-3).squeeze(1)

    
    def calculate_ray_intersection(self, rays_o, rays_d, return_normals=False):
        query =  np.stack([
            rays_o, rays_d
        ], axis=1).reshape([-1, 6]).astype(np.float32)
        o3d_query = o3d.core.Tensor(query, o3d.core.float32)

        rc_results = self.scene.cast_rays(o3d_query)
        t = rc_results['t_hit'].numpy().reshape(-1, 1).astype(rays_o.dtype)

        if return_normals:
            normals = rc_results['primitive_normals'].numpy().reshape(-1, 3).astype(rays_o.dtype)
            return t, normals

        return t

    def cast_rays(self, rays_o, rays_d):
        query =  np.stack([
            rays_o, rays_d
        ], axis=1).reshape([-1, 6]).astype(np.float32)
        o3d_query = o3d.core.Tensor(query, o3d.core.float32)

        rc_results = self.scene.cast_rays(o3d_query)
        t = rc_results['t_hit'].numpy().astype(rays_o.dtype)

        mask = np.isfinite(t)
        hit_xyz = rays_o + rays_d * t[..., None]
        return mask, hit_xyz




