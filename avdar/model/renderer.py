import os
import logging
import math

import torch
import torch.nn as nn
import torchaudio.functional as F
from torch.utils.tensorboard.writer import SummaryWriter

import scipy.fft
import scipy.signal

import torchaudio
import numpy as np

import matplotlib.pyplot as plt

from . import module_registry

from ..geometry.mesh_scene import RayTracingScene
from ..utils.sample_utils import sample_sphere, fibonacci_sphere, sample_hemisphere
from ..core.typing import *
from  ..utils.hrtf_utils import compute_hrirs
from ..utils.nn_utils import LinearInterpolation

logger = logging.getLogger(__name__)

class RirRenderer(nn.Module):

    diffuse_model: nn.Module
    late_model: nn.Module
    spec_model: nn.Module
    source_resp: nn.Module
    pcd_feat_extractor: nn.Module

    # TODO: maxbounce to maxorder
    def __init__(
        self,
        scene: RayTracingScene,
        speed_of_sound: float,
        sample_rate: int,
        frequency_min: int, 
        frequency_max: int,
        frequency_num: int,
        filter_length: int,
        rir_length: int,
        src_opts: Dict[str, Any],
        src_ir_opts: Dict[str, Any],
        spec_ir_opts: Dict[str, Any],
        late_ir_opts: Dict[str, Any], 
        feat_extractor_opts: Dict[str, Any],
        diffuse_ir_opts: Dict[str, Any],
        hrtf_opts: Dict[str, Any] = None,
        n_ambient_samples: int = 1024,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cpu')
    ):
        super().__init__()
        self.scene = scene

        ## Store the parameters
        self.speed_of_sound = speed_of_sound
        self.sample_rate = sample_rate
        self.filter_length = filter_length
        self.rir_length = rir_length
        self.nyq = self.sample_rate / 2
        self.n_key_freqs = frequency_num
        self.device = device # don't load anything into device in `__init__`
        self.dtype = dtype
        self.n_ambient_samples = n_ambient_samples

        logger.info(f'Number of ambient samples: {self.n_ambient_samples}')

        self.n_freq_samples = 1 + 2 ** int(math.ceil(math.log2(self.filter_length)))
        frequencies = torch.logspace(math.log10(frequency_min), math.log10(frequency_max), frequency_num)

        shared_opts = {
            'dtype': dtype,
            'device': device,
        }

        self.diffuse_model = self.load_sub_model(diffuse_ir_opts['name'], diffuse_ir_opts['options'], shared_opts)
        self.source_dir_model = self.load_sub_model(src_ir_opts['name'], src_ir_opts['options'], shared_opts)
        self.pcd_feat_extractor = self.load_sub_model(feat_extractor_opts['name'], feat_extractor_opts['options'], shared_opts)
        self.spatial_feat_extractor = self.pcd_feat_extractor
        self.spec_model = self.load_sub_model(spec_ir_opts['name'], spec_ir_opts['options'], shared_opts)
        self.late_model = self.load_sub_model(late_ir_opts['name'], late_ir_opts['options'], shared_opts)
        self.source_resp = self.load_sub_model(src_opts['name'], src_opts['options'], shared_opts)
        
        frequency_indices = (frequencies / self.nyq * self.n_freq_samples + .5).int()
        print(frequency_indices)
        self.frequency_interpolator = LinearInterpolation(frequency_indices, self.n_freq_samples)

        window = torch.Tensor(scipy.fft.fftshift(scipy.signal.get_window("hamming", filter_length, fftbins=False)))
        self.window = nn.Parameter(window, requires_grad=False)

        self.decay = nn.Parameter(torch.Tensor([5]), requires_grad=True)

        logger.info(f'Diffuse Model: {diffuse_ir_opts["name"]} {type(self.diffuse_model)}')
        logger.info(f'Source Dir Model: {src_ir_opts["name"]} {type(self.source_dir_model)}')
        logger.info(f'Source Response: {src_opts["name"]} {type(self.source_resp)}')
        logger.info(f'Feature Extractor: {feat_extractor_opts["name"]} {type(self.pcd_feat_extractor)}')
        logger.info(f'Specular Model: {spec_ir_opts["name"]} {type(self.spec_model)}')
        logger.info(f'Late Model: {late_ir_opts["name"]} {type(self.late_model)}')


    def load_sub_model(self, name, kwargs, shared_kwargs):
        model = None

        if name != 'none':
            model = module_registry.build_shared(name, kwargs, shared_kwargs)
        
        return model

    def attenuation_and_decay(
        self, 
        ir: Float[Tensor, "T"],
        non_increasing: bool = False,
        attenuation_start_index: int = 5
    ):
        times = torch.arange(ir.shape[-1], device=ir.device) / self.sample_rate
        attn = 1 / (1e-6 + times * self.speed_of_sound)
        attn[:attenuation_start_index] = 0
        decay = nn.functional.sigmoid(self.decay) ** times # shape (rir_length)
        ir = ir * attn * decay if not non_increasing else ir * torch.clamp(decay * attn, max=1.0)
        return ir

    def prepare_tensors(self, mc_path_samples):

        path_xyzs = mc_path_samples['path_xyzs']
        directions = mc_path_samples['directions']
        lengths = mc_path_samples['lengths']

       
        path_split_size = tuple(path.shape[0] for path in path_xyzs)
        path_xyz_tensor = torch.from_numpy(np.concatenate(path_xyzs, axis=0)).float().to(self.device) # [N, 3]
        start_directions = torch.from_numpy(np.array([d[0] for d in directions], dtype=np.float32)).to(self.device)
        lengths = torch.tensor(lengths).float().to(self.device)

        if 'path_cos_thetas' not in mc_path_samples:
            end_directions = torch.from_numpy(np.array([d[-1] for d in directions], dtype=np.float32)).to(self.device)
            return {
                'path_splits': path_split_size,
                'path_xyzs': path_xyz_tensor,
                'first_d': start_directions,
                'last_d': end_directions,
                'last_d_numpy': end_directions.cpu().numpy(),
                'lengths': lengths,
            }

        path_costheta = mc_path_samples['path_cos_thetas']
        path_deviate_sinphi = mc_path_samples['path_deviate_sin_phi']
        path_local_transforms = mc_path_samples['path_local_transforms']
        path_lengths = mc_path_samples['path_lengths']

        last_d = mc_path_samples['last_directions']
        end_directions = torch.from_numpy(np.array(last_d, dtype=np.float32)).to(self.device)

        path_costheta_tensor = torch.from_numpy(np.concatenate(path_costheta, axis=0)).float().to(self.device) # [N]
        path_lengths_tensor = torch.from_numpy(np.concatenate(path_lengths, axis=0)).float().to(self.device) # [N]
        path_deviate_sinphi_tensor = torch.from_numpy(np.concatenate(path_deviate_sinphi, axis=0)).float().to(self.device) # [N]
        path_local_trans_tensor = torch.from_numpy(np.concatenate(path_local_transforms, axis=0)).float().to(self.device) # [N, 3, 3]


        return {
            'path_splits': path_split_size,
            'path_xyzs': path_xyz_tensor,
            'path_costheta': path_costheta_tensor,
            'path_deviate_sinphi': path_deviate_sinphi_tensor,
            'path_local_transforms': path_local_trans_tensor,
            'path_lengths': path_lengths_tensor,
            'first_d': start_directions,
            'last_d': end_directions,
            'last_d_numpy': np.array(last_d, dtype=np.float32),
            'lengths': lengths,
        }
    
    def render_rir_early(self, mc_samples, rotation=None, return_temp=False, face_d=None, left_d=None):

        if len(mc_samples['path_xyzs']) == 0:
            if face_d is not None:
                return torch.zeros(2, self.rir_length, dtype=self.dtype, device=self.device)
            return torch.zeros(self.rir_length, dtype=self.dtype, device=self.device)


        tensors = self.prepare_tensors(mc_samples)
        num_paths = tensors['first_d'].shape[0]

        path_splits: Tuple[int] = tensors['path_splits']
        xyzs: Float[Tensor, "N 3"] = tensors['path_xyzs'] # N = number of all path vertices

        if 'path_costheta' in tensors:
            xyz_std_base: Float[Tensor, "N 3"] = tensors['path_deviate_sinphi'] * tensors['path_lengths']
            xyz_std2_base: Float[Tensor, "N 3"] = xyz_std_base * xyz_std_base
            tangent_1_std2: Float[Tensor, "N"] = xyz_std2_base / (tensors['path_costheta'] + 1e-6) # [N]
            tangent_0_std2: Float[Tensor, "N"] = tangent_1_std2 / (tensors['path_costheta'] + 1e-6) # [N]
            tangent_0_mean: Float[Tensor, "N 3"] = tensors['path_local_transforms'][:, 0]
            tangent_1_mean: Float[Tensor, "N 3"] = tensors['path_local_transforms'][:, 1]
            diag_cov: Float[Tensor, "N 3"] =  tangent_0_std2.unsqueeze(-1) * torch.square(tangent_0_mean) + tangent_1_std2.unsqueeze(-1) * torch.square(tangent_1_mean)
        else:
            diag_cov: Float[Tensor, "N 3"] = torch.zeros((xyzs.shape[0], 3), dtype=self.dtype, device=self.device)

        features: Float[Tensor, "N F"] = self.spatial_feat_extractor(xyzs, diag_cov)
        resp: Float[Tensor, "N R"] = self.spec_model(xyzs, None, features, diag_cov)
        point_response: Float[Tensor, "N R"] = resp

        path_reflectance: Float[Tensor, "P R"] = torch.ones((num_paths, point_response.shape[-1]), dtype=self.dtype, device=self.device)
        for i, path_reflections in enumerate(torch.split(point_response, path_splits)):
            if path_reflections.shape[0] != 0:
                path_reflectance[i] = torch.prod(path_reflections, dim=-2)
        
        if path_reflectance.shape[-1] == self.n_key_freqs:
            path_reflectance: Float[Tensor, "P R"] = self.frequency_interpolator(path_reflectance)
            

        path_delays: Float[Tensor, "P"] = torch.round(tensors['lengths'] / self.speed_of_sound * self.sample_rate).int()

        # start_directions = PathSpace.start_directions(directions).to(dtype=self.dtype, device=self.device)
        start_directions: Float[Tensor, "P 3"] = tensors['first_d']

        start_dir_response: Float[Tensor, "P R"] = self.source_dir_model(start_directions, rotation)
        start_dir_response: Float[Tensor, "P R"] = self.frequency_interpolator(start_dir_response)
        directivity_amp: Float[Tensor, "P R"] = torch.exp(start_dir_response)


        'minimum phase filter'
        frequency_response: Float[Tensor, "P R"] = directivity_amp * path_reflectance
        phases: Float[Tensor, "P R"] = hilbert_one_sided(frequency_response, device=self.device)
        fx2: Float[Tensor, "P R"] = frequency_response * torch.exp(1j*phases)
        out_full: Float[Tensor, "P T"] = torch.fft.irfft(fx2)
        out: Float[Tensor, "P T"] = out_full[...,:self.filter_length] * self.window

        reflection_kernels: Float[Tensor, "P T"] = torch.zeros((num_paths, self.rir_length), dtype=self.dtype, device=self.device)
        reflection_kernels[:, :out.shape[-1]] = out

        reflection_kernels = propogate(reflection_kernels, path_delays)

        source_kernel: Float[Tensor, "T"] = self.source_resp()

        if face_d is not None:
            last_d_numpy = tensors['last_d_numpy']
            hrirs = compute_hrirs(last_d_numpy, face_d, left_d)
            hrirs = torch.from_numpy(hrirs).to(self.device).float()

            reflection_kernels = torch.unsqueeze(reflection_kernels, dim=1) # [P, 1, len]
            reflection_kernels = torchaudio.functional.fftconvolve(reflection_kernels, hrirs, )

            rir_early = torch.sum(reflection_kernels, dim=0)
            rir_early = F.fftconvolve(source_kernel[None,...], rir_early) [..., :self.rir_length]
        else:
            rir_early = torch.sum(reflection_kernels, dim=0)
            rir_early = F.fftconvolve(source_kernel, rir_early) [:self.rir_length]

        rir_early: Float[Tensor, "T"] = self.attenuation_and_decay(rir_early)

        if return_temp:
            return rir_early, {
                'source_kernel': source_kernel,
                'reflection_kernels': reflection_kernels,
                'raw_reflections': out, 
                'path_delay': path_delays,
            }

        return rir_early
    
    def render_rir_ambient(
        self, 
        src_xyz: Float[Tensor, "3"],
        dst_xyz: Float[Tensor, "3"],
        src_orient: Float[Tensor, "4"],
        face_d=None,
        left_d=None
    ) -> Float[Tensor, "T"]:
        r'''
        Neural rendering of the acoustic field.

        Args:
            src_xyz: torch.Tensor, shape (3)
                - source points
            dst_xyz: torch.Tensor, shape (3)
                - destination points
            src_orient: torch.Tensor, shape (4)
                - source orientation

        '''
        batch_size = 1
        n_ambient_samples = self.n_ambient_samples
        rays_d = sample_sphere(n_ambient_samples * batch_size, dtype=self.dtype, device=src_xyz.device)
        rays_o = dst_xyz.unsqueeze(0).repeat(n_ambient_samples * batch_size, 1)

        mask, hit_xyz = self.scene.cast_rays(rays_o.cpu().numpy(), rays_d.cpu().numpy())
        mask, hit_xyz = torch.from_numpy(mask).to(src_xyz.device), torch.from_numpy(hit_xyz).to(src_xyz.device).to(self.dtype)
        
        hit_xyz = hit_xyz[mask]
        query_dst_xyz = dst_xyz.unsqueeze(0).repeat(n_ambient_samples * batch_size, 1)[mask]
        query_src_xyz = src_xyz.unsqueeze(0).repeat(n_ambient_samples * batch_size, 1)[mask]
        query_src_orient = src_orient.unsqueeze(0).repeat(n_ambient_samples * batch_size, 1)[mask]

        path_ir = self.diffuse_model(hit_xyz, -rays_d[mask], query_src_xyz, query_src_orient)
        dist_to_dst = torch.norm(hit_xyz - query_dst_xyz, dim=-1)
        dist_to_src = torch.norm(hit_xyz - query_src_xyz, dim=-1)
        min_delay = torch.round(dist_to_src / self.speed_of_sound * self.sample_rate).int()
        min_delay[min_delay <= 0] = 1
        delay = torch.round(dist_to_dst / self.speed_of_sound * self.sample_rate).int()

        path_ir = mask_ir(path_ir, min_delay)
        path_ir = propogate(path_ir, delay)

        tot_sample_ir = torch.zeros(n_ambient_samples * batch_size, path_ir.shape[-1], dtype=self.dtype, device=src_xyz.device)
        tot_sample_ir[mask] = path_ir

        tot_sample_ir = tot_sample_ir.reshape(n_ambient_samples, -1)

        if face_d is None:
            est_ir = torch.zeros(self.rir_length, dtype=self.dtype, device=src_xyz.device)
            est_ir[:tot_sample_ir.shape[-1]] = torch.sum(tot_sample_ir, dim=0) / n_ambient_samples
        else:
            est_ir = torch.zeros(2, self.rir_length, dtype=self.dtype, device=src_xyz.device)
            hrirs = compute_hrirs(-rays_d.cpu().numpy(), face_d, left_d)
            hrirs = torch.from_numpy(hrirs).to(self.device).float()

            tot_sample_ir = torch.unsqueeze(tot_sample_ir, dim=1) # [P, 1, len]
            tot_sample_ir = torchaudio.functional.fftconvolve(tot_sample_ir, hrirs, )[..., :self.rir_length]

            est_ir[..., :tot_sample_ir.shape[-1]] = torch.sum(tot_sample_ir, dim=0) / n_ambient_samples


        est_ir = self.attenuation_and_decay(est_ir)

        return est_ir
  
    
    def forward(self, paths, directions, lengths, visible_point_indices=None, rotation=None, source_xyz=None, listener_xyz=None, rot_quat=None, mc_samples=None, face_d = None, left_d=None):

        rir_early = self.render_rir_early(mc_samples, rotation, False, face_d, left_d)


        rir_ambient = torch.zeros_like(rir_early)
        if self.diffuse_model is not None:
            source_xyz = source_xyz.to(self.device)
            listener_xyz = listener_xyz.to(self.device)
            rot_quat = rot_quat.to(self.device)
            rir_ambient = self.render_rir_ambient(source_xyz, listener_xyz, rot_quat, face_d, left_d)


        if self.late_model is not None:
            # rir_full = rir_early + rir_ambient +  self.attenuation_and_decay(self.late_model(), non_increasing=True, attenuation_start_index=1000)
            rir_full = rir_early + rir_ambient +  self.late_model()
        else:
            rir_full = rir_early + rir_ambient

        return {
            'rir_early': rir_early,
            'rir_ambient': rir_ambient,
            'rir_full': rir_full,
        }

    @torch.no_grad()
    def model_visulization_mc(self, output_dir, swriter: Optional[SummaryWriter], epoch, i, 
                           mc_samples, visible_point_indices=None, rotation=None,source_xyz=None, listener_xyz=None, rot_quat=None, 
                           gt_rir: Optional[torch.Tensor] = None):
        
        rir_early = self.render_rir_early(mc_samples, rotation)
        rir_full = rir_early

        rir_ambient = torch.zeros_like(rir_early)
        if self.diffuse_model is not None:
            source_xyz = source_xyz.to(self.device)
            listener_xyz = listener_xyz.to(self.device)
            rot_quat = rot_quat.to(self.device)
            rir_ambient = self.render_rir_ambient(source_xyz, listener_xyz, rot_quat)
            # rir_full =  rir_full + rir_ambient

        if self.late_model is not None:
            # rir_full = self.late_model(rir_early, rir_ambient)
            rir_full = rir_full + rir_ambient +  self.late_model()
        else:
            rir_full = rir_early + rir_ambient


        plt.figure(figsize=(10, 5))
        plt.subplot(3, 1, 1)
        plt.plot(rir_early.cpu().numpy()[:2000], label='Early RIR', linewidth=0.5)
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(rir_full.cpu().numpy()[:2000], label='Full RIR', linewidth=0.5)
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(self.source_resp().cpu().numpy(), label='Source Response', linewidth=0.5)
        plt.title(f'RIR Early and Full')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'render_{i}.png'))
        plt.close()

        if gt_rir is not None:
            plt.figure(figsize=(10, 5))
            plt.subplot(2, 1, 1)
            plt.plot(rir_full.cpu().numpy()[:2000], label='Full RIR', linewidth=0.5)
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(gt_rir.cpu().numpy()[:2000], label='GT RIR', linewidth=0.5)
            # plt.title(f'RIR Full vs GT first 2000 samples')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'comp_{i}.png'))
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.subplot(2, 1, 1)
            plt.plot(rir_full.cpu().numpy(), label='Full RIR', linewidth=0.5)
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(gt_rir.cpu().numpy(), label='GT RIR', linewidth=0.5)
            # plt.title(f'RIR Full vs GT full')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'full_comp_{i}.png'))
            plt.close()

       
        bds = fibonacci_sphere(2048).to(dtype=self.dtype, device=self.device)
        dirs = self.source_dir_model(bds, None).detach().cpu().numpy()

        bds = bds.cpu().numpy()

        mask = bds[:, 2] < 0
        # plt.figure()
        cm = plt.get_cmap('inferno')
        normd = dirs / dirs.max()
        fig, ax = plt.subplots(4, 4, figsize=(10, 10))
        step = max(self.source_dir_model.directivity.shape[1] // 16, 1)
        for i in range(min(self.source_dir_model.directivity.shape[1], 16)):
            ax[i//4, i%4].scatter(bds[mask, 0], bds[mask, 1], c=cm(normd[mask, i * step]))
            ax[i//4, i%4].axis('equal')
            ax[i//4, i%4].axis('off')
            # show colorbar
            ax[i//4, i%4].set_title(f'{i * step}')

        # plt.colorbar(ax[0, 0].collections[0], ax=ax, orientation='horizontal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'directivity-{epoch}-neg.png'))
        # plt.close()
        plt.close(fig)

        mask = bds[:, 2] > 0
        plt.figure()
        cm = plt.get_cmap('inferno')
        normd = dirs / dirs.max()
        fig, ax = plt.subplots(4, 4, figsize=(10, 10))
        for i in range(min(self.source_dir_model.directivity.shape[1], 16)):
            ax[i//4, i%4].scatter(bds[mask, 0], bds[mask, 1], c=cm(normd[mask, i * step]))
            ax[i//4, i%4].axis('equal')
            ax[i//4, i%4].axis('off')
            # show colorbar
            ax[i//4, i%4].set_title(f'{i * step}')

        # plt.colorbar(ax[0, 0].collections[0], ax=ax, orientation='horizontal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'directivity-{epoch}-pos.png'))
        plt.close()

def random_sphere(n_samples, dtype=torch.float32, device=torch.device('cpu')):
    
    phi = torch.acos(torch.rand(n_samples, dtype=dtype, device=device) * 2 - 1) - torch.pi / 2
    theta = torch.rand(n_samples, dtype=dtype, device=device) * 2 * torch.pi

    x = torch.cos(phi) * torch.cos(theta)
    y = torch.cos(phi) * torch.sin(theta)
    z = torch.sin(phi)

    return torch.stack([x, y, z], dim=-1)

def load_sub_model(name, kwargs, shared_kwargs):
    model = None

    if name != 'none':
        model = module_registry.build_shared(name, kwargs, shared_kwargs)
    
    return model

def hilbert_one_sided(x, device=torch.device('cpu')):
    """
    Returns minimum phases for a given log-frequency response x.
    Assume x.shape[-1] is ODD
    """
    N = 2*x.shape[-1] - 1
    Xf = torch.fft.irfft(x, n=N)
    h = torch.zeros(N).to(device)
    h[0] = 1
    h[1:(N + 1) // 2] = 2
    x = torch.fft.rfft(Xf * h)
    return torch.imag(x)


def safe_log(x, eps=1e-9):
    """Prevents Taking the log of a non-positive number"""
    safe_x = torch.where(x <= eps, eps, x)
    return torch.log(safe_x)


def mask_ir(ir, delay):
    B, T = ir.shape
    mask = torch.arange(T, device=ir.device).unsqueeze(0) >= delay.unsqueeze(-1)
    return ir * mask.float()
def quat_to_rot(quats):
    r'''
    Convert quaternions to rotation matrices.
    
    Args:
        quats: torch.Tensor, shape (B, 4)
            - quaternions
    
    Returns:
        rot: torch.Tensor, shape (B, 3, 3)
            - rotation matrices
    '''
    x, y, z, w = torch.split(quats, 1, dim=1)
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz, xw, yw, zw = x * y, y * z, x * z, x * w, y * w, z * w

    rot = torch.stack([1 - 2 * (y2 + z2), 2 * (xy - zw), 2 * (xz + yw),
                    2 * (xy + zw), 1 - 2 * (x2 + z2), 2 * (yz - xw),
                    2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (x2 + y2)], dim=1).view(-1, 3, 3)
    return rot

def propogate(batch_ir, ind_shift):
    '''
    Args:
        batch_ir: torch.Tensor, shape (B, T)
            - batch of impulse responses
        ind_shift: torch.Tensor, shape (B,)
            - indices to shift the impulse responses
    '''
    B, T = batch_ir.shape

    if B == 0:
        return batch_ir
    
    ir_indices = torch.arange(T, device=batch_ir.device).unsqueeze(0)
    ind_shift = ind_shift.unsqueeze(-1)
    valid = (ir_indices + ind_shift) < T
    batch_ir = batch_ir * valid
    batch_fft_ir = torch.fft.rfft(batch_ir)
    freq_indices = torch.arange(batch_fft_ir.shape[1], device=batch_fft_ir.device).unsqueeze(0)
    batch_fft_ir = batch_fft_ir * torch.exp(-2j * torch.pi * ind_shift * freq_indices / T)
    batch_ir = torch.fft.irfft(batch_fft_ir, n=T)
    return batch_ir

module_registry.add('rir_renderer', RirRenderer)