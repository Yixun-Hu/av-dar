from ..data import AcousticDataset, data_registry

import numpy as np
import os
import torch

import trimesh

from ..geometry.mesh_scene import RayTracingScene

import librosa
import logging
import pathlib

logger = logging.getLogger(__name__)

class DiffRIRDataset(AcousticDataset):

    INIT_KEYS = AcousticDataset.INIT_KEYS + [
        'rir_length', 'split', 'train_indices', 'val_indices', 'n_data', 
        'n_points', 'cache_dir', 'sample_rate', 
        'speed_of_sound', 'source_xyz', 'mesh_path']

    def __init__(self, 
                 name, scene_name, path, 
                 rir_length, split,
                 train_indices, val_indices, n_data, 
                 n_points,
                 cache_dir,
                 source_xyz, 
                 sample_rate,
                 mesh_path,
                 speed_of_sound,
                 dtype=torch.float32):
        super().__init__(name, scene_name, path)

        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = list(set(range(n_data)) - set(train_indices) - set(val_indices))
        self.n_data = n_data

        source_xyz = torch.tensor(source_xyz).to(dtype=dtype)
        self.source_xyz = source_xyz

        self.speed_of_sound = speed_of_sound
        self.sample_rate = sample_rate
        self.mesh_path = mesh_path
        self.rir_length = rir_length
        self.split = split
        self.dtype = dtype

        self.scene_mesh = None

        self.load_data()
        
        self.points = []
        self.source_idx = None
        self.listener_base_idx = None
        if n_points is not None and n_points > 0:
            self.init_points(n_points, cache_dir)
        self.n_points = len(self.points)

        self.indices = []
        if 'train' in split:
            self.indices += self.train_indices
        if 'val' in split:
            self.indices += self.val_indices
        if 'test' in split:
            self.indices += self.test_indices

        if split == 'inference':
            self.indices = list(range(n_data))
        
    def get_mesh(self):

        if self.scene_mesh is None:
            self.scene_mesh = RayTracingScene(self.mesh_path)
            logger.info(f'Loaded mesh from {self.mesh_path}')

        return self.scene_mesh
    
    def get_mesh_path(self):
        return self.mesh_path

    def load_data(self):
        rirs = np.load(os.path.join(self.path, "RIRs.npy"))
        rirs_resampled = librosa.resample(rirs, orig_sr=48000, target_sr=self.sample_rate)
        
        self.xyzs = torch.from_numpy(np.load(os.path.join(self.path, "xyzs.npy"))).to(dtype=self.dtype)
        self.RIRs = torch.from_numpy(rirs_resampled).to(dtype=self.dtype)
    
    def __getitem__(self, idx):
        rir = self.RIRs[self.indices[idx]][:self.rir_length]
        listener_xyz = self.xyzs[self.indices[idx]]
        return {
            'rir': rir,
            'source_xyz': self.source_xyz,
            'listener_xyz': listener_xyz,
            'source_rotation_quat': torch.tensor([0, 0, 0, 1], dtype=self.dtype),
        }

    def __len__(self):
        return len(self.indices)
    
data_registry.add("diff_rir", DiffRIRDataset)