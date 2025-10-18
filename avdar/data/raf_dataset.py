from . import AcousticDataset, data_registry

import os
import torch
import numpy as np

from ..utils.io_utils import load_json
from ..utils.nn_utils import SpatialEncoder
from ..geometry.mesh_scene import RayTracingScene

import trimesh
import librosa
from scipy.spatial.transform import Rotation

import open3d as o3d

import logging
logger = logging.getLogger(__name__)

N_MIC_PER_EARFUL_TOWER = 36

class RafDataset(AcousticDataset):

    INIT_KEYS = AcousticDataset.INIT_KEYS \
        + ['rir_length', 'split', 'n_points', 
           'speed_of_sound', 'sample_rate', 'cache_dir', 
           'split_config_file', 'mesh_path']

    def __init__(self, 
                 name: str, 
                 scene_name: str, 
                 path: str,
                 split, 
                 n_points, 
                 speed_of_sound, 
                 sample_rate, 
                 rir_length,
                 cache_dir, 
                 split_config_file, 
                 mesh_path,
                 dtype=torch.float32):
        super().__init__(name, scene_name, path)

        self.split = split
        self.dtype = dtype

        self.speed_of_sound = speed_of_sound
        self.sample_rate = sample_rate
        self.rir_length = rir_length

        split_data = load_json(split_config_file)
        self.train_indices = split_data['train'][0]
        self.val_indices = split_data['validation'][0]
        self.test_indices = split_data['test'][0]
        
        self.mesh_path = mesh_path
        self.scene_mesh = None

        if split == 'train':
            self.indices = self.train_indices
        elif split == 'val':
            self.indices = self.val_indices
        elif split == 'test':
            self.indices = self.test_indices
        elif split == 'inference':
            self.indices = self.test_indices + self.val_indices + self.train_indices
        else:
            raise ValueError(f'Invalid split: {split}, must be one of ["train", "val", "test"]')
        
        print(f'Number of indices: {len(self.indices)}')

        # use warmup, no load at init
        self.rir_data = {}
        self.metadata = self.load_meta_data()

        self.init_points(n_points, cache_dir)
        self.n_points = len(self.points)

    def load_meta_data(self):
        all_rx_pos_path = self.path / 'metadata' / 'all_rx_pos.txt'
        all_tx_pos_path = self.path / 'metadata' / 'all_tx_pos.txt'

        all_rx_pos = self.load_meta_txt(all_rx_pos_path)
        all_tx_pos = self.load_meta_txt(all_tx_pos_path)

        return {
            'all_rx_pos': all_rx_pos,
            'all_tx_pos': all_tx_pos[:, -3:],
            'all_tx_rot': all_tx_pos[:, :4]
        }
    
    def get_mesh(self):

        if self.scene_mesh is None:
            self.scene_mesh = RayTracingScene(self.get_mesh_path())

        return self.scene_mesh
    
    def get_mesh_path(self):
        return self.mesh_path

        
    def index_to_listener_idx(self, index: str):
        return int(index)

    def index_to_source_idx(self, index: str):
        return int(index) // N_MIC_PER_EARFUL_TOWER
    
    def load_meta_txt(self, path):
        with open(path, 'r') as fp:
            lines = fp.read().strip().split('\n')
        data = [[float(number) for number in line.split(',')] for line in lines if line.strip()]
        return np.array(data)
    
    def num_speaker(self):
        return self.metadata['all_tx_pos'].shape[0] // N_MIC_PER_EARFUL_TOWER
    
    def num_listener(self):
        return self.metadata['all_rx_pos'].shape[0]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        index = self.indices[idx]

        if index not in self.rir_data:
            rir_path = self.path / 'data' / index / 'rir.wav'
            gt_rir = torch.tensor(librosa.load(rir_path, sr=self.sample_rate)[0][:self.rir_length], dtype=self.dtype)
            self.rir_data[index] = gt_rir
        else:
            gt_rir = self.rir_data[index]
        
        listener_xyz = self.metadata['all_rx_pos'][int(index)]
        source_xyz = self.metadata['all_tx_pos'][int(index)]
        source_rot_quat = self.metadata['all_tx_rot'][int(index)]


        source_rot_matrix = Rotation.from_quat(source_rot_quat).as_matrix()
        return {
            'rir': gt_rir,
            'source_rotation_quat': torch.tensor(source_rot_quat, dtype=self.dtype),
            'source_rotation': torch.from_numpy(source_rot_matrix).to(dtype=self.dtype),
            'source_xyz': torch.tensor(source_xyz, dtype=self.dtype),
            'listener_xyz': torch.tensor(listener_xyz, dtype=self.dtype),
        }


data_registry.add("real_acoustic_field", RafDataset)