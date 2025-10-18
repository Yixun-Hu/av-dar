import torch
import torch.nn as nn

import open3d as o3d

from ..utils.nn_utils import SpatialEncoder
from ..model import module_registry

from typing import Optional
from ..core.typing import *

class PositionalEncodingBasedAcousticField(nn.Module):
    def __init__(
        self, 
        xyz_min: float, 
        xyz_max: float, 
        xyz_order: int, 
        dir_order: int, 
        quat_order: int, 
        output_dim: int, 
        n_layers: int, 
        n_features: int, 
        start_index: int = 0,
        normalized: bool = False,
        disabled: bool = False,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        
        self.xyz_min = xyz_min
        self.xyz_max = xyz_max

        self.disabled = disabled

        self.dtype = dtype
        self.normalized = normalized
        self.xyz_encoder = SpatialEncoder(3, xyz_order, 1, data_type=dtype)
        self.quat_encoder = SpatialEncoder(4, quat_order, 1, data_type=dtype)
        self.direction_encoder = SpatialEncoder(3, dir_order, 1, data_type=dtype)
        self.start_index = start_index

        self.input_dim = xyz_order * 3 * 2 + quat_order * 4 + dir_order * 3
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, n_features),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(n_features, n_features), nn.ReLU()) for _ in range(n_layers)],
            nn.Linear(n_features, output_dim - start_index)
        )

    def normalize(self, xyz):
        '''
        Normalize the input xyz to [0, 1]
        
        Args:
            xyz: torch.Tensor, shape (B, 3)
                - input xyz coordinates
        '''

        return (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)

    def forward(
        self, 
        xyz: Float[Tensor, "N 3"],
        w_in: Float[Tensor, "N 3"],
        src_xyzs: Float[Tensor, "N 3"],
        src_orients: Float[Tensor, "N 4"]
    ) -> Float[Tensor, "N D"]:
        
        B = xyz.shape[0]

        xyz = self.normalize(xyz)
        src_xyzs = self.normalize(src_xyzs)
        src_orients = (src_orients + 1) / 2
        w_in = (w_in + 1) / 2

        xyz_pe = self.xyz_encoder(xyz)
        src_xyz_pe = self.xyz_encoder(src_xyzs)
        src_orient_pe = self.quat_encoder(src_orients)
        w_in_pe = self.direction_encoder(w_in)

        input_feat = torch.cat([xyz_pe, src_xyz_pe, src_orient_pe, w_in_pe], dim=-1)
        output_feat = self.mlp(input_feat)

        output_feat_full = torch.zeros(
            B, self.output_dim, dtype=self.dtype, device=xyz.device
        )
        output_feat_full[:, self.start_index:] = output_feat

        if self.disabled:
            output_feat_full = torch.zeros_like(output_feat_full)

        if self.normalized:
            output_feat_full = output_feat_full - torch.mean(output_feat_full, dim=-1, keepdim=True)
        return output_feat_full

module_registry.add('positional_encoding_acoustic_field', PositionalEncodingBasedAcousticField, ['dtype'])