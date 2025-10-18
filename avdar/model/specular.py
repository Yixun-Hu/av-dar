from . import module_registry


import torch
import torch.nn as nn
from ..core.typing import *
from ..utils.nn_utils import SpatialEncoderWithCovariance

import logging
logger = logging.getLogger(__name__)

class SpecularPeMLP(nn.Module):
    def __init__(self,
                 dim_feat: int, 
                 dim_out: int,
                 n_layers: int,
                 hidden_size: int,
                 pe_order: int,
                 fuse_pe: bool=False,
                 fuse_vision: bool=False,
                 use_residual: bool=True,
                 activation: str='sigmoid',
                 xyz_scale = [20, 5, 20],
                 dtype=torch.float32,
                ):
        super().__init__()
        self.feat_encoder = nn.Sequential(
            nn.Linear(dim_feat + 3 * pe_order, hidden_size), nn.ReLU(),
        )

        self.pe_projector = None
        if fuse_pe:
            self.pe_projector = nn.Linear(3 * pe_order, hidden_size)

        self.vision_projector = None
        if fuse_vision:
            self.vision_projector = nn.Linear(dim_feat, hidden_size)

        # FIXME: the fuser should be after each layer (Linear + ReLU), not for each component separately.
        module_list = nn.ModuleList()
        for i in range(n_layers):
            module_list.append(nn.Linear(hidden_size, hidden_size))
            module_list.append(nn.LeakyReLU())
            
        self.decoder_layers = module_list

        self.output_layer = nn.Linear(hidden_size, dim_out)

        self.use_residual = use_residual

        self.xyz_encoder = SpatialEncoderWithCovariance(3, pe_order, xyz_scale, data_type=dtype)
        self.act_fn = {
            'abs': lambda x: torch.abs(x),
            'square': lambda x: torch.square(x),
            'sigmoid': nn.Sigmoid(),
            'trunc_sigmoid': lambda x: torch.sigmoid(x) * .7 + .005,
            'sin': lambda x: torch.sin(x * 10),
            'exp': lambda x: torch.exp(x),
            'exp_softplus': lambda x: torch.exp(-torch.nn.functional.softplus(x)),
            'identity': lambda x: x,
        }[activation]
        self.activation = self.act_fn
    
    def forward(
        self, 
        xyz: Float[Tensor, "N 3"],
        normal: Any, # unused
        feat: Float[Tensor, "N F"],
        xyz_std2: Float[Tensor, "N 3"]
    ):

        vision_fuser = 0
        if self.vision_projector is not None:
            vision_fuser = self.vision_projector(feat)
    
        xyz_feat = self.xyz_encoder(xyz, xyz_std2)
        feat = torch.cat([feat, xyz_feat], dim=-1)
        feat_ = self.feat_encoder(feat)

        pe_fuser = 0
        if self.pe_projector is not None:
            pe_fuser = self.pe_projector(xyz_feat)

        for layer in self.decoder_layers:
            if self.use_residual:
                feat_ = layer(feat_) + feat_
            else:
                feat_ = layer(feat_)
            feat_ = feat_ + pe_fuser + vision_fuser # FIXME: See above comment

        feat_ = self.output_layer(feat_)
        
        refl = self.act_fn(feat_)
        return refl  

module_registry.add('specular_pe_mlp', SpecularPeMLP, ['dtype'])