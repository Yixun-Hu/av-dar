from . import module_registry

import torch
import torch.nn as nn

import numpy as np

from ..utils.nn_utils import SpatialEncoder
from ..utils.sample_utils import fibonacci_sphere
from ..core.typing import *

class DirectionalSource(nn.Module):
    def __init__(self,
                 n_base_directions: int,
                 out_dim: int,
                 sharpness: float,
                 dtype=torch.float32):
        super().__init__()
        self.sharpness = sharpness
        self.base_directions = nn.Parameter(fibonacci_sphere(n_base_directions).to(dtype), requires_grad=False)
        self.directivity = nn.Parameter(torch.ones(n_base_directions, out_dim, dtype=dtype), requires_grad=True)

    def forward(self, direction: Float[Tensor, "N 3"], rotation: Optional[Float[Tensor, "N 3"]] = None) -> Float[Tensor, "N D"]:
        if rotation is not None:
            direction = torch.einsum('ij,bj->bi', rotation.T.to(dtype=direction.dtype, device=direction.device), direction)
        dots = direction @ self.base_directions.T
        weights = torch.exp(-self.sharpness*(1 - dots))
        weights = weights/(torch.sum(weights, dim=-1).view(-1, 1))
        weighted = weights.unsqueeze(-1) * self.directivity
        directivity_profile = torch.sum(weighted, dim=1)
        return directivity_profile
    

module_registry.add("parameterized_directional_source", DirectionalSource, ['dtype'])
