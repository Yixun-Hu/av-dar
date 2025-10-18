from . import module_registry

import torch
import torch.nn as nn

import math

from ..core.typing import *
    
class ResidualParaDirectModel(nn.Module):
    def __init__(self,
                 rir_length: int,
                 dtype=torch.float32):
        super().__init__()

        self.rir_length = rir_length
        self.dtype = dtype

        self.residual = nn.Parameter(torch.zeros(self.rir_length, dtype=dtype), requires_grad=True)
        self.activation = nn.Sigmoid()

    def forward(self) -> Float[Tensor, "L"]:
        return self.residual

module_registry.add("residual_para_direct", ResidualParaDirectModel, ['dtype'])