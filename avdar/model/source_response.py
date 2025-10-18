from . import module_registry

import torch
import torch.nn as nn

from ..core.typing import *


class SourceParameterModel(nn.Module):
    def __init__(self, filter_length: int, forced_offset: int, dtype=torch.float32):

        assert 0 <= forced_offset < filter_length

        super().__init__()
        self.filter_length = filter_length
        self.forced_offset = forced_offset
        self.dtype = dtype

        source_response = torch.Tensor([0.0] * self.filter_length).to(dtype)
        source_response[forced_offset] = 0.02
        source_response[forced_offset+1] = -0.007
        source_response[forced_offset+2] = 0.002
        
        self.source_response = nn.Parameter(source_response, requires_grad=True)

        print("! force delay: ", forced_offset)

    def forward(self) -> Float[Tensor, "L"]:
        kernel = torch.zeros_like(self.source_response)
        kernel[self.forced_offset:] = self.source_response[self.forced_offset:]
        
        return kernel
    
class SourceParameterWindowedModel(nn.Module):
    def __init__(self, filter_length: int, forced_offset: int, window_size: int, dtype=torch.float32):

        assert 0 <= forced_offset < forced_offset+ window_size < filter_length

        super().__init__()
        self.filter_length = filter_length
        self.forced_offset = forced_offset
        self.window_size = window_size
        self.dtype = dtype

        source_response = torch.Tensor([0.0] * self.window_size).to(dtype)

        source_response[0] = 0.02
        source_response[1] = -0.007
        source_response[2] = 0.002
        
        self.source_response = nn.Parameter(source_response, requires_grad=True)

        print("! force delay: ", forced_offset)
        print("! window size: ", window_size)

    def forward(self) -> Float[Tensor, "L"]:
        kernel = torch.zeros(self.filter_length, dtype=self.dtype, device=self.source_response.device)
        kernel[self.forced_offset:self.forced_offset+self.window_size] = self.source_response
        return kernel
    

module_registry.add("src_para", SourceParameterModel, ['dtype'])
module_registry.add("src_para_windowed", SourceParameterWindowedModel, ['dtype'])