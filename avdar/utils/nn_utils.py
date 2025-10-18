import torch
import torch.nn as nn

import os

AVDAR_DISABLE_PE_COVARIANCE = os.environ.get('AVDAR_DISABLE_PE_COVARIANCE', False)

if AVDAR_DISABLE_PE_COVARIANCE:
    print('Covariance disabled')
    print('\n' * 30) # to remind the user that covariance is disabled


class SpatialEncoder(nn.Module):
    """Positional Encoder"""
    def __init__(self, input_dim, output_dim, scale, split_dim=False, base=2, data_type=torch.float32):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.split_dim = split_dim
        self.data_type = data_type
        self.scale = scale

        sin_freqs = (torch.pi * torch.pow(
            base, torch.arange(0, output_dim // 2) ) * 2).to(dtype=data_type)
        cos_freqs = (torch.pi * torch.pow(
            base, torch.arange(0, (output_dim + 1) // 2) ) * 2).to(dtype=data_type)
        
        self.sin_freqs = nn.Parameter(sin_freqs, requires_grad=False)
        self.cos_freqs = nn.Parameter(cos_freqs, requires_grad=False)
        
        
    def forward(self, x: torch.Tensor):

        x = x / self.scale
        input_shape, dim = x.shape[:-1], x.shape[-1]

        sin_phases = x.view(-1, dim, 1) * self.sin_freqs
        cos_phases = x.view(-1, dim, 1) * self.cos_freqs
        output_flat = torch.cat([torch.cos(cos_phases), torch.sin(sin_phases)], -1)

        if not self.split_dim:
            output = output_flat.reshape(*input_shape, self.output_dim * dim)
        else:
            output = output_flat.reshape(*input_shape, dim, self.output_dim)
        
        return output
    
class SpatialEncoderWithCovariance(nn.Module):
    """Positional Encoder with Covariance, see Mip-NeRF for formula"""
    def __init__(self, input_dim, output_dim, scale, split_dim=False, base=2, data_type=torch.float32):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.split_dim = split_dim
        self.data_type = data_type

        scale = torch.tensor(scale, dtype=data_type)
        if scale.ndim == 0:
            scale = scale.repeat(input_dim)

        ## 2 * pi * f, period <= 1
        sin_phases = torch.pow(base, torch.arange(0, output_dim // 2)) # [1, 2, 4, ...]
        cos_phases = torch.pow(base, torch.arange(0, (output_dim + 1) // 2)) # [1, 2, 4, ...]

        sin_freqs = (2 * torch.pi * sin_phases).to(dtype=data_type)
        cos_freqs = (2 * torch.pi * cos_phases).to(dtype=data_type)
        sin_freqs_2 = sin_freqs * sin_freqs
        cos_freqs_2 = cos_freqs * cos_freqs
        
        self.sin_freqs = nn.Parameter(sin_freqs, requires_grad=False)
        self.cos_freqs = nn.Parameter(cos_freqs, requires_grad=False)
        self.sin_freqs_2 = nn.Parameter(sin_freqs_2, requires_grad=False)
        self.cos_freqs_2 = nn.Parameter(cos_freqs_2, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)


    def forward(self, x: torch.Tensor, std2_x: torch.Tensor):
        x = x / self.scale
        input_shape, dim = x.shape[:-1], x.shape[-1]

        sin_phases = x.view(-1, dim, 1) * self.sin_freqs
        cos_phases = x.view(-1, dim, 1) * self.cos_freqs
        sin_std2 = std2_x.view(-1, dim, 1) * (self.sin_freqs_2 / (self.scale * self.scale)[..., None])
        cos_std2 = std2_x.view(-1, dim, 1) * (self.cos_freqs_2 / (self.scale * self.scale)[..., None])
        
        pe_mean = torch.cat([torch.cos(cos_phases), torch.sin(sin_phases)], -1)
        pe_std2 = torch.cat([cos_std2, sin_std2], -1)
        pe_std2 = torch.exp(- pe_std2 / 2)

        if AVDAR_DISABLE_PE_COVARIANCE:
            output_flat = pe_mean
        else:
            output_flat = pe_mean * pe_std2

        if not self.split_dim:
            output = output_flat.reshape(*input_shape, self.output_dim * dim)
        else:
            output = output_flat.reshape(*input_shape, dim, self.output_dim)
        # import IPython; IPython.embed(); exit()
        return output

class LinearInterpolation(nn.Module):
    """
    Linear interpolation layer
    """
    def __init__(self, sample_indices, output_dim, data_type=torch.float32):
        
        assert max(sample_indices) < output_dim, 'Max sample index must be less than output dimension'

        super().__init__()

        self.input_dim = len(sample_indices)
        self.output_dim = output_dim
        self.data_type = data_type 

        interpolator = torch.zeros((self.input_dim, self.output_dim), dtype=data_type)
        diffs = torch.diff(sample_indices)
        
        # 0th row
        linterp = torch.cat((torch.arange(sample_indices[0]) / sample_indices[0], 1 - torch.arange(diffs[0]) / diffs[0]))
        interpolator[0, 0 : sample_indices[1]] = linterp

        # last row
        linterp = torch.cat((torch.arange(diffs[-1]) / diffs[-1], torch.ones(output_dim - sample_indices[-1])))
        interpolator[-1, sample_indices[-2]:] = linterp

        # middle rows
        for i in range(1, self.input_dim - 1):
            linterp = torch.cat((torch.arange(diffs[i - 1]) / diffs[i - 1], 1 - torch.arange(diffs[i]) / diffs[i]))
            interpolator[i, sample_indices[i - 1]:sample_indices[i + 1]] = linterp

        self.interpolator = nn.Parameter(interpolator, requires_grad=False)

        
    def forward(self, x: torch.Tensor):
        assert x.shape[-1] == self.input_dim, 'Input dimension does not match'
        return (x.unsqueeze(-1) * self.interpolator).sum(-2)
    
class KnnAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k_neighbors, num_heads, dim_kqv, norm_pt=False, dtype=torch.float32):
        super(KnnAttentionLayer, self).__init__()
        self.k_neighbors = k_neighbors
        self.num_heads = num_heads
        self.dim_kqv = dim_kqv
        self.norm_pt = norm_pt
        self.dtype = dtype

        self.weight_keys = nn.Linear(in_channels, dim_kqv * num_heads, bias=False)
        self.weight_values = nn.Linear(in_channels, dim_kqv * num_heads, bias=False)
        self.weight_query = nn.Linear(in_channels, dim_kqv * num_heads, bias=False)
        self.aggregate = nn.Sequential(nn.Linear(dim_kqv * num_heads, dim_kqv * num_heads), nn.ReLU(), nn.Linear(dim_kqv * num_heads, num_heads))
        self.weight_out = nn.Linear(dim_kqv * num_heads, out_channels)
        self.proj_position = nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, dim_kqv * num_heads))
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, xyzs: torch.Tensor, features: torch.Tensor, k_graph: torch.Tensor):
        '''
        : xyzs: [B, N, 3]
        : features: [B, N, in_channels]
        : k_graph: [B, N, k_neighbors]
        '''
        B, N, _ = xyzs.shape
        K = self.k_neighbors
        H = self.num_heads
        D = self.dim_kqv

        neighbor_features = torch.stack([features[b, k_graph[b]] for b in range(B)], dim=0) # [B, N, K, C]
        neighbor_xyzs = torch.stack([xyzs[b, k_graph[b]] for b in range(B)], dim=0) # [B, N, K, 3]

        rel_pos = xyzs.unsqueeze(2) - neighbor_xyzs # [B, N, K, 3]
        pe = self.proj_position(rel_pos) # [B, N, K, num_heads * dim_kqv]
        keys = self.weight_keys(neighbor_features) # [B, N, K, num_heads * dim_kqv]
        values = self.weight_values(neighbor_features) # [B, N, K, num_heads * dim_kqv]
        queries = self.weight_query(features) # [B, N, num_heads * dim_kqv]
        attn = (queries.unsqueeze(2) - keys + pe) # [B, N, K, num_heads * dim_kqv]

        attn = self.aggregate(attn) # [B, N, K, num_heads]
        attn = self.softmax(attn) # [B, N, K, num_heads]
        fused_value = attn.unsqueeze(-1) * (values + pe).view(B, N, K, H, D) # [B, N, K, num_heads, dim_kqv]
        fused_value = fused_value.sum(dim=2) # [B, N, num_heads, dim_kqv]
        fused_value = fused_value.view(B, N, -1) # [B, N, num_heads * dim_kqv]
        out = self.weight_out(fused_value) # [B, N, out_channels]
        return out
    

    def external_query_forward(self, xyzs_query: torch.Tensor, features_query: torch.Tensor, xyzs: torch.Tensor, features: torch.Tensor, k_graph_query: torch.Tensor):
        '''
        : xyzs_query: [B, Nq, 3]
        : features_query: [B, Nq, in_channels]
        : xyzs: [B, N, 3]
        : features: [B, N, in_channels]
        : k_graph_query: [B, Nq, k_neighbors]
        '''
        B, Nq, _ = xyzs_query.shape
        K = self.k_neighbors
        _, N, _ = xyzs.shape
        H = self.num_heads
        D = self.dim_kqv

        neighbor_features = torch.stack([features[b, k_graph_query[b]] for b in range(B)], dim=0) # [B, Nq, K, C]
        neighbor_xyzs = torch.stack([xyzs[b, k_graph_query[b]] for b in range(B)], dim=0) # [B, Nq, K, 3]

        rel_pos = xyzs_query.unsqueeze(2) - neighbor_xyzs # [B, Nq, K, 3]
        pe = self.proj_position(rel_pos) # [B, Nq, K, num_heads * dim_kqv]
        keys = self.weight_keys(neighbor_features) # [B, Nq, K, num_heads * dim_kqv]
        values = self.weight_values(neighbor_features) # [B, Nq, K, num_heads * dim_kqv]
        queries = self.weight_query(features_query) # [B, Nq, num_heads * dim_kqv]
        attn = (queries.unsqueeze(2) - keys + pe) # [B, Nq, K, num_heads * dim_kqv]
        attn = self.aggregate(attn) # [B, Nq, K, num_heads]
        attn = self.softmax(attn) # [B, Nq, K, num_heads]
        fused_value = attn.unsqueeze(-1) * (values + pe).view(B, Nq, K, H, D) # [B, Nq, K, num_heads, dim_kqv]
        fused_value = fused_value.sum(dim=2) # [B, Nq, num_heads, dim_kqv]
        fused_value = fused_value.view(B, Nq, -1) # [B, Nq, num_heads * dim_kqv]
        out = self.weight_out(fused_value) # [B, Nq, out_channels]
        return out


def hilbert_one_sided(x):
    """
    Returns minimum phases for a given log-frequency response x.
    Assume x.shape[-1] is ODD
    """
    N = 2 * x.shape[-1] - 1
    Xf = torch.fft.irfft(x, n=N)
    h = torch.zeros(N).to(x.device)
    h[0] = 1
    h[1:(N + 1) // 2] = 2
    x = torch.fft.rfft(Xf * h)
    return torch.imag(x)
