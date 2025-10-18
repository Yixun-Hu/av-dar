import os

import torch
import torch.nn as nn

import numpy as np  
from sklearn.neighbors import NearestNeighbors

import logging

from ..utils.pcd_utils import pcd_downsample
from ..utils.nn_utils import SpatialEncoder, SpatialEncoderWithCovariance, KnnAttentionLayer

from ..core.typing import *
from ..model import module_registry

logger = logging.getLogger(__name__)

class MultiviewCrossAttnFeature(nn.Module):
    def __init__(self, 
                 voxels, 
                 voxel_size, 
                 voxel_features,
                 voxel_feature_scores,
                 voxel_feature_camera_ids,
                 extrinsics,
                 num_heads,
                 ## Key, Query
                 dim_key,
                 dim_value,
                 ## Spatial Dimensions
                 pe_order,
                 dtype=torch.float32):
        super(MultiviewCrossAttnFeature, self).__init__()

        ## Constants
        self.voxel_size = voxel_size
        self.num_heads = num_heads
        self.dtype = dtype

        ## Dimensions
        self.vision_feature_dim = voxel_features.shape[-1]
        self.spatial_input_dim = pe_order * 3
        self.vision_input_dim = self.vision_feature_dim + 16
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.out_channels = dim_value * num_heads

        ## Tensors
        self.voxels = torch.from_numpy(voxels).to(dtype)
        self.vmin, _ = torch.min(self.voxels, dim=0) # (3)
        self.vmax, _ = torch.max(self.voxels, dim=0) # (3)
        self.voxel_features = torch.from_numpy(voxel_features).to(dtype)
        self.voxel_feature_scores = torch.from_numpy(voxel_feature_scores).to(dtype)
        self.mask = torch.where(self.voxel_feature_scores < 0, -torch.inf, 0.0).to(dtype)

        print(self.mask.shape)
        print(self.voxel_features.shape)
        print(self.voxel_feature_scores.shape)
        # exit()
        # import
        self.extrinsics = torch.from_numpy(extrinsics.reshape(-1, 16)).to(dtype)

        ## Numpy Arrays
        self.voxel_feature_camera_ids = voxel_feature_camera_ids

        ## neural network layers
        self.spatial_encoder = SpatialEncoder(3, pe_order, 1.0, data_type=dtype)

        # Cross Attention
        self.weight_keys = nn.Linear(self.vision_input_dim, dim_key * num_heads, bias=False)
        self.weight_values = nn.Linear(self.vision_feature_dim, dim_value * num_heads, bias=False)
        self.weight_query = nn.Linear(self.spatial_input_dim, dim_key * num_heads, bias=False)

    def get_query(self, vox_xyzs):
        batch_size = vox_xyzs.shape[0]
        
        pe = self.spatial_encoder(vox_xyzs)
        
        query = self.weight_query(pe) # (B, H * D)
        query = query.view(batch_size, 1, self.num_heads, self.dim_key).permute(0, 2, 1, 3) # (B, H, N, D), N = 1
        
        return query
    
    def get_key_value(self, vox_indices):
        r'''
        Input:
            vox_indices: torch.Tensor, shape (B)
                - voxel indices
        '''
        batch_size = vox_indices.shape[0]
        vox_features = self.voxel_features[vox_indices] # (B, N, D)

        value = self.weight_values(vox_features) # (B, N, H * D)
        value = value.view(batch_size, -1, self.num_heads, self.dim_value).permute(0, 2, 1, 3) # (B, H, N, D)

        camera_ids = self.voxel_feature_camera_ids[vox_indices] # (B, N)
        # print(camera_ids.shape)
        extrinsics = self.extrinsics[camera_ids] # (B, N, 16)
        # print(extrinsics.shape)

        aug_vision_features = torch.cat([vox_features, extrinsics], dim=-1) # (B, N, D + 16)
        key = self.weight_keys(aug_vision_features) # (B, N, H * D)
        key = key.view(batch_size, -1, self.num_heads, self.dim_key).permute(0, 2, 1, 3) # (B, H, N, D)

        return key, value
    
    def normalize_xyz(self, xyzs):
        return (xyzs - self.vmin) / (self.vmax - self.vmin) - .5


    def move_to(self, device):
        '''
        Need Manual Move
        '''

        self.vmin = self.vmin.to(device)
        self.vmax = self.vmax.to(device)
        self.voxels = self.voxels.to(device)
        self.voxel_features = self.voxel_features.to(device)
        self.voxel_feature_scores = self.voxel_feature_scores.to(device)
        self.mask = self.mask.to(device)
        self.extrinsics = self.extrinsics.to(device)

    def cross_attention(self, query, key, value, mask = None):
        r'''
        Cross attention
        
        Input:
            query: torch.Tensor, shape (B, H, 1, D)
                - query
            key: torch.Tensor, shape (B, H, M, D)
                - key
            value: torch.Tensor, shape (B, H, M, Dv)
                - value
            mask: torch.Tensor, shape (B, H, 1, M)
                - mask
        '''

        if mask is None:
            mask = 0.0
        
        batch_size = query.shape[0]
        # print(query.shape, key.shape, mask.shape)
        kq = torch.matmul(query, key.permute(0, 1, 3, 2)) / np.sqrt(self.dim_key) + mask # (B, H, 1, M)
        kq = nn.functional.softmax(kq, dim=-1) # (B, H, 1, M)
        
        attn = torch.matmul(kq, value) # (B, H, 1, Dv)
        attn = attn.permute(0, 2, 1, 3).reshape(batch_size, -1) # (B, H * Dv)
        return attn
    

    def forward(self, vox_indices: torch.Tensor) -> torch.Tensor:
        r'''
        Input:
            xyz: torch.Tensor, shape (B, 1)
                - input voxel indices
        '''

        # Find nearest voxel
        vox_xyzs = self.voxels[vox_indices]
        vox_xyzs = (vox_xyzs - self.vmin) / (self.vmax - self.vmin) - .5

        query = self.get_query(vox_xyzs)
        key, value = self.get_key_value(vox_indices)

        mask = self.mask[vox_indices].unsqueeze(1).unsqueeze(1) # (B, 1, 1, M)

        # Cross Attention
        attn = self.cross_attention(query, key, value, mask)

        return attn


class PointTransformerBlock(nn.Module):
    def __init__(
        self, 
        input_ch: int, 
        output_ch: int, 
        num_hidden_layers: int, 
        hidden_dim: int, 
        k_neighbors: int, 
        num_heads: int, 
        dim_kqv: int, 
        res_connection: bool = True, 
        dtype: torch.dtype = torch.float32
    ):
        super(PointTransformerBlock, self).__init__()
        self.input_layer = nn.Linear(input_ch, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        self.num_hidden_layers = num_hidden_layers
        for i in range(num_hidden_layers):
            self.hidden_layers.append(
                KnnAttentionLayer(hidden_dim, hidden_dim, k_neighbors, num_heads, dim_kqv, dtype=dtype)
            )
        self.output_layer = nn.Linear(hidden_dim, output_ch)

        self.activation = nn.LeakyReLU()
        self.res_connection = res_connection

    def forward(
        self, 
        xyzs: Float[Tensor, "B N 3"],
        features: Float[Tensor, "B N C"],
        k_graph: Int[Tensor, "B N K"]
    ):
        '''
        : xyzs: [B, N, 3]
        : features: [B, N, in_channels]
        : k_graph: [N, k_neighbors] <- indices
        '''
        # import IPython; IPython.embed();

        features = self.input_layer(features) # (B, N, hidden_dim)
        for i in range(self.num_hidden_layers):
            res = features if self.res_connection else 0
            features = self.hidden_layers[i](xyzs, features, k_graph) + res
        out = self.output_layer(features)
        out = self.activation(out)
        return out

class MultiViewCrossAttentionVoxTransformerExtractor(nn.Module):
    def __init__(
        self, 
        out_channels: int,
        ## Voxel Features
        voxels_path: str,
        voxel_features_path: str,
        voxel_feature_scores_path: str,
        voxel_feature_camera_ids_path: str,
        extrinsics_path: str,
        voxel_size: float,
        ## Cross Attention
        num_heads: int,
        dim_key: int,
        dim_value: int,
        pe_order: int,
        ## Point Transformer
        k_neighbors: int,
        num_hidden_layers: int,
        use_res_connection: bool,
        dim_kqv_point: int,
        ## Query
        k_neighbors_query: int,
        ## Misc
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32
    ):
        super(MultiViewCrossAttentionVoxTransformerExtractor, self).__init__()
        voxels = np.load(voxels_path)
        voxel_features = np.load(voxel_features_path)
        voxel_feature_scores = np.load(voxel_feature_scores_path)
        voxel_feature_camera_ids = np.load(voxel_feature_camera_ids_path)
        extrinsics = np.load(extrinsics_path)

        self.dtype = dtype
        self.device = device
        self.out_channels = out_channels

        ## Nearst Neighbors
        self.nearest_voxels = NearestNeighbors(n_neighbors=k_neighbors_query, algorithm='auto').fit(voxels)

        nearest_k_voxels = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(voxels)
        knn_graph = nearest_k_voxels.kneighbors(voxels, return_distance=False) # (N, k_neighbors)
        self.knn_graph = knn_graph

        ## Sub-graph query
        neighbors = [{i, } for i in range(voxels.shape[0])]
        for i in range(num_hidden_layers):
            neighbors = [
                nbr | set(map(int, np.unique(knn_graph[list(nbr)].flatten())))
                for nbr in neighbors
            ]
        self.neighbor_voxels = neighbors

        ### NN Layers
        ## Cross Attention
        self.cross_attn = MultiviewCrossAttnFeature(
            voxels, 
            voxel_size, 
            voxel_features,
            voxel_feature_scores,
            voxel_feature_camera_ids,
            extrinsics,
            num_heads,
            dim_key,
            dim_value,
            pe_order,
            dtype=dtype
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.cross_attn.out_channels, self.cross_attn.out_channels),
            nn.LeakyReLU(),
            nn.Linear(self.cross_attn.out_channels, self.cross_attn.out_channels),
            nn.LeakyReLU(),
        )
                                 
        ## Point Transformer
        input_dim = self.cross_attn.out_channels
        self.transformer = PointTransformerBlock(
            input_dim, input_dim, num_hidden_layers, input_dim, k_neighbors, num_heads, dim_kqv_point, use_res_connection, dtype=dtype
        )

        self.gather_layer = KnnAttentionLayer(input_dim, out_channels, k_neighbors_query, num_heads, dim_kqv_point, dtype=dtype)
        self.activation = nn.LeakyReLU()

        v_range = voxels.max(axis=0) - voxels.min(axis=0)
        self.pe_encoder = SpatialEncoderWithCovariance(3, pe_order, v_range * 1.5, data_type=dtype)
        self.pe_input_proj = nn.Linear(pe_order * 3, input_dim)
        self.pe_output_proj = nn.Linear(pe_order * 3, out_channels)

        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.layer_norm_3 = nn.LayerNorm(out_channels)

        self.cached = False
        self.cached_voxel_features = None

    def set_cache(self, cached):
        if self.cached == cached:
            return
        
        if cached:
            assert self.training == False, 'Cannot cache during training'
            self.cross_attn.move_to(self.device)
            with torch.no_grad():
                self.cache_voxel_features = self.prepare_voxel_features(
                    torch.arange(self.cross_attn.voxels.shape[0], dtype=torch.long), torch.from_numpy(self.knn_graph).long()
                )[0]
        else:
            self.cache_voxel_features = None
        self.cached = cached    


    def prepare_voxel_features(self, queried_indices, sub_k_graph):
        voxel_features = self.cross_attn(queried_indices)
        voxel_features = self.ffn(voxel_features)
        vox_features = self.transformer.forward(
            self.cross_attn.voxels[queried_indices].unsqueeze(0), # [1, N, 3]
            voxel_features.unsqueeze(0), # [1, N, D]
            sub_k_graph.unsqueeze(0) # [1, N, k_neighbors]
        ) # [1, N, D] 


        return vox_features

    def forward(
        self, 
        xyzs: Float[Tensor, "B 3"],
        xyzs_std2: Float[Tensor, "B 3"]=None
    ) -> Float[Tensor, "B D"]:
        '''
        : xyzs: [B, 3]
        '''
        if xyzs.shape[0] == 0:
            return torch.zeros((0, self.out_channels), device=self.device)

        if xyzs.device != self.cross_attn.voxel_features.device:
            self.cross_attn.move_to(xyzs.device)

        query_indices = self.nearest_voxels.kneighbors(
            xyzs.cpu().numpy(), return_distance=False
        ) # (B, k_neighbors_query)

        all_queried_indices = set()
        for idx in query_indices.flatten():
            all_queried_indices |= self.neighbor_voxels[idx]
        all_queried_indices = torch.tensor(sorted(list(all_queried_indices)), dtype=torch.long)
        
        sub_graph_index_map = torch.zeros(self.cross_attn.voxels.shape[0], dtype=torch.long)
        sub_graph_index_map[all_queried_indices] = torch.arange(all_queried_indices.shape[0])
        sub_k_graph = sub_graph_index_map[self.knn_graph[all_queried_indices]] # (N, k_neighbors)
        sub_graph_query_indices = sub_graph_index_map[query_indices] # (B, k_neighbors_query)

        if not self.cached or self.cache_voxel_features is None:
            vox_features = self.prepare_voxel_features(all_queried_indices, sub_k_graph)
        else:
            vox_features = self.cache_voxel_features[all_queried_indices][None]
        
        pe = self.pe_encoder.forward(
            xyzs, xyzs_std2 if xyzs_std2 is not None else torch.zeros_like(xyzs)
        )
        xyz_pe_query = self.pe_input_proj.forward(pe)
        xyz_features = self.gather_layer.external_query_forward(
            xyzs.unsqueeze(0), # [1, B, 3],
            xyz_pe_query.unsqueeze(0), # [1, B, D],
            self.cross_attn.voxels[all_queried_indices].unsqueeze(0), # [1, N, 3],
            vox_features, # [1, N, D],
            sub_graph_query_indices.unsqueeze(0) # [1, B, k_neighbors]
        ).squeeze(0) # [B, D]
        out = self.activation(xyz_features)
        out = self.pe_output_proj(pe) + out
        return out


module_registry.add('mv_xformer_attn_only_feat_extractor', MultiViewCrossAttentionVoxTransformerExtractor, ['dtype', 'device'])