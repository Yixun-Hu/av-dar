import numpy as np

def pcd_downsample(xyzs, voxel_size, res=1000, center_mode='random'):
    """
    Downsamples the point cloud using voxel grid filtering
    """
    v_coods = np.floor(xyzs / voxel_size).astype(np.int64)

    voxel_id = v_coods[:, 0] * res * res * 4 + v_coods[:, 1] * res * 4 + v_coods[:, 2]

    unique_voxel_ids, unique_indices, rev_indices = np.unique(voxel_id, return_index=True, return_inverse=True)

    # center_mode \in ['random', 'mean']
    if center_mode == 'random':
        down_pcd = xyzs[unique_indices]
    elif center_mode == 'mean':
        down_pcd = np.zeros((len(unique_indices), 3))
        label_cnt = np.bincount(rev_indices)
        mask = label_cnt > 0
        down_pcd[mask, 0] = np.bincount(rev_indices, weights=xyzs[:, 0])[mask] / label_cnt[mask]
        down_pcd[mask, 1] = np.bincount(rev_indices, weights=xyzs[:, 1])[mask] / label_cnt[mask]
        down_pcd[mask, 2] = np.bincount(rev_indices, weights=xyzs[:, 2])[mask] / label_cnt[mask]
    else:
        raise ValueError('center_mode should be in [random  mean]')

    return down_pcd, rev_indices