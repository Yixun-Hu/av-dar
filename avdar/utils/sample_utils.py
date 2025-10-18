import torch
import math
import numpy as np

from typing import Tuple

from scipy.spatial.transform import Rotation as R

# borrowed and adapted (tensorized) from https://masonlwang.com/hearinganythinganywhere/
def fibonacci_sphere(n_samples):
    """Distributes n_samples on a unit fibonacci_sphere"""
    phi = math.pi * (math.sqrt(5.) - 1.)

    i = torch.arange(n_samples, dtype=torch.float32)
    y = 1 - i / (n_samples - 1) * 2
    radius = torch.sqrt(1 - y * y)
    theta = phi * i
    x = torch.cos(theta) * radius
    z = torch.sin(theta) * radius

    return torch.stack([x, y, z], dim=-1)

class CachedFibonacciSphere:
    cache = {}
    
    @classmethod
    def get(cls, n_samples):
        if n_samples not in cls.cache:
            cls.cache[n_samples] = fibonacci_sphere(n_samples).numpy().astype(np.float32)
        return cls.cache[n_samples]

def sample_points_from_surfaces(surfaces, n_samples):
    """
    Samples n_samples points from the surfaces
    """

    areas = [s.area() for s in surfaces]
    tot_area = np.sum(areas)
    points = [s._uniform_samples(int(n_samples * s.area() / tot_area + .5)) for s in surfaces]

    xyzs = np.concatenate(points, axis=0)
    normals = np.concatenate([np.tile(s.normal(), (len(points[i]), 1)) for i, s in enumerate(surfaces)], axis=0)
    surface_ids = np.concatenate([np.full(len(points[i]), i) for i in range(len(surfaces))], axis=0)

    xyzs = torch.from_numpy(xyzs)
    normals = torch.from_numpy(normals)
    surface_ids = torch.from_numpy(surface_ids)

    return xyzs, normals, surface_ids

def sample_uniform_sphere(n_samples):
    """Distributes n_samples on a unit sphere"""
    r = max(5, int(n_samples * 0.1 + .5))
    points = np.random.randn(n_samples + r, 3)
    mask = np.linalg.norm(points, axis=1) >  1e-5
    points = points[mask][:n_samples]
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    return points


def sample_fibonacci_sphere(n_samples, random_rotation=True):
    points = CachedFibonacciSphere.get(n_samples).astype(np.float32)
    if random_rotation:
        random_rotation = R.random().as_matrix().astype(np.float32)
        points = np.dot(points, random_rotation)
    return points

    
def generate_pink_noise(N, vol_factor = 0.04, freq_threshold=25, fs=48000):
    """
    Generates Pink Noise

    Parameters
    ----------
    N: length of audio in samples
    vol_factor: scaling factor to adjust volume to approximately match direct-line volume
    thres: frequency floor in hertz, below which the pink noise will not have any energy
    fs: sampling rate

    Returns
    -------
    pink_noise: (N,) generated pink noise
    """
    X_white = torch.fft.rfft(torch.randn(N))
    freqs = torch.fft.rfftfreq(N)
    
    normalized_freq_threshold = freq_threshold/fs
    
    pink_noise_spectrum = 1/torch.where(freqs<normalized_freq_threshold, float('inf'), torch.sqrt(freqs))
    pink_noise_spectrum = pink_noise_spectrum / torch.sqrt(torch.mean(pink_noise_spectrum**2))
    X_pink = X_white * pink_noise_spectrum
    pink_noise = torch.fft.irfft(X_pink*vol_factor)
    return pink_noise

def sample_sphere(n_samples, dtype, device):
    """
    Samples n_samples points from a unit sphere
    """
    u = torch.rand(n_samples, dtype=dtype, device=device)
    v = torch.rand(n_samples, dtype=dtype, device=device)

    theta = 2 * math.pi * u
    phi = torch.acos(2 * v - 1)

    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    xyz = torch.stack([x, y, z], dim=-1) # shape (n_samples, 3)
    return xyz

def sample_hemisphere(n_samples, dtype, device, normal=None):
    """
    Samples n_samples points from a unit hemisphere
    """
    u = torch.rand(n_samples, dtype=dtype, device=device)
    v = torch.rand(n_samples, dtype=dtype, device=device)

    theta = 2 * math.pi * u
    phi = torch.acos(v)

    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    xyz = torch.stack([x, y, z], dim=-1) # shape (n_samples, 3)

    if normal is not None:
        # Rotate the hemisphere to align with the normal z->normal
        if normal.dim() == 1:
            normal = normal.unsqueeze(0)
        t, bt = find_tangents(normal)
        xyz = t * xyz[:, :1] + bt * xyz[:, 1:2] + normal * xyz[:, 2:3]

    return xyz


def find_tangents(normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # adapted from nori
    x = normals[:, 0]
    y = normals[:, 1]
    z = normals[:, 2]
    z2 = z*z

    bt = torch.zeros_like(normals)

    mask = x.abs() > y.abs()
    inv_len = 1 / torch.sqrt(x[mask]*x[mask] + z2[mask])
    bt[mask, 0] = z[mask] * inv_len
    bt[mask, 2] = -x[mask] * inv_len

    mask = ~mask
    inv_len = 1 / torch.sqrt(y[mask]*y[mask] + z2[mask])
    bt[mask, 1] = z[mask] * inv_len
    bt[mask, 2] = -y[mask] * inv_len

    t = torch.cross(bt, normals, dim=-1)
    return t, bt


def sample_hemisphere_numpy(n_samples, dtype, normal: np.ndarray=None):
    """
    Samples n_samples points from a unit hemisphere
    """
    u = np.random.rand(n_samples).astype(dtype)
    v = np.random.rand(n_samples).astype(dtype)

    theta = 2 * np.pi * u
    phi = np.arccos(v)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    xyz = np.stack([x, y, z], axis=-1) # shape (n_samples, 3)

    if normal is not None:
        # Rotate the hemisphere to align with the normal z->normal
        if normal.ndim == 1:
            normal = normal.unsqueeze(0)
        t, bt = find_tangents_numpy(normal)
        xyz = t * xyz[:, :1] + bt * xyz[:, 1:2] + normal * xyz[:, 2:3]

    return xyz


def find_tangents_numpy(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # adapted from nori
    x = normals[:, 0]
    y = normals[:, 1]
    z = normals[:, 2]
    z2 = z*z

    bt = np.zeros_like(normals)

    mask = np.abs(x) > np.abs(y)
    inv_len = 1 / np.sqrt(x[mask]*x[mask] + z2[mask])
    bt[mask, 0] = z[mask] * inv_len
    bt[mask, 2] = -x[mask] * inv_len

    mask = ~mask
    inv_len = 1 / np.sqrt(y[mask]*y[mask] + z2[mask])
    bt[mask, 1] = z[mask] * inv_len
    bt[mask, 2] = -y[mask] * inv_len

    t = np.cross(bt, normals, axis=-1)
    return t, bt