from abc import ABC, abstractmethod
from torch.utils.data import Dataset

import pathlib
import numpy as np
import trimesh

from ..utils.logging_utils import getLogger
from ..utils.registry_utils import Registry
from ..utils.import_utils import import_children

logger = getLogger(__name__)

class AcousticDataset(Dataset, ABC):

    INIT_KEYS = ['name', 'scene_name', 'path']
    
    def __init__(self, name, scene_name, path):

        super().__init__()

        self.name = name
        self.scene_name = scene_name
        self.path = pathlib.Path(path)
        self.points: np.ndarray = None

    @abstractmethod
    def get_mesh(self) -> trimesh.Trimesh:
        raise NotImplementedError('get_mesh not implemented')
    
    @abstractmethod
    def get_mesh_path(self) -> str:
        raise NotImplementedError('get_mesh_path not implemented')

    def get_surface_xyzs(self):
        return self.points
    
    def init_points(self, n_points, cache_dir=None):
        points_path = None if cache_dir is None else pathlib.Path(cache_dir) / f'points_{n_points}.npy'
        if points_path is not None and points_path.exists():
            self.points = np.load(points_path)
            return

        logger.info(f'Initializing points...')
        
        mesh: trimesh.Trimesh = trimesh.load(self.get_mesh_path())
        n_faces = mesh.faces.shape[0]

        points = mesh.sample(n_points, return_index=False)
        self.points = points

        if cache_dir is not None:
            logger.info(f'Saving points to cache: {points_path}')
            points_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(points_path, self.points)


data_registry = Registry("data", AcousticDataset)
import_children(__file__, __name__)
