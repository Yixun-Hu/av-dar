"""
Config classes for the project.
Used for type hinting and validation.
We do not use this for config loading.
Instead, we use DictConfig.
"""

from dataclasses import dataclass
from .typing import *


@dataclass
class VisualizationConfig:
    wave_visualization: bool
    directivity_visualization: bool
    path_visualization: bool


@dataclass
class DatasetConfig:
    name: str
    scene_name: str
    path: str
    sample_rate: int
    rir_length: int
    options: Dict[str, Any]

@dataclass
class ObjectConfig:
    name: str
    options: Dict[str, Any]

@dataclass
class RendererConfig:
    frequency_min: int
    frequency_max: int
    frequency_num: int
    filter_length: int

    src_opts: ObjectConfig
    src_ir_opts: ObjectConfig
    spec_ir_opts: ObjectConfig
    late_ir_opts: ObjectConfig
    diffuse_ir_opts: ObjectConfig


@dataclass
class TrainConfig:
    exp_name: str

    start_bounce: int
    max_bounce: int
    
    start_growing_epoch: int
    growing_interval: int

    save_interval: Optional[int]
    eval_interval: Optional[int]
    vis_interval: Optional[int]

    decay_loss: Optional[bool]
    decay_loss_lambda: Optional[float]

    lambdas: Dict[str, float]

    pink_noise_supervision: bool
    pink_start_epoch: int
    lambda_pink: float

    n_epochs: int

    hrirs_cache_dir: str

    learning_rate: Dict[str, Any]
    batch_size: int
    shuffle: bool
    sampler_opts: ObjectConfig
    loss_opts: ObjectConfig
    clip_value: float

    renderer: str
    model: RendererConfig

    visualization: VisualizationConfig

    optimizer: str

    lr_scheduler: Optional[ObjectConfig] = None




@dataclass
class BaseConfig:
    dataset: DatasetConfig
    train: TrainConfig
    no_terminal: bool
    device: str
    working_dir: str
    tensorboard: str
    state_dict_path: str
    seed: int