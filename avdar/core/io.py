import torch
import torch.optim as optim
import numpy as np

import logging

from .base_config import DatasetConfig, BaseConfig
from ..data import data_registry

from ..utils.io_utils import load_json
from ..model.renderer import RirRenderer

logger = logging.getLogger(__name__)

OPTIM_CLASS = {
    'Adam': optim.Adam,
    'AdamW': optim.AdamW,
    'SGD': optim.SGD,
    'RAdam': optim.RAdam,
}

def build_dataset(cfg: DatasetConfig, base_cfg: BaseConfig, cache_dir: str, inference_only: bool = False):
    dataset_cls = data_registry.get(cfg.name)
    opt = dict(
        name=cfg.name,
        scene_name=cfg.scene_name,
        path=cfg.path,
        cache_dir=cache_dir,
    )

    all_opt = { **cfg, **base_cfg , **{'cache_dir': cache_dir}}
    
    if 'options' in cfg:
        all_opt.update(cfg.options)

    if 'split_file' in all_opt:
        split_file = all_opt['split_file']
        split_cfg = load_json(split_file)
        opt['train_indices'] = split_cfg['train']
        opt['val_indices'] = split_cfg['val']
        opt['n_data'] = split_cfg['n_data']

    for key in dataset_cls.INIT_KEYS:
        if key in opt or key == 'split':
            continue

        opt[key] = all_opt.get(key, None)

    train_opt = {**opt, **{'split': 'train'}}
    val_opt = {**opt, **{'split': 'val'}}
    test_opt = {**opt, **{'split': 'test'}}

    if inference_only:
        inference_opt = {**opt, **{'split': 'inference'}}
        dataset_inference = dataset_cls(**inference_opt)
        return dataset_inference, dataset_inference, dataset_inference, dataset_inference

    dataset_train = dataset_cls(**train_opt)
    dataset_val = dataset_cls(**val_opt)
    dataset_test = dataset_cls(**test_opt)

    return dataset_train, dataset_val, dataset_test, None


def build_from_config(config: BaseConfig, working_dir: str, cache_dir: str, resume: bool = False, dataset_dict: dict = None, inference_only: bool = False):
    dataset_train, dataset_val, dataset_test, dataset_inference = build_dataset(config.dataset, config, cache_dir, inference_only=inference_only)
    
    output = {
        'dataset_train': dataset_train,
        'dataset_val': dataset_val,
        'dataset_test': dataset_test,
        'dataset_inference': dataset_inference,
    }
    
    output.update(
        build_refl_model_from_config(config, dataset_train, cache_dir, resume, output)
    )

    return output


def build_refl_model_from_config(
    config: BaseConfig, dataset_train, cache_dir: str, resume: bool = False, dataset_dict: dict = None
):

    mesh_scene = dataset_dict['dataset_train'].get_mesh()
    
    rir_renderer = RirRenderer(
        mesh_scene, dataset_train.speed_of_sound,
        config.dataset.sample_rate, rir_length=config.dataset.rir_length,  dtype=dataset_train.dtype, device=config.device, **config.train.model
    )
    rir_renderer.to(config.device)

    late_params = [p for name, p in rir_renderer.named_parameters() if 'late_model' in name]
    spec_params = [p for name, p in rir_renderer.named_parameters() if 'spec_model' in name]
    source_ir_params = [p for name, p in rir_renderer.named_parameters() if 'source_resp' in name]
    source_params = [p for name, p in rir_renderer.named_parameters() if 'source_dir_model' in name]
    feature_extractor_params = [p for name, p in rir_renderer.named_parameters() if 'pcd_feat_extractor' in name]
    diffuse_params = [p for name, p in rir_renderer.named_parameters() if 'diffuse_model' in name and 'basis' not in name]
    diffuse_basis_params = [p for name, p in rir_renderer.named_parameters() if 'diffuse_model' in name and 'basis' in name]
    base_params = [
        p for name, p in rir_renderer.named_parameters() 
        if ('late_model' not in name) \
            and ('spec_model' not in name) \
            and ('source_dir_model' not in name) \
            and ('source_resp' not in name) \
            and ('diffuse_model' not in name) \
            and ('pcd_feat_extractor' not in name)]
    
    optimizer_cls = OPTIM_CLASS[config.train.optimizer]

    if 'source_ir' not in config.train.learning_rate:
        logger.warning('No source learning rate specified, using base learning rate')
        config.train.learning_rate['source_ir'] = config.train.learning_rate['base']

    optimizer = optimizer_cls(
        [{'params': base_params}, 
         {'params': source_params, 'lr': config.train.learning_rate['source']}, 
         {'params': source_ir_params, 'lr': config.train.learning_rate['source_ir']},
         {'params': spec_params, 'lr': config.train.learning_rate['spec']}, 
         {'params': late_params, 'lr': config.train.learning_rate['late']}, 
         {'params': diffuse_params, 'lr': config.train.learning_rate['diffuse']},
         {'params': diffuse_basis_params, 'lr': config.train.learning_rate['diffuse_basis']},
         {'params': feature_extractor_params, 'lr': config.train.learning_rate['feature_extractor']},],
        lr=config.train.learning_rate['base'])

    if resume:
        checkpoint = torch.load(config.state_dict_path, map_location=config.device)
        rir_renderer.load_state_dict(checkpoint['model_state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            logger.warning('Optimizer state dict does not match model state dict, skipping')

    return {
        'rir_renderer': rir_renderer,
        'optimizer': optimizer,
    }