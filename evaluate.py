import torch
import torch.nn as nn
import torch.optim as optim

import pathlib

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import numpy as np

import datetime

from typing import List

from avdar.core.base_config import BaseConfig
from avdar.core.io import build_from_config
from avdar.core.run import train_step, eval_step_rir
from avdar.geometry.pathspace import SpecularPathSampler
from avdar.model.renderer import RirRenderer
from avdar.utils.loss_utils import RafC50Error, RafEdtError, RafT60Error, RafLoudnessError
from avdar.utils.io_utils import save_json

import argparse

import logging
from tqdm import tqdm


logger = logging.getLogger(__name__)


def main(config_dir, state_dict_name, device):
    with hydra.initialize_config_dir(config_dir=str(pathlib.Path(config_dir).absolute()), version_base="1.2"):
        config: BaseConfig = hydra.compose(config_name='config', overrides=[
            f'device={device}',
            f'state_dict_path={pathlib.Path(config_dir) / state_dict_name}',
            'no_terminal=False',
        ])
        
    cache_dir = pathlib.Path(config.working_dir) / (config.dataset.name + '_' + config.dataset.scene_name) 
    build_dict = build_from_config(config, working_dir=config.working_dir, cache_dir=cache_dir, resume=True)
    dataset_train = build_dict['dataset_train']
    dataset_val = build_dict['dataset_val']
    dataset_test = build_dict['dataset_test']
    rir_renderer: RirRenderer = build_dict['rir_renderer'].to(device)
    # optimizer = build_dict['optimizer']
    rir_renderer.eval()

    max_path_length = config.train.max_bounce

    print(f"using: {config.train['sampler_opts']['name']}")
    config.train['sampler_opts']['options']['num_sample_directions'] = 8192
    path_sampler = SpecularPathSampler.from_config(config.train['sampler_opts'], max_path_length, dataset_test.get_mesh_path())

    ## To avoid loading the dataset in the eval loop, we just iterate over it once
    print('Loading dataset...')
    for _ in tqdm(dataset_test):
        pass


    logger.info(f'Evaluation...')
    eval_loss_dict = eval_step_rir(
        config, rir_renderer, dataset_test, path_sampler, 
        {
            'C50': RafC50Error(dataset_test.sample_rate), 
            'EDT': RafEdtError(dataset_test.sample_rate), 
            'T60': RafT60Error(dataset_test.sample_rate, 20),
            'Loudness': RafLoudnessError(dataset_test.sample_rate),
        }
    )

    base_path = pathlib.Path(config_dir) / 'eval_results-{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    base_path.mkdir(parents=True, exist_ok=True)
    save_json(eval_loss_dict, base_path / f'eval_losses_{state_dict_name}.json')


parser = argparse.ArgumentParser(description='Evaluate a model')
parser.add_argument('--config_dir', type=str, help='Path to the config directory', required=True)
parser.add_argument('--state_dict_name', type=str, help='Name of the state dict file', default='weight_final.pt')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
if __name__ == '__main__':
    args = parser.parse_args()
    config_dir = args.config_dir
    state_dict_name = args.state_dict_name
    device = args.device
    main(config_dir, state_dict_name, device)